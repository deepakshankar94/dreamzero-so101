#!/usr/bin/env python3

import dataclasses
import logging
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import tyro
from tianshou.data import Batch
from torch.distributed.device_mesh import init_device_mesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_utils.policy_server import PolicyServerConfig, WebsocketPolicyServer
from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from openpi_client.base_policy import BasePolicy


LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "/root/dreamzero/checkpoints/dreamzero_high_camera_updated_run1"
    attention_backend: str = "FA2"
    enable_dit_cache: bool = False
    dynamo_recompile_limit: int = 800
    dynamo_cache_size_limit: int = 1000


class HighCameraUpdatedPolicy(BasePolicy):
    FRAMES_PER_CHUNK = 4

    def __init__(self, policy: GrootSimPolicy):
        self._policy = policy
        self._frame_buffers: dict[str, list[np.ndarray]] = {
            "video.overhead": [],
            "video.grey_arm": [],
        }
        self._current_session_id: str | None = None
        self._is_first_call = True

    def _reset_model_cache(self) -> None:
        action_head = getattr(self._policy.trained_model, "action_head", None)
        if action_head is None:
            return

        action_head.current_start_frame = 0
        action_head.language = None
        action_head.clip_feas = None
        action_head.ys = None
        action_head.kv_cache1 = None
        action_head.kv_cache_neg = None
        action_head.crossattn_cache = None
        action_head.crossattn_cache_neg = None

    def _reset_state(self) -> None:
        for key in self._frame_buffers:
            self._frame_buffers[key] = []
        self._is_first_call = True
        self._reset_model_cache()

    def _get_array(self, obs: dict[str, Any], keys: list[str]) -> np.ndarray | None:
        for key in keys:
            if key in obs:
                value = obs[key]
                if torch.is_tensor(value):
                    value = value.detach().cpu().numpy()
                return np.asarray(value)
        return None

    def _get_language(self, obs: dict[str, Any]) -> str:
        for key in ("prompt", "annotation.task_index", "annotation.language.action_text"):
            if key in obs:
                value = obs[key]
                if isinstance(value, np.ndarray):
                    value = value.item() if value.size == 1 else value[0]
                if isinstance(value, list):
                    value = value[0]
                return str(value)
        return ""

    def _append_frames(self, obs: dict[str, Any]) -> None:
        camera_aliases = {
            "video.overhead": ["observation/overhead", "video.overhead", "observation/exterior_image_0_left"],
            "video.grey_arm": ["observation/grey_arm", "video.grey_arm", "observation/exterior_image_1_left"],
        }

        for target_key, aliases in camera_aliases.items():
            frames = self._get_array(obs, aliases)
            if frames is None:
                continue
            if frames.ndim == 3:
                self._frame_buffers[target_key].append(frames)
            elif frames.ndim == 4:
                self._frame_buffers[target_key].extend(list(frames))
            else:
                raise ValueError(
                    f"{target_key} must be HWC or THWC, got shape {frames.shape}"
                )

    def _build_video_obs(self) -> dict[str, np.ndarray]:
        num_frames = 1 if self._is_first_call else self.FRAMES_PER_CHUNK
        converted: dict[str, np.ndarray] = {}

        for key, buffer in self._frame_buffers.items():
            if not buffer:
                raise ValueError(f"Missing camera stream for {key}")

            if len(buffer) >= num_frames:
                frames_to_use = buffer[-num_frames:]
            else:
                frames_to_use = buffer.copy()
                while len(frames_to_use) < num_frames:
                    frames_to_use.insert(0, buffer[0])

            converted[key] = np.stack(frames_to_use, axis=0).astype(np.uint8)

        return converted

    def _build_state_obs(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        joint_position = self._get_array(
            obs,
            ["observation/joint_position", "state.joint_position"],
        )
        gripper_position = self._get_array(
            obs,
            ["observation/gripper_position", "state.gripper_position"],
        )

        if joint_position is None or gripper_position is None:
            combined_state = self._get_array(obs, ["observation/state", "state"])
            if combined_state is not None:
                combined_state = np.asarray(combined_state, dtype=np.float32).reshape(-1)
                if combined_state.shape[0] >= 6:
                    joint_position = combined_state[:5]
                    gripper_position = combined_state[5:6]

        if joint_position is None or gripper_position is None:
            raise ValueError("Expected joint/gripper state in observation")

        joint_position = np.asarray(joint_position, dtype=np.float32).reshape(-1)
        gripper_position = np.asarray(gripper_position, dtype=np.float32).reshape(-1)

        if joint_position.shape[0] != 5:
            raise ValueError(f"Expected 5 joint values, got shape {joint_position.shape}")
        if gripper_position.shape[0] != 1:
            raise ValueError(f"Expected 1 gripper value, got shape {gripper_position.shape}")

        return {
            "state.joint_position": joint_position.reshape(1, 5),
            "state.gripper_position": gripper_position.reshape(1, 1),
        }

    def _convert_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._append_frames(obs)

        converted = self._build_video_obs()
        converted.update(self._build_state_obs(obs))
        converted["annotation.task_index"] = self._get_language(obs)
        return converted

    def _to_numpy_tree(self, value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        if isinstance(value, Batch):
            return {k: self._to_numpy_tree(v) for k, v in value.items()}
        if isinstance(value, dict):
            return {k: self._to_numpy_tree(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_numpy_tree(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._to_numpy_tree(v) for v in value)
        return value

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        session_id = obs.get("session_id")
        if session_id is not None and session_id != self._current_session_id:
            self._reset_state()
            self._current_session_id = session_id

        converted_obs = self._convert_observation(obs)
        result_batch, _ = self._policy.lazy_joint_forward_causal(Batch(obs=converted_obs))

        action_dict = self._to_numpy_tree(result_batch.act)
        self._is_first_call = False
        return action_dict

    def reset(self, reset_info: dict[str, Any]) -> None:
        self._current_session_id = reset_info.get("session_id")
        self._reset_state()


def _init_mesh() -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to serve DreamZero checkpoints.")

    if not dist.is_initialized():
        dist.init_process_group("nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("ip",),
    )


def main(args: Args) -> None:
    os.environ["ATTENTION_BACKEND"] = args.attention_backend
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"

    # These compiled scheduler/update paths specialize on per-step Python ints
    # such as step_index/order during autoregressive sampling. That creates many
    # valid graphs even with fixed client-side tensor shapes.
    torch._dynamo.config.recompile_limit = args.dynamo_recompile_limit
    torch._dynamo.config.cache_size_limit = args.dynamo_cache_size_limit

    device_mesh = _init_mesh()
    rank = dist.get_rank()

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.HIGH_CAMERA_UPDATED,
        model_path=args.model_path,
        device="cuda",
        device_mesh=device_mesh,
    )

    server_config = PolicyServerConfig(
        # Advertise the raw camera resolution expected by the eval transform.
        image_resolution=(480, 640),
        needs_wrist_camera=False,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,
        action_space="joint_position",
    )

    if rank != 0:
        LOGGER.info("Single-process serving expected; rank %s idling.", rank)
        dist.barrier()
        return

    server = WebsocketPolicyServer(
        policy=HighCameraUpdatedPolicy(policy),
        server_config=server_config,
        host=args.host,
        port=args.port,
    )
    LOGGER.info("Serving %s on %s:%s", args.model_path, args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
