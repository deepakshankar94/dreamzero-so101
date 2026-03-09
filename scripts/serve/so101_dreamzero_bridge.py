#!/usr/bin/env python3

"""Bridge a local SO101 follower arm to a DreamZero high_camera_updated websocket server.

Example:
    python scripts/serve/so101_dreamzero_bridge.py \
      --server-host 127.0.0.1 \
      --server-port 8000 \
      --robot-port /dev/ttyACM0 \
      --robot-id my_so101 \
      --overhead-camera /dev/video2 \
      --arm-camera /dev/video0 \
      --task "pick white cuboid and place on blue notepad"
"""

from __future__ import annotations

import argparse
import logging
import time
import uuid
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_utils.policy_client import WebsocketClientPolicy


LOGGER = logging.getLogger(__name__)

DEFAULT_MOTOR_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _import_lerobot():
    try:
        from lerobot.robots import make_robot_from_config, so101_follower
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        return make_robot_from_config, so101_follower.SO101FollowerConfig, OpenCVCameraConfig
    except ImportError:
        from lerobot.common.robots import make_robot_from_config, so101_follower
        from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        return make_robot_from_config, so101_follower.SO101FollowerConfig, OpenCVCameraConfig


def _maybe_import_cv2():
    try:
        import cv2

        return cv2
    except ImportError:
        return None


def _to_hwc_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {arr.shape}")

    # Convert CHW to HWC if needed.
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


class SO101DreamZeroBridge:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cv2 = _maybe_import_cv2()

        make_robot_from_config, so101_follower_config, open_cv_camera_config = _import_lerobot()

        camera_cfg = {
            args.overhead_camera_key: open_cv_camera_config(
                index_or_path=args.overhead_camera,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
            ),
            args.arm_camera_key: open_cv_camera_config(
                index_or_path=args.arm_camera,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
            ),
        }

        robot_cfg = so101_follower_config(
            port=args.robot_port,
            id=args.robot_id,
            cameras=camera_cfg,
        )
        self.robot = make_robot_from_config(robot_cfg)
        self.client = WebsocketClientPolicy(host=args.server_host, port=args.server_port)
        self.session_id = args.session_id or str(uuid.uuid4())
        self.motor_keys: list[str] | None = None

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        if not self.args.resize_before_send:
            return image

        if self.cv2 is None:
            raise RuntimeError("OpenCV is required for --resize-before-send.")

        target_h = self.args.send_height
        target_w = self.args.send_width
        if image.shape[0] == target_h and image.shape[1] == target_w:
            return image

        return self.cv2.resize(image, (target_w, target_h), interpolation=self.cv2.INTER_AREA)

    def _infer_motor_keys(self, observation: dict[str, Any]) -> list[str]:
        if hasattr(self.robot, "_motors_ft"):
            keys = list(self.robot._motors_ft.keys())
            if len(keys) == 6:
                return keys

        keys = [k for k in DEFAULT_MOTOR_KEYS if k in observation]
        if len(keys) == 6:
            return keys

        observed_pos_keys = [k for k in observation.keys() if k.endswith(".pos")]
        if len(observed_pos_keys) == 6:
            return observed_pos_keys

        raise ValueError(
            "Could not infer SO101 motor key order from observation. "
            f"Available keys: {sorted(observation.keys())}"
        )

    def _build_request(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.motor_keys is None:
            self.motor_keys = self._infer_motor_keys(observation)
            LOGGER.info("Motor key order: %s", self.motor_keys)

        overhead = _to_hwc_uint8(observation[self.args.overhead_camera_key])
        grey_arm = _to_hwc_uint8(observation[self.args.arm_camera_key])
        overhead = self._resize_if_needed(overhead)
        grey_arm = self._resize_if_needed(grey_arm)

        state = np.array([observation[k] for k in self.motor_keys], dtype=np.float32)
        joint_position = state[:5]
        gripper_position = state[5:6]

        return {
            "observation/overhead": overhead,
            "observation/grey_arm": grey_arm,
            "observation/joint_position": joint_position,
            "observation/gripper_position": gripper_position,
            "prompt": self.args.task,
            "session_id": self.session_id,
        }

    def _build_robot_action(
        self,
        action_chunk: dict[str, Any],
        idx: int,
    ) -> dict[str, float]:
        assert self.motor_keys is not None

        joint = np.asarray(action_chunk["action.joint_position"], dtype=np.float32)[idx].reshape(-1)
        gripper = np.asarray(action_chunk["action.gripper_position"], dtype=np.float32)[idx].reshape(-1)
        if joint.shape[0] != 5:
            raise ValueError(f"Expected 5 joint actions, got shape {joint.shape}")
        if gripper.shape[0] != 1:
            raise ValueError(f"Expected 1 gripper action, got shape {gripper.shape}")

        action = np.concatenate([joint, gripper], axis=0)
        return {motor_key: float(action[i]) for i, motor_key in enumerate(self.motor_keys)}

    def run(self) -> None:
        LOGGER.info(
            "Connecting SO101 %s on %s to DreamZero server %s:%s",
            self.args.robot_id,
            self.args.robot_port,
            self.args.server_host,
            self.args.server_port,
        )
        LOGGER.info("Server metadata: %s", self.client.get_server_metadata())
        LOGGER.info("Session id: %s", self.session_id)

        self.robot.connect()
        try:
            while True:
                observation = self.robot.get_observation()
                request = self._build_request(observation)
                action_chunk = self.client.infer(request)

                available_steps = len(np.asarray(action_chunk["action.joint_position"]))
                steps_to_execute = min(self.args.actions_per_infer, available_steps)

                for idx in range(steps_to_execute):
                    robot_action = self._build_robot_action(action_chunk, idx)
                    self.robot.send_action(robot_action)
                    time.sleep(self.args.action_sleep_s)
        finally:
            try:
                self.client.reset({"session_id": self.session_id})
            except Exception as exc:
                LOGGER.warning("Failed to reset remote session cleanly: %s", exc)

            if hasattr(self.robot, "disconnect"):
                self.robot.disconnect()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--robot-port", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--overhead-camera", required=True)
    parser.add_argument("--arm-camera", required=True)
    parser.add_argument("--overhead-camera-key", default="overhead")
    parser.add_argument("--arm-camera-key", default="grey_arm")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--resize-before-send", action="store_true")
    parser.add_argument("--send-width", type=int, default=320)
    parser.add_argument("--send-height", type=int, default=176)
    parser.add_argument("--task", required=True)
    parser.add_argument("--session-id")
    parser.add_argument(
        "--actions-per-infer",
        type=int,
        default=1,
        help="How many actions from each returned chunk to execute before requesting a new chunk.",
    )
    parser.add_argument(
        "--action-sleep-s",
        type=float,
        default=0.10,
        help="Sleep after each executed action.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_arg_parser().parse_args()
    bridge = SO101DreamZeroBridge(args)
    bridge.run()


if __name__ == "__main__":
    main()
