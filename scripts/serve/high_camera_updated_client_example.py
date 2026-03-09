#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_utils.policy_client import WebsocketClientPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="pick white cuboid and place on blue notepad")
    parser.add_argument("--session-id", default="so101-session-001")
    args = parser.parse_args()

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    print("Server metadata:", client.get_server_metadata())

    overhead = np.zeros((176, 320, 3), dtype=np.uint8)
    grey_arm = np.zeros((176, 320, 3), dtype=np.uint8)
    joint_position = np.zeros(5, dtype=np.float32)
    gripper_position = np.zeros(1, dtype=np.float32)

    obs = {
        "observation/overhead": overhead,
        "observation/grey_arm": grey_arm,
        "observation/joint_position": joint_position,
        "observation/gripper_position": gripper_position,
        "prompt": args.prompt,
        "session_id": args.session_id,
    }

    action = client.infer(obs)
    print("Action keys:", list(action.keys()))
    for key, value in action.items():
        print(key, np.asarray(value).shape)

    client.reset({"session_id": args.session_id})


if __name__ == "__main__":
    main()
