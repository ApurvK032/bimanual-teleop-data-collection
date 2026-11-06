from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np
from websockets.sync.client import connect

from widowxai_quest_teleop.config import load_config
from widowxai_quest_teleop.model import WidowXAIModel
from widowxai_quest_teleop.viewer import close_passive_viewer


def parse_ik_state(raw: str, from_id: str | None = None) -> tuple[np.ndarray, float] | None:
    try:
        payload = json.loads(raw)
        if payload.get("type") != "ik_state":
            return None
        if from_id is not None and payload.get("teleop_id") != from_id:
            return None
        q_arm = np.asarray(payload["q_arm"], dtype=float).reshape(6)
        gripper_q = float(payload["gripper_q"])
        if not np.all(np.isfinite(q_arm)) or not np.isfinite(gripper_q):
            return None
        return q_arm, gripper_q
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Passive WidowXAI MuJoCo viewer for broadcast IK state"
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--url", help="defaults to quest.websocket_url in the config")
    parser.add_argument("--from-id", help="only render this teleop label")
    args = parser.parse_args()

    config = load_config(args.config)
    url = args.url or config["quest"]["websocket_url"]
    robot = WidowXAIModel(config["model"]["xml_path"])
    data = mujoco.MjData(robot.model)
    q_start = robot.clamp_joints(
        np.asarray(config["model"]["simulation_start_q_rad"], dtype=float)
    )
    robot.set_viewer_qpos(data, q_start, 0.044)

    from mujoco import viewer as mujoco_viewer

    viewer = mujoco_viewer.launch_passive(robot.model, data)
    try:
        viewer.sync()
        print(f"viewer connected to {url}; listening for ik_state")
        with connect(url, open_timeout=2.0, close_timeout=1.0, compression=None) as websocket:
            while viewer.is_running():
                latest = None
                try:
                    raw = websocket.recv(timeout=0.1)
                except TimeoutError:
                    viewer.sync()
                    continue
                state = parse_ik_state(raw, args.from_id)
                if state is not None:
                    latest = state
                # Rendering is display-rate; discard intermediate messages and
                # apply the newest available 200 Hz state, as a passive viewer.
                for _ in range(63):
                    try:
                        raw = websocket.recv(timeout=0.0)
                    except TimeoutError:
                        break
                    state = parse_ik_state(raw, args.from_id)
                    if state is not None:
                        latest = state
                if latest is not None:
                    robot.set_viewer_qpos(data, latest[0], latest[1])
                viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        close_passive_viewer(viewer)


if __name__ == "__main__":
    main()
