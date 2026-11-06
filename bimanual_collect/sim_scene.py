"""Check and picture the dual-arm teleop scene in MuJoCo without a Quest or an arm.

Builds the same combined scene as `teleop/scripts/run_dual_sim.py --tabletop` (both arms at the profile's
measured base transforms + the collidable tabletop), runs the same offline path checks, and renders two
views at home pose from where the operator stands: eye level and higher up.

    python -m bimanual_collect sim-scene
    python -m bimanual_collect sim-scene --table-width-m 1.2 --table-depth-m 0.8
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from widowxai_quest_teleop.config import DUAL_ARM_SIDES, load_config, parse_dual_arm_config  # noqa: E402
from widowxai_quest_teleop.dual_arm_model import DualArmCollisionModel  # noqa: E402
from widowxai_quest_teleop.dual_arm_preflight import offline_checks  # noqa: E402

from . import rig  # noqa: E402

TELEOP = rig.load().teleop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=rig.load().arm_profile, help="two-arm teleop profile")
    parser.add_argument("--table-width-m", type=float, default=1.0)
    parser.add_argument("--table-depth-m", type=float, default=0.7)
    parser.add_argument("--table-thickness-m", type=float, default=0.04)
    parser.add_argument("--base-rear-inset-m", type=float, default=0.0508)
    parser.add_argument("--out", type=Path, default=rig.load().state / "sim_scene.png")
    args = parser.parse_args()
    # Resolve our own paths first: after the chdir, a relative --out would land inside the teleop repo.
    args.config, args.out = args.config.resolve(), args.out.resolve()

    os.chdir(TELEOP)  # the profile's model and calibration paths are relative to the repo
    config = load_config(str(args.config))
    config["simulation_environment"] = {
        "tabletop": {
            "width_m": args.table_width_m,
            "depth_m": args.table_depth_m,
            "thickness_m": args.table_thickness_m,
            "rear_edge_x_m": -args.base_rear_inset_m,
            "top_z_m": 0.0,
        }
    }
    arms_config = parse_dual_arm_config(config)
    offline_checks(config, arms_config)

    scene = DualArmCollisionModel(
        {side: arms_config[side].placement for side in DUAL_ARM_SIDES},
        xml_path=config["model"]["xml_path"],
        clearance_m=float(config["safety"]["cross_arm_clearance_m"]),
        tabletop=config["simulation_environment"]["tabletop"],
    )
    home = np.asarray(config["model"]["simulation_start_q_rad"], dtype=float)
    data = mujoco.MjData(scene.model)
    gripper = float(config["hardware"].get("gripper_open_m", 0.04))
    scene.set_viewer_qpos(data, {side: home for side in DUAL_ARM_SIDES}, {side: gripper for side in DUAL_ARM_SIDES})

    # Picture only: the repo's floor plane sits at the table-top height (z = 0) and hides the table from low
    # angles. Hide it in this rendered copy (the checks above use their own models).
    scene.model.geom_rgba[mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")][3] = 0.0
    scene.model.vis.global_.offwidth = max(scene.model.vis.global_.offwidth, 960)
    scene.model.vis.global_.offheight = max(scene.model.vis.global_.offheight, 720)
    renderer = mujoco.Renderer(scene.model, height=720, width=960)
    views = []
    # Azimuth 0 looks along +X (robot forward), i.e. from where the operator stands behind the arms.
    # Not straight down: the repo's floor and tabletop top share z = 0 and flicker from above.
    for azimuth, elevation, distance in ((0.0, -25.0, 1.6), (0.0, -62.0, 1.45)):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.25, 0.0, 0.08]
        camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
        renderer.update_scene(data, camera)
        views.append(renderer.render().copy())
    renderer.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(views, axis=1)).save(args.out)
    print(f"bases: left {arms_config['left'].placement.position_m.tolist()} "
          f"right {arms_config['right'].placement.position_m.tolist()} m; "
          f"table {args.table_width_m} x {args.table_depth_m} m")
    print(f"rendered home pose (operator eye level | higher) -> {args.out}")


if __name__ == "__main__":
    main()
