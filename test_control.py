import argparse
from typing import Any, Dict
from visualization_tools import show_rgb_depth, create_tracking_video
from obstacle_detection import get_ball_position, ball_kalman_update
from control import get_actions, quat_error
from grasping import generate_and_select_grasps
import numpy as np
import yaml
from tqdm import tqdm
import open3d as o3d
import mujoco

from mujoco_app.mj_simulation import MjSim
import xml.etree.ElementTree as ET
import os


def get_mesh_path_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the first <mesh> tag in <asset>
    mesh_tag = root.find(".//asset/mesh")
    if mesh_tag is not None:
        relative_path = mesh_tag.get("file")
        # Convert relative path to absolute
        xml_dir = os.path.dirname(os.path.abspath(xml_path))
        return os.path.normpath(os.path.join(xml_dir, relative_path))

    raise ValueError("No mesh file found in the provided XML.")


# get a quaternion that points from the current pos to the target pos to prevent weird poses we could encounter when sampling orientations
def get_lookat_quat(current_pos, target_pos):
    direction = target_pos - current_pos
    direction /= np.linalg.norm(direction)

    up = np.array([0, 0, 1])

    x_axis = np.cross(up, direction)
    x_axis /= np.linalg.norm(x_axis) + 1e-6
    y_axis = np.cross(direction, x_axis)

    rot_matrix = np.stack([x_axis, y_axis, direction], axis=1)

    target_quat = np.zeros(4)
    mujoco.mju_mat2Quat(target_quat, rot_matrix.flatten())

    return target_quat

def runner(config: Dict[str, Any], num_experiments: int):
    cam_cfg = config.get("mujoco", {}).get("camera", {})
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)
    near = cam_cfg.get("near", 0.01)
    far = cam_cfg.get("far", 5.0)
    fovy = cam_cfg.get("fovy", 58.0)
    xml_path = config.get("mujoco", {}).get("grasp_object", {}).get("xml", "")
    model_path = get_mesh_path_from_xml(xml_path)
    print(model_path)

    pos_errors = []
    rot_errors = []

    sim = MjSim(config)
    for _ in range(num_experiments):
        sim.reset()

        # For sim stabilization
        for _ in range(1000):
            sim.step()

        rgb, depth, intrinsic, extrinsic = sim.render_camera(
            "static",
            width=width,
            height=height,
            near=near,
            far=far,
            fovy=fovy,
        )

        desired_poses = []
        images = []

        site_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, "fingertip")
        initial_ee_pos = sim.data.site_xpos[site_id].copy()
        ee_pos = sim.data.site_xpos[site_id].copy()
        desired_pos = initial_ee_pos + [0, 0, 0.1]
        desired_quat = get_lookat_quat(desired_pos, ee_pos)

        for t in tqdm(range(300), dynamic_ncols=True):
            sim.step()

            images.append(rgb)
            desired_poses.append(desired_pos.copy())

            rgb, depth, intrinsic, extrinsic = sim.render_camera(
                "static",
                width=width,
                height=height,
                near=near,
                far=far,
                fovy=fovy,
            )
            q_indices = [qidx for qidx, _ in sim.robot.arm_pairs]
            q = np.array([sim.data.qpos[qidx] for qidx in q_indices])
            joint_positions = get_actions(
                sim, desired_pos, desired_quat, 0.1, 0.001, dt=0.05
            )
            joint_positions[-1] = q[-1]
            sim.set_arm_joint_positions(joint_positions, clamp=True, sync=True)

            if t % 25 == 0 and t > 0:
                ee_pos = sim.data.site_xpos[site_id].copy()
                current_quat = np.zeros(4)
                mujoco.mju_mat2Quat(current_quat, sim.data.site_xmat[site_id])
                pos_errors.append(np.linalg.norm(desired_pos - ee_pos))
                rot_errors.append(quat_error(desired_quat, current_quat))

                desired_pos = initial_ee_pos + np.random.uniform(0.00, 0.1, (3,)) + [0, 0, 0.1]


    estimates = {"desired": desired_poses}

    print("Position error (mean, std)", f"({np.array(pos_errors).mean()}, {np.array(pos_errors).std()})")
    print("Rotation error (mean, std)", f"({np.array(rot_errors).mean()}, {np.array(rot_errors).std()})")

    create_tracking_video(
        images,
        estimates,
        intrinsic,
        extrinsic,
        output_path="tracking_output.mp4",
        fps=20,
    )

    sim.close()
    print("Simulation completed.")


def main(config_path: str):
    # Changing the object wouldnt make sense here because this script is object agnostic anyway
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if not os.path.exists("visualizations/"):
        os.mkdir("visualizations")
    # You can make runner for one experiment
    runner(config, 10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/test_config_mj.yaml")
    args = parser.parse_args()
    main(config_path=args.config)

