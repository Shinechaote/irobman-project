import argparse
from typing import Any, Dict
from visualization_tools import show_rgb_depth, create_tracking_video
from pose_estimation import segment_and_align
from obstacle_detection import get_ball_position, ball_kalman_update

import numpy as np
import yaml
from tqdm import tqdm
import open3d as o3d

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

# Experiment runner example
def runner(config: Dict[str, Any], num_experiments: int):
    # Configure in YAML
    cam_cfg = config.get("mujoco", {}).get("camera", {})
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)
    near = cam_cfg.get("near", 0.01)
    far = cam_cfg.get("far", 5.0)
    fovy = cam_cfg.get("fovy", 58.0)
    xml_path = config.get("mujoco", {}).get("grasp_object", {}).get("xml", "")
    model_path = get_mesh_path_from_xml(xml_path)
    print(model_path)

    sim = MjSim(config)
    for _ in range(num_experiments):
        sim.reset()

        # For sim stabilization
        for _ in tqdm(range(1000), dynamic_ncols=True):
            sim.step()

        rgb, depth, intrinsic, extrinsic = sim.render_camera(
            "static",
            width=width,
            height=height,
            near=near,
            far=far,
            fovy=fovy,
        )

        red_range = (np.array([0, 150, 50]), np.array([10, 255, 255]))
        orange_range = (np.array([11, 150, 50]), np.array([25, 255, 255]))
        red_pos = get_ball_position(rgb, depth, intrinsic, extrinsic, red_range, visualize=False)
        orange_pos = get_ball_position(rgb, depth, intrinsic, extrinsic, orange_range, visualize=False)
        red_state_world, orange_state_world = np.zeros(6), np.zeros(6)
        red_cov, orange_cov = np.eye(6), np.eye(6)

        red_state_world[:3] = red_pos
        orange_state_world[:3] = orange_pos

        pred_red = []
        pred_orange = []
        images = []

        pose, _ = segment_and_align(
            rgb, depth, intrinsic, model_path, depth_scale=1.0)

        # lower iterations per step for reaching the target pose
        print("Moving to target pose...")
        for t in tqdm(range(1000), dynamic_ncols=True):
            sim.step()

            images.append(rgb)
            pred_red.append(red_state_world[:3])
            pred_orange.append(orange_state_world[:3])

            # Showcasing some operations that can be done with the simulation
            rgb, depth, intrinsic, extrinsic = sim.render_camera(
                "static",
                width=width,
                height=height,
                near=near,
                far=far,
                fovy=fovy,
            )
            red_state_world, red_cov = ball_kalman_update(rgb, depth, intrinsic, extrinsic, red_range, red_state_world, red_cov)
            orange_state_world, orange_cov = ball_kalman_update(rgb, depth, intrinsic, extrinsic, orange_range, orange_state_world, orange_cov)

            # Account for that we can only measure the distance to the surface of the ball and thus need to subtract the radius
            cam_pos = np.linalg.inv(extrinsic)[:3, 3]
            ball_radius = 0.06 
            red_ray_direction = cam_pos - red_state_world[:3]
            red_ray_direction /= np.linalg.norm(red_ray_direction)

            orange_ray_direction = cam_pos - orange_state_world[:3]
            orange_ray_direction /= np.linalg.norm(orange_ray_direction)

            # These are the estimated centers of the balls in the world frame
            red_estimated_center_world = red_state_world[:3] - (red_ray_direction * ball_radius)
            orange_estimated_center_world = orange_state_world[:3] - (orange_ray_direction * ball_radius)

            # Robot should not collide with obstacles
            # This condition must be there
            if sim.check_robot_obstacle_collision():
                print("Collision!")
                break
    
    estimates = {"red": pred_red, "orange": pred_orange}
    create_tracking_video(images, estimates, intrinsic, extrinsic, output_path="tracking_output.mp4", fps=100)

    sim.close()
    print("Simulation completed.")


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # You can make runner for one experiment
    runner(config, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/test_config_mj.yaml"
    )
    args = parser.parse_args()
    main(config_path=args.config)
