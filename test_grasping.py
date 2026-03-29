from pose_estimation import estimate_pose
from solve import get_mesh_path_from_xml
from tqdm import tqdm
import yaml
import os
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from mujoco_app.mj_simulation import MjSim
import cv2
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from grasping import generate_and_select_grasps

def runner(config, num_experiments, object_name="a"):
    # Configure in YAML
    cam_cfg = config.get("mujoco", {}).get("camera", {})
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)
    near = cam_cfg.get("near", 0.01)
    far = cam_cfg.get("far", 5.0)
    fovy = cam_cfg.get("fovy", 58.0)
    xml_path = config.get("mujoco", {}).get("grasp_object", {}).get("xml", "")
    model_path = get_mesh_path_from_xml(xml_path)

    forces = []

    sim = MjSim(config)
    for i in tqdm(range(num_experiments), dynamic_ncols=True):
        sim.reset()

        # For sim stabilization
        for _ in range(1000):
            sim.step()

        rgb, depth, intrinsic, extrinsic = sim.render_camera(
            "user_cam",
            width=width,
            height=height,
            near=near,
            far=far,
            fovy=fovy,
        )
        pose, _ = estimate_pose(
            rgb, depth, intrinsic, extrinsic, model_path, depth_scale=1.0, visualize=False, vis_title=(object_name + ".png") if i == 0 else None)

        desired_pos, desired_quat, estimated_grasp_strength = generate_and_select_grasps(
            model_path,
            pose,
            num_samples=500,
            max_gripper_width=0.08,
            gripper_depth=0.103,
            weight_distance=0.5,
            visualize=False
        )

        forces.append(estimated_grasp_strength)

    sim.close()

    return forces


if __name__ == "__main__":
    with open("configs/test_config_mj.yaml", "r") as f:
        config = yaml.safe_load(f)
    # You can make runner for one experiment
    if not os.path.exists("visualizations/"):
        os.mkdir("visualizations")
    forces = {}
    for obj in ["Banana", "MasterChefCan", "Pear", "CrackerBox", "PowerDrill", "TomatoSoupCan"]:
        xml_path = config["mujoco"]["grasp_object"]["xml"]
        split_xml_path = xml_path.split("/")
        split_xml_path[-2] = "Ycb" + obj
        xml_path = os.path.join(*split_xml_path)
        config["mujoco"]["grasp_object"]["xml"] = xml_path

        forces[obj] = runner(config, 10, obj)

    for key in forces:
        print(f"testimated grasp strength (Mean, Std) for {key}  ({np.mean(forces[key])}, {np.std(forces[key])})")

