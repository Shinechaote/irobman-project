import argparse
from typing import Any, Dict
from visualization_tools import show_rgb_depth, create_tracking_video
from pose_estimation import estimate_pose
from obstacle_detection import get_ball_position, ball_kalman_update
from control import get_actions
from grasping import generate_and_select_grasps
import numpy as np
import yaml
from tqdm import tqdm
import open3d as o3d
import mujoco

from mujoco_app.mj_simulation import MjSim
import xml.etree.ElementTree as ET
import os

def check_object_in_basket(sim: MjSim) -> dict:
    """Check if the grasp object is inside the basket.

    Returns a dict with keys:
        - in_basket: bool - True if object is in basket
        - in_x, in_y, in_z: bool - Individual bound checks
        - object_pos: list - [x, y, z] position of object
    """
    # Get basket parameters
    basket_center = sim.ids.get("basket_center")
    basket_dims = sim.ids.get("basket_dims")
    basket_height = sim.ids.get("basket_height")

    # Default if basket not configured
    if basket_center is None or basket_dims is None or basket_height is None:
        return {
            "in_basket": False,
            "in_x": False,
            "in_y": False,
            "in_z": False,
            "object_pos": None,
        }

    # Get grasp object body ID
    grasp_obj_info = sim.ids.get("grasp_object", {})
    obj_body_name = grasp_obj_info.get("body_name", "sample_object")

    obj_body_id = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_BODY, obj_body_name
    )

    if obj_body_id < 0:
        return {
            "in_basket": False,
            "in_x": False,
            "in_y": False,
            "in_z": False,
            "object_pos": None,
        }

    # Get object position
    obj_pos = sim.data.xpos[obj_body_id].copy()

    # Check bounds
    in_x = abs(obj_pos[0] - basket_center[0]) < basket_dims[0] / 2.0
    in_y = abs(obj_pos[1] - basket_center[1]) < basket_dims[1] / 2.0
    basket_bottom = basket_center[2] - basket_height / 2.0
    in_z = obj_pos[2] > basket_bottom

    return {
        "in_basket": in_x and in_y and in_z,
        "in_x": in_x,
        "in_y": in_y,
        "in_z": in_z,
        "object_pos": obj_pos.tolist(),
    }

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
def runner(config: Dict[str, Any], num_experiments: int, render_video=False, obj=""):
    # Configure in YAML
    has_obstacles = config.get("mujoco", {}).get("obstacle_toggle", True)
    cam_cfg = config.get("mujoco", {}).get("camera", {})
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)
    near = cam_cfg.get("near", 0.01)
    far = cam_cfg.get("far", 5.0)
    fovy = cam_cfg.get("fovy", 58.0)
    xml_path = config.get("mujoco", {}).get("grasp_object", {}).get("xml", "")
    seed = config.get("mujoco", {}).get("seed", 42)
    model_path = get_mesh_path_from_xml(xml_path)
    np.random.seed(seed)
    print(model_path)

    sim = MjSim(config)
    num_collisions = 0
    num_success = 0
    for _ in range(num_experiments):
        sim.reset()

        # For sim stabilization
        for _ in tqdm(range(1000), dynamic_ncols=True):
            sim.step()

        rgb, depth, intrinsic, extrinsic = sim.render_camera(
            "user_cam",
            width=width,
            height=height,
            near=near,
            far=far,
            fovy=fovy,
        )

        # get pose estimate using user cam and then use static cam for obstacle tracking and rendering the video
        pose, _ = estimate_pose(
            rgb,
            depth,
            intrinsic,
            extrinsic,
            model_path,
            depth_scale=1.0,
            visualize=False,
        )

        rgb, depth, intrinsic, extrinsic = sim.render_camera(
            "static",
            width=width,
            height=height,
            near=near,
            far=far,
            fovy=fovy,
        )

        # If you want to visualize the predicted pose

        # from visualization_tools import visualize_pose_on_image
        # import cv2
        # source_mesh = o3d.io.read_triangle_mesh(model_path)
        # flip_mat = np.diag([-1, -1, -1, 1])
        # viz_2d = visualize_pose_on_image(
        #     rgb, source_mesh, flip_mat @ extrinsic @ pose, intrinsic)
        # cv2.imshow("6D Pose Projection", cv2.cvtColor(viz_2d, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        red_range = (np.array([0, 150, 50]), np.array([10, 255, 255]))
        orange_range = (np.array([11, 150, 50]), np.array([25, 255, 255]))
        if has_obstacles:
            red_pos = get_ball_position(
                rgb, depth, intrinsic, extrinsic, red_range, visualize=False
            )
            orange_pos = get_ball_position(
                rgb, depth, intrinsic, extrinsic, orange_range, visualize=False
            )
        else:
            # just move them far away but still have them have a good height
            red_pos = [-10, -10, 1.05]
            orange_pos = [-10, -10, 1.05]

        red_state_world, orange_state_world = np.zeros(6), np.zeros(6)
        red_cov, orange_cov = np.eye(6), np.eye(6)

        red_state_world[:3] = red_pos
        orange_state_world[:3] = orange_pos

        desired_poses = []
        pred_red = []
        pred_orange = []
        images = []

        site_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, "fingertip")
        ee_pos = sim.data.site_xpos[site_id].copy()

        grasp_pos, grasp_quat, _ = generate_and_select_grasps(
            model_path,
            pose,
            num_samples=500,
            max_gripper_width=0.08,
            gripper_depth=0.103,
            weight_distance=0.5,
            visualize=False
        )
        
        for i in range(10):
            if grasp_pos is not None:
                break
            if i == 0:
                print("Retrying pose and grasp estimation until valid pose is found")

            rgb, depth, intrinsic, extrinsic = sim.render_camera(
                "user_cam",
                width=width,
                height=height,
                near=near,
                far=far,
                fovy=fovy,
            )

            # from visualization_tools import visualize_pose_on_image
            # import cv2
            # source_mesh = o3d.io.read_triangle_mesh(model_path)
            # flip_mat = np.diag([-1, -1, -1, 1])
            # viz_2d = visualize_pose_on_image(
            #     rgb, source_mesh, flip_mat @ extrinsic @ pose, intrinsic)
            # cv2.imshow("6D Pose Projection", cv2.cvtColor(viz_2d, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # get pose estimate using user cam and then use static cam for obstacle tracking and rendering the video
            pose, _ = estimate_pose(
                rgb,
                depth,
                intrinsic,
                extrinsic,
                model_path,
                depth_scale=1.0,
                visualize=False,
            )

            # from visualization_tools import visualize_pose_on_image
            # import cv2
            # source_mesh = o3d.io.read_triangle_mesh(model_path)
            # flip_mat = np.diag([-1, -1, -1, 1])
            # viz_2d = visualize_pose_on_image(
            #     rgb, source_mesh, flip_mat @ extrinsic @ pose, intrinsic)
            # cv2.imshow("6D Pose Projection", cv2.cvtColor(viz_2d, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            grasp_pos, grasp_quat, _ = generate_and_select_grasps(
                        model_path,
                        pose,
                        num_samples=500,
                        max_gripper_width=0.08,
                        gripper_depth=0.103,
                        weight_distance=0.5,
                        visualize=False
                    )

        # just accept failure
        if grasp_pos is None:
            desired_pos = ee_pos.copy() + [0, 0, 0.1]
            desired_quat = get_lookat_quat(desired_pos, ee_pos.copy())
        else:
            desired_pos = grasp_pos.copy()
            desired_quat = grasp_quat.copy()

        move_down = 0
        for t in tqdm(range(2000), dynamic_ncols=True):
            sim.step()

            rgb, depth, intrinsic, extrinsic = sim.render_camera(
                "static",
                width=width,
                height=height,
                near=near,
                far=far,
                fovy=fovy,
            )

            max_delta_pos = 0.01
            ee_pos = sim.data.site_xpos[site_id].copy()
            basket_center = [0.5, 0.52, 0.765]
            num_grasp_move_steps = 200
            num_gripper_steps = 200
            num_move_up_steps = 200
            num_move_above_goal = 400
            num_move_down = 100

            target_pos = desired_pos.copy() 

            if move_down == 0:
                if t <= num_grasp_move_steps:
                    target_pos = grasp_pos.copy()
                elif t > num_grasp_move_steps and t <= num_grasp_move_steps + num_gripper_steps:
                    gripper_val = np.linspace(0.03, 0.0, num_gripper_steps)[min(t - num_grasp_move_steps, num_gripper_steps - 1)]
                    sim._set_gripper_opening(gripper_val)
                    target_pos = grasp_pos.copy()
                elif t > num_grasp_move_steps + num_gripper_steps and t <= num_grasp_move_steps + num_gripper_steps + num_move_up_steps:
                    target_pos = np.array([grasp_pos[0], grasp_pos[1], red_state_world[2] + 0.1])
                elif t > num_grasp_move_steps + num_gripper_steps + num_move_up_steps and t <= num_grasp_move_steps + num_gripper_steps + num_move_up_steps + num_move_above_goal:
                    target_pos = np.array([basket_center[0], basket_center[1], red_state_world[2] + 0.1])
                    max_delta_pos = 0.005
                elif np.linalg.norm(ee_pos[:2] - red_state_world[:2]) > 0.15 and np.linalg.norm(ee_pos[:2] - orange_state_world[:2]) > 0.15:
                    target_pos = np.array(basket_center.copy())
                    move_down = 1

            if move_down > 0:
                move_down += 1
                target_pos = np.array(basket_center.copy())
                if move_down >= num_move_down:
                    sim._set_gripper_opening(0.03)
                if move_down >= num_move_down + 25:
                    target_pos[2] += 0.4

            delta_pos = target_pos - desired_pos
            distance = np.linalg.norm(delta_pos)

            if distance > max_delta_pos:
                desired_pos += (delta_pos / distance) * max_delta_pos
            else:
                desired_pos = target_pos.copy()

            if render_video:
                images.append(rgb)
                desired_poses.append(desired_pos.copy())
                pred_red.append(red_state_world[:3])
                pred_orange.append(orange_state_world[:3])

            rgb, depth, intrinsic, extrinsic = sim.render_camera(
                "static",
                width=width,
                height=height,
                near=near,
                far=far,
                fovy=fovy,
            )
            if has_obstacles:
                red_state_world, red_cov = ball_kalman_update(
                    rgb, depth, intrinsic, extrinsic, red_range, red_state_world, red_cov
                )
                orange_state_world, orange_cov = ball_kalman_update(
                    rgb,
                    depth,
                    intrinsic,
                    extrinsic,
                    orange_range,
                    orange_state_world,
                    orange_cov,
                )

            q_indices = [qidx for qidx, _ in sim.robot.arm_pairs]
            q = np.array([sim.data.qpos[qidx] for qidx in q_indices])
            joint_positions = get_actions(
                sim, desired_pos, desired_quat, 0.1, 0.001, dt=0.05
            )
            joint_positions[-1] = q[-1]
            sim.set_arm_joint_positions(joint_positions, clamp=True, sync=True)

            # Robot should not collide with obstacles
            # This condition must be there
            # Only makes sense if there are actual obstacles
            if has_obstacles:
                if sim.check_robot_obstacle_collision():
                    print("Collision!")
                    num_collisions += 1
                    break

            if check_object_in_basket(sim)["in_basket"]:
                num_success += 1
                break

    if render_video:
        estimates = {"red": pred_red, "orange": pred_orange, "desired": desired_poses}

        rgb, depth, intrinsic, extrinsic = sim.render_camera(
            "static",
            width=width,
            height=height,
            near=near,
            far=far,
            fovy=fovy,
        )
        create_tracking_video(
            images,
            estimates,
            intrinsic,
            extrinsic,
            output_path=f"{obj}_tracking_output.mp4",
            fps=100,
        )

    sim.close()
    print("Simulation completed.")
    print("Num Collisions:", num_collisions)
    print("Num Success:", num_success)

    return num_success, num_collisions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/test_config_mj.yaml")
    parser.add_argument("--render_video", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    stats = {}

    for obj in ["Banana", "MasterChefCan", "Pear", "CrackerBox", "PowerDrill", "TomatoSoupCan"]:
        xml_path = config["mujoco"]["grasp_object"]["xml"]
        split_xml_path = xml_path.split("/")
        split_xml_path[-2] = "Ycb" + obj
        xml_path = os.path.join(*split_xml_path)
        config["mujoco"]["grasp_object"]["xml"] = xml_path
        stats[obj] = runner(config, 10, args.render_video, obj=obj)

    for obj in stats:
        print(f"Stats for {obj}: Num Successful {stats[obj][0]}, Num Collisions {stats[obj][1]}")
