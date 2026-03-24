from tqdm import tqdm
import yaml
import numpy as np
from obstacle_detection import get_ball_position, ball_kalman_update
import argparse
import mujoco
from visualization_tools import create_tracking_video

from mujoco_app.mj_simulation import MjSim
from scipy.spatial.transform import Rotation as R

# Experiment runner example
def runner(config, num_experiments):
    # Configure in YAML
    cam_cfg = config.get("mujoco", {}).get("camera", {})
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)
    near = cam_cfg.get("near", 0.01)
    far = cam_cfg.get("far", 5.0)
    fovy = cam_cfg.get("fovy", 58.0)

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
        red_state, orange_state = np.zeros(6), np.zeros(6)
        red_cov, orange_cov = np.eye(6), np.eye(6)

        red_state[:3] = red_pos
        orange_state[:3] = orange_pos

        pred_red = []
        gt_red = []
        pred_orange = []
        gt_orange = []

        images = []

        for t in tqdm(range(100), dynamic_ncols=True):
            sim.step()

            images.append(rgb)

            # Get IDs of obstacles
            red_g_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_lr")
            orange_g_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_tb")

            red_gt_world = np.append(sim.data.xpos[red_g_id].copy(), 1.0)
            orange_gt_world = np.append(sim.data.xpos[orange_g_id].copy(), 1.0)

            # Do some axis magic because apparently the axes are flipped in mujoco!!
            flip_mat = np.diag([-1, -1, -1, 1])
            red_gt_world = np.linalg.inv(extrinsic) @ flip_mat @ extrinsic @ red_gt_world
            orange_gt_world = np.linalg.inv(extrinsic) @ flip_mat @ extrinsic @ orange_gt_world

            # Account for that we can only measure the distance to the surface of the ball and thus need to subtract the radius
            cam_pos = np.linalg.inv(extrinsic)[:3, 3]
            ball_radius = 0.06 
            red_ray_direction = cam_pos - red_state[:3]
            red_ray_direction /= np.linalg.norm(red_ray_direction)
            red_estimate = red_state[:3] - (red_ray_direction * ball_radius)

            orange_ray_direction = cam_pos - orange_state[:3]
            orange_ray_direction /= np.linalg.norm(orange_ray_direction)
            orange_estimate = orange_state[:3] - (orange_ray_direction * ball_radius)

            gt_red.append(red_gt_world[:3])
            gt_orange.append(orange_gt_world[:3])

            pred_red.append(red_estimate[:3])
            pred_orange.append(orange_estimate[:3])

            # Showcasing some operations that can be done with the simulation
            rgb, depth, intrinsic, extrinsic = sim.render_camera(
                "static",
                width=width,
                height=height,
                near=near,
                far=far,
                fovy=fovy,
            )
            # pose, _ = segment_and_align(
            #     rgb, depth, intrinsic, model_path, depth_scale=1.0)
            red_state, red_cov = ball_kalman_update(rgb, depth, intrinsic, extrinsic, red_range, red_state, red_cov)
            orange_state, orange_cov = ball_kalman_update(rgb, depth, intrinsic, extrinsic, orange_range, orange_state, orange_cov)

            # Robot should not collide with obstacles
            # This condition must be there
            if sim.check_robot_obstacle_collision():
                print("Collision!")
                break

        # estimates = {"red": pred_red, "orange": pred_orange}
        estimates = {"red": gt_red, "orange": gt_orange}
        create_tracking_video(images, estimates, intrinsic, extrinsic, output_path="tracking_output.mp4", fps=100)
        import matplotlib.pyplot as plt
        def visualize_trajectory(gt_points, pred_points, title="Ball Tracking"):
            gt = np.array(gt_points)
            pred = np.array(pred_points)
            
            plt.figure(figsize=(8, 6))
            # Plot X and Y coordinates (Top-down view)
            plt.plot(gt[:, 0], gt[:, 1], 'g-', label="Ground Truth", alpha=0.6)
            plt.plot(pred[:, 0], pred[:, 1], 'r--', label="Kalman Estimate")
            
            # Draw a circle at the start and end
            plt.scatter(gt[0, 0], gt[0, 1], c='green', marker='o')
            plt.scatter(pred[-1, 0], pred[-1, 1], c='red', marker='x')
            
            plt.xlabel("World X (meters)")
            plt.ylabel("World Y (meters)")
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            plt.savefig(title)

        # Correct comparison block
        mse_red = np.linalg.norm(np.array(pred_red) - np.array(gt_red), axis=1).mean()
        mse_orange = np.linalg.norm(np.array(pred_orange) - np.array(gt_orange), axis=1).mean()

        print(f"Red Ball MSE: {mse_red:.4f}")
        print(f"Orange Ball MSE: {mse_orange:.4f}")

        visualize_trajectory(gt_red, pred_red, "Red Ball Tracking Evaluation")
        visualize_trajectory(gt_orange, pred_orange, "Orange Ball Tracking Evaluation")

    
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
