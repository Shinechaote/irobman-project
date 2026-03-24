import argparse
from typing import Any, Dict

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml
from tqdm import tqdm
import open3d as o3d

from mujoco_app.mj_simulation import MjSim
import copy
import cv2
import xml.etree.ElementTree as ET
import os

def show_rgb_depth(
    rgb,
    depth,
    cam_name,
    near=None,
    far=None,
    cmap="viridis",
    figsize=(10, 4),
    title_rgb="RGB",
    title_depth="Depth",
):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title(cam_name + "_" + title_rgb)
    axes[0].axis("off")

    depth_vis = depth.astype(np.float32)

    if near is not None and far is not None:
        depth_vis = np.clip(depth_vis, near, far)

    # Hide invalid depth
    depth_vis[~np.isfinite(depth_vis)] = np.nan

    axes[1].matshow(depth_vis, cmap=plt.cm.viridis)
    axes[1].set_title(cam_name + "_" + title_depth)

    plt.tight_layout()
    plt.savefig("a.png")


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


def visualize_pose_on_image(rgb_img, mesh, transformation_matrix, K):
    K = np.array(K, dtype=np.float32)
    # Extract Rotation vector and Translation vector from the 4x4 matrix
    rvec, _ = cv2.Rodrigues(transformation_matrix[:3, :3].astype(np.float32))
    tvec = transformation_matrix[:3, 3].astype(np.float32)

    canvas = rgb_img.copy()
    if canvas.shape[2] == 3:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    vertices = np.asarray(mesh.vertices).astype(np.float32)
    img_pts, _ = cv2.projectPoints(vertices, rvec, tvec, K, None)

    for pt in img_pts:
        cv2.circle(canvas, tuple(pt[0].astype(int)), 1, (0, 255, 0), -1)

    # 10cm axes
    axis_length = 0.1
    axis_points = np.float32([[0, 0, 0], [axis_length, 0, 0], [
                             0, axis_length, 0], [0, 0, axis_length]])
    img_axis, _ = cv2.projectPoints(axis_points, rvec, tvec, K, None)

    origin = tuple(img_axis[0][0].astype(int))
    # x (red), y (green), z (blue)
    cv2.line(canvas, origin, tuple(
        img_axis[1][0].astype(int)), (0, 0, 255), 2)  # X - Red
    cv2.line(canvas, origin, tuple(
        img_axis[2][0].astype(int)), (0, 255, 0), 2)  # Y - Green
    cv2.line(canvas, origin, tuple(
        img_axis[3][0].astype(int)), (255, 0, 0), 2)  # Z - Blue

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    source_temp.transform(transformation)

    print("Visualizing Alignment: Yellow is Model, Cyan is Scene.")
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      window_name="ICP Alignment Result",
                                      width=1024, height=768)


def execute_global_registration(source, target, voxel_size):
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Compute FPFH features
    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    distance_threshold = voxel_size * 1.5

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,  # Number of RANSAC points to sample
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )

    return result_ransac.transformation


def visualize_bounding_box(rgb_img, bbox):
    x_min, x_max, y_min, y_max = bbox
    viz_img = rgb_img.copy()
    box_color = (0, 255, 0)
    thickness = 2
    top_left = (x_min, y_min)
    bottom_right = (x_max, y_max)

    cv2.rectangle(viz_img, top_left, bottom_right, box_color, thickness)

    bgr_viz = cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Bounding Box", bgr_viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return viz_img


def segment_and_align(rgb_img, depth_img, intrinsic_matrix, model_path, depth_scale=1.0, visualize=False):
    y_min, y_max = int(rgb_img.shape[0] * 0.4), rgb_img.shape[0]
    x_min, x_max = 0, int(rgb_img.shape[1] * 0.6)
    bbox = (x_min, x_max, y_min, y_max)

    if visualize:
        visualize_bounding_box(rgb_img, bbox)

    cropped_rgb = rgb_img[y_min:y_max, x_min:x_max].copy()
    cropped_depth = depth_img[y_min:y_max, x_min:x_max].copy()

    adj_intrinsic = intrinsic_matrix.copy()
    adj_intrinsic[0, 2] -= x_min
    adj_intrinsic[1, 2] -= y_min

    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    o3d_intrinsic.set_intrinsics(
        width=cropped_rgb.shape[1], height=cropped_rgb.shape[0],
        fx=adj_intrinsic[0, 0], fy=adj_intrinsic[1, 1],
        cx=adj_intrinsic[0, 2], cy=adj_intrinsic[1, 2]
    )

    color_raw = o3d.geometry.Image(cropped_rgb)
    depth_raw = o3d.geometry.Image(cropped_depth.astype(np.float32))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=depth_scale, convert_rgb_to_intensity=False
    )

    full_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, o3d_intrinsic)

    # Remove the Table
    plane_model, inliers = full_pcd.segment_plane(distance_threshold=0.01,
                                                  ransac_n=3,
                                                  num_iterations=1000)
    object_candidates = full_pcd.select_by_index(inliers, invert=True)

    cl, ind = object_candidates.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0)
    scene_pcd = object_candidates.select_by_index(ind)

    source_mesh = o3d.io.read_triangle_mesh(model_path)
    source_pcd = source_mesh.sample_points_uniformly(number_of_points=5000)

    scene_pcd.estimate_covariances()
    source_pcd.estimate_covariances()

    voxel_size = 0.01

    initial_trans = execute_global_registration(
        source_pcd, scene_pcd, voxel_size)

    if visualize:
        draw_registration_result(source_pcd, scene_pcd, initial_trans)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, scene_pcd, max_correspondence_distance=0.1,
        init=initial_trans,
        estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    )

    final_pose_cam = reg_p2p.transformation

    if visualize:
        draw_registration_result(source_pcd, scene_pcd, final_pose_cam)

        viz_2d = visualize_pose_on_image(
            rgb_img, source_mesh, final_pose_cam, intrinsic_matrix)
        cv2.imshow("6D Pose Projection", cv2.cvtColor(viz_2d, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return reg_p2p.transformation, scene_pcd


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

        # lower iterations per step for reaching the target pose
        print("Moving to target pose...")
        for t in tqdm(range(100000), dynamic_ncols=True):
            sim.step()

            # Showcasing some operations that can be done with the simulation
            rgb, depth, intrinsic, extrinsic = sim.render_camera(
                "user_cam",
                width=width,
                height=height,
                near=near,
                far=far,
                fovy=fovy,
            )
            pose, _ = segment_and_align(
                rgb, depth, intrinsic, model_path, depth_scale=1.0)
            show_rgb_depth(rgb, depth, "static")
            return

            ee_body_name = sim.robot_settings.get("ee_body_name", "hand")
            ee_body_id = mujoco.mj_name2id(
                sim.model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name
            )
            ee_pos = sim.data.xpos[ee_body_id].copy()
            # Robot should not collide with obstacles
            # This condition must be there
            if sim.check_robot_obstacle_collision():
                print("Collision!")
                break
    sim.close()
    print("Simulation completed.")


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # You can make runner for one experiment
    runner(config, 10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/test_config_mj.yaml"
    )
    args = parser.parse_args()
    main(config_path=args.config)
