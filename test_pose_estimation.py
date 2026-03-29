from solve import estimate_pose, get_mesh_path_from_xml
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
import copy


from scipy.spatial import cKDTree

def compute_adds_metric(model_points, t_est, R_est, t_gt, R_gt):
    t_est = np.array(t_est).reshape(3, 1)
    t_gt = np.array(t_gt).reshape(3, 1)

    pts = np.asarray(model_points).T

    pts_est = (R_est @ pts) + t_est
    pts_gt = (R_gt @ pts) + t_gt

    # Transpose back to (N, 3) for KDTree
    pts_est_T = pts_est.T
    pts_gt_T = pts_gt.T

    # Build a spatial tree using the gt
    tree = cKDTree(pts_gt_T)

    # For every point in the estimated model, find the distance to the closest point in gt model
    distances, _ = tree.query(pts_est_T, k=1)

    return np.mean(distances)

def debug_visualize_meshes_3d(model_path, t_est, rot_est, t_gt, rot_gt):
    """
    Renders the estimated and ground truth poses as colored 3D meshes.
    """
    # Load base mesh
    mesh = o3d.io.read_triangle_mesh(model_path)
    mesh.compute_vertex_normals()
    
    # Create Estimated Mesh (Painted Red)
    mesh_est = copy.deepcopy(mesh)
    pose_est = np.eye(4)
    pose_est[:3, :3] = rot_est
    pose_est[:3, 3] = t_est.flatten()
    mesh_est.transform(pose_est)
    mesh_est.paint_uniform_color([1, 0, 0]) 
    
    # Create Ground Truth Mesh (Painted Green)
    mesh_gt = copy.deepcopy(mesh)
    pose_gt = np.eye(4)
    pose_gt[:3, :3] = rot_gt
    pose_gt[:3, 3] = t_gt.flatten()
    mesh_gt.transform(pose_gt)
    mesh_gt.paint_uniform_color([0, 1, 0]) 
    
    # Add a coordinate frame to show where the camera is (0,0,0)
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    
    print("Close the Open3D window to continue...")
    o3d.visualization.draw_geometries(
        [mesh_est, mesh_gt, cam_frame], 
        window_name="RED: Estimated | GREEN: Ground Truth"
    )


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

    errors = []

    sim = MjSim(config)
    for i in tqdm(range(num_experiments), dynamic_ncols=True):
        sim.reset()

        # For sim stabilization
        for _ in range(1000):
            sim.step()

        # lower iterations per step for reaching the target pose
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
        pose, _ = estimate_pose(
            rgb, depth, intrinsic, extrinsic, model_path, depth_scale=1.0, visualize=False, vis_title=(object_name + ".png") if i == 0 else None)

        t_est = pose[:3, 3]
        rot_est = pose[:3, :3]

        # Get grasp object body ID
        grasp_obj_info = sim.ids.get("grasp_object", {})
        obj_body_name = grasp_obj_info.get("body_name", "sample_object")

        obj_body_id = mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_BODY, obj_body_name
        )
        t_gt_world = sim.data.xpos[obj_body_id].copy()
        mj_quat = sim.data.xquat[obj_body_id].copy()
        scipy_quat = np.array([mj_quat[1], mj_quat[2], mj_quat[3], mj_quat[0]])
        rot_gt_world = R.from_quat(scipy_quat).as_matrix()

        # 3. Build the 4x4 Ground Truth matrix in the World Frame
        pose_gt_world = np.eye(4)
        pose_gt_world[:3, :3] = rot_gt_world
        pose_gt_world[:3, 3] = t_gt_world

        # 4. Convert World GT to Camera Frame GT using the extrinsic matrix
        # (Assuming your extrinsic matrix maps World -> Camera coordinates)
        flip_matrix = np.diag([-1, -1, -1, 1])
        pose_gt_cam = np.linalg.inv(extrinsic) @ flip_matrix @ extrinsic @ pose_gt_world

        # Extract the camera-frame ground truth translation and rotation
        t_gt = pose_gt_cam[:3, 3]
        rot_gt = pose_gt_cam[:3, :3]

        source_mesh = o3d.io.read_triangle_mesh(model_path)
        source_pcd = source_mesh.sample_points_uniformly(number_of_points=5000)
        error = compute_adds_metric(
            source_pcd.points, t_est, rot_est, t_gt, rot_gt)
        errors.append(error)

    sim.close()

    return errors


if __name__ == "__main__":
    with open("configs/test_config_mj.yaml", "r") as f:
        config = yaml.safe_load(f)
    # You can make runner for one experiment
    if not os.path.exists("visualizations/"):
        os.mkdir("visualizations")
    errors = {}
    for obj in ["Banana", "MasterChefCan", "Pear", "CrackerBox", "PowerDrill", "TomatoSoupCan"]:
        xml_path = config["mujoco"]["grasp_object"]["xml"]
        split_xml_path = xml_path.split("/")
        split_xml_path[-2] = "Ycb" + obj
        xml_path = os.path.join(*split_xml_path)
        config["mujoco"]["grasp_object"]["xml"] = xml_path

        errors[obj] = runner(config, 10, obj)

    for key in errors:
        print(f"ADD-S metric (Median, [Quartiles]) for {key}: ({np.median(errors[key])}, {np.quantile(errors[key], [0.25, 0.75])})")
