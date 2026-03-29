import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

def generate_and_select_grasps(model_path, obj_transform,
                               num_samples=500, max_gripper_width=0.08, 
                               gripper_depth=0.1, weight_distance=0.5,
                               visualize=False):

    weight_orientation = 0.5 
    
    # matrix for enforcing the grasp comes from above (in the mujoco coordinate system)
    top_down_mat = np.array([
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0, -1]
    ])
    target_quat = R.from_matrix(top_down_mat).as_quat()

    source_mesh = o3d.io.read_triangle_mesh(model_path)
    source_pcd = source_mesh.sample_points_uniformly(number_of_points=5000)

    source_pcd.estimate_normals()

    points = np.asarray(source_pcd.points)
    normals = np.asarray(source_pcd.normals)

    R_obj = obj_transform[:3, :3]
    t_obj = obj_transform[:3, 3]
    
    world_points = (R_obj @ points.T).T + t_obj
    world_normals = (R_obj @ normals.T).T 

    # Ensure normals are normalized after rotation
    world_normals = world_normals / np.linalg.norm(world_normals, axis=1, keepdims=True)

    best_score = -np.inf
    best_pose = None
    
    # for visualization
    all_candidate_matrices = []

    num_points = len(world_points)
    sampled_indices = np.random.choice(num_points, num_samples, replace=False)

    for idx1 in sampled_indices:
        p1 = world_points[idx1]
        n1 = world_normals[idx1]

        # 3. Find points within gripper width
        distances = np.linalg.norm(world_points - p1, axis=1)
        valid_width_mask = (distances > 0.01) & (distances <= max_gripper_width)
        
        if not np.any(valid_width_mask):
            continue
            
        candidate_indices = np.where(valid_width_mask)[0]
        
        # evaluate antipodal force
        dots = np.sum(world_normals[candidate_indices] * n1, axis=1)
        best_pair_idx = candidate_indices[np.argmin(dots)]
        best_dot = np.min(dots)

        if best_dot > -0.7:
            continue

        p2 = world_points[best_pair_idx]

        grasp_center = (p1 + p2) / 2.0
        z_axis = -n1
        
        # filter out grasps that do not fulfill the top-down grasp constraint
        if np.dot(z_axis, np.array([0, 0, -1])) < 0.7:
            continue
            
        y_axis = p2 - p1
        y_axis = y_axis / np.linalg.norm(y_axis)
        y_axis = y_axis - np.dot(y_axis, z_axis) * z_axis
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)

        rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
        quat = R.from_matrix(rot_matrix).as_quat()
        
        if visualize:
            cand_mat = np.eye(4)
            cand_mat[:3, :3] = rot_matrix
            cand_mat[:3, 3] = grasp_center
            all_candidate_matrices.append(cand_mat)

        grasp_strength = -best_dot

        # score grasps by grasp strength and proximity to the grasping from the top constraint
        dot_product = np.clip(np.abs(np.dot(quat, target_quat)), 0.0, 1.0)
        angle_diff = 2.0 * np.arccos(dot_product) # Angle in radians
        
        score = grasp_strength - (weight_orientation * angle_diff)

        if score > best_score:
            best_score = score
            best_grasp_strength = grasp_strength
            best_pose = (grasp_center, quat, rot_matrix)

    if best_pose is None:
        print("Warning: No valid grasps found")
        return None, None, -1

    if visualize:
        vis_geometries = []
        
        world_pcd = o3d.geometry.PointCloud()
        world_pcd.points = o3d.utility.Vector3dVector(world_points)
        world_pcd.normals = o3d.utility.Vector3dVector(world_normals)
        world_pcd.paint_uniform_color([0.6, 0.6, 0.6]) # Neutral Gray
        vis_geometries.append(world_pcd)
        
        for mat in all_candidate_matrices:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.015)
            frame.transform(mat)
            vis_geometries.append(frame)

        best_mat = np.eye(4)
        best_mat[:3, :3] = best_pose[2]
        best_mat[:3, 3] = best_pose[0]
        
        best_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
        best_frame.transform(best_mat)
        vis_geometries.append(best_frame)
        
        o3d.visualization.draw_geometries(vis_geometries, window_name="Grasp Candidates Evaluator")

    # rotate quat to match the gripper's geometry
    quat = (R.from_euler("x", -90, degrees=True) * R.from_quat(best_pose[1])).as_quat()

    return best_pose[0], quat, best_grasp_strength
