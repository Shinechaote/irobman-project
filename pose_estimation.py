import numpy as np
import open3d as o3d
import cv2
from visualization_tools import visualize_bounding_box, visualize_pose_on_image, draw_registration_result

# Compute FPFH features and use them to get rough global estimate with RANSAC
def execute_global_registration(source, target, voxel_size):
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

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

def estimate_pose(rgb_img, depth_img, intrinsic_matrix, extrinsic_matrix, model_path, depth_scale=1.0, visualize=False, vis_title=None):
    voxel_size = 0.01

    source_mesh = o3d.io.read_triangle_mesh(model_path)
    source_pcd = source_mesh.sample_points_uniformly(number_of_points=5000)

    source_pcd.estimate_covariances()

    voxel_size = 0.01
    
    max_retries = 5
    min_fitness_threshold = 0.6
    max_correspondence_dist = 0.02
    
    best_fitness = -np.inf
    best_transformation = None
    h, w = rgb_img.shape[:2]

    # Base bounding box settings
    base_y_min, base_y_max = int(h * 0.4), h
    base_x_min, base_x_max = 0, int(w * 0.6)
    
    jitter_y = int(h * 0.15)
    jitter_x = int(w * 0.15)


    for attempt in range(max_retries):
        print(f"--- Registration attempt {attempt + 1}/{max_retries} ---")
        
        # dynamically generate the bounding box
        if attempt == 0:
            y_min, y_max = base_y_min, base_y_max
            x_min, x_max = base_x_min, base_x_max
        else:
            y_min = np.clip(base_y_min + np.random.randint(-jitter_y, jitter_y), 0, h - 50)
            y_max = np.clip(base_y_max + np.random.randint(-jitter_y, jitter_y), y_min + 50, h)
            x_min = np.clip(base_x_min + np.random.randint(-jitter_x, jitter_x), 0, w - 50)
            x_max = np.clip(base_x_max + np.random.randint(-jitter_x, jitter_x), x_min + 50, w)

        # crop point cloud
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

        # remove the table
        plane_model, inliers = full_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        object_candidates = full_pcd.select_by_index(inliers, invert=True)

        cl, ind = object_candidates.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        scene_pcd = object_candidates.select_by_index(ind)
        
        if not scene_pcd.has_points():
            print("Warning: Cropped area is empty")
            continue
            
        scene_pcd.estimate_covariances()
        
        # first get rough guess using global registration and then refine it via icp
        initial_trans = execute_global_registration(source_pcd, scene_pcd, voxel_size)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, scene_pcd, max_correspondence_distance=max_correspondence_dist,
            init=initial_trans,
            estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
        )
        
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source_pcd, scene_pcd, max_correspondence_dist, reg_p2p.transformation
        )
        
        fitness = evaluation.fitness
        rmse = evaluation.inlier_rmse
        print(f"Attempt {attempt + 1}, Fitness: {fitness:.4f}, RMSE: {rmse:.4f}")
        
        if fitness > best_fitness and reg_p2p.transformation is not None:
            best_fitness = fitness
            best_transformation = reg_p2p.transformation
            
        if fitness >= min_fitness_threshold:
            break
            
    if best_fitness < min_fitness_threshold:
        print(f"Warning: Could not reach fitness threshold after {max_retries} attempts. Using best found ({best_fitness:.4f}).")


    # mujoco axes and Open3D/OpenCV axes are a pain!
    inv_extrinsic = extrinsic_matrix.copy()
    inv_extrinsic[:3, :3] = extrinsic_matrix[:3, :3].T
    inv_extrinsic[:3, 3] = -extrinsic_matrix[:3, :3].T @ extrinsic_matrix[:3, 3]
    flip_mat = np.diag([-1, -1, -1, 1])

    if visualize:
        draw_registration_result(source_pcd, scene_pcd, best_transformation)

        viz_2d = visualize_pose_on_image(
            rgb_img, source_mesh, best_transformation, intrinsic_matrix)
        cv2.imshow("6D Pose Projection", cv2.cvtColor(viz_2d, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    if vis_title is not None:
        viz_2d = visualize_pose_on_image(
            rgb_img, source_mesh, best_transformation, intrinsic_matrix)
        cv2.imwrite("visualizations/" + vis_title, cv2.cvtColor(viz_2d, cv2.COLOR_RGB2BGR))

    final_pose_world = inv_extrinsic @ flip_mat @ best_transformation

    return final_pose_world, scene_pcd
