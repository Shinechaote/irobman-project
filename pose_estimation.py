import numpy as np
import open3d as o3d
import cv2
from visualization_tools import visualize_bounding_box, visualize_pose_on_image, draw_registration_result

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
