import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import copy
import cv2

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

def show_masked_color(img, mask, title="Masked Color"):
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    plt.imshow(masked_img)
    plt.title(title)
    plt.axis('off')
    plt.savefig("masks.png")

def create_tracking_video(image_stack, estimates, intrinsic, extrinsic, output_path="tracking_output.mp4", fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (image_stack[0].shape[1], image_stack[0].shape[0]))

    for i in range(len(image_stack)):
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(image_stack[i], cv2.COLOR_RGB2BGR)

        colors = {"orange": (0, 165, 255), "red": (0, 0, 255), "desired": (0, 255, 0)}
        
        for color in estimates.keys():
            color_val = colors[color]
            if i < len(estimates[color]):
                # 1. Get the 3D position from the Kalman State
                pos_world = estimates[color][i]
                
                # 2. Project World 3D -> Camera 3D -> Pixel 2D
                # Transform to Camera Space: P_cam = T * P_world
                pos_world_homo = np.append(pos_world, 1.0)
                flip_mat = np.diag([-1, -1, -1, 1])
                pos_cam = flip_mat @ extrinsic @ pos_world_homo
                
                # Project to Pixel Space: p = K * (P_cam / z)
                if pos_cam[2] > 0:  # Check if in front of camera
                    u = int((intrinsic[0, 0] * pos_cam[0] / pos_cam[2]) + intrinsic[0, 2])
                    v = int((intrinsic[1, 1] * pos_cam[1] / pos_cam[2]) + intrinsic[1, 2])
                    
                    # 3. Draw Marker and Label
                    cv2.drawMarker(frame, (u, v), color_val, cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(frame, f"Tracked {color}", (u + 15, v - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_val, 2)
                    
                    # Optional: Draw a "trail" of previous positions
                    for j in range(max(0, i-10), i):
                        prev_pos = estimates[color][j][:3]
                        p_prev_cam = extrinsic @ np.append(prev_pos, 1.0)
                        u_p = int((intrinsic[0, 0] * p_prev_cam[0] / p_prev_cam[2]) + intrinsic[0, 2])
                        v_p = int((intrinsic[1, 1] * p_prev_cam[1] / p_prev_cam[2]) + intrinsic[1, 2])
                        cv2.circle(frame, (u_p, v_p), 2, color_val, -1)

        out.write(frame)

    out.release()
    print(f"Video saved to {output_path}")

def draw_ball_center(rgb_image, center, color_name="Ball"):
    output_img = rgb_image.copy()
    u, v = center

    cv2.circle(output_img, (u, v), 5, (0, 255, 0), -1) 

    length = 15
    cv2.line(output_img, (u - length, v), (u + length, v), (0, 255, 0), 2)
    cv2.line(output_img, (u, v - length), (u, v + length), (0, 255, 0), 2)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Bounding Box", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return output_img

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
