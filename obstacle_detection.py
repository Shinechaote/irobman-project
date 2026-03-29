import cv2
import numpy as np
from visualization_tools import draw_ball_center, show_masked_color

def get_ball_only(mask):
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_cnt = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_cnt) < 500:
        return None
    
    
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest_cnt], -1, 255, -1)
    
    return clean_mask

def get_linear_motion_params(state_estimate, dt, jitter_scale):
    # state_estimate is [x, y, z, vx, vy, vz]
    
    motion_update = np.zeros(6)
    motion_update[0:3] = state_estimate[3:6] * dt
    
    # motion derivative
    F = np.eye(6)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    
    # motion noise
    Q = np.eye(6) * 0.01
    Q[0:3, 0:3] *= jitter_scale 
    
    return motion_update, F, Q

def sensor_model_linear(state):
    return state[0:3]

def sensor_deriv_linear(state):
    H = np.zeros((3, 6))
    H[0:3, 0:3] = np.eye(3)
    return H

def kalman_filter(motion_update, motion_deriv, motion_noise, sensor_model, sensor_noise, sensor_deriv, state_estimate, measurement, covariance):
    a_priori_est = state_estimate + motion_update
    a_priori_cov = motion_deriv @ covariance @ motion_deriv.T + motion_noise

    s_k = sensor_deriv(a_priori_est) @ a_priori_cov @ sensor_deriv(a_priori_est).T + sensor_noise
    kalman_gain = a_priori_cov @ sensor_deriv(a_priori_est).T @ np.linalg.inv(s_k)
    updated_state_estimate = a_priori_est + kalman_gain @ (measurement - sensor_model(a_priori_est))
    updated_covariance = (np.eye(covariance.shape[0]) - kalman_gain @ sensor_deriv(a_priori_est)) @ a_priori_cov

    return np.squeeze(updated_state_estimate), updated_covariance

def ball_kalman_update(rgb_image, depth_img, intrinsic, extrinsic, color_range, state_estimate, covariance):
    dt = 0.01 

    m_up, m_f, m_q = get_linear_motion_params(state_estimate, dt, 0.03)

    pixel = get_ball_centroid(rgb_image, color_range) 
    if pixel:
        z_meas = calculate_world_pos(pixel[0], pixel[1], depth_img, intrinsic, extrinsic)
        
        new_est, new_cov = kalman_filter(
            motion_update=m_up,
            motion_deriv=m_f,
            motion_noise=m_q,
            sensor_model=sensor_model_linear,
            sensor_noise=np.eye(3) * 0.005,
            sensor_deriv=sensor_deriv_linear,
            state_estimate=state_estimate,
            measurement=z_meas,
            covariance=covariance
        )
    else:
        # If ball is occluded, only do the motion update
        new_est = state_estimate + m_up
        new_cov = m_f @ covariance @ m_f.T + m_q
        
    return new_est, new_cov

def get_ball_centroid(rgb_image, color_range, visualize=False):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, color_range[0], color_range[1])

    mask = get_ball_only(mask)
    if mask is None:
        return None

    if visualize:
        show_masked_color(rgb_image, mask)
    
    # calculate moments to find the center
    M = cv2.moments(mask)
    if M["m00"] != 0:
        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])
        return (u, v)
    return None

def calculate_world_pos(u, v, depth_map, K, extrinsic):
    z = depth_map[v, u]
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x_c = (u - cx) * z / fx
    y_c = (v - cy) * z / fy
    
    point_cam = np.array([x_c, y_c, z, 1.0])
    
    inv_extrinsic = np.eye(4)
    inv_extrinsic[:3, :3] = extrinsic[:3, :3].T
    inv_extrinsic[:3, 3] = -(inv_extrinsic[:3, :3] @ extrinsic[:3, 3])
    flip_mat = np.diag([-1, -1, -1, 1])
    point_world = inv_extrinsic @ flip_mat @ point_cam
    
    return point_world[:3]

def get_ball_position(rgb_img, depth_img, intrinsic, extrinsic, ball_color, visualize=False):

    position = get_ball_centroid(rgb_img, ball_color, visualize=False)

    pos_3d = calculate_world_pos(position[0], position[1], depth_img, intrinsic, extrinsic)
    if visualize:
        draw_ball_center(rgb_img, position, color_name="Red") 

    return pos_3d

