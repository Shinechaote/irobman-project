import mujoco
import numpy as np

# https://math.stackexchange.com/questions/3572459/how-to-compute-the-orientation-error-between-two-3d-coordinate-frames
def quat_error(q_des, q_curr):
    q_inv = np.array([q_curr[0], -q_curr[1], -q_curr[2], -q_curr[3]])

    w1, x1, y1, z1 = q_des
    w2, x2, y2, z2 = q_inv

    q_err = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

    if q_err[0] < 0:
        q_err = -q_err

    return 2.0 * q_err[1:]

def get_actions(sim, desired_pos, desired_quat, alpha, lm_penalty_term, dt=0.002):
    m = sim.model
    d = sim.data

    # Create a copy of the data so we can calculate the forward kinematics without setting the actual simulation's values
    d_ik = mujoco.MjData(m)
    d_ik.qpos = d.qpos.copy()

    ee_body_name = sim.robot_settings.get("ee_body_name", "hand")
    ee_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "fingertip")

    arm_pairs = sim.robot.arm_pairs
    arm_vindices = [vidx for _, vidx in arm_pairs]
    q_indices = [qidx for qidx, _ in arm_pairs]

    q = np.array([d.qpos[qidx] for qidx in q_indices])

    lam = lm_penalty_term
    max_iters = 10
    tol = 1e-4

    for i in range(max_iters):

        for i, qidx in enumerate(q_indices):
            d_ik.qpos[qidx] = q[i]
            
        mujoco.mj_kinematics(m, d_ik)
        mujoco.mj_comPos(m, d_ik) 

        ee_pos = d_ik.site_xpos[site_id].copy()

        current_quat = np.zeros(4)
        mujoco.mju_mat2Quat(current_quat, d_ik.site_xmat[site_id])

        # weigh positional error lower than rotational error
        w_pos = 1.0
        w_rot = 0.5
        W = np.diag([w_pos]*3 + [w_rot]*3)

        pos_error = desired_pos - ee_pos
        rot_error = quat_error(desired_quat, current_quat)
        error = np.concatenate((w_pos * pos_error, w_rot * rot_error))

        if np.linalg.norm(error) < tol:
            break

        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d_ik, jacp, jacr, ee_pos, ee_body_id)

        J = np.vstack((jacp, jacr))[:, arm_vindices]

        J_w = W @ J
        e_w = W @ error

        H = J_w.T @ J_w + (lam ** 2) * np.eye(len(q))
        g = J_w.T @ (alpha * e_w)

        # more stable because we don't actually have to compute the inverse
        dq = np.linalg.solve(H, g)

        q_candidate = q + dq

        for i, qidx in enumerate(q_indices):
            d_ik.qpos[qidx] = q_candidate[i]
            
        mujoco.mj_kinematics(m, d_ik)
        mujoco.mj_comPos(m, d_ik)

        ee_pos_new = d_ik.site_xpos[site_id].copy()
        mujoco.mju_mat2Quat(current_quat, d_ik.site_xmat[site_id])

        pos_err_new = desired_pos - ee_pos_new
        rot_err_new = quat_error(desired_quat, current_quat)
        error_new = np.concatenate((w_pos * pos_err_new, w_rot * rot_err_new))

        # only accept new q if error is lower than before
        if np.linalg.norm(error_new) < np.linalg.norm(error):
            q = q_candidate
            lam *= 0.8
        else:
            lam *= 2.0

    return q

