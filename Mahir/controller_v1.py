import numpy as np

def controller(state, target_pos, dt):
    # Controller gains
    Kp_pos = np.array([0.5, 0.5, 1.0])  # Proportional gain for position
    Kd_pos = np.array([0.3, 0.3, 0.6])  # Derivative gain for velocity
    Kp_yaw = 1.0                       # Proportional gain for yaw

    # Unpack current state
    current_pos = np.array(state[0:3])
    current_yaw = state[5]

    # Calculate position error
    pos_error = np.array(target_pos[0:3]) - current_pos

    # Desired velocity (PD Control for Position)
    desired_velocity = Kp_pos * pos_error

    # Yaw control (Proportional Control for Yaw)
    yaw_error = target_pos[3] - current_yaw
    yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi  # Normalize yaw error between [-pi, pi]
    yaw_rate_setpoint = Kp_yaw * yaw_error

    # Assuming drone dynamics and simulation already include damping,
    # we do not explicitly calculate velocity errors here.

    # Limit desired velocities
    desired_velocity = np.clip(desired_velocity, -1, 1)
    yaw_rate_setpoint = np.clip(yaw_rate_setpoint, -1.74533, 1.74533)

    output = (
        desired_velocity[0],
        desired_velocity[1],
        desired_velocity[2],
        yaw_rate_setpoint
    )

    return output
