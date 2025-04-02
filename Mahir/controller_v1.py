import numpy as np

# Persistent state for PID (globals used for compatibility with simulator)
integral_error_pos = np.zeros(3)
integral_error_yaw = 0.0
previous_error_pos = np.zeros(3)
previous_error_yaw = 0.0
last_target = np.array([0.0, 0.0, 0.0, 0.0]) 

def controller(state, target_pos, dt):
    global integral_error_pos, integral_error_yaw
    global previous_error_pos, previous_error_yaw
    global last_target
    MAX_YAW_ERROR = np.radians(30) 

    # PID gains
    Kp_pos = np.array([0.6, 0.6, 6.0])
    Kd_pos = np.array([0.05, 0.05, 0.3])   # Tune slowly
    Ki_pos = np.array([0.02, 0.02, 0.1])

    Kp_yaw = 0.01
    Kd_yaw = 0.0
    Ki_yaw = 0.001

    # Extract current state
    current_pos = np.array(state[0:3])
    current_yaw = state[5]

    # Compute errors
    pos_error = np.array(target_pos[0:3]) - current_pos
    yaw_error = (target_pos[3] - current_yaw + np.pi) % (2 * np.pi) - np.pi
    yaw_error_clipped = np.clip(yaw_error, -MAX_YAW_ERROR, MAX_YAW_ERROR)

    # Integral error
    integral_error_pos += pos_error * dt
    integral_error_yaw += yaw_error_clipped * dt

    # Derivative error
    d_pos_error = (pos_error - previous_error_pos) / dt
    d_yaw_error = (yaw_error_clipped - previous_error_yaw) / dt

    # PID control
    velocity_cmd = Kp_pos * pos_error + Kd_pos * d_pos_error + Ki_pos * integral_error_pos
    yaw_rate_cmd = Kp_yaw * yaw_error_clipped + Kd_yaw * d_yaw_error + Ki_yaw * integral_error_yaw

    # Clip outputs
    velocity_cmd = np.clip(velocity_cmd, -1, 1)
    yaw_rate_cmd = np.clip(yaw_rate_cmd, -0.5, 0.5)

    # Save previous error
    previous_error_pos = pos_error
    previous_error_yaw = yaw_error

    return velocity_cmd[0], velocity_cmd[1], velocity_cmd[2], yaw_rate_cmd
