import numpy as np

# Parameter settings
dt = 0.1  # Time step
v_max = 1.0  # Maximum velocity
yaw_rate_max = 1.0  # Maximum yaw rate
position_tolerance = 0.05  # Position error tolerance
yaw_tolerance = np.radians(2)  # Yaw error tolerance (2 degrees)

def wrap_angle(angle):
    """ Normalize angle to [-π, π] """
    return (angle + np.pi) % (2 * np.pi) - np.pi

# UAV dynamics model
def system_model(state, control_input, dt):
    """
    Simple UAV dynamics model
    state: [x, y, z, roll, pitch, yaw] (current state)
    control_input: [v_x_body, v_y_body, v_z, yaw_rate] (control input)
    dt: time step
    Returns: Next state
    """
    x, y, z, _, _, yaw = state
    v_x_body, v_y_body, v_z, yaw_rate = control_input

    # Convert velocity from body frame to world frame
    v_x_world = v_x_body * np.cos(yaw) - v_y_body * np.sin(yaw)
    v_y_world = v_x_body * np.sin(yaw) + v_y_body * np.cos(yaw)

    # Update position
    x_new = x + v_x_world * dt
    y_new = y + v_y_world * dt
    z_new = z + v_z * dt
    yaw_new = wrap_angle(yaw + yaw_rate * dt)  # Ensure yaw angle does not overflow

    return np.array([x_new, y_new, z_new, 0, 0, yaw_new])

# DOBC Controller
def controller(state, target_pos, dt):
    """
    Compute UAV velocity commands using DOBC controller
    state: [x, y, z, roll, pitch, yaw] (current state)
    target_pos: [x_target, y_target, z_target, yaw_target] (target state)
    dt: time step
    Returns: (vx_body, vy_body, vz, yaw_rate)
    """
    # Current state
    x0, y0, z0, _, _, yaw0 = state
    x_target, y_target, z_target, yaw_target = target_pos

    # Compute position error
    error_pos = np.array([x_target - x0, y_target - y0, z_target - z0])

    # Compute vx, vy in world frame
    Kp_pos = 0.8  # Position proportional gain
    v_x_world = Kp_pos * error_pos[0]
    v_y_world = Kp_pos * error_pos[1]
    v_z_world = Kp_pos * error_pos[2]

    # Convert to UAV body frame
    v_x_body = v_x_world * np.cos(yaw0) + v_y_world * np.sin(yaw0)
    v_y_body = -v_x_world * np.sin(yaw0) + v_y_world * np.cos(yaw0)

    # Yaw control (PID method)
    Kp_yaw = 1.0  # Yaw angle proportional gain
    error_yaw = wrap_angle(yaw_target - yaw0)  # Compute yaw error
    yaw_rate = Kp_yaw * error_yaw  # Rotate UAV gradually towards `yaw_target`

    # Predict system state
    control_input = np.array([v_x_body, v_y_body, v_z_world, yaw_rate])
    predicted_state = system_model(state, control_input, dt)

    # Compute disturbance compensation
    wc = 2
    Kd = 0.9
    disturbance = state - predicted_state
    disturbance_estimate = wc * disturbance  # Estimate disturbance

    # Adjust control input using disturbance compensation
    v_x_compensated = v_x_body + Kd * disturbance_estimate[0]
    v_y_compensated = v_y_body + Kd * disturbance_estimate[1]
    v_z_compensated = v_z_world + Kd * disturbance_estimate[2]
    yaw_rate_compensated = yaw_rate + Kd * disturbance_estimate[5]

    # Limit velocity and yaw rate
    v_x_compensated = np.clip(v_x_compensated, -v_max, v_max)
    v_y_compensated = np.clip(v_y_compensated, -v_max, v_max)
    v_z_compensated = np.clip(v_z_compensated, -v_max, v_max)
    yaw_rate_compensated = np.clip(yaw_rate_compensated, -yaw_rate_max, yaw_rate_max)

    output = (v_x_compensated, v_y_compensated, v_z_compensated, yaw_rate_compensated)

    return output


