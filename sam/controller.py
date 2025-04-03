import numpy as np

# Persistent storage for integral and previous errors
integral_error_pos = np.array([0.0, 0.0, 0.0])
integral_error_yaw = 0.0
previous_yaw_error = 0.0
filtered_derivative_yaw = 0.0  # Low-pass filter state

# Anti-windup limits
INTEGRAL_POS_LIMIT = np.array([2.0, 2.0, 2.0])
INTEGRAL_YAW_LIMIT = 1.0
FILTER_ALPHA = 0.7  # Smoothing factor for low-pass filter


def controller(state, target_pos, dt):
    global integral_error_pos, integral_error_yaw, previous_yaw_error, filtered_derivative_yaw

    # PID Gains
    Kp_pos = np.array([0.35, 0.35, 0.55])
    Ki_pos = np.array([0.015, 0.015, 0.02])
    Kd_pos = np.array([0.12, 0.12, 0.35])

    Kp_yaw = 0.35
    Ki_yaw = 0.015
    Kd_yaw = 0.25

    # Unpack current state
    current_pos = np.array(state[0:3])
    current_vel = np.array(state[3:6])
    current_yaw = state[5]

    # Compute position error
    pos_error = np.array(target_pos[0:3]) - current_pos
    distance_to_target = np.linalg.norm(pos_error)

    # Update integral error with anti-windup
    integral_error_pos += pos_error * dt
    integral_error_pos = np.clip(integral_error_pos, -INTEGRAL_POS_LIMIT, INTEGRAL_POS_LIMIT)

    # Compute derivative term
    derivative_pos = -current_vel

    # PID control for position
    desired_velocity = (
        Kp_pos * pos_error +
        Ki_pos * integral_error_pos +
        Kd_pos * derivative_pos
    )

    # Yaw control
    yaw_error = np.arctan2(np.sin(target_pos[3] - current_yaw), np.cos(target_pos[3] - current_yaw))
    integral_error_yaw += yaw_error * dt
    integral_error_yaw = np.clip(integral_error_yaw, -INTEGRAL_YAW_LIMIT, INTEGRAL_YAW_LIMIT)

    # Derivative calculation with low-pass filter
    if dt > 0:
        raw_derivative_yaw = (yaw_error - previous_yaw_error) / dt
        filtered_derivative_yaw = FILTER_ALPHA * raw_derivative_yaw + (1 - FILTER_ALPHA) * filtered_derivative_yaw
    else:
        print("Warning: dt is zero, skipping derivative calculation.")
        filtered_derivative_yaw = 0.0

    previous_yaw_error = yaw_error

    # PID control for yaw
    yaw_rate_setpoint = (
        Kp_yaw * yaw_error +
        Ki_yaw * integral_error_yaw +
        Kd_yaw * filtered_derivative_yaw
    )

    # Limit desired velocities
    desired_velocity = np.clip(desired_velocity, -1, 1.5)
    yaw_rate_setpoint = np.clip(yaw_rate_setpoint, -1.74533, 1.74533)

    output = (
        desired_velocity[0],
        desired_velocity[1],
        desired_velocity[2],
        yaw_rate_setpoint
    )

    return output
