 def controller(state, target_pos, dt):
  """
  Controller function to calculate target velocities and yaw rate.
  :param state: Current state [position_x, position_y, position_z, roll, pitch, yaw]
  :param target_pos: Target position and orientation [target_x, target_y, target_z, target_yaw]
  :param dt: Time step
  :return: (velocity_x_setpoint, velocity_y_setpoint, velocity_z_setpoint, yaw_rate_setpoint)
  """
  
  # Extract current position and target position
  position_x, position_y, position_z, roll, pitch, yaw = state
  target_x, target_y, target_z, target_yaw = target_pos
  
  # PID controller gains (adjust as needed)
  Kp_pos = 2.0 # Position control proportional gain
  Ki_pos = 0.2 # Position control integral gain
  Kd_pos = 0.2 # Position control derivative gain
  
  Kp_vel = 1.0 # Velocity control proportional gain
  Ki_vel = 0.1 # Velocity control integral gain
  Kd_vel = 0.2 # Velocity control derivative gain
  
  Kp_yaw = 1.0 # Yaw control proportional gain
  Ki_yaw = 0.2 # Yaw control integral gain
  Kd_yaw = 0.1 # Yaw control derivative gain
  
  # Initialize PID controller states (integral and previous error)
  integral_pos = 0
  prev_error_pos = 0
  
  integral_vel = 0
  prev_error_vel = 0
  
  integral_yaw = 0
  prev_error_yaw = 0
  
  # Calculate position error and generate target velocities
  # Position control (outer PID controller)
  error_x = target_x - position_x
  integral_pos += error_x * dt
  derivative_x = (error_x - prev_error_pos) / dt if dt > 0 else 0
  velocity_x_setpoint = Kp_pos * error_x + Ki_pos * integral_pos + Kd_pos * derivative_x
  prev_error_pos = error_x
  
  error_y = target_y - position_y
  integral_pos += error_y * dt
  derivative_y = (error_y - prev_error_pos) / dt if dt > 0 else 0
  velocity_y_setpoint = Kp_pos * error_y + Ki_pos * integral_pos + Kd_pos * derivative_y
  prev_error_pos = error_y
  
  error_z = target_z - position_z
  integral_pos += error_z * dt
  derivative_z = (error_z - prev_error_pos) / dt if dt > 0 else 0
  velocity_z_setpoint = Kp_pos * error_z + Ki_pos * integral_pos + Kd_pos * derivative_z
  prev_error_pos = error_z
  
  # Velocity control (inner PID controllers)
  # Generate actual velocity commands based on target velocity
  velocity_x_setpoint, integral_vel, prev_error_vel = pid_control(velocity_x_setpoint, 0, integral_vel, prev_error_vel, Kp_vel, Ki_vel, Kd_vel, dt)
  velocity_y_setpoint, integral_vel, prev_error_vel = pid_control(velocity_y_setpoint, 0, integral_vel, prev_error_vel, Kp_vel, Ki_vel, Kd_vel, dt)
  velocity_z_setpoint, integral_vel, prev_error_vel = pid_control(velocity_z_setpoint, 0, integral_vel, prev_error_vel, Kp_vel, Ki_vel, Kd_vel, dt)
  
  # Yaw control (independent PID controller)
  yaw_rate_setpoint, integral_yaw, prev_error_yaw = pid_control(target_yaw, yaw, integral_yaw, prev_error_yaw, Kp_yaw, Ki_yaw, Kd_yaw, dt)
  
  # Return target velocities and yaw rate
  return (velocity_x_setpoint, velocity_y_setpoint, velocity_z_setpoint, yaw_rate_setpoint)
  
  
  def pid_control(setpoint, measurement, integral, prev_error, kp, ki, kd, dt):
  """
  PID control function
  :param setpoint: Target value
  :param measurement: Current measurement value
  :param integral: Integral term
  :param prev_error: Previous error term
  :param kp: Proportional gain
  :param ki: Integral gain
  :param kd: Derivative gain
  :param dt: Time step
  :return: Control output, updated integral term, and error
  """
  # Calculate error
  error = setpoint - measurement
  integral += error * dt # Update integral term
  derivative = (error - prev_error) / dt if dt > 0 else 0 # Calculate derivative term
  
  # Calculate PID output
  output = kp * error + ki * integral + kd * derivative
  
  # Return updated control output, integral term, and error
  return output, integral, error
