import numpy as np
import csv
import os
from datetime import datetime

# ————————————————
# Lab constants & thresholds
# ————————————————
TELLO_VICON_NAME = "tello_marker2"
TELLO_ID          = "101"

POSITION_ERROR = 0.5     # m, when to switch to next waypoint
YAW_ERROR      = 0.5     # rad, (unused here but available)

# clamp linear velocities so that run.py’s *100 → cm/s* is within ±100cm/s
MAX_SPEED     = 0.5      # m/s  
# clamp yaw rate so that run.py’s deg(v_yaw) is within ±100°/s → 100° = 1.745rad
MAX_YAW_RATE  = np.radians(50)  

# how much yaw-error we allow into the PID
MAX_YAW_ERROR = np.radians(30)

POSITION_YAW_ENABLE_THRESH = 0.2  # meters

# PID gains for [x, y, z]
Kp_pos = np.array([0.6,   0.6,   6.0])
Kd_pos = np.array([0.05,  0.05,  0.3])
Ki_pos = np.array([0.02,  0.02,  0.1])

Kp_pos = np.array([0.3,   0.3,   2.0])
Kd_pos = np.array([0.025,  0.025,  0.1])
Ki_pos = np.array([0.01,  0.01,  0.03])

Kd_pos = np.array([0.025,  0.035,  0.1]) 

# PID gains for yaw
Kp_yaw = 0.5
Kd_yaw = 0.02
Ki_yaw = 0.002

# ————————————————
# Persistent PID state
# ————————————————
_last_time_ms      = None
_integral_error_p  = np.zeros(3)
_integral_error_y  = 0.0
_prev_error_p      = np.zeros(3)
_prev_error_y      = 0.0

# ————————————————
# CSV Logging setup
# ————————————————
LOG_DIR  = "logs"
LOG_FILE = os.path.join(LOG_DIR, f"log_{datetime.now():%Y%m%d_%H%M%S}.csv")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "time",
        "px","py","pz","roll","pitch","yaw",
        "tx","ty","tz","tyaw",
        "ex","ey","ez","eyaw_raw",
        "vx","vy","vz","yaw_rate"
    ])


def log_to_csv(state, target, err_p, err_y_raw, vel, yaw_rate):
    """Append one line of data to the flight log."""
    row = [
        datetime.now().isoformat(),
        *state,
        *target,
        *err_p,
        err_y_raw,
        *vel,
        yaw_rate
    ]
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)


def controller(state, target_pos, timestamp_ms):
    """
    PID controller for Tello + Vicon lab.

    Args:
      state        : [x(m), y(m), z(m), roll, pitch, yaw(rad)]
      target_pos   : (x(m), y(m), z(m), yaw(rad))
      timestamp_ms : current time in milliseconds

    Returns:
      vx, vy, vz   : setpoint m/s   (clamped ±MAX_SPEED)
      yaw_rate     : setpoint rad/s (clamped ±MAX_YAW_RATE)
    """
    global _last_time_ms, _integral_error_p, _integral_error_y, _prev_error_p, _prev_error_y

    # ————————————————
    # 1) Compute dt in seconds
    # ————————————————
    if _last_time_ms is None:
        dt = 0.0
    else:
        dt = (timestamp_ms - _last_time_ms) / 1000.0
    _last_time_ms = timestamp_ms

    # ————————————————
    # 2) Extract and compute raw errors
    # ————————————————
    pos      = np.array(state[0:3])
    yaw      = state[5]
    err_p    = np.array(target_pos[0:3]) - pos
    raw_err_y= (target_pos[3] - yaw + np.pi) % (2*np.pi) - np.pi
    err_y    = np.clip(raw_err_y, -MAX_YAW_ERROR, MAX_YAW_ERROR)

    # ————————————————
    # 3) Update integral & derivative (only if dt>0)
    # ————————————————
    if dt > 0:
        _integral_error_p += err_p * dt
        _integral_error_y += err_y * dt
        d_err_p = (err_p - _prev_error_p) / dt
        d_err_y = (err_y - _prev_error_y) / dt
    else:
        d_err_p = np.zeros(3)
        d_err_y = 0.0

    # ————————————————
    # 4) PID Law
    # ————————————————
    pos_dist = np.linalg.norm(err_p)
    vel_cmd    = (Kp_pos * err_p) + (Kd_pos * d_err_p) + (Ki_pos * _integral_error_p)
    if pos_dist >= POSITION_YAW_ENABLE_THRESH:
        yaw_rate = 0.0
        _integral_error_y = 0.0
    else:
        yaw_rate   = (Kp_yaw * err_y) + (Kd_yaw * d_err_y) + (Ki_yaw * _integral_error_y)

    # ————————————————
    # 5) Clip to lab limits
    # ————————————————
    
    cy, sy = np.cos(yaw), np.sin(yaw)
    vx_w, vy_w = vel_cmd[0], vel_cmd[1]
    vel_cmd[0] =  cy*vx_w + sy*vy_w
    vel_cmd[1] = -sy*vx_w + cy*vy_w
    vel_cmd    = np.clip(vel_cmd, -MAX_SPEED, MAX_SPEED)
    yaw_rate   = float(np.clip(yaw_rate, -MAX_YAW_RATE, MAX_YAW_RATE))

    # ————————————————
    # 6) Save for next derivative
    # ————————————————
    _prev_error_p = err_p
    _prev_error_y = err_y

    # ————————————————
    # 7) Log it!
    # ————————————————


    if abs(vel_cmd[0]) > 0.5:
        vel_cmd[0] = ((vel_cmd[0])/abs(vel_cmd[0]))*0.5

    if abs(vel_cmd[1]) > 0.5:
        vel_cmd[1] = ((vel_cmd[1])/abs(vel_cmd[1]))*0.5

    if abs(vel_cmd[2]) > 0.5:
        vel_cmd[2] = ((vel_cmd[2])/abs(vel_cmd[2]))*0.5

    log_to_csv(state, target_pos, err_p, raw_err_y, vel_cmd, yaw_rate)


    return vel_cmd[0], vel_cmd[1], vel_cmd[2], yaw_rate