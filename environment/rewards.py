import numpy as np
from dataclasses import dataclass, field


@dataclass
class RewardCfg:
    # reward for each timestep alive
    time_w: float = 1.0

    # Penalty for angle deviation
    angle_w: float = -1.0

    # Penalize pendulum velocity (encourage stillness at top)
    pend_vel_w: float = -0.5

    # Penalize arm deviation from center
    arm_pos_w: float = -0.0

    # Penalize arm velocity (smooth control)
    arm_vel_w: float = -0.0



def rot_pend_reward(obs, terminated, cfg, voltage=0.0):
    """
    obs: [theta_arm, theta_pend, dtheta_arm, dtheta_pend]
    """
    if terminated:
        return 0.0

    theta_arm = obs[0]
    theta_pend = obs[1]
    dtheta_arm = obs[2]
    dtheta_pend = obs[3]

    reward = 0.0

    reward += cfg.time_w

    reward += cfg.angle_w * (theta_pend**2)

    # Penalize pendulum velocity
    reward += cfg.pend_vel_w * dtheta_pend ** 2

    # Penalize arm deviation from center
    reward += cfg.arm_pos_w * theta_arm ** 2

    # Penalize arm velocity
    reward += cfg.arm_vel_w * dtheta_arm ** 2

    return reward
