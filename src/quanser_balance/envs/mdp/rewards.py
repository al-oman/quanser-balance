import numpy as np
from dataclasses import dataclass, field


@dataclass
class RewardCfg:
    # Pendulum upright reward: cos(theta_pend) → +1 at top, -1 at bottom
    upright_w: float = 1.0

    sideways_w: float = -0.0 #maybe stupid

    angle_w: float = -0.0

    # Energy-based reward: encourage injecting energy to reach upright
    # Uses normalized pendulum energy: E/E_target where E_target = m*g*L (potential at top)
    energy_w: float = 0.0

    # Penalize arm deviation from center
    arm_pos_w: float = -1.0

    # Penalize arm velocity (smooth control)
    arm_vel_w: float = -0.01

    # Penalize pendulum velocity (encourage stillness at top)
    pend_vel_w: float = -0.5

    # Penalize large voltage commands (efficiency)
    voltage_w: float = -0.01

    # Linear bonus that ramps from 0 at threshold to balance_bonus at upright
    balance_bonus: float = 0.0
    balance_threshold: float = 0.3  # rad — region where bonus kicks in

    # Pendulum physical params (for energy reward)
    pend_mass: float = 0.024
    pend_length: float = 0.129
    gravity: float = 9.81


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

    reward += cfg.angle_w * (theta_pend**2)

    # Cosine reward: +1 when upright (theta=0), -1 when hanging (theta=pi)
    reward += cfg.upright_w * np.cos(theta_pend)

    # Sine penalty:
    reward += cfg.sideways_w * np.sin(theta_pend) ** 2

    # Energy-based reward for swing-up
    if cfg.energy_w != 0.0:
        half_L = cfg.pend_length / 2.0
        E_pot = cfg.pend_mass * cfg.gravity * half_L * (np.cos(theta_pend) - 1.0)  # 0 at top, negative below
        E_kin = 0.5 * cfg.pend_mass * (half_L * dtheta_pend) ** 2
        E_total = E_pot + E_kin
        E_target = 0.0  # energy at upright with zero velocity
        energy_error = abs(E_total - E_target)
        reward += cfg.energy_w * np.exp(-10.0 * energy_error)

    # Penalize arm deviation from center
    reward += cfg.arm_pos_w * theta_arm ** 2

    # Penalize arm velocity
    reward += cfg.arm_vel_w * dtheta_arm ** 2

    # Penalize pendulum velocity
    reward += cfg.pend_vel_w * dtheta_pend ** 2

    # Penalize voltage
    reward += cfg.voltage_w * voltage ** 2

    # Linear bonus near upright: ramps from 0 at threshold to balance_bonus at theta=0
    if abs(theta_pend) < cfg.balance_threshold:
        reward += cfg.balance_bonus * (1.0 - abs(theta_pend) / cfg.balance_threshold)

    return reward
