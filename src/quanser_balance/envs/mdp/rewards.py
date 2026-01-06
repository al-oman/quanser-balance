import numpy as np

class RewardCfg:
    def __init__(self, theta_w=1.0, pos_w=-1.0):
        self.theta_w = theta_w
        self.pos_w = pos_w

def rot_pend_reward(obs, terminated, cfg):
    """
    Uses cfg class to define reward weights
    """
    if terminated:
        return 0.0
    theta_reward = cfg.theta_w * np.cos(obs[1])
    pos_reward = cfg.pos_w * obs[0] ** 2
    return theta_reward + pos_reward