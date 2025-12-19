import gymnasium as gym
import mujoco
import numpy as np
from importlib.resources import files

class InvertedPendulumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, render_mode=None):
        xml = files("quanser_balance.sim.assets").joinpath("inverted_pendulum.xml")
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation: concatenate qpos + qvel
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

    def reset(self, *, seed=None, options=None):
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        self.data.qpos[1] = 0.2
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = 0.0
        mujoco.mj_step(self.model, self.data)
        return self._get_obs(), 0.0, False, False, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
