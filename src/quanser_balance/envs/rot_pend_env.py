__credits__ = ["Kallinteris-Andreas"]

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from quanser_balance.envs.mdp.rewards import rot_pend_reward, RewardCfg
from quanser_balance.envs.mdp.curriculum import RotaryPendulumCurriculum

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 2.04,
}

from pathlib import Path
ASSETS_DIR = Path(__file__).parent / "assets"


class RotaryPendulumEnv(MujocoEnv, utils.EzPickle):
    """
    ## Description
    This environment is the Cartpole environment, based on the work of Barto, Sutton, and Anderson in ["Neuronlike adaptive elements that can solve difficult learning control problems"](https://ieeexplore.ieee.org/document/6313077),
    just like in the classic environments, but now powered by the Mujoco physics simulator - allowing for more complex experiments (such as varying the effects of gravity).
    This environment consists of a cart that can be moved linearly, with a pole attached to one end and having another end free.
    The cart can be pushed left or right, and the goal is to balance the pole on top of the cart by applying forces to the cart.


    ## Action Space
    The agent take a 1-element vector for actions.

    The action space is a continuous `(action)` in `[-3, 3]`, where `action` represents
    the numerical force applied to the cart (with magnitude representing the amount of
    force and sign representing the direction)

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint |Type (Unit)|
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
    | 0   | Force applied on the cart | -3          | 3           | slider                           | slide | Force (N) |


    ## Observation Space
    The observation space consists of the following parts (in order):
    - *qpos (2 element):* Position values of the robot's cart and pole.
    - *qvel (2 elements):* The velocities of cart and pole (their derivatives).

    The observation space is a `Box(-Inf, Inf, (4,), float64)` where the elements are as follows:

    | Num | Observation                                   | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
    | --- | --------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------- |
    | 0   | position of the cart along the linear surface | -Inf | Inf | slider                           | slide | position (m)              |
    | 1   | vertical angle of the pole on the cart        | -Inf | Inf | hinge                            | hinge | angle (rad)               |
    | 2   | linear velocity of the cart                   | -Inf | Inf | slider                           | slide | velocity (m/s)            |
    | 3   | angular velocity of the pole on the cart      | -Inf | Inf | hinge                            | hinge | angular velocity (rad/s)  |


    ## Rewards
    The goal is to keep the inverted pendulum stand upright (within a certain angle limit) for as long as possible - as such, a reward of +1 is given for each timestep that the pole is upright.

    The pole is considered upright if:
    $|angle| < 0.2$.

    and `info` also contains the reward.


    ## Starting State
    The initial position state is $\\mathcal{U}_{[-reset\\_noise\\_scale \times I_{2}, reset\\_noise\\_scale \times I_{2}]}$.
    The initial velocity state is $\\mathcal{U}_{[-reset\\_noise\\_scale \times I_{2}, reset\\_noise\\_scale \times I_{2}]}$.

    where $\\mathcal{U}$ is the multivariate uniform continuous distribution.


    ## Episode End
    ### Termination
    The environment terminates when the Inverted Pendulum is unhealthy.
    The Inverted Pendulum is unhealthy if any of the following happens:

    1. Any of the state space values is no longer finite.
    2. The absolute value of the vertical angle between the pole and the cart is greater than 0.2 radians.

    ### Truncation
    The default duration of an episode is 1000 timesteps.


    ## Arguments
    InvertedPendulum provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
    These parameters can be applied during `gymnasium.make` in the following way:

    ```python
    import gymnasium as gym
    env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1)
    ```

    | Parameter               | Type       | Default                 | Description                                                                                   |
    |-------------------------|------------|-------------------------|-----------------------------------------------------------------------------------------------|
    | `xml_file`              | **str**    |`"inverted_pendulum.xml"`| Path to a MuJoCo model                                                                        |
    | `reset_noise_scale`     | **float**  | `0.01`                  | Scale of random perturbations of initial position and velocity (see `Starting State` section) |

    ## Version History
    * v5:
        - Minimum `mujoco` version is now 2.3.3.
        - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
        - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
        - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the MuJoCo environments.
        - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
        - Fixed bug: `healthy_reward` was given on every step (even if the Pendulum is unhealthy), now it is only given if the Pendulum is healthy (not terminated) (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/500)).
        - Added `xml_file` argument.
        - Added `reset_noise_scale` argument to set the range of initial states.
        - Added `info["reward_survive"]` which contains the reward.
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3.
    * v3: This environment does not have a v3 release. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
    * v2: All continuous control environments now use mujoco-py >= 1.5. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
    * v1: max_time_steps raised to 1000 for robot based tasks (including inverted pendulum).
    * v0: Initial versions release.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    # Qube Servo 3 motor parameters
    MOTOR_KT = 0.0422       # Torque constant (N·m/A)
    MOTOR_KM = 0.0422       # Back-EMF constant (V/(rad/s))
    MOTOR_RM = 7.5          # Terminal resistance (Ω)
    VOLTAGE_MAX = 10.0      # Recommended operating voltage limit (V)
    VOLTAGE_DEADBAND = 0.65 # Amplifier deadband (V)

    def __init__(
        self,
        xml_file: str | None = None,
        frame_skip: int = 2,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.01,
        reset_theta_range=0.05,
        reset_theta_dot_range=0.01,
        curriculum_stage: int = 2,
        **kwargs,
    ):
        if xml_file is None:
            xml_file = ASSETS_DIR / "rot_pend.xml"
        xml_file = str(xml_file)

        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, curriculum_stage=curriculum_stage, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

        self._reset_noise_scale = reset_noise_scale
        self.reward_cfg = RewardCfg(energy_w=0.5)
        self.curriculum = RotaryPendulumCurriculum(start_stage=curriculum_stage)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # Override action space: policy outputs voltage, not torque
        self.action_space = Box(
            low=-self.VOLTAGE_MAX,
            high=self.VOLTAGE_MAX,
            shape=(1,),
            dtype=np.float64,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }

    def reward(self, obs, terminated, voltage=0.0):
        return rot_pend_reward(obs, terminated, self.reward_cfg, voltage=voltage)
    
    def voltage_to_torque(self, voltage, arm_angular_vel):
        """
        Convert voltage command to motor torque using DC motor model.

        V = Rm * i + Km * ω   (voltage equation)
        τ = Kt * i            (torque equation)

        Solving for τ:
            i = (V - Km * ω) / Rm
            τ = Kt * (V - Km * ω) / Rm
        """
        current = (voltage - self.MOTOR_KM * arm_angular_vel) / self.MOTOR_RM
        torque = self.MOTOR_KT * current
        return torque

    def step(self, action):
        # Policy outputs voltage; apply deadband and convert to torque for MuJoCo
        voltage = np.clip(action[0], -self.VOLTAGE_MAX, self.VOLTAGE_MAX)
        if abs(voltage) < self.VOLTAGE_DEADBAND:
            voltage = 0.0
        arm_angular_vel = self.data.qvel[0]
        torque = self.voltage_to_torque(voltage, arm_angular_vel)
        self.do_simulation(np.array([torque]), self.frame_skip)

        observation = self._get_obs()

        pend_limit = self.curriculum.termination_limits
        terminated = bool(
            not np.isfinite(observation).all()
            or np.abs(observation[0]) > (np.pi - 0.2)
            or np.abs(observation[1]) > pend_limit
        )

        reward = self.reward(observation, terminated, voltage=voltage)

        info = {"reward": reward}

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def reset_model(self):
        pend_angle = self.curriculum.get_initial_pend_angle(self.np_random)
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = np.array([0.0, pend_angle]) + self.np_random.uniform(
            size=self.model.nq, low=noise_low, high=noise_high
        )
        qvel = np.zeros(self.model.nv) + self.np_random.uniform(
            size=self.model.nv, low=noise_low, high=noise_high
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        """
        Added clipping to observation to enforce limits on rotary arm
        """

        obs = np.concatenate([self.data.qpos, self.data.qvel]).ravel()

        obs = np.clip(
        obs,
        [-3*np.pi/2, -20.0, -50.0, -50.0],   # min
        [ 3*np.pi/2,  20.0,  50.0,  50.0],   # max
    )

        return obs
