from gymnasium.envs.registration import register

register(
    id="RotPendEnv-v0",
    entry_point="environment.rot_pend_env:RotaryPendulumEnv",
    max_episode_steps=4000,  # 4000 * 0.002s = 8s
)
