from gymnasium.envs.registration import register

register(
    id="CustomInvertedPendulumEnv-v0",
    entry_point="quanser_balance.envs.inverted_pendulum_env:InvertedPendulumEnv",
)

register(
    id="InvPendEnv-v0",
    entry_point="quanser_balance.envs.inv_pend_env_2:InvertedPendulumEnv",
)