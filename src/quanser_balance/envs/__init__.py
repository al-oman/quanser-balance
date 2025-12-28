from gymnasium.envs.registration import register

register(
    id="RotPendEnv-v0",
    entry_point="quanser_balance.envs.rot_pend_env:RotaryPendulumEnv",
)

register(
    id="InvPendEnv-v0",
    entry_point="quanser_balance.envs.inv_pend_env_2:InvertedPendulumEnv",
)