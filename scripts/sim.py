import gymnasium as gym
import quanser_balance.envs  # triggers register()

env = gym.make("InvPendEnv-v0", render_mode="human")
obs, info = env.reset(seed=123, options={"low": -0.1, "high": 0.1})

over = False
total_reward = 0

while not over:
    action = [0.0]
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    env.render()
    over = terminated or truncated
    print(terminated, truncated)

env.close()