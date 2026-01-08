import gymnasium as gym
import quanser_balance.envs  # triggers register()
from quanser_balance.rl.PPO import CustomPPO

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = ROOT_DIR / "outputs" / "rotpend" / "ppo"


env = gym.make("RotPendEnv-v0", render_mode="human")
model = CustomPPO.load(OUTPUTS_DIR / "rotpend_ppo_model_3", env=env)
obs, info = env.reset()

try:
    obs, info = env.reset(seed=123, options={"low": -0.1, "high": 0.1})

    over = False
    total_reward = 0

    while not over:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        over = terminated or truncated
        print(action, reward)
finally:
    env.close()