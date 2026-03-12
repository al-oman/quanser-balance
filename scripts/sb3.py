import gymnasium as gym
import quanser_balance.envs  # triggers register()

from quanser_balance.rl.PPO import CustomPPO
from stable_baselines3.common.env_util import make_vec_env

from datetime import datetime
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
LOGS_DIR = ROOT_DIR / "logs" / "rotpend" / "ppo"
OUTPUTS_DIR = ROOT_DIR / "outputs" / "rotpend" / "ppo"

SAVE_TAG = datetime.now().strftime("%m_%d_%H_%M")
LOAD = False

if LOAD:
    train_env = make_vec_env("RotPendEnv-v0", n_envs=1)
    model = CustomPPO.load(OUTPUTS_DIR / "rotpend_ppo_model_2", env=train_env)
    model.learn(total_timesteps=100_000,
                reset_num_timesteps=False)
    model.save(OUTPUTS_DIR / f"rotpend_ppo_{SAVE_TAG}")
    train_env.close()
else:
    train_env = make_vec_env("RotPendEnv-v0", n_envs=1)

    model = CustomPPO(
        "MlpPolicy",
        train_env,
        verbose=1, 
        tensorboard_log=str(LOGS_DIR),
    )

    model.learn(total_timesteps=250_000)
    model.save(OUTPUTS_DIR / f"rotpend_ppo_{SAVE_TAG}")
    train_env.close()

env = gym.make("RotPendEnv-v0", render_mode="human")
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