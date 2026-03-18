import argparse
import gymnasium as gym
import quanser_balance.envs  # triggers register()

from quanser_balance.rl.PPO import CustomPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, default=None,
                    help="Path to a saved model to resume training from")
parser.add_argument("--steps", type=int, default=500_000,
                    help="Total training timesteps")
parser.add_argument("--curriculum", type=int, default=2,
                    help="Curriculum stage (0=balance, 1=recover, 2=swing-up)")
args = parser.parse_args()

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
LOGS_DIR = ROOT_DIR / "logs" / "rotpend" / "ppo"
OUTPUTS_DIR = ROOT_DIR / "outputs" / "rotpend" / "ppo"

SAVE_TAG = datetime.now().strftime("%m_%d_%H_%M")
TOTAL_TIMESTEPS = args.steps
LOAD = args.load

# Eval callback saves the best model with step count in the filename
env_kwargs = {"curriculum_stage": args.curriculum}
eval_env = make_vec_env("RotPendEnv-v0", n_envs=1, env_kwargs=env_kwargs)
best_model_dir = OUTPUTS_DIR / f"best_{SAVE_TAG}"
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(best_model_dir),
    log_path=str(best_model_dir),
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
    verbose=1,
)

if LOAD:
    load_path = OUTPUTS_DIR / LOAD
    train_env = make_vec_env("RotPendEnv-v0", n_envs=8, env_kwargs=env_kwargs)
    model = CustomPPO.load(load_path, env=train_env)
    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                reset_num_timesteps=False,
                callback=eval_callback)
    model.save(OUTPUTS_DIR / f"rotpend_ppo_{SAVE_TAG}")
    train_env.close()
else:
    train_env = make_vec_env("RotPendEnv-v0", n_envs=8, env_kwargs=env_kwargs)

    model = CustomPPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=str(LOGS_DIR),
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(OUTPUTS_DIR / f"rotpend_ppo_{SAVE_TAG}_final")
    train_env.close()

eval_env.close()

env = gym.make("RotPendEnv-v0", render_mode="human", curriculum_stage=args.curriculum)
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
