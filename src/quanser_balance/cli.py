import argparse
from pathlib import Path

import gymnasium as gym
import quanser_balance.envs  # triggers env registration

from quanser_balance.rl.PPO import CustomPPO
from stable_baselines3.common.env_util import make_vec_env


# ─────────────────────────────────────────────
# Paths
# cli.py → src/package/cli.py
# project root → parents[2]
# ─────────────────────────────────────────────

PACKAGE_DIR = Path(__file__).resolve().parent          # src/package
SRC_DIR = PACKAGE_DIR.parent                           # src
ROOT_DIR = SRC_DIR.parent                              # project root

LOGS_DIR_DEFAULT = ROOT_DIR / "logs" / "rotpend" / "ppo"
OUTPUTS_DIR_DEFAULT = ROOT_DIR / "outputs" / "rotpend" / "ppo"


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rotpend",
        description="Train, finetune, or evaluate RotPend with CustomPPO",
    )

    # Mode
    parser.add_argument(
        "--mode",
        choices=("train", "finetune", "eval"),
        default="train",
        help="Execution mode",
    )

    # Environment
    parser.add_argument("--env-id", default="RotPendEnv-v0")
    parser.add_argument("--n-envs", type=int, default=1)

    # Training
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Number of timesteps for this training call",
    )
    parser.add_argument(
        "--reset-num-timesteps",
        action="store_true",
        help="Reset timestep counter (NOT recommended for finetuning)",
    )

    # Paths
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR_DEFAULT)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR_DEFAULT)
    parser.add_argument(
        "--load-path",
        type=Path,
        default=OUTPUTS_DIR_DEFAULT / "rotpend_ppo_model_2.zip",
        help="Model checkpoint to load",
    )
    parser.add_argument(
        "--save-name",
        default="rotpend_ppo_model",
        help="Base filename for saved models",
    )

    # Evaluation
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--deterministic", action="store_true")

    return parser.parse_args()


