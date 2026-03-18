import gymnasium as gym
import numpy as np
import csv
from pathlib import Path
import quanser_balance.envs  # triggers register()

# --- Step test parameters (match test.py) ---
STEP_VOLTAGE = 2.0     # voltage applied during the step (V)
T_SETTLE     = 1.0     # seconds before step (baseline)
T_STEP       = 1.0     # duration the step voltage is held (s)
T_AFTER      = 1.0     # seconds to record after step ends
RENDER_EVERY = 20      # render every Nth step (500Hz/20 = 25 FPS)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = SCRIPT_DIR.parent / "outputs" / "step_response_sim.csv"

env = gym.make("RotPendEnv-v0", render_mode="human", max_episode_steps=100_000)
dt = env.unwrapped.dt  # sim step duration

total_time = T_SETTLE + T_STEP + T_AFTER
total_steps = int(total_time / dt)

log = []

try:
    obs, info = env.reset(seed=123)
    t = 0.0

    for step in range(total_steps):
        # Step function: voltage on only during [T_SETTLE, T_SETTLE + T_STEP)
        if T_SETTLE <= t < T_SETTLE + T_STEP:
            voltage = STEP_VOLTAGE
        else:
            voltage = 0.0

        obs, reward, terminated, truncated, info = env.step(np.array([voltage]))
        if step % RENDER_EVERY == 0:
            env.render()

        theta_1     = obs[0]
        theta_2     = obs[1]
        theta_1_dot = obs[2]
        theta_2_dot = obs[3]
        actual_voltage = info["voltage"]
        current        = info["current"]

        log.append([t, actual_voltage, theta_1, theta_2, theta_1_dot, theta_2_dot, current])

        print(f"t={t:5.3f}s  V={actual_voltage:+6.2f}  "
              f"theta_1={theta_1:+.4f}  theta_2={theta_2:+.4f}  "
              f"theta_1_dot={theta_1_dot:+7.2f}  theta_2_dot={theta_2_dot:+7.2f}  "
              f"I={current:+.4f}")

        t += dt

        if terminated or truncated:
            print(f"Episode ended at t={t:.3f}s")
            break

finally:
    env.close()

    if log:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "voltage", "motor_pos", "pend_pos",
                             "motor_spd", "pend_spd", "current"])
            writer.writerows(log)
        print(f"Saved {len(log)} samples to {OUTPUT_CSV}")
