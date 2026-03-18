import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

hw = pd.read_csv(OUTPUT_DIR / "step_response.csv")
sim = pd.read_csv(OUTPUT_DIR / "step_response_sim.csv")

fig, axes = plt.subplots(6, 1, sharex=True, figsize=(10, 14))

# theta_1 (motor/arm position)
axes[0].plot(hw["time"], hw["motor_pos"], label="hardware")
axes[0].plot(sim["time"], sim["motor_pos"], label="sim", linestyle="--")
axes[0].set_ylabel("theta_1 (rad)")
axes[0].legend()

# theta_2 (pendulum position)
axes[1].plot(hw["time"], hw["pend_pos"], label="hardware")
axes[1].plot(sim["time"], sim["pend_pos"], label="sim", linestyle="--")
axes[1].set_ylabel("theta_2 (rad)")
axes[1].legend()

# theta_1_dot (motor velocity)
axes[2].plot(hw["time"], hw["motor_spd"], label="hardware")
axes[2].plot(sim["time"], sim["motor_spd"], label="sim", linestyle="--")
axes[2].set_ylabel("theta_1_dot (rad/s)")
axes[2].legend()

# theta_2_dot (pendulum velocity)
axes[3].plot(hw["time"], hw["pend_spd"], label="hardware")
axes[3].plot(sim["time"], sim["pend_spd"], label="sim", linestyle="--")
axes[3].set_ylabel("theta_2_dot (rad/s)")
axes[3].legend()

# voltage
axes[4].plot(hw["time"], hw["voltage"], label="hardware")
axes[4].plot(sim["time"], sim["voltage"], label="sim", linestyle="--")
axes[4].set_ylabel("voltage (V)")
axes[4].legend()

# current
axes[5].plot(hw["time"], hw["current"], label="hardware")
axes[5].plot(sim["time"], sim["current"], label="sim", linestyle="--")
axes[5].set_ylabel("current (A)")
axes[5].set_xlabel("time (s)")
axes[5].legend()

fig.suptitle("Step Response: Hardware vs Sim")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "step_comparison.png", dpi=150)
plt.show()
