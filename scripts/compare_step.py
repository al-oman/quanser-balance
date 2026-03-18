import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

hw = pd.read_csv(OUTPUT_DIR / "step_response.csv")
sim = pd.read_csv(OUTPUT_DIR / "step_response_sim.csv")

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

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

# voltage
axes[2].plot(hw["time"], hw["voltage"], label="hardware")
axes[2].plot(sim["time"], sim["voltage"], label="sim", linestyle="--")
axes[2].set_ylabel("voltage (V)")
axes[2].set_xlabel("time (s)")
axes[2].legend()

fig.suptitle("Step Response: Hardware vs Sim")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "step_comparison.png", dpi=150)
plt.show()
