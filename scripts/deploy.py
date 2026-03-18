"""
Deploy a trained PPO model to the physical Qube Servo 3.

Reads arm/pendulum angles from encoders and angular velocities from
other_in channels, constructs the observation vector matching the sim env
[theta_arm, theta_pend, dtheta_arm, dtheta_pend], runs the policy,
and writes the voltage output to the motor.
"""

from quanser.hardware import HIL, HILError
from quanser_balance.rl.PPO import CustomPPO

import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = ROOT_DIR / "outputs" / "rotpend" / "ppo"
XML_PATH = ROOT_DIR / "src" / "quanser_balance" / "envs" / "assets" / "rot_pend.xml"

# ── Config ───────────────────────────────────────────────────────────
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "rotpend_ppo_model_3"
VOLTAGE_MAX = 10.0
Ts = 0.002  # 500 Hz control loop
COUNTS_PER_REV = 2048  # 512 lines * 4 (quadrature)
RAD_PER_COUNT = 2 * np.pi / COUNTS_PER_REV

# Observation clipping (must match env._get_obs)
OBS_LOW  = np.array([-np.pi/2, -20.0, -50.0, -50.0])
OBS_HIGH = np.array([ np.pi/2,  20.0,  50.0,  50.0])

# ── Load policy (no env needed — we build obs manually) ──────────────
model = CustomPPO.load(OUTPUTS_DIR / MODEL_NAME)

# ── MuJoCo visualizer ───────────────────────────────────────────────
mj_model = mujoco.MjModel.from_xml_path(str(XML_PATH))
mj_data = mujoco.MjData(mj_model)
viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

# ── Hardware setup ───────────────────────────────────────────────────
card = HIL("qube_servo3_usb", "0")

encoder_channels     = np.array([0, 1], dtype=np.uint32)      # 0=arm, 1=pendulum
analog_out_channels  = np.array([0], dtype=np.uint32)
digital_out_channels = np.array([0], dtype=np.uint32)
other_in_channels    = np.array([14000, 14001], dtype=np.uint32)  # hw velocities
other_out_channels   = np.array([11000, 11001, 11002], dtype=np.uint32)  # LED RGB

encoder_buffer  = np.zeros(2, dtype=np.int32)
other_in_buffer = np.zeros(2, dtype=np.float64)

try:
    # Reset encoders
    card.set_encoder_counts(encoder_channels, 2, np.zeros(2, dtype=np.int32))

    # Enable amplifier
    card.write_digital(digital_out_channels, 1, np.array([1], dtype=np.int8))

    # LED green = running
    card.write_other(other_out_channels, 3, np.array([0, 1, 0], dtype=np.float64))

    print(f"Deploying model: {MODEL_NAME}")
    print("Running... Ctrl+C to stop\n")

    while True:
        # Read sensors
        card.read_encoder(encoder_channels, 2, encoder_buffer)
        card.read_other(other_in_channels, 2, other_in_buffer)

        theta_arm   = -(encoder_buffer[0] * RAD_PER_COUNT)
        theta_pend  = (encoder_buffer[1] * RAD_PER_COUNT) + np.pi
        dtheta_arm  = -(other_in_buffer[0] * RAD_PER_COUNT)
        dtheta_pend = other_in_buffer[1] * RAD_PER_COUNT

        # Build observation (same format as env)
        obs = np.array([theta_arm, theta_pend, dtheta_arm, dtheta_pend], dtype=np.float64)
        obs = np.clip(obs, OBS_LOW, OBS_HIGH)

        # Run policy
        action, _ = model.predict(obs, deterministic=True)
        voltage = float(np.clip(action[0], -VOLTAGE_MAX, VOLTAGE_MAX))

        # Write voltage to motor
        card.write_analog(analog_out_channels, 1, np.array([voltage], dtype=np.float64))

        # Update MuJoCo visualizer with measured angles
        mj_data.qpos[0] = theta_arm
        mj_data.qpos[1] = theta_pend
        mj_data.qvel[0] = dtheta_arm
        mj_data.qvel[1] = dtheta_pend
        mujoco.mj_forward(mj_model, mj_data)
        viewer.sync()

        print(f"arm={theta_arm:+.3f} pend={theta_pend:+.3f} | "
              f"darm={dtheta_arm:+.1f} dpend={dtheta_pend:+.1f} | "
              f"V={voltage:+.2f}")

        time.sleep(Ts)

except HILError as ex:
    print(f"HIL Error: {ex.get_error_message()}")

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    # Zero voltage
    card.write_analog(analog_out_channels, 1, np.array([0.0], dtype=np.float64))
    # LED red = stopped
    card.write_other(other_out_channels, 3, np.array([1, 0, 0], dtype=np.float64))
    # Disable amplifier
    card.write_digital(digital_out_channels, 1, np.array([0], dtype=np.int8))
    card.close()
    viewer.close()
    print("Hardware shutdown complete.")
