import gymnasium as gym
import quanser_balance.envs  # triggers register()
from quanser_balance.rl.PPO import CustomPPO

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
OUTPUTS_DIR = ROOT_DIR / "outputs" / "rotpend" / "ppo"

model = CustomPPO.load(OUTPUTS_DIR / "rotpend_ppo_model_2")

from quanser.hardware import HIL, HILError
from quanser.common import GenericError
import numpy as np
import math
import time
import sys
import traceback

card = HIL("qube_servo3_usb", "0")

# encoder channels for angle readings
encoder_channels = np.array([0, 1], dtype=np.uint32)
encoder_buffer = np.zeros(2, dtype=np.int32)

# 'other input' channels for tachometer readings
input_channels = np.array([14000, 14001], dtype=np.uint32)
input_buffer = np.zeros(2, dtype=np.float64)

# analog channel for writing voltage
analog_channels  = np.array([0], dtype=np.uint32)   

# digital channel for enabling amplifier
digital_channels = np.array([0], dtype=np.uint32)

def get_voltage(theta, alpha, theta_dot, alpha_dot):
    obs = np.array([theta, alpha, theta_dot, alpha_dot], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    voltage = action[0]
    # Saturate
    voltage = max(min(voltage, 10), -10)
    return voltage

def test_voltage(theta, alpha, theta_dot, alpha_dot):
    obs = np.array([theta, alpha, theta_dot, alpha_dot], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    voltage = action[0]
    # Saturate
    voltage = max(min(voltage, 10), -10)
    return 3.0

try:
    card.set_encoder_counts(encoder_channels, 2,
                            encoder_buffer)
    
    # Enable amplifier
    card.write_digital(digital_channels, 1, np.array([1], dtype=np.int8))

    print("Reading encoders... Ctrl+C to stop")
    
    Ts=0.05

    while True:
        card.read_encoder(encoder_channels, 2, encoder_buffer)

        arm_counts = encoder_buffer[0]
        pend_counts = encoder_buffer[1]

        theta = 2 * math.pi / 2048 * arm_counts
        alpha = 2 * math.pi / 2048 * pend_counts

        card.read_other(input_channels, 2, input_buffer)
        theta_dot = 2 * math.pi / 2048 * input_buffer[0]
        alpha_dot = 2 * math.pi / 2048 * input_buffer[1]


        # Simple test input
        # voltage = 0.25 * math.sin(time.time())
        voltage = get_voltage(theta, alpha, theta_dot, alpha_dot)

        # Write voltage
        card.write_analog(analog_channels, 1,
                          np.array([voltage], dtype=np.float64))

        print(f"arm = {theta: .3f} rad | pend = {alpha: .3f} rad | V = {voltage: .2f} V | theta_dot = {theta_dot: .3f} rad/s | alpha_dot = {alpha_dot: .3f} rad/s")

        time.sleep(0.05)

except KeyboardInterrupt:
    time.sleep(0.5)
    print("\nStopping...")

except Exception as e:
    traceback.print_exc()
    print("An error occurred:", e)
    traceback.print_exc()
finally:
    card.write_digital(
    np.array([0], dtype=np.uint32),
    1,
    np.array([0], dtype=np.int8)
)
    card.close()
    time.sleep(0.5)
    print("closed the card")
    sys.exit(0)



# try:
#     obs, info = env.reset(seed=123, options={"low": -0.1, "high": 0.1})

#     over = False
#     total_reward = 0

#     while not over:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = env.step(action)
#         total_reward += reward
#         env.render()
#         over = terminated or truncated
#         # print(action, reward)
# finally:
#     env.close()