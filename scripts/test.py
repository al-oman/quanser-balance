from quanser.hardware import HIL, HILError, MAX_STRING_LENGTH
import time
import math
import numpy as np
import csv
from pathlib import Path

# --- Step test parameters ---
STEP_VOLTAGE = 2.0     # voltage applied during the step (V)
T_SETTLE     = 1.0     # seconds to wait before step (collect baseline)
T_STEP       = 1.0     # duration the step voltage is held (s)
T_AFTER      = 1.0     # seconds to record after step ends
Ts           = 0.002   # sample period — 500 Hz

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = SCRIPT_DIR.parent / "outputs" / "step_response.csv"

# Open device
card = HIL("qube_servo3_usb", "0")

# Channels
encoder_channels     = np.array([0, 1], dtype=np.uint32)
analog_in_channels   = np.array([0], dtype=np.uint32)
analog_out_channels  = np.array([0], dtype=np.uint32)
digital_in_channels  = np.array([0, 1, 2], dtype=np.uint32)
digital_out_channels = np.array([0], dtype=np.uint32)
other_in_channels    = np.array([14000, 14001], dtype=np.uint32)
other_out_channels   = np.array([11000, 11001, 11002], dtype=np.uint32)

# Buffers
encoder_buffer    = np.zeros(2, dtype=np.int32)
analog_in_buffer  = np.zeros(1, dtype=np.float64)
digital_in_buffer = np.zeros(3, dtype=np.int8)
other_in_buffer   = np.zeros(2, dtype=np.float64)

# Data log
log = []

try:
    # Reset encoders
    card.set_encoder_counts(
        encoder_channels, 2, np.zeros(2, dtype=np.int32))

    # Enable amplifier
    card.write_digital(
        digital_out_channels, 1, np.array([1], dtype=np.int8))

    # Set LED green
    card.write_other(
        other_out_channels, 3, np.array([0, 1, 0], dtype=np.float64))

    total_time = T_SETTLE + T_STEP + T_AFTER
    print(f"Step test: {STEP_VOLTAGE} V for {T_STEP}s "
          f"(settle {T_SETTLE}s, after {T_AFTER}s, total {total_time}s)")

    t_start = time.perf_counter()
    t_elapsed = 0.0

    while t_elapsed < total_time:
        loop_start = time.perf_counter()
        t_elapsed = loop_start - t_start

        # Read sensors
        card.read_encoder(encoder_channels, 2, encoder_buffer)
        card.read_analog(analog_in_channels, 1, analog_in_buffer)
        card.read_digital(digital_in_channels, 3, digital_in_buffer)
        card.read_other(other_in_channels, 2, other_in_buffer)

        motor_pos = encoder_buffer[0] * 2 * np.pi / 2048
        pend_pos  = encoder_buffer[1] * 2 * np.pi / 2048 - np.pi
        motor_spd = other_in_buffer[0] * 2 * np.pi / 2048
        pend_spd  = other_in_buffer[1] * 2 * np.pi / 2048
        current   = analog_in_buffer[0]

        # Step function: voltage on only during [T_SETTLE, T_SETTLE + T_STEP)
        if T_SETTLE <= t_elapsed < T_SETTLE + T_STEP:
            voltage = STEP_VOLTAGE
        else:
            voltage = 0.0

        card.write_analog(
            analog_out_channels, 1,
            np.array([voltage], dtype=np.float64))

        # Log data
        log.append([t_elapsed, voltage, motor_pos, pend_pos,
                    motor_spd, pend_spd, current])

        print(f"t={t_elapsed:5.3f}s  V={voltage:+6.2f}  "
              f"theta_1={motor_pos:+.4f}  theta_2={pend_pos:+.4f}  "
              f"theta_1_dot={motor_spd:+7.2f}  theta_2_dot={pend_spd:+7.2f}  "
              f"I={current:+.4f}")

        # Maintain constant sample rate
        dt = time.perf_counter() - loop_start
        if dt < Ts:
            time.sleep(Ts - dt)

    print("Step test complete.")

except HILError as ex:
    print("Error: %s" % ex.get_error_message())

except KeyboardInterrupt:
    print("\nStopping early...")

finally:
    # Zero voltage
    card.write_analog(
        analog_out_channels, 1, np.array([0.0], dtype=np.float64))
    # LED red
    card.write_other(
        other_out_channels, 3, np.array([1, 0, 0], dtype=np.float64))
    # Disable amplifier
    card.write_digital(
        digital_out_channels, 1, np.array([0], dtype=np.int8))
    card.close()

    # Save logged data to CSV
    if log:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "voltage", "motor_pos", "pend_pos",
                             "motor_spd", "pend_spd", "current"])
            writer.writerows(log)
        print(f"Saved {len(log)} samples to {OUTPUT_CSV}")