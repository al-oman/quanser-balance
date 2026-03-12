from quanser.hardware import HIL, HILError
import time
import math
import numpy as np

# Open device
card = HIL("qube_servo3_usb", "0")

# Channels
encoder_channels = np.array([0], dtype=np.uint32)
analog_channels  = np.array([0], dtype=np.uint32)
digital_channels = np.array([0], dtype=np.uint32)

encoder_buffer = np.zeros(1, dtype=np.int32)

try:
    # Reset encoder
    card.set_encoder_counts(encoder_channels, 1, np.array([0], dtype=np.int32))

    # Enable amplifier
    card.write_digital(digital_channels, 1, np.array([1], dtype=np.int8))

    print("Running... Ctrl+C to stop")

    Ts = 0.01  # 100 Hz loop

    while True:
        # Read encoder
        card.read_encoder(encoder_channels, 1, encoder_buffer)
        counts = encoder_buffer[0]

        # Convert to radians
        theta = 2 * math.pi / (512 * 4) * counts

        # Simple test input
        # voltage = 1.0 * math.sin(time.time())*0
        voltage = 0.5

        # Saturate
        voltage = max(min(voltage, 10), -10)

        # Write voltage
        card.write_analog(analog_channels, 1,
                          np.array([voltage], dtype=np.float64))

        print(f"theta = {theta:.3f} rad, V = {voltage:.2f}")

        time.sleep(Ts)

except HILError as ex:
    print("Error: %s" % ex.get_error_message())

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    # Shutdown cleanly
    card.write_analog(analog_channels, 1, np.array([0], dtype=np.float64))
    time.sleep(0.1)
    card.write_analog(analog_channels, 1, np.array([0], dtype=np.float64))

    card.write_digital(digital_channels, 1, np.array([0], dtype=np.int8))
    card.close()
