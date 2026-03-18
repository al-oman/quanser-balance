from quanser.hardware import HIL, HILError, MAX_STRING_LENGTH
import time
import math
import numpy as np

# Open device
card = HIL("qube_servo3_usb", "0")

# Board-specific options (important for Servo 3)
# card.set_card_specific_options(
#     'deadband_compensation=0.3;pwm_en=0;enc0_velocity=3.0;enc1_velocity=3.0;'
#     'min_diode_compensation=0.3;max_diode_compensation=1.5',
#     MAX_STRING_LENGTH
# )

encoder_line_counts = 512

# Channels
encoder_channels    = np.array([0, 1], dtype=np.uint32)
analog_in_channels  = np.array([0], dtype=np.uint32)
analog_out_channels = np.array([0], dtype=np.uint32)
digital_in_channels = np.array([0, 1, 2], dtype=np.uint32)
digital_out_channels = np.array([0], dtype=np.uint32)
other_in_channels   = np.array([14000, 14001], dtype=np.uint32)
other_out_channels  = np.array([11000, 11001, 11002], dtype=np.uint32)

# Buffers
encoder_buffer    = np.zeros(2, dtype=np.int32)
analog_in_buffer  = np.zeros(1, dtype=np.float64)
digital_in_buffer = np.zeros(3, dtype=np.int8)
other_in_buffer   = np.zeros(2, dtype=np.float64)

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

    print("Running... Ctrl+C to stop")
    Ts = 0.002  # 500 Hz

    t_step = 2.0
    t_stop = 2.5

    while True:
        card.read_encoder(encoder_channels, 2, encoder_buffer)
        card.read_analog(analog_in_channels, 1, analog_in_buffer)
        card.read_digital(digital_in_channels, 3, digital_in_buffer)
        card.read_other(other_in_channels, 2, other_in_buffer)

        motor_pos = encoder_buffer[0] * 2 * np.pi / 2048
        pend_pos  = encoder_buffer[1] * 2 * np.pi / 2048
        motor_spd = other_in_buffer[0] * 2 * np.pi / 2048
        pend_spd  = other_in_buffer[1] * 2 * np.pi / 2048
        current   = analog_in_buffer[0]

        # Your control logic here
        voltage = 1.0 * math.sin(time.time())
        voltage = np.clip(voltage, -10, 10)

        voltage = 0.4

        card.write_analog(
            analog_out_channels, 1,
            np.array([voltage], dtype=np.float64))

        print(f"Motor: {motor_pos:.3f} rad, Pend: {pend_pos:.3f} rad, "
            f"Motor Spd: {motor_spd:.1f} rad/s, Pend Spd: {pend_spd:.1f} rad/s, "
            f"Current: {current:.3f} A, Voltage: {voltage:.3f} V")

        time.sleep(Ts)

except HILError as ex:
    print("Error: %s" % ex.get_error_message())

except KeyboardInterrupt:
    print("\nStopping...")

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