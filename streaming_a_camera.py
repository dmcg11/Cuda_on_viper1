import cv2
import numpy as np
import time
import smbus2

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE        = 5
WIDTH         = 1920
HEIGHT        = 1280
FPS           = 30
BAYER_PAT     = cv2.COLOR_BayerBG2BGR

DISPLAY_SCALE = 0.5
DISPLAY_W     = int(WIDTH  * DISPLAY_SCALE)
DISPLAY_H     = int(HEIGHT * DISPLAY_SCALE)

# I2C config
I2C_BUS       = 1
SENSOR_ADDR   = 0x6C >> 1  # smbus uses 7-bit address = 0x36
# ─────────────────────────────────────────────────────────────────────────────

CTRL_WIN = "Camera Controls"

# ── I2C helpers ───────────────────────────────────────────────────────────────
def write_reg(bus, addr, reg16, val8):
    """Write 8-bit value to 16-bit register address."""
    reg_hi = (reg16 >> 8) & 0xFF
    reg_lo = reg16 & 0xFF
    bus.write_i2c_block_data(addr, reg_hi, [reg_lo, val8])

def read_reg(bus, addr, reg16):
    """Read 8-bit value from 16-bit register address."""
    reg_hi = (reg16 >> 8) & 0xFF
    reg_lo = reg16 & 0xFF
    bus.write_i2c_block_data(addr, reg_hi, [reg_lo])
    return bus.read_byte(addr)

# ── Sensor control functions ──────────────────────────────────────────────────
def set_exposure(bus, rows):
    """
    Set DCG exposure time in rows.
    Registers 0x3501[7:0] = rows[15:8], 0x3502[7:0] = rows[7:0]
    Valid range: 1 to VTS-10 (default VTS=0x0468=1128, so max ~1118)
    """
    rows = max(1, min(rows, 1118))
    write_reg(bus, SENSOR_ADDR, 0x3501, (rows >> 8) & 0xFF)
    write_reg(bus, SENSOR_ADDR, 0x3502, rows & 0xFF)

def set_analog_gain(bus, gain_x16):
    """
    Set HCG analog gain. Format is 4.4 bits.
    gain_x16 = desired_gain * 16 (e.g. 16=1x, 32=2x, 248=15.5x)
    Register 0x3508 bits[3:0] = gain[7:4] (integer part)
    Register 0x3509 bits[7:4] = gain[3:0] (fractional part)
    """
    gain_x16 = max(16, min(gain_x16, 248))
    reg_3508 = (gain_x16 >> 4) & 0x0F
    reg_3509 = (gain_x16 & 0x0F) << 4
    write_reg(bus, SENSOR_ADDR, 0x3508, reg_3508)
    write_reg(bus, SENSOR_ADDR, 0x3509, reg_3509)

def set_awb_gain(bus, r_gain, g_gain, b_gain):
    """
    Set HCG AWB gains for R, G (Gb+Gr averaged), B channels.
    Format is 5.10 bits: value = gain * 1024 (e.g. 1x = 0x0400 = 1024)
    Valid range: 1x to 31.999x
    Registers:
      B:  0x5180 (hi), 0x5181 (lo)
      Gb: 0x5182 (hi), 0x5183 (lo)
      Gr: 0x5184 (hi), 0x5185 (lo)
      R:  0x5186 (hi), 0x5187 (lo)
    """
    def encode(gain):
        val = int(gain * 1024)
        val = max(1024, min(val, 0x7FFF))
        return (val >> 8) & 0x7F, val & 0xFF

    b_hi,  b_lo  = encode(b_gain)
    g_hi,  g_lo  = encode(g_gain)
    r_hi,  r_lo  = encode(r_gain)

    # B channel
    write_reg(bus, SENSOR_ADDR, 0x5180, b_hi)
    write_reg(bus, SENSOR_ADDR, 0x5181, b_lo)
    # Gb channel
    write_reg(bus, SENSOR_ADDR, 0x5182, g_hi)
    write_reg(bus, SENSOR_ADDR, 0x5183, g_lo)
    # Gr channel
    write_reg(bus, SENSOR_ADDR, 0x5184, g_hi)
    write_reg(bus, SENSOR_ADDR, 0x5185, g_lo)
    # R channel
    write_reg(bus, SENSOR_ADDR, 0x5186, r_hi)
    write_reg(bus, SENSOR_ADDR, 0x5187, r_lo)


# ── Camera and UI ─────────────────────────────────────────────────────────────
def open_camera():
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'BG12'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    cap.set(cv2.CAP_PROP_CONVERT_RGB,  0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open /dev/video5")
    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps}fps")
    return cap


def create_controls():
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WIN, 500, 300)
    # Exposure: 1-1118 rows
    cv2.createTrackbar("Exposure (rows)", CTRL_WIN,  64, 1118, lambda x: None)
    # Analog gain: 16-248 (1x-15.5x in units of 1/16)
    cv2.createTrackbar("Analog Gain x16", CTRL_WIN,  16,  248, lambda x: None)
    # AWB gains: 100-800 stored as x100 (1.0x-8.0x)
    cv2.createTrackbar("AWB R x100",      CTRL_WIN, 100,  800, lambda x: None)
    cv2.createTrackbar("AWB G x100",      CTRL_WIN, 100,  800, lambda x: None)
    cv2.createTrackbar("AWB B x100",      CTRL_WIN, 100,  800, lambda x: None)
    # Software brightness for display only
    cv2.createTrackbar("Brightness",      CTRL_WIN,  25,  100, lambda x: None)


def get_controls():
    exposure   = max(cv2.getTrackbarPos("Exposure (rows)", CTRL_WIN), 1)
    analog_gain= max(cv2.getTrackbarPos("Analog Gain x16", CTRL_WIN), 16)
    awb_r      = max(cv2.getTrackbarPos("AWB R x100",      CTRL_WIN), 1) / 100.0
    awb_g      = max(cv2.getTrackbarPos("AWB G x100",      CTRL_WIN), 1) / 100.0
    awb_b      = max(cv2.getTrackbarPos("AWB B x100",      CTRL_WIN), 1) / 100.0
    brightness = max(cv2.getTrackbarPos("Brightness",       CTRL_WIN), 1) / 100.0
    return exposure, analog_gain, awb_r, awb_g, awb_b, brightness


def main():
    # Open I2C bus
    try:
        bus = smbus2.SMBus(I2C_BUS)
        print(f"I2C bus {I2C_BUS} opened, sensor address 0x{SENSOR_ADDR:02X}")
    except Exception as e:
        print(f"Warning: Could not open I2C bus: {e}")
        bus = None

    cap = open_camera()
    create_controls()

    # GPU mats
    gpu_bayer  = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_16UC1)
    gpu_bgr16  = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_16UC3)
    gpu_b16    = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_16UC1)
    gpu_g16    = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_16UC1)
    gpu_r16    = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_16UC1)
    gpu_b8     = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC1)
    gpu_g8     = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC1)
    gpu_r8     = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC1)
    gpu_bgr8   = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC3)

    bayer16    = np.empty((HEIGHT, WIDTH), dtype=np.uint16)
    stream     = cv2.cuda_Stream()

    prev_controls   = (None,) * 6
    pending_controls = (None,) * 6
    debounce_count   = 0
    DEBOUNCE_FRAMES  = 3   # wait this many stable frames before writing to sensor
    alpha_b = alpha_g = alpha_r = 0.25

    print("Press 'q' to quit")

    frame_count = 0
    fps_display = 0.0
    t0          = time.time()

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        exposure, analog_gain, awb_r, awb_g, awb_b, brightness = get_controls()
        controls = (exposure, analog_gain, awb_r, awb_g, awb_b, brightness)

        # Debounced register writes — only write after value stable for N frames
        if controls != pending_controls:
            pending_controls = controls
            debounce_count   = 0
        else:
            debounce_count += 1

        if debounce_count == DEBOUNCE_FRAMES and controls != prev_controls and bus is not None:
            try:
                if exposure    != prev_controls[0]: set_exposure(bus, exposure)
                if analog_gain != prev_controls[1]: set_analog_gain(bus, analog_gain)
                if controls[2:5] != prev_controls[2:5]:
                    set_awb_gain(bus, awb_r, awb_g, awb_b)
            except Exception as e:
                print(f"I2C write error: {e}")

            alpha_b = alpha_g = alpha_r = brightness
            prev_controls = controls

        # ── CPU: Zero-copy reinterpret ────────────────────────────────────────
        np.copyto(bayer16, raw_frame.view(np.uint16).reshape(HEIGHT, WIDTH))

        # ── GPU: Upload → Demosaic → Scale to 8-bit ───────────────────────────
        gpu_bayer.upload(bayer16, stream)
        cv2.cuda.demosaicing(gpu_bayer, BAYER_PAT, gpu_bgr16, stream=stream)
        cv2.cuda.split(gpu_bgr16, [gpu_b16, gpu_g16, gpu_r16], stream=stream)
        gpu_b16.convertTo(cv2.CV_8UC1, alpha_b, gpu_b8, 0)
        gpu_g16.convertTo(cv2.CV_8UC1, alpha_g, gpu_g8, 0)
        gpu_r16.convertTo(cv2.CV_8UC1, alpha_r, gpu_r8, 0)
        cv2.cuda.merge([gpu_b8, gpu_g8, gpu_r8], gpu_bgr8, stream=stream)

        # ── Download ──────────────────────────────────────────────────────────
        stream.waitForCompletion()
        bgr8 = gpu_bgr8.download()

        # ── Resize for display ────────────────────────────────────────────────
        display = cv2.resize(bgr8, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)

        # ── FPS counter ───────────────────────────────────────────────────────
        frame_count += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            t0 = time.time()

        cv2.putText(display, f"FPS: {fps_display:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"Exp:{exposure} Gain:{analog_gain/16:.1f}x  "
                             f"AWB R:{awb_r:.2f} G:{awb_g:.2f} B:{awb_b:.2f}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.namedWindow("RAW12 Camera (GPU debayer)", cv2.WINDOW_NORMAL)
        cv2.imshow("RAW12 Camera (GPU debayer)", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if bus:
        bus.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
