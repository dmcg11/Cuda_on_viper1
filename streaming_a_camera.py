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

I2C_BUS       = 1
SENSOR_ADDR   = 0x6C >> 1  # 7-bit = 0x36

AWB_INTERVAL  = 15   # frames between AWB recalculations
AWB_SMOOTH    = 0.7  # smoothing factor
# ─────────────────────────────────────────────────────────────────────────────

CTRL_WIN = "Camera Controls"

# ── I2C helpers ───────────────────────────────────────────────────────────────
def write_reg(bus, addr, reg16, val8):
    reg_hi = (reg16 >> 8) & 0xFF
    reg_lo = reg16 & 0xFF
    bus.write_i2c_block_data(addr, reg_hi, [reg_lo, val8])

# ── Sensor control ────────────────────────────────────────────────────────────
def set_exposure(bus, rows):
    rows = max(1, min(rows, 1118))
    write_reg(bus, SENSOR_ADDR, 0x3501, (rows >> 8) & 0xFF)
    write_reg(bus, SENSOR_ADDR, 0x3502, rows & 0xFF)

def set_analog_gain(bus, gain_x16):
    gain_x16 = max(16, min(gain_x16, 248))
    write_reg(bus, SENSOR_ADDR, 0x3508, (gain_x16 >> 4) & 0x0F)
    write_reg(bus, SENSOR_ADDR, 0x3509, (gain_x16 & 0x0F) << 4)

def set_awb_gain(bus, r_gain, g_gain, b_gain):
    def encode(gain):
        val = max(1024, min(int(gain * 1024), 0x7FFF))
        return (val >> 8) & 0x7F, val & 0xFF
    b_hi, b_lo = encode(b_gain)
    g_hi, g_lo = encode(g_gain)
    r_hi, r_lo = encode(r_gain)
    for reg, hi, lo in [(0x5180, b_hi, b_lo), (0x5182, g_hi, g_lo),
                        (0x5184, g_hi, g_lo), (0x5186, r_hi, r_lo)]:
        write_reg(bus, SENSOR_ADDR, reg,     hi)
        write_reg(bus, SENSOR_ADDR, reg + 1, lo)

def compute_awb_gains(bgr8, cur_r, cur_g, cur_b):
    """Gray world AWB with midtone masking and smoothing."""
    gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
    mask = (gray > 30) & (gray < 220)
    if mask.sum() < 1000:
        return cur_r, cur_g, cur_b
    b, g, r = cv2.split(bgr8)
    r_avg = float(r[mask].mean()) + 1e-6
    g_avg = float(g[mask].mean()) + 1e-6
    b_avg = float(b[mask].mean()) + 1e-6
    target = (r_avg + g_avg + b_avg) / 3.0
    new_r = np.clip((target / r_avg), 0.5, 8.0)
    new_b = np.clip((target / b_avg), 0.5, 8.0)
    new_g = 1.0
    return (AWB_SMOOTH * cur_r + (1 - AWB_SMOOTH) * new_r,
            new_g,
            AWB_SMOOTH * cur_b + (1 - AWB_SMOOTH) * new_b)

# ── Camera ────────────────────────────────────────────────────────────────────
def open_camera():
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'BG12'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    cap.set(cv2.CAP_PROP_CONVERT_RGB,  0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open /dev/video5")
    print(f"Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS)}fps")
    return cap

# ── Controls UI ───────────────────────────────────────────────────────────────
def create_controls():
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WIN, 500, 280)
    cv2.createTrackbar("Exposure (rows)", CTRL_WIN,  375, 1118, lambda x: None)
    cv2.createTrackbar("Analog Gain x16", CTRL_WIN,  139,  248, lambda x: None)
    cv2.createTrackbar("AWB R x100",      CTRL_WIN,   82,  800, lambda x: None)
    cv2.createTrackbar("AWB G x100",      CTRL_WIN,  100,  800, lambda x: None)
    cv2.createTrackbar("AWB B x100",      CTRL_WIN,  800,  800, lambda x: None)
    # AWB mode: 0 = manual, 1 = auto
    cv2.createTrackbar("Auto WB (1=on)",  CTRL_WIN,    0,    1, lambda x: None)

def get_controls():
    return (
        max(cv2.getTrackbarPos("Exposure (rows)", CTRL_WIN), 1),
        max(cv2.getTrackbarPos("Analog Gain x16", CTRL_WIN), 16),
        max(cv2.getTrackbarPos("AWB R x100",      CTRL_WIN), 1) / 100.0,
        max(cv2.getTrackbarPos("AWB G x100",      CTRL_WIN), 1) / 100.0,
        max(cv2.getTrackbarPos("AWB B x100",      CTRL_WIN), 1) / 100.0,
        cv2.getTrackbarPos("Auto WB (1=on)",  CTRL_WIN) == 1,
    )

def sync_wb_sliders(r, g, b):
    """Push computed AWB gains back to the sliders so user can see them."""
    cv2.setTrackbarPos("AWB R x100", CTRL_WIN, int(np.clip(r * 100, 1, 800)))
    cv2.setTrackbarPos("AWB G x100", CTRL_WIN, int(np.clip(g * 100, 1, 800)))
    cv2.setTrackbarPos("AWB B x100", CTRL_WIN, int(np.clip(b * 100, 1, 800)))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    try:
        bus = smbus2.SMBus(I2C_BUS)
        print(f"I2C bus {I2C_BUS} opened, sensor 0x{SENSOR_ADDR:02X}")
    except Exception as e:
        print(f"Warning: I2C unavailable: {e}")
        bus = None

    cap = open_camera()
    create_controls()

    gpu_bayer = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_16UC1)
    gpu_bgr16 = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_16UC3)
    gpu_b16   = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_16UC1)
    gpu_g16   = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_16UC1)
    gpu_r16   = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_16UC1)
    gpu_b8    = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC1)
    gpu_g8    = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC1)
    gpu_r8    = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC1)
    gpu_bgr8  = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC3)

    bayer16 = np.empty((HEIGHT, WIDTH), dtype=np.uint16)
    stream  = cv2.cuda_Stream()

    prev_controls    = (None,) * 6
    pending_controls = (None,) * 6
    debounce_count   = 0
    DEBOUNCE_FRAMES  = 3

    # AWB state
    awb_r, awb_g, awb_b = 1.0, 1.0, 1.0
    awb_initialized      = False

    alpha = 1.0
    frame_count = fps_display = 0
    t0 = time.time()

    print("Press 'q' to quit")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        exposure, analog_gain, man_r, man_g, man_b, auto_wb = get_controls()

        # ── Debounced I2C writes ──────────────────────────────────────────────
        controls = (exposure, analog_gain, man_r, man_g, man_b, auto_wb)
        if controls != pending_controls:
            pending_controls = controls
            debounce_count   = 0
        else:
            debounce_count  += 1

        if debounce_count == DEBOUNCE_FRAMES and controls != prev_controls and bus:
            try:
                if exposure    != prev_controls[0]: set_exposure(bus, exposure)
                if analog_gain != prev_controls[1]: set_analog_gain(bus, analog_gain)
                # Only write manual AWB if auto WB is OFF
                if not auto_wb and controls[2:5] != prev_controls[2:5]:
                    set_awb_gain(bus, man_r, man_g, man_b)
            except Exception as e:
                print(f"I2C error: {e}")
            alpha         = 1.0
            prev_controls = controls

        # ── CPU: reinterpret ──────────────────────────────────────────────────
        np.copyto(bayer16, raw_frame.view(np.uint16).reshape(HEIGHT, WIDTH))

        # ── GPU pipeline ──────────────────────────────────────────────────────
        gpu_bayer.upload(bayer16, stream)
        cv2.cuda.demosaicing(gpu_bayer, BAYER_PAT, gpu_bgr16, stream=stream)
        cv2.cuda.split(gpu_bgr16, [gpu_b16, gpu_g16, gpu_r16], stream=stream)
        gpu_b16.convertTo(cv2.CV_8UC1, alpha, gpu_b8, 0)
        gpu_g16.convertTo(cv2.CV_8UC1, alpha, gpu_g8, 0)
        gpu_r16.convertTo(cv2.CV_8UC1, alpha, gpu_r8, 0)
        cv2.cuda.merge([gpu_b8, gpu_g8, gpu_r8], gpu_bgr8, stream=stream)
        stream.waitForCompletion()
        bgr8 = gpu_bgr8.download()

        # ── Auto WB ───────────────────────────────────────────────────────────
        if auto_wb and bus:
            if not awb_initialized:
                # Fast convergence on first enable
                for _ in range(20):
                    awb_r, awb_g, awb_b = compute_awb_gains(bgr8, awb_r, awb_g, awb_b)
                awb_initialized = True
                try:
                    set_awb_gain(bus, awb_r, awb_g, awb_b)
                except Exception as e:
                    print(f"AWB I2C error: {e}")
                sync_wb_sliders(awb_r, awb_g, awb_b)
            elif frame_count % AWB_INTERVAL == 0:
                awb_r, awb_g, awb_b = compute_awb_gains(bgr8, awb_r, awb_g, awb_b)
                try:
                    set_awb_gain(bus, awb_r, awb_g, awb_b)
                except Exception as e:
                    print(f"AWB I2C error: {e}")
                sync_wb_sliders(awb_r, awb_g, awb_b)
        else:
            awb_initialized = False  # reset so it re-converges next time enabled

        # ── Display ───────────────────────────────────────────────────────────
        display = cv2.resize(bgr8, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)

        frame_count += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            t0 = time.time()

        wb_label = "AWB:AUTO" if auto_wb else f"AWB:MAN R:{man_r:.2f} G:{man_g:.2f} B:{man_b:.2f}"
        cv2.putText(display, f"FPS:{fps_display:.1f}  Exp:{exposure}  Gain:{analog_gain/16:.1f}x",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, wb_label,
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

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
