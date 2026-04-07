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
AWB_SMOOTH    = 0.7  # EMA smoothing factor
BITS          = 12   # sensor ADC bit depth
MAX_RAW       = (1 << BITS) - 1  # 4095
# ─────────────────────────────────────────────────────────────────────────────

CTRL_WIN = "Camera Controls"

# ── I2C helpers ───────────────────────────────────────────────────────────────
def write_reg(bus, addr, reg16, val8):
    reg_hi = (reg16 >> 8) & 0xFF
    reg_lo = reg16 & 0xFF
    print(f"I2C write: reg=0x{reg16:04X} val=0x{val8:02X}")
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
        val = max(0, min(int(gain * 1024), 0x7FFF))
        return (val >> 8) & 0x7F, val & 0xFF
    b_hi, b_lo = encode(b_gain)
    g_hi, g_lo = encode(g_gain)
    r_hi, r_lo = encode(r_gain)
    for reg, hi, lo in [(0x5180, b_hi, b_lo), (0x5182, g_hi, g_lo),
                        (0x5184, g_hi, g_lo), (0x5186, r_hi, r_lo)]:
        write_reg(bus, SENSOR_ADDR, reg,     hi)
        write_reg(bus, SENSOR_ADDR, reg + 1, lo)

def compute_awb_gains(bgr8, cur_r, cur_g, cur_b):
    """Gray world AWB with midtone masking and EMA smoothing."""
    gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
    mask = (gray > 30) & (gray < 220)
    if mask.sum() < 1000:
        return cur_r, cur_g, cur_b
    b, g, r = cv2.split(bgr8)
    r_avg = float(r[mask].mean()) + 1e-6
    g_avg = float(g[mask].mean()) + 1e-6
    b_avg = float(b[mask].mean()) + 1e-6
    target = (r_avg + g_avg + b_avg) / 3.0
    new_r = np.clip(target / r_avg, 0.5, 8.0)
    new_b = np.clip(target / b_avg, 0.5, 8.0)
    return (AWB_SMOOTH * cur_r + (1 - AWB_SMOOTH) * new_r,
            1.0,
            AWB_SMOOTH * cur_b + (1 - AWB_SMOOTH) * new_b)

# ── ISP helpers ───────────────────────────────────────────────────────────────
def build_display_lut(gamma: float, tonemap_strength: float) -> np.ndarray:
    """uint8→uint8 LUT: Reinhard global tonemap then gamma encode."""
    x = np.arange(256, dtype=np.float32) / 255.0
    if tonemap_strength > 0.01:
        x = x / (1.0 + x * tonemap_strength)
        peak = x[-1]
        if peak > 0:
            x /= peak
    x = np.power(np.clip(x, 1e-6, 1.0), 1.0 / max(gamma, 0.1))
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)

def build_wb_lut(r_gain: float, g_gain: float, b_gain: float) -> np.ndarray:
    """Per-channel gain LUT (256×3, BGR order) for software white balance."""
    x = np.arange(256, dtype=np.float32)
    b_lut = np.clip(x * b_gain, 0, 255).astype(np.uint8)
    g_lut = np.clip(x * g_gain, 0, 255).astype(np.uint8)
    r_lut = np.clip(x * r_gain, 0, 255).astype(np.uint8)
    return np.stack([b_lut, g_lut, r_lut], axis=1)   # shape (256, 3)

def apply_wb_lut(bgr8: np.ndarray, wb_lut: np.ndarray) -> np.ndarray:
    """Apply a per-channel LUT (256×3, BGR) to a BGR image."""
    b, g, r = cv2.split(bgr8)
    b = cv2.LUT(b, wb_lut[:, 0])
    g = cv2.LUT(g, wb_lut[:, 1])
    r = cv2.LUT(r, wb_lut[:, 2])
    return cv2.merge([b, g, r])

def apply_clahe(bgr8: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE on the L channel in LAB — enhances local contrast without noise amp."""
    lab = cv2.cvtColor(bgr8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

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
    cv2.resizeWindow(CTRL_WIN, 500, 560)
    cv2.createTrackbar("Exposure (rows)", CTRL_WIN, 1000, 1118, lambda x: None)
    cv2.createTrackbar("Analog Gain x16", CTRL_WIN,   16,  248, lambda x: None)
    cv2.createTrackbar("AWB R x100",      CTRL_WIN,   82,  800, lambda x: None)
    cv2.createTrackbar("AWB G x100",      CTRL_WIN,  100,  800, lambda x: None)
    cv2.createTrackbar("AWB B x100",      CTRL_WIN,  800,  800, lambda x: None)
    # Brightness %: 100 = neutral (1×), 200 = 2×, 50 = 0.5×
    cv2.createTrackbar("Brightness %",    CTRL_WIN,  100,  200, lambda x: None)
    # Black level in 12-bit counts; typical OV sensor ≈ 64
    cv2.createTrackbar("Black Level",     CTRL_WIN,   64,  256, lambda x: None)
    # Gamma × 10: 22 = 2.2 (sRGB)
    cv2.createTrackbar("Gamma x10",       CTRL_WIN,   22,   40, lambda x: None)
    # Reinhard strength × 10: 0 = off, 15 = moderate
    cv2.createTrackbar("Tonemap x10",     CTRL_WIN,   15,  100, lambda x: None)
    # CLAHE clip × 10: 0 = off, 20 = moderate (2.0), applied on display frame
    cv2.createTrackbar("CLAHE clip x10",  CTRL_WIN,   20,   80, lambda x: None)
    # Auto WB: default ON — writes gains to sensor over I2C (or applies software)
    cv2.createTrackbar("Auto WB (1=on)",  CTRL_WIN,    1,    1, lambda x: None)

def get_controls():
    return (
        max(cv2.getTrackbarPos("Exposure (rows)", CTRL_WIN), 1),
        max(cv2.getTrackbarPos("Analog Gain x16", CTRL_WIN), 16),
        max(cv2.getTrackbarPos("AWB R x100",      CTRL_WIN), 1) / 100.0,
        max(cv2.getTrackbarPos("AWB G x100",      CTRL_WIN), 1) / 100.0,
        max(cv2.getTrackbarPos("AWB B x100",      CTRL_WIN), 1) / 100.0,
        max(cv2.getTrackbarPos("Brightness %",    CTRL_WIN), 1) / 100.0,
        max(cv2.getTrackbarPos("Black Level",     CTRL_WIN), 0),
        max(cv2.getTrackbarPos("Gamma x10",       CTRL_WIN), 10) / 10.0,
        cv2.getTrackbarPos("Tonemap x10",      CTRL_WIN) / 10.0,
        cv2.getTrackbarPos("CLAHE clip x10",   CTRL_WIN) / 10.0,
        cv2.getTrackbarPos("Auto WB (1=on)",   CTRL_WIN) == 1,
    )

def sync_wb_sliders(r, g, b):
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

    # Pre-allocate GPU buffers
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

    # 11 controls (removed AutoStretch — not needed with proper WB)
    prev_controls    = (None,) * 11
    pending_controls = (None,) * 11
    debounce_count   = 0
    DEBOUNCE_FRAMES  = 3

    # AWB state (software path, used when bus=None)
    awb_r, awb_g, awb_b = 1.0, 1.0, 1.0
    awb_initialized      = False

    # Cached LUTs — rebuilt only when their params change
    lut            = build_display_lut(2.2, 1.5)
    lut_gamma      = 2.2
    lut_tonemap    = 1.5
    wb_lut         = build_wb_lut(1.0, 1.0, 1.0)
    wb_lut_gains   = (1.0, 1.0, 1.0)

    frame_count = fps_display = 0
    t0 = time.time()

    print("Press 'q' to quit, 's' to save snapshot manually")

    SNAPSHOT_DELAY = 5.0
    snapshot_saved = False
    start_time     = time.time()

    VIDEO_FRAMES = 15
    video_buffer = []
    video_saved  = False

    cv2.namedWindow("RAW12 Camera (GPU debayer)", cv2.WINDOW_NORMAL)

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # ── Read controls ─────────────────────────────────────────────────────
        (exposure, analog_gain, man_r, man_g, man_b,
         brightness, black_level, gamma, tonemap_strength,
         clahe_clip, auto_wb) = get_controls()

        controls = (exposure, analog_gain, man_r, man_g, man_b,
                    brightness, black_level, gamma, tonemap_strength,
                    clahe_clip, auto_wb)

        if controls != pending_controls:
            pending_controls = controls
            debounce_count   = 0
        else:
            debounce_count  += 1

        if debounce_count == DEBOUNCE_FRAMES and controls != prev_controls and bus:
            try:
                if exposure    != prev_controls[0]: set_exposure(bus, exposure)
                if analog_gain != prev_controls[1]: set_analog_gain(bus, analog_gain)
                if not auto_wb and controls[2:5] != prev_controls[2:5]:
                    set_awb_gain(bus, man_r, man_g, man_b)
            except Exception as e:
                print(f"I2C error: {e}")
            prev_controls = controls

        # ── Compute base 16→8 scale every frame ───────────────────────────────
        # map [black_level, MAX_RAW] → [0, 255] with brightness multiplier
        effective_range = max(MAX_RAW - black_level, 1)
        base_scale = (255.0 / effective_range) * brightness
        base_beta  = -black_level * base_scale

        # ── WB gains for the GPU convertTo ────────────────────────────────────
        # When hardware (I2C) handles WB, all channels use the same scale.
        # When no bus, apply WB in the GPU convertTo at zero extra GPU cost.
        if bus:
            sc_r, sc_g, sc_b = base_scale, base_scale, base_scale
            bt_r, bt_g, bt_b = base_beta,  base_beta,  base_beta
        else:
            # Use auto-computed or manual gains for software WB
            sw_r = awb_r if auto_wb else man_r
            sw_g = awb_g if auto_wb else man_g
            sw_b = awb_b if auto_wb else man_b
            sc_r = base_scale * sw_r
            sc_g = base_scale * sw_g
            sc_b = base_scale * sw_b
            bt_r = base_beta  * sw_r
            bt_g = base_beta  * sw_g
            bt_b = base_beta  * sw_b

        # ── Rebuild gamma/tonemap LUT only when params change ─────────────────
        if gamma != lut_gamma or tonemap_strength != lut_tonemap:
            lut         = build_display_lut(gamma, tonemap_strength)
            lut_gamma   = gamma
            lut_tonemap = tonemap_strength

        # ── CPU: reinterpret raw bytes as uint16 Bayer ────────────────────────
        np.copyto(bayer16, raw_frame.view(np.uint16).reshape(HEIGHT, WIDTH))

        # ── GPU pipeline ──────────────────────────────────────────────────────
        # demosaic + black-level subtract + scale to 8-bit, all in one pass
        gpu_bayer.upload(bayer16, stream)
        cv2.cuda.demosaicing(gpu_bayer, BAYER_PAT, gpu_bgr16, stream=stream)
        cv2.cuda.split(gpu_bgr16, [gpu_b16, gpu_g16, gpu_r16], stream=stream)
        gpu_b16.convertTo(cv2.CV_8UC1, sc_b, gpu_b8, bt_b)
        gpu_g16.convertTo(cv2.CV_8UC1, sc_g, gpu_g8, bt_g)
        gpu_r16.convertTo(cv2.CV_8UC1, sc_r, gpu_r8, bt_r)
        cv2.cuda.merge([gpu_b8, gpu_g8, gpu_r8], gpu_bgr8, stream=stream)
        stream.waitForCompletion()
        bgr8 = gpu_bgr8.download()

        # ── Gamma + tonemap LUT (vectorized, full-res) ────────────────────────
        bgr8 = cv2.LUT(bgr8, lut)

        # ── Hardware AWB (I2C path) ───────────────────────────────────────────
        if auto_wb and bus:
            if not awb_initialized:
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
        elif auto_wb and not bus:
            # Software AWB: update gains from this frame, apply next frame
            if not awb_initialized:
                for _ in range(20):
                    awb_r, awb_g, awb_b = compute_awb_gains(bgr8, awb_r, awb_g, awb_b)
                awb_initialized = True
                sync_wb_sliders(awb_r, awb_g, awb_b)
            elif frame_count % AWB_INTERVAL == 0:
                awb_r, awb_g, awb_b = compute_awb_gains(bgr8, awb_r, awb_g, awb_b)
                sync_wb_sliders(awb_r, awb_g, awb_b)
        else:
            awb_initialized = False

        # ── Snapshot / video buffer (full-res, pre-CLAHE) ─────────────────────
        if not snapshot_saved and (time.time() - start_time) >= SNAPSHOT_DELAY:
            cv2.imwrite("/tmp/camera_snapshot.jpg", bgr8, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print("Auto snapshot saved: /tmp/camera_snapshot.jpg")
            snapshot_saved = True

        if snapshot_saved and not video_saved:
            video_buffer.append(bgr8.copy())
            if len(video_buffer) >= VIDEO_FRAMES:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter("/tmp/camera_clip.mp4", fourcc, 10, (WIDTH, HEIGHT))
                for f in video_buffer:
                    out.write(f)
                out.release()
                print(f"Video clip saved: /tmp/camera_clip.mp4 ({VIDEO_FRAMES} frames)")
                video_saved = True

        # ── Display path: resize first, then apply CLAHE (4× cheaper) ─────────
        display = cv2.resize(bgr8, (DISPLAY_W, DISPLAY_H),
                             interpolation=cv2.INTER_LINEAR)

        if clahe_clip > 0.0:
            display = apply_clahe(display, clip_limit=clahe_clip)

        # ── FPS counter ───────────────────────────────────────────────────────
        frame_count += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            t0 = time.time()

        # ── OSD overlay ───────────────────────────────────────────────────────
        wb_label = "AWB:AUTO" if auto_wb else f"AWB:MAN R:{man_r:.2f} G:{man_g:.2f} B:{man_b:.2f}"
        cv2.putText(display, f"FPS:{fps_display:.1f}  Exp:{exposure}  Gain:{analog_gain/16:.1f}x",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, wb_label,
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        clahe_tag = f"CLAHE:{clahe_clip:.1f}" if clahe_clip > 0 else "CLAHE:off"
        cv2.putText(display,
                    f"BL:{black_level}  Bright:{brightness:.2f}x  G:{gamma:.1f}  TM:{tonemap_strength:.1f}  {clahe_tag}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        cv2.imshow("RAW12 Camera (GPU debayer)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("/tmp/camera_snapshot.jpg", bgr8, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print("Manual snapshot saved: /tmp/camera_snapshot.jpg")

    if bus:
        bus.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
