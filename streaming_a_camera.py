import cv2
import numpy as np
import time

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE        = 5
WIDTH         = 1920
HEIGHT        = 1280
FPS           = 30
BAYER_PAT     = cv2.COLOR_BayerBG2BGR  # BGGR pattern

DISPLAY_SCALE = 0.5
DISPLAY_W     = int(WIDTH  * DISPLAY_SCALE)
DISPLAY_H     = int(HEIGHT * DISPLAY_SCALE)

# Auto WB update interval in frames
AWB_INTERVAL  = 5
# Smoothing factor — higher = slower/smoother AWB adaptation (0.0-1.0)
AWB_SMOOTH    = 0.7
# ─────────────────────────────────────────────────────────────────────────────

CTRL_WIN = "White Balance Controls"

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
    cv2.resizeWindow(CTRL_WIN, 500, 250)
    # Manual override gains — only used when AWB is OFF
    cv2.createTrackbar("R Gain  x100", CTRL_WIN, 100, 400, lambda x: None)
    cv2.createTrackbar("G Gain  x100", CTRL_WIN, 100, 400, lambda x: None)
    cv2.createTrackbar("B Gain  x100", CTRL_WIN, 100, 400, lambda x: None)
    cv2.createTrackbar("Brightness",   CTRL_WIN,  50, 100, lambda x: None)
    # AWB toggle: 0 = manual, 1 = auto
    cv2.createTrackbar("AWB (0=off)",  CTRL_WIN,   1,   1, lambda x: None)


def get_controls():
    r_gain     = max(cv2.getTrackbarPos("R Gain  x100", CTRL_WIN), 1) / 100.0
    g_gain     = max(cv2.getTrackbarPos("G Gain  x100", CTRL_WIN), 1) / 100.0
    b_gain     = max(cv2.getTrackbarPos("B Gain  x100", CTRL_WIN), 1) / 100.0
    brightness = max(cv2.getTrackbarPos("Brightness",   CTRL_WIN), 1) / 100.0
    awb_on     = cv2.getTrackbarPos("AWB (0=off)",  CTRL_WIN) == 1
    return r_gain, g_gain, b_gain, brightness, awb_on


def compute_awb_gains(bgr8, current_r, current_g, current_b):
    """
    Gray world AWB with midtone masking.
    Excludes very dark pixels (lens border) and very bright pixels (blown highlights)
    so they don't skew the color estimate.
    """
    b, g, r = cv2.split(bgr8)

    # Only use pixels in the midtone range (not too dark, not blown out)
    gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
    mask = (gray > 30) & (gray < 220)

    if mask.sum() < 1000:
        # Not enough valid pixels, return unchanged
        return current_r, current_g, current_b

    r_avg = float(r[mask].mean()) + 1e-6
    g_avg = float(g[mask].mean()) + 1e-6
    b_avg = float(b[mask].mean()) + 1e-6

    # Target: make all channels equal to the mean of all three
    target = (r_avg + g_avg + b_avg) / 3.0

    new_r = target / r_avg
    new_g = target / g_avg
    new_b = target / b_avg

    # Normalize so G stays at 1.0 (G is reference)
    new_r = new_r / new_g
    new_b = new_b / new_g
    new_g = 1.0

    # Clamp gains to reasonable range
    new_r = np.clip(new_r, 0.5, 3.0)
    new_b = np.clip(new_b, 0.5, 3.0)

    # Smooth with previous gains to avoid flickering
    smooth_r = AWB_SMOOTH * current_r + (1 - AWB_SMOOTH) * new_r
    smooth_b = AWB_SMOOTH * current_b + (1 - AWB_SMOOTH) * new_b

    return smooth_r, new_g, smooth_b


def main():
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

    prev_gains  = (None, None, None, None, None)
    alpha_b = alpha_g = alpha_r = 0.25

    # AWB state — start neutral
    awb_r, awb_g, awb_b = 1.0, 1.0, 1.0
    awb_initialized = False

    print("Press 'q' to quit. Toggle AWB slider to switch between auto and manual WB.")

    frame_count = 0
    fps_display = 0.0
    t0          = time.time()

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        r_gain, g_gain, b_gain, brightness, awb_on = get_controls()

        # ── Use AWB gains or manual gains ─────────────────────────────────────
        if awb_on:
            eff_r, eff_g, eff_b = awb_r, awb_g, awb_b
        else:
            eff_r, eff_g, eff_b = r_gain, g_gain, b_gain

        gains = (eff_r, eff_g, eff_b, brightness, awb_on)
        if gains != prev_gains:
            # Scale factor: map 12-bit (0-4095) to 8-bit (0-255)
            # 255 / 4095 = 0.0623
            scale = brightness * 0.0623
            alpha_b    = scale * eff_b
            alpha_g    = scale * eff_g
            alpha_r    = scale * eff_r
            prev_gains = gains

        # ── CPU: Reinterpret + black level correction ────────────────────────
        # Sensor outputs 16-bit left-shifted values (12-bit << 4)
        # Shift right by 4 to get true 12-bit, subtract black level 110, clamp
        raw16 = raw_frame.view(np.uint16).reshape(HEIGHT, WIDTH).astype(np.int32)
        bayer16 = np.clip((raw16 >> 4) - 110, 0, 4095).astype(np.uint16)

        # ── GPU: Upload → Demosaic ────────────────────────────────────────────
        gpu_bayer.upload(bayer16, stream)
        cv2.cuda.demosaicing(gpu_bayer, BAYER_PAT, gpu_bgr16, stream=stream)

        # ── GPU: Split → brightness+WB per channel → merge ───────────────────
        cv2.cuda.split(gpu_bgr16, [gpu_b16, gpu_g16, gpu_r16], stream=stream)
        gpu_b16.convertTo(cv2.CV_8UC1, alpha_b, gpu_b8, 0)
        gpu_g16.convertTo(cv2.CV_8UC1, alpha_g, gpu_g8, 0)
        gpu_r16.convertTo(cv2.CV_8UC1, alpha_r, gpu_r8, 0)
        cv2.cuda.merge([gpu_b8, gpu_g8, gpu_r8], gpu_bgr8, stream=stream)

        # ── Download ──────────────────────────────────────────────────────────
        stream.waitForCompletion()
        bgr8 = gpu_bgr8.download()

        # ── AWB: recompute gains every N frames ───────────────────────────────
        if awb_on and frame_count % AWB_INTERVAL == 0:
            awb_r, awb_g, awb_b = compute_awb_gains(bgr8, awb_r, awb_g, awb_b)
            # Force alpha recalc next frame
            prev_gains = (None, None, None, None, None)

        # ── Resize for display ────────────────────────────────────────────────
        display = cv2.resize(bgr8, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)

        # ── FPS counter ───────────────────────────────────────────────────────
        frame_count += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            t0 = time.time()

        wb_label = "AWB" if awb_on else f"R:{eff_r:.2f} G:{eff_g:.2f} B:{eff_b:.2f}"
        cv2.putText(display, f"FPS: {fps_display:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"{wb_label}  Bri:{brightness:.2f}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.namedWindow("RAW12 Camera (GPU debayer)", cv2.WINDOW_NORMAL)
        cv2.imshow("RAW12 Camera (GPU debayer)", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
