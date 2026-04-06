import cv2
import numpy as np
import time

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = 5
WIDTH       = 1920
HEIGHT      = 1280
FPS         = 30
BAYER_PAT   = cv2.COLOR_BayerBG2BGR  # BGGR pattern
# ─────────────────────────────────────────────────────────────────────────────

# Trackbar window name
CTRL_WIN = "White Balance Controls"

def unpack_raw12_unpacked(raw_bytes, width, height):
    raw = np.frombuffer(raw_bytes, dtype=np.uint16)
    return raw.reshape(height, width)


def apply_white_balance(bgr8, r_gain, g_gain, b_gain):
    b, g, r = cv2.split(bgr8)
    r = np.clip(r.astype(np.float32) * r_gain, 0, 255).astype(np.uint8)
    g = np.clip(g.astype(np.float32) * g_gain, 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.float32) * b_gain, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])


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
    """Create a separate window with trackbars for WB and brightness control."""
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WIN, 500, 200)

    # Gains stored as integers 0–400, divide by 100 to get float (0.0–4.0)
    # Default: R=160 (1.6x), G=100 (1.0x), B=140 (1.4x)
    cv2.createTrackbar("R Gain  x100", CTRL_WIN, 160, 400, lambda x: None)
    cv2.createTrackbar("G Gain  x100", CTRL_WIN, 100, 400, lambda x: None)
    cv2.createTrackbar("B Gain  x100", CTRL_WIN, 140, 400, lambda x: None)

    # Brightness: stored as 1–100, maps to alpha 1/100 – 1/1
    # Default: 25 → alpha = 1/4 (same as before)
    cv2.createTrackbar("Brightness", CTRL_WIN,  25, 100, lambda x: None)


def get_controls():
    r_gain     = max(cv2.getTrackbarPos("R Gain  x100", CTRL_WIN), 1) / 100.0
    g_gain     = max(cv2.getTrackbarPos("G Gain  x100", CTRL_WIN), 1) / 100.0
    b_gain     = max(cv2.getTrackbarPos("B Gain  x100", CTRL_WIN), 1) / 100.0
    brightness = max(cv2.getTrackbarPos("Brightness",   CTRL_WIN), 1) / 100.0
    return r_gain, g_gain, b_gain, brightness


def main():
    cap = open_camera()
    create_controls()

    # Pre-allocate GPU mats
    gpu_bayer  = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_16UC1)
    gpu_bgr16  = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_16UC3)
    gpu_bgr8   = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC3)

    stream = cv2.cuda_Stream()

    print("Press 'q' to quit, 'w' to print channel averages for WB tuning")

    frame_count = 0
    fps_display = 0.0
    t0 = time.time()

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Read controls every frame
        r_gain, g_gain, b_gain, brightness = get_controls()

        # ── CPU: Reinterpret buffer as uint16 Bayer ───────────────────────────
        bayer16 = unpack_raw12_unpacked(raw_frame.tobytes(), WIDTH, HEIGHT)

        # ── GPU: Upload → Demosaic → Scale to 8-bit ───────────────────────────
        gpu_bayer.upload(bayer16, stream)
        cv2.cuda.demosaicing(gpu_bayer, BAYER_PAT, gpu_bgr16, stream=stream)
        gpu_bgr16.convertTo(cv2.CV_8UC3, brightness, gpu_bgr8, 0)

        # ── Download and apply white balance ──────────────────────────────────
        stream.waitForCompletion()
        bgr8 = gpu_bgr8.download()
        bgr8 = apply_white_balance(bgr8, r_gain, g_gain, b_gain)

        # ── FPS counter ───────────────────────────────────────────────────────
        frame_count += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            t0 = time.time()

        # ── Overlay ───────────────────────────────────────────────────────────
        cv2.putText(bgr8, f"FPS: {fps_display:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(bgr8, f"R:{r_gain:.2f} G:{g_gain:.2f} B:{b_gain:.2f}  Brightness:{brightness:.2f}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow("RAW12 Camera (GPU debayer)", bgr8)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            b, g, r = cv2.split(bgr8)
            print(f"Channel averages — R: {r.mean():.1f}  G: {g.mean():.1f}  B: {b.mean():.1f}")
            print(f"Suggested gains  — R: {g.mean()/max(r.mean(),1):.2f}x  B: {g.mean()/max(b.mean(),1):.2f}x")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
