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

CTRL_WIN = "White Balance Controls"

def unpack_raw12_unpacked(raw_bytes, width, height):
    raw = np.frombuffer(raw_bytes, dtype=np.uint16)
    return raw.reshape(height, width)


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
    cv2.resizeWindow(CTRL_WIN, 500, 200)
    cv2.createTrackbar("R Gain  x100", CTRL_WIN, 160, 400, lambda x: None)
    cv2.createTrackbar("G Gain  x100", CTRL_WIN, 100, 400, lambda x: None)
    cv2.createTrackbar("B Gain  x100", CTRL_WIN, 140, 400, lambda x: None)
    cv2.createTrackbar("Brightness",   CTRL_WIN,  25, 100, lambda x: None)


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

    # Per-channel GPU mats for white balance
    gpu_b      = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC1)
    gpu_g      = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC1)
    gpu_r      = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC1)
    gpu_b_wb   = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC1)
    gpu_g_wb   = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC1)
    gpu_r_wb   = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC1)
    gpu_wb     = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC3)

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

        r_gain, g_gain, b_gain, brightness = get_controls()

        # ── CPU: Reinterpret buffer as uint16 Bayer ───────────────────────────
        bayer16 = unpack_raw12_unpacked(raw_frame.tobytes(), WIDTH, HEIGHT)

        # ── GPU: Upload → Demosaic → Scale to 8-bit ───────────────────────────
        gpu_bayer.upload(bayer16, stream)
        cv2.cuda.demosaicing(gpu_bayer, BAYER_PAT, gpu_bgr16, stream=stream)
        gpu_bgr16.convertTo(cv2.CV_8UC3, brightness, gpu_bgr8, 0)

        # ── GPU: White balance — split, scale each channel, merge ─────────────
        # convertTo(rtype, alpha, dst, beta) scales and clips to uint8 range
        cv2.cuda.split(gpu_bgr8, [gpu_b, gpu_g, gpu_r], stream=stream)
        gpu_b.convertTo(cv2.CV_8UC1, b_gain, gpu_b_wb, 0)
        gpu_g.convertTo(cv2.CV_8UC1, g_gain, gpu_g_wb, 0)
        gpu_r.convertTo(cv2.CV_8UC1, r_gain, gpu_r_wb, 0)
        cv2.cuda.merge([gpu_b_wb, gpu_g_wb, gpu_r_wb], gpu_wb, stream=stream)

        # ── Download ──────────────────────────────────────────────────────────
        stream.waitForCompletion()
        bgr8 = gpu_wb.download()

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
