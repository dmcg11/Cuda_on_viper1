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
    prev_gains = (None, None, None, None)
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

        r_gain, g_gain, b_gain, brightness = get_controls()
        gains = (r_gain, g_gain, b_gain, brightness)

        if gains != prev_gains:
            alpha_b    = brightness * b_gain
            alpha_g    = brightness * g_gain
            alpha_r    = brightness * r_gain
            prev_gains = gains

        # ── CPU: Zero-copy reinterpret ────────────────────────────────────────
        np.copyto(bayer16, raw_frame.view(np.uint16).reshape(HEIGHT, WIDTH))

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
        cv2.putText(display, f"R:{r_gain:.2f} G:{g_gain:.2f} B:{b_gain:.2f}  Bri:{brightness:.2f}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.namedWindow("RAW12 Camera (GPU debayer)", cv2.WINDOW_NORMAL)
        cv2.imshow("RAW12 Camera (GPU debayer)", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
