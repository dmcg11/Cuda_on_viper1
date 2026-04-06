import cv2
import numpy as np
import time
import threading

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = 5
WIDTH       = 1920
HEIGHT      = 1280
FPS         = 30
BAYER_PAT   = cv2.COLOR_BayerBG2BGR  # BGGR pattern
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


def display_thread(frame_holder, stop_event):
    """
    Runs in a separate thread — handles imshow and waitKey independently
    so it never blocks the capture/GPU pipeline.
    """
    while not stop_event.is_set():
        frame = frame_holder.get('frame')
        fps   = frame_holder.get('fps', 0.0)
        if frame is not None:
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("RAW12 Camera (GPU debayer)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break
        elif key == ord('w') and frame is not None:
            b, g, r = cv2.split(frame)
            print(f"Channel averages — R: {r.mean():.1f}  G: {g.mean():.1f}  B: {b.mean():.1f}")
            print(f"Suggested gains  — R: {g.mean()/max(r.mean(),1):.2f}x  B: {g.mean()/max(b.mean(),1):.2f}x")


def main():
    cap = open_camera()
    create_controls()

    gpu_bayer  = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_16UC1)
    gpu_bgr16  = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_16UC3)
    gpu_bgr8   = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC3)
    bayer16    = np.empty((HEIGHT, WIDTH), dtype=np.uint16)

    lut_r = np.arange(256, dtype=np.uint8)
    lut_g = np.arange(256, dtype=np.uint8)
    lut_b = np.arange(256, dtype=np.uint8)

    stream     = cv2.cuda_Stream()
    stop_event = threading.Event()

    # Shared dict between capture thread and display thread
    frame_holder = {'frame': None, 'fps': 0.0}

    # Start display thread
    disp_thread = threading.Thread(target=display_thread,
                                   args=(frame_holder, stop_event),
                                   daemon=True)
    disp_thread.start()

    print("Press 'q' to quit, 'w' to print channel averages for WB tuning")

    frame_count = 0
    fps_display = 0.0
    t0          = time.time()
    prev_gains  = (None, None, None, None)

    while not stop_event.is_set():
        ret, raw_frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        r_gain, g_gain, b_gain, brightness = get_controls()
        gains = (r_gain, g_gain, b_gain, brightness)
        if gains != prev_gains:
            lut_r = np.clip(np.arange(256) * r_gain, 0, 255).astype(np.uint8)
            lut_g = np.clip(np.arange(256) * g_gain, 0, 255).astype(np.uint8)
            lut_b = np.clip(np.arange(256) * b_gain, 0, 255).astype(np.uint8)
            prev_gains = gains

        # ── CPU: Zero-copy reinterpret ────────────────────────────────────────
        np.copyto(bayer16, raw_frame.view(np.uint16).reshape(HEIGHT, WIDTH))

        # ── GPU: Upload → Demosaic → Scale to 8-bit ───────────────────────────
        gpu_bayer.upload(bayer16, stream)
        cv2.cuda.demosaicing(gpu_bayer, BAYER_PAT, gpu_bgr16, stream=stream)
        gpu_bgr16.convertTo(cv2.CV_8UC3, brightness, gpu_bgr8, 0)

        # ── Download ──────────────────────────────────────────────────────────
        stream.waitForCompletion()
        bgr8 = gpu_bgr8.download()

        # ── White balance via LUT ─────────────────────────────────────────────
        b_ch, g_ch, r_ch = cv2.split(bgr8)
        bgr8 = cv2.merge([cv2.LUT(b_ch, lut_b), cv2.LUT(g_ch, lut_g), cv2.LUT(r_ch, lut_r)])

        # ── FPS counter ───────────────────────────────────────────────────────
        frame_count += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            t0 = time.time()

        # ── Hand frame to display thread (non-blocking) ───────────────────────
        frame_holder['frame'] = bgr8
        frame_holder['fps']   = fps_display

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
