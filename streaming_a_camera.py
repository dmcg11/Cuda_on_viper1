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

    gpu_bayer  = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_16UC1)
    gpu_bgr16  = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_16UC3)
    gpu_bgr8   = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC3)
    bayer16    = np.empty((HEIGHT, WIDTH), dtype=np.uint16)

    lut_r = np.arange(256, dtype=np.uint8)
    lut_g = np.arange(256, dtype=np.uint8)
    lut_b = np.arange(256, dtype=np.uint8)

    stream = cv2.cuda_Stream()

    print("Press 'q' to quit")
    print("Profiling first 60 frames then printing timings...")

    frame_count  = 0
    fps_display  = 0.0
    t0           = time.time()
    prev_gains   = (None, None, None, None)

    # Profiling accumulators
    t_capture = t_unpack = t_gpu = t_download = t_wb = t_display = 0.0
    PROFILE_FRAMES = 60

    while True:
        # ── Capture ───────────────────────────────────────────────────────────
        t = time.perf_counter()
        ret, raw_frame = cap.read()
        t_capture += time.perf_counter() - t
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

        # ── Unpack ────────────────────────────────────────────────────────────
        t = time.perf_counter()
        np.copyto(bayer16, raw_frame.view(np.uint16).reshape(HEIGHT, WIDTH))
        t_unpack += time.perf_counter() - t

        # ── GPU pipeline ──────────────────────────────────────────────────────
        t = time.perf_counter()
        gpu_bayer.upload(bayer16, stream)
        cv2.cuda.demosaicing(gpu_bayer, BAYER_PAT, gpu_bgr16, stream=stream)
        gpu_bgr16.convertTo(cv2.CV_8UC3, brightness, gpu_bgr8, 0)
        t_gpu += time.perf_counter() - t

        # ── Download ──────────────────────────────────────────────────────────
        t = time.perf_counter()
        stream.waitForCompletion()
        bgr8 = gpu_bgr8.download()
        t_download += time.perf_counter() - t

        # ── White balance ─────────────────────────────────────────────────────
        t = time.perf_counter()
        b_ch, g_ch, r_ch = cv2.split(bgr8)
        bgr8 = cv2.merge([cv2.LUT(b_ch, lut_b), cv2.LUT(g_ch, lut_g), cv2.LUT(r_ch, lut_r)])
        t_wb += time.perf_counter() - t

        # ── Display ───────────────────────────────────────────────────────────
        t = time.perf_counter()
        frame_count += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            t0 = time.time()

        cv2.putText(bgr8, f"FPS: {fps_display:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(bgr8, f"R:{r_gain:.2f} G:{g_gain:.2f} B:{b_gain:.2f}  Bri:{brightness:.2f}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imshow("RAW12 Camera (GPU debayer)", bgr8)
        t_display += time.perf_counter() - t

        # ── Print profile after N frames then reset ───────────────────────────
        if frame_count == 0 and fps_display > 0 and \
           (t_capture + t_unpack + t_gpu + t_download + t_wb + t_display) > 0:
            total = t_capture + t_unpack + t_gpu + t_download + t_wb + t_display
            print(f"\n── Profile over ~{PROFILE_FRAMES} frames ──────────────────")
            print(f"  cap.read()   : {t_capture*1000/PROFILE_FRAMES:6.2f} ms/frame")
            print(f"  unpack       : {t_unpack*1000/PROFILE_FRAMES:6.2f} ms/frame")
            print(f"  GPU pipeline : {t_gpu*1000/PROFILE_FRAMES:6.2f} ms/frame")
            print(f"  download     : {t_download*1000/PROFILE_FRAMES:6.2f} ms/frame")
            print(f"  white bal    : {t_wb*1000/PROFILE_FRAMES:6.2f} ms/frame")
            print(f"  display      : {t_display*1000/PROFILE_FRAMES:6.2f} ms/frame")
            print(f"  TOTAL        : {total*1000/PROFILE_FRAMES:6.2f} ms/frame  →  {PROFILE_FRAMES/total:.1f} FPS")
            t_capture = t_unpack = t_gpu = t_download = t_wb = t_display = 0.0

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
