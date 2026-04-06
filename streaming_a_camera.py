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

def unpack_raw12_unpacked(raw_bytes, width, height):
    """
    Unpacked RAW12 — each pixel stored in 2 bytes (uint16, little-endian).
    No bit manipulation needed, just reinterpret and reshape.
    """
    raw = np.frombuffer(raw_bytes, dtype=np.uint16)
    return raw.reshape(height, width)


def open_camera():
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'BG12'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    cap.set(cv2.CAP_PROP_CONVERT_RGB,  0)  # Disable auto-conversion

    if not cap.isOpened():
        raise RuntimeError("Failed to open /dev/video5")

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps}fps")

    return cap


def main():
    cap = open_camera()

    # Pre-allocate GPU mats
    gpu_bayer  = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_16UC1)
    gpu_bgr16  = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_16UC3)
    gpu_bgr8   = cv2.cuda_GpuMat(HEIGHT, WIDTH, cv2.CV_8UC3)

    stream = cv2.cuda_Stream()

    print("Press 'q' to quit")

    frame_count = 0
    fps_display = 0.0
    t0 = time.time()

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # ── CPU: Reinterpret buffer as uint16 Bayer ───────────────────────────
        bayer16 = unpack_raw12_unpacked(raw_frame.tobytes(), WIDTH, HEIGHT)

        # ── GPU: Upload ───────────────────────────────────────────────────────
        gpu_bayer.upload(bayer16, stream)

        # ── GPU: Demosaic BGGR Bayer → BGR 16-bit ────────────────────────────
        cv2.cuda.demosaicing(gpu_bayer, BAYER_PAT, gpu_bgr16, stream=stream)

        # ── GPU: Scale to 8-bit (alpha=1/16 maps 12-bit range to 8-bit) ──────
        # args: (rtype, alpha, dst, beta)
        gpu_bgr16.convertTo(cv2.CV_8UC3, 1/16, gpu_bgr8, 0)

        # ── Download to CPU for display ───────────────────────────────────────
        stream.waitForCompletion()
        bgr8 = gpu_bgr8.download()

        # ── FPS counter ───────────────────────────────────────────────────────
        frame_count += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            t0 = time.time()

        cv2.putText(bgr8, f"FPS: {fps_display:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("RAW12 Camera (GPU debayer)", bgr8)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
