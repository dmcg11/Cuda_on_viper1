import cv2
import numpy as np
import time

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = 5
WIDTH       = 1920
HEIGHT      = 1280
FPS         = 30
BAYER_PAT   = cv2.COLOR_BayerBG2BGR  # BGGR pattern

# ── White Balance Gains (tune these to fix color cast) ────────────────────────
# Values > 1.0 boost that channel, < 1.0 reduce it
# Current: reduce green, boost red slightly to fix green tint
WB_R = 1.6
WB_G = 1.0
WB_B = 1.4
# ─────────────────────────────────────────────────────────────────────────────

def unpack_raw12_unpacked(raw_bytes, width, height):
    """
    Unpacked RAW12 — each pixel stored in 2 bytes (uint16, little-endian).
    """
    raw = np.frombuffer(raw_bytes, dtype=np.uint16)
    return raw.reshape(height, width)


def apply_white_balance(bgr8):
    """
    Apply per-channel gain for white balance correction on CPU.
    """
    b, g, r = cv2.split(bgr8)
    r = np.clip(r.astype(np.float32) * WB_R, 0, 255).astype(np.uint8)
    g = np.clip(g.astype(np.float32) * WB_G, 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.float32) * WB_B, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])


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

    print("Press 'q' to quit, 'w' to print current channel averages for WB tuning")

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

        # ── GPU: Upload → Demosaic → Scale to 8-bit ───────────────────────────
        gpu_bayer.upload(bayer16, stream)
        cv2.cuda.demosaicing(gpu_bayer, BAYER_PAT, gpu_bgr16, stream=stream)
        gpu_bgr16.convertTo(cv2.CV_8UC3, 1/4, gpu_bgr8, 0)

        # ── Download and apply white balance ──────────────────────────────────
        stream.waitForCompletion()
        bgr8 = gpu_bgr8.download()
        bgr8 = apply_white_balance(bgr8)

        # ── FPS counter ───────────────────────────────────────────────────────
        frame_count += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            t0 = time.time()

        cv2.putText(bgr8, f"FPS: {fps_display:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(bgr8, f"WB R:{WB_R:.2f} G:{WB_G:.2f} B:{WB_B:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow("RAW12 Camera (GPU debayer)", bgr8)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            # Print channel averages to help tune WB gains
            b, g, r = cv2.split(bgr8)
            print(f"Channel averages — R: {r.mean():.1f}  G: {g.mean():.1f}  B: {b.mean():.1f}")
            print(f"Ideal ratio     — R: {g.mean()/r.mean():.2f}x  B: {g.mean()/b.mean():.2f}x  (multiply current gains by these)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
