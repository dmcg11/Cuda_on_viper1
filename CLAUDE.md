# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Platform

**Jetson Xavier NX** running JetPack 5.1.x (R35), CUDA 11.4, OpenCV 4.8.0 built from source with CUDA support. The GPU compute capability is SM 7.2 (Volta). The camera sensor at `/dev/video5` is an OV-series RAW12 Bayer sensor at I2C address `0x36` on bus 1.

Ensure CUDA is in PATH before running anything:
```bash
export PATH=/usr/local/cuda-11.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
```

Verify OpenCV has CUDA:
```bash
python3 -c "import cv2; print(cv2.__version__); print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"
```

## Running the camera stream

```bash
python3 streaming_a_camera.py
```

Press `q` to quit, `s` to manually save a snapshot to `/tmp/camera_snapshot.jpg`.

## Architecture: `streaming_a_camera.py`

Single-file application — all logic lives here. Key design decisions:

**GPU pipeline (per frame):**
1. Raw BG12 frame captured via V4L2 → reinterpreted as `uint16` on CPU via `np.copyto`
2. Uploaded to `cv2.cuda_GpuMat` → GPU demosaicing (`BayerBG → BGR`, 16-bit) → channel split → per-channel scale to 8-bit (brightness alpha applied here) → merge → download

**I2C sensor control:**
- Writes happen immediately after `cap.read()` returns (start of frame readout) to avoid mid-frame register changes causing banding
- Trackbar changes are debounced: a change must be stable for `DEBOUNCE_FRAMES=3` frames before the I2C write fires
- Registers controlled: exposure (`0x3501/3502`), analog gain (`0x3508/3509`), AWB gains (`0x5180–0x5187`)
- AWB gain format: 5.10 fixed-point (`val = gain * 1024`, max `0x7FFF`)

**Auto white balance:**
- Gray-world algorithm with midtone masking (pixels 30–220)
- On first enable: 20 iterations for fast convergence, then every `AWB_INTERVAL=15` frames
- Smoothing: exponential moving average with `AWB_SMOOTH=0.7`
- Computed gains are written back to the I2C sensor AND reflected on the sliders

**Auto snapshot/video:**
- 5 seconds after start: saves JPEG to `/tmp/camera_snapshot.jpg`
- Immediately after snapshot: buffers 15 frames, saves `/tmp/camera_clip.mp4` at 10 fps

## OpenCV build (from source)

See README.md for the full build procedure. Critical flags: `CUDA_ARCH_BIN=7.2`, `WITH_CUDA=ON`, `WITH_CUDNN=ON`, `OPENCV_DNN_CUDA=ON`, `WITH_GSTREAMER=ON`. Python path fix required: symlink `site-packages/cv2` → `dist-packages/cv2`. If `CUDA devices: 0`, the build found no `nvcc` — clean and rebuild from Step 6.
