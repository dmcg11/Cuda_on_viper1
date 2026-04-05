# Cuda_on_viper1
cuda projects with Viper1


# Building OpenCV with CUDA on Jetson Xavier NX (JetPack 5.x)

**Target:** Jetson Xavier NX, JetPack 5.1.x (R35), CUDA 11.4, OpenCV 4.8.0

---

## Step 1 — Verify Your JetPack Version

```bash
cat /etc/nv_tegra_release
```

You should see `R35` (JetPack 5.x). This guide targets that version.

---

## Step 2 — Add CUDA to Your PATH

CUDA is installed at `/usr/local/cuda-11.4` but is not in PATH by default. Fix this permanently:

```bash
echo 'export PATH=/usr/local/cuda-11.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify it works:

```bash
nvcc --version
# Should print: Cuda compilation tools, release 11.4
```

If `nvcc` is not found at all, install JetPack components first:

```bash
sudo apt-get install -y nvidia-jetpack
```

---

## Step 3 — Install Dependencies

```bash
sudo apt-get install -y \
    build-essential cmake git pkg-config \
    libjpeg-dev libtiff-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgtk2.0-dev libcanberra-gtk* \
    libxvidcore-dev libx264-dev libgtk-3-dev \
    libtbb2 libtbb-dev libdc1394-22-dev \
    libv4l-dev v4l-utils \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    python3-dev python3-numpy \
    libatlas-base-dev gfortran \
    libhdf5-serial-dev hdf5-tools
```

Also install numpy for Python bindings:

```bash
pip3 install numpy
```

---

## Step 4 — Add Swap Space (Prevents OOM During Build)

The build is memory-intensive. Add 8GB of swap before starting:

```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Step 5 — Clone OpenCV and opencv_contrib

```bash
mkdir -p ~/opencv_build && cd ~/opencv_build

git clone --branch 4.8.0 --depth 1 https://github.com/opencv/opencv.git
git clone --branch 4.8.0 --depth 1 https://github.com/opencv/opencv_contrib.git
```

---

## Step 6 — Run CMake

```bash
mkdir -p ~/opencv_build/opencv/build
cd ~/opencv_build/opencv/build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.4 \
      -D CUDA_ARCH_BIN="7.2" \
      -D CUDA_ARCH_PTX="" \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D CUDA_FAST_MATH=ON \
      -D WITH_CUBLAS=ON \
      -D ENABLE_NEON=ON \
      -D WITH_GSTREAMER=ON \
      -D BUILD_opencv_python3=ON \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
      -D WITH_TBB=ON \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF \
      .. 2>&1 | tee cmake_output.log
```

> **Important:** The trailing `..` is required — it points CMake to the source directory.

---

## Step 7 — Verify CUDA Was Detected Before Building

After CMake finishes, confirm CUDA was found before spending 40 minutes on a broken build:

```bash
grep -E "CUDA|cuDNN|NVIDIA" cmake_output.log
```

You must see all of these lines:

```
--   NVIDIA CUDA:                   YES (ver 11.4, CUFFT CUBLAS FAST_MATH)
--     NVIDIA GPU arch:             72
--     cuDNN:                       YES (ver 8.x.x)
```

If CUDA shows `NO`, do not proceed — go back and check that `nvcc --version` works.

---

## Step 8 — Build (20–40 minutes)

```bash
cd ~/opencv_build/opencv/build
make -j$(nproc)
```

- The percentage counter will stall at 98–99% during linking — this is normal, do not kill it.
- If the build crashes with an out-of-memory error, restart with fewer jobs: `make -j2`

---

## Step 9 — Install

```bash
sudo make install
sudo ldconfig
```

---

## Step 10 — Fix Python Path

OpenCV installs to `site-packages` but Python looks in `dist-packages`. Fix with a symlink:

```bash
sudo ln -sf /usr/local/lib/python3.8/site-packages/cv2 \
            /usr/local/lib/python3.8/dist-packages/cv2
```

---

## Step 11 — Verify Everything Works

```bash
python3 -c "import cv2; print(cv2.__version__); print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"
```

Expected output:
```
4.8.0
CUDA devices: 1
```

Full build info check:
```bash
python3 -c "
import cv2
info = cv2.getBuildInformation()
for line in info.split('\n'):
    if any(x in line for x in ['CUDA', 'cuDNN', 'NVIDIA', 'GStreamer', 'Python']):
        print(line)
"
```

---

## Quick GPU Smoke Test

Run this to confirm CUDA is actually accelerating operations:

```python
import cv2
import numpy as np
import time

img = np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)

# CPU
start = time.time()
for _ in range(100):
    cv2.GaussianBlur(img, (21, 21), 0)
cpu_time = time.time() - start

# GPU
gpu_img = cv2.cuda_GpuMat()
gpu_img.upload(img)
gpu_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (21, 21), 0)
start = time.time()
for _ in range(100):
    gpu_filter.apply(gpu_img)
gpu_time = time.time() - start

print(f"CPU: {cpu_time:.3f}s")
print(f"GPU: {gpu_time:.3f}s")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")
```

---

## Key CMake Flags Reference

| Flag | Value | Purpose |
|---|---|---|
| `CUDA_ARCH_BIN` | `7.2` | Xavier NX is Volta SM 7.2 — compiles for exact arch, avoids JIT overhead |
| `OPENCV_DNN_CUDA` | `ON` | Enables CUDA backend for the DNN module |
| `CUDA_FAST_MATH` | `ON` | Enables `--use_fast_math` in NVCC for significant speedup |
| `WITH_CUBLAS` | `ON` | Uses cuBLAS for accelerated matrix operations |
| `WITH_CUDNN` | `ON` | Enables cuDNN for DNN inference acceleration |
| `ENABLE_NEON` | `ON` | ARM NEON SIMD for CPU-side operations |
| `CUDA_TOOLKIT_ROOT_DIR` | `/usr/local/cuda-11.4` | Explicit path so CMake doesn't miss CUDA if not in PATH |

---

## Common Pitfalls

- **`nvcc` not in PATH during CMake** — CUDA will be silently skipped. Always run `nvcc --version` before cmake.
- **Wrong trailing argument in cmake** — the `..` at the end is required and must be the last argument.
- **Build OOM crash** — add swap in Step 4 and/or reduce to `make -j2`.
- **`CUDA devices: 0` after install** — means CMake ran without nvcc visible. Clean build dir (`rm -rf *`) and redo from Step 6.
- **`No module named cv2`** — the site-packages vs dist-packages symlink in Step 10 is missing.
