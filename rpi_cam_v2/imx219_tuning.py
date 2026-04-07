#!/usr/bin/env python3
"""
IMX219 RAW8 Camera Tuning Script  (Jetson / Tegra VI edition)
=============================================================
Performs software AWB, AEC/AGC, and black-level correction on RAW8 Bayer
frames from an IMX219 sensor on a Jetson (tegra-video driver).

Capture uses direct V4L2 ioctls + mmap – NOT OpenCV's generic V4L2 backend –
because Tegra VI nodes require an explicit VIDIOC_S_FMT('RGGB') before the
device can be opened for streaming.

Sensor registers are written back over I2C (smbus2) to close the AE/AG loop
on hardware, not just in software.

Register map (linux/drivers/media/i2c/imx219.c + Sony IMX219PQH5-C datasheet)
  0x0157          ANA_GAIN_GLOBAL_A   8-bit   range 0-232
  0x0158-0x0159   DIG_GAIN_GLOBAL_A   12-bit  range 0x0100-0x0FFF (=1x-~16x)
  0x015A-0x015B   COARSE_INTEG_TIME_A 16-bit  range 4-65535 lines
  0x0160-0x0161   FRM_LENGTH_A        16-bit  VTS

Your v4l2-ctl output shows: "cam_v1 1-0008" -> I2C bus 1, address 0x08.
Default --i2c-addr is 0x08.

Usage
-----
  python3 imx219_tuning.py --device /dev/video5 --i2c-bus 1 --i2c-addr 0x08

  Optional flags:
    --width  1920  --height 1080      (default)
    --target-brightness 100           (0-255, default 100)
    --no-awb                          disable gray-world AWB
    --no-aec                          disable auto-exposure/gain

Runtime keys (OpenCV window must have focus):
    q / ESC  quit
    r        reset all gains + exposure to sensor defaults
    s        save current debayered+WB frame as PNG
    p        print current sensor register state

Dependencies
------------
  pip install smbus2 numpy opencv-python
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# ------------------------------------------------------------------------------
# smbus2 - optional; register writes silently skipped if absent
# ------------------------------------------------------------------------------
try:
    import smbus2
    HAS_SMBUS = True
except ImportError:
    HAS_SMBUS = False
    print("[WARN] smbus2 not found - I2C writes disabled.  pip install smbus2")


# ==============================================================================
# Camera capture  (Tegra VI / OpenCV V4L2)
# ==============================================================================
# The Tegra VI driver works with cv2.VideoCapture when:
#   - CAP_PROP_FOURCC is set to the native sensor format BEFORE reading
#   - CAP_PROP_CONVERT_RGB is 0  (do not let OpenCV convert the raw data)
# For IMX219 RAW8 the fourcc is 'RGGB' (BayerRG8).
# For the previous RAW12 sensor in streaming_a_camera.py it was 'BG12'.

class TegraCapture:
    """OpenCV V4L2 capture configured for Tegra VI RAW8 (RGGB) output."""

    def __init__(self, device_index: int, width: int, height: int, fps: int = 30):
        self._cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open /dev/video{device_index}")

        # Must set fourcc and disable RGB conversion BEFORE the first read
        self._cap.set(cv2.CAP_PROP_FOURCC,
                      cv2.VideoWriter_fourcc(*'RGGB'))   # RAW8 Bayer RGGB
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS,          fps)
        self._cap.set(cv2.CAP_PROP_CONVERT_RGB,  0)     # keep raw Bayer bytes

        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAP] /dev/video{device_index}  "
              f"{self.width}x{self.height}  RAW8 RGGB")

    def read(self):
        """Return (H, W) uint8 numpy array or None on failure."""
        ret, frame = self._cap.read()
        if not ret:
            return None
        # Frame arrives as a flat or 2-channel array depending on driver;
        # reshape to (H, W) uint8 in all cases.
        raw = frame.reshape((self.height, self.width))
        return raw

    def release(self):
        self._cap.release()


# ==============================================================================
# IMX219 register definitions
# ==============================================================================
REG_ANA_GAIN     = 0x0157
ANA_GAIN_MIN     = 0
ANA_GAIN_MAX     = 232
ANA_GAIN_DEFAULT = 0

REG_DIG_GAIN_HI  = 0x0158
REG_DIG_GAIN_LO  = 0x0159
DIG_GAIN_MIN     = 0x0100
DIG_GAIN_MAX     = 0x0FFF
DIG_GAIN_DEFAULT = 0x0100

REG_EXPOSURE_HI  = 0x015A
REG_EXPOSURE_LO  = 0x015B
EXPOSURE_MIN     = 4
EXPOSURE_MAX     = 65535
EXPOSURE_DEFAULT = 0x0640

BAYER_BLACK_LEVEL = 16    # RAW8 pedestal  (native 10-bit 64 >> 2)
BAYER_WHITE_LEVEL = 255


# ==============================================================================
# I2C  (16-bit CCI addressing)
# ==============================================================================
class IMX219I2C:
    def __init__(self, bus, addr):
        self._addr = addr
        if not HAS_SMBUS:
            self._bus = None
            return
        self._bus = smbus2.SMBus(bus)
        print(f"[I2C] /dev/i2c-{bus}  addr=0x{addr:02X}")
        self._verify()

    def _verify(self):
        hi  = self._read8(0x0000)
        lo  = self._read8(0x0001)
        cid = (hi << 8) | lo
        ok  = "OK" if cid == 0x0219 else "UNEXPECTED - check --i2c-addr"
        print(f"[I2C] Chip ID 0x{cid:04X}  {ok}")

    def _read8(self, reg):
        if self._bus is None:
            return 0
        wb = smbus2.i2c_msg.write(self._addr, [(reg >> 8) & 0xFF, reg & 0xFF])
        rb = smbus2.i2c_msg.read(self._addr, 1)
        self._bus.i2c_rdwr(wb, rb)
        return list(rb)[0]

    def write8(self, reg, val):
        if self._bus is None:
            return
        msg = smbus2.i2c_msg.write(
            self._addr, [(reg >> 8) & 0xFF, reg & 0xFF, val & 0xFF])
        self._bus.i2c_rdwr(msg)

    def close(self):
        if self._bus is not None:
            self._bus.close()


# ==============================================================================
# Sensor controller
# ==============================================================================
class IMX219Controller:
    LINE_TIME_US = 3448.0 / 182_400_000.0 * 1e6  # ~18.90 us

    def __init__(self, i2c):
        self._i2c           = i2c
        self.ana_code       = ANA_GAIN_DEFAULT
        self.dig_code       = DIG_GAIN_DEFAULT
        self.exposure_lines = EXPOSURE_DEFAULT
        self._flush()

    @property
    def analog_gain(self):
        return 256.0 / max(1, 256 - self.ana_code)

    @analog_gain.setter
    def analog_gain(self, g):
        g = max(1.0, g)
        self.ana_code = int(np.clip(round(256.0 - 256.0 / g),
                                    ANA_GAIN_MIN, ANA_GAIN_MAX))
        self._i2c.write8(REG_ANA_GAIN, self.ana_code)

    @property
    def digital_gain(self):
        return self.dig_code / 256.0

    @digital_gain.setter
    def digital_gain(self, g):
        self.dig_code = int(np.clip(round(g * 256.0),
                                    DIG_GAIN_MIN, DIG_GAIN_MAX))
        self._i2c.write8(REG_DIG_GAIN_HI, (self.dig_code >> 8) & 0x0F)
        self._i2c.write8(REG_DIG_GAIN_LO,  self.dig_code        & 0xFF)

    @property
    def exposure_us(self):
        return self.exposure_lines * self.LINE_TIME_US

    @exposure_us.setter
    def exposure_us(self, us):
        self.exposure_lines = int(np.clip(round(us / self.LINE_TIME_US),
                                           EXPOSURE_MIN, EXPOSURE_MAX))
        self._i2c.write8(REG_EXPOSURE_HI, (self.exposure_lines >> 8) & 0xFF)
        self._i2c.write8(REG_EXPOSURE_LO,  self.exposure_lines        & 0xFF)

    def reset(self):
        self.ana_code       = ANA_GAIN_DEFAULT
        self.dig_code       = DIG_GAIN_DEFAULT
        self.exposure_lines = EXPOSURE_DEFAULT
        self._flush()

    def _flush(self):
        self._i2c.write8(REG_ANA_GAIN,    self.ana_code)
        self._i2c.write8(REG_DIG_GAIN_HI, (self.dig_code >> 8) & 0x0F)
        self._i2c.write8(REG_DIG_GAIN_LO,  self.dig_code        & 0xFF)
        self._i2c.write8(REG_EXPOSURE_HI, (self.exposure_lines >> 8) & 0xFF)
        self._i2c.write8(REG_EXPOSURE_LO,  self.exposure_lines        & 0xFF)

    def print_state(self):
        print(f"  Analog  gain : code={self.ana_code:3d}  -> {self.analog_gain:.3f}x")
        print(f"  Digital gain : code=0x{self.dig_code:04X}  -> {self.digital_gain:.3f}x")
        print(f"  Exposure     : {self.exposure_lines} lines  ~{self.exposure_us:.0f} us")


# ==============================================================================
# Image processing
# ==============================================================================
def subtract_black(raw, bl=BAYER_BLACK_LEVEL):
    out = raw.astype(np.int16) - bl
    np.clip(out, 0, 255, out=out)
    scale = 255.0 / (BAYER_WHITE_LEVEL - bl)
    return np.clip(out * scale, 0, 255).astype(np.uint8)


def debayer(raw):
    return cv2.cvtColor(raw, cv2.COLOR_BayerRG2BGR)


def gray_world_gains(bgr):
    b  = bgr[:, :, 0].astype(np.float32)
    g  = bgr[:, :, 1].astype(np.float32)
    r  = bgr[:, :, 2].astype(np.float32)
    mg = np.mean(g) + 1e-6
    return mg / (np.mean(r) + 1e-6), 1.0, mg / (np.mean(b) + 1e-6)


def apply_gains(bgr, gr, gg, gb):
    out = bgr.astype(np.float32)
    out[:, :, 2] *= gr
    out[:, :, 1] *= gg
    out[:, :, 0] *= gb
    return np.clip(out, 0, 255).astype(np.uint8)


# ==============================================================================
# AEC / AGC
# ==============================================================================
class AEController:
    def __init__(self, target=100.0, tol=8.0, max_exp_us=33000.0,
                 max_ana=8.0, max_dig=4.0, k=0.3):
        self.target  = target
        self.tol     = tol
        self.max_exp = max_exp_us
        self.max_ana = max_ana
        self.max_dig = max_dig
        self.k       = k

    @staticmethod
    def measure(bgr):
        h, w = bgr.shape[:2]
        roi  = bgr[int(h*.2):int(h*.8), int(w*.2):int(w*.8)]
        return float(np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))

    def step(self, brt, ctrl):
        err = self.target - brt
        if abs(err) < self.tol:
            return
        ratio = 1.0 + self.k * err / self.target

        if ratio > 1.0:
            new_exp = ctrl.exposure_us * ratio
            if new_exp <= self.max_exp:
                ctrl.exposure_us = new_exp; return
            ctrl.exposure_us = self.max_exp
            r2 = new_exp / self.max_exp
            new_ana = ctrl.analog_gain * r2
            if new_ana <= self.max_ana:
                ctrl.analog_gain = new_ana; return
            ctrl.analog_gain = self.max_ana
            ctrl.digital_gain = min(ctrl.digital_gain * (new_ana / self.max_ana),
                                    self.max_dig)
        else:
            new_dig = ctrl.digital_gain * ratio
            if new_dig >= 1.0:
                ctrl.digital_gain = new_dig; return
            ctrl.digital_gain = 1.0
            new_ana = ctrl.analog_gain * ratio
            if new_ana >= 1.0:
                ctrl.analog_gain = new_ana; return
            ctrl.analog_gain = 1.0
            min_exp = EXPOSURE_MIN * IMX219Controller.LINE_TIME_US
            ctrl.exposure_us = max(ctrl.exposure_us * ratio, min_exp)


# ==============================================================================
# Main loop
# ==============================================================================
def run(args):
    # Extract integer index from device string e.g. "/dev/video5" -> 5
    dev_idx = int(''.join(filter(str.isdigit, args.device)) or 0)
    cap  = TegraCapture(dev_idx, args.width, args.height)
    i2c  = IMX219I2C(args.i2c_bus, args.i2c_addr)
    ctrl = IMX219Controller(i2c)
    ae   = AEController(target=args.target_brightness)

    awb       = [1.0, 1.0, 1.0]   # [gr, gg, gb]
    alpha     = 0.05
    frame_n   = 0
    save_next = False

    print("\nKeys (window must have focus):")
    print("  q/ESC quit  |  r reset  |  s save frame  |  p print state\n")

    while True:
        raw = cap.read()
        if raw is None:
            continue
        frame_n += 1

        raw_bl = subtract_black(raw)
        bgr    = debayer(raw_bl)

        if not args.no_aec:
            brt = ae.measure(bgr)
            ae.step(brt, ctrl)

        if not args.no_awb and frame_n % 5 == 0:
            gr, gg, gb = gray_world_gains(bgr)
            gr = float(np.clip(gr, 0.5, 3.0))
            gb = float(np.clip(gb, 0.5, 3.0))
            awb[0] = alpha * gr + (1 - alpha) * awb[0]
            awb[2] = alpha * gb + (1 - alpha) * awb[2]

        bgr_wb = apply_gains(bgr, *awb)

        if save_next:
            fname = f"imx219_{frame_n:05d}.png"
            cv2.imwrite(fname, bgr_wb)
            print(f"[SAVE] {fname}")
            save_next = False

        disp = bgr_wb.copy()
        brt  = ae.measure(disp)
        osd  = [
            f"Frame {frame_n}",
            f"Brightness: {brt:.1f} / target {args.target_brightness}",
            f"AnaGain : {ctrl.analog_gain:.2f}x  (code {ctrl.ana_code})",
            f"DigGain : {ctrl.digital_gain:.2f}x  (0x{ctrl.dig_code:04X})",
            f"Exposure: {ctrl.exposure_us:.0f} us  ({ctrl.exposure_lines} lines)",
            f"AWB  R={awb[0]:.3f}  G={awb[1]:.3f}  B={awb[2]:.3f}",
        ]
        for i, t in enumerate(osd):
            cv2.putText(disp, t, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                        cv2.LINE_AA)

        dh, dw = disp.shape[:2]
        if dw > 1280:
            s = 1280 / dw
            disp = cv2.resize(disp, (int(dw * s), int(dh * s)))

        cv2.imshow("IMX219 Tuning", disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            ctrl.reset()
            awb[:] = [1.0, 1.0, 1.0]
            print("[RESET]")
        elif key == ord('s'):
            save_next = True
        elif key == ord('p'):
            ctrl.print_state()
            print(f"  AWB  R={awb[0]:.3f}  G={awb[1]:.3f}  B={awb[2]:.3f}")
            print(f"  Brightness: {ae.measure(bgr):.1f}")

    cap.release()
    cv2.destroyAllWindows()
    i2c.close()
    print("[DONE]")


# ==============================================================================
# CLI
# ==============================================================================
def _parse():
    p = argparse.ArgumentParser(description="IMX219 RAW8 tuning - Jetson/Tegra VI")
    p.add_argument("--device",   default="/dev/video5")
    p.add_argument("--i2c-bus",  type=int, default=1)
    p.add_argument("--i2c-addr", type=lambda x: int(x, 0), default=0x08,
                   help="Sensor I2C address (default 0x08 from 'cam_v1 1-0008')")
    p.add_argument("--width",    type=int, default=1920)
    p.add_argument("--height",   type=int, default=1080)
    p.add_argument("--target-brightness", type=float, default=100.0)
    p.add_argument("--no-awb",   action="store_true")
    p.add_argument("--no-aec",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
