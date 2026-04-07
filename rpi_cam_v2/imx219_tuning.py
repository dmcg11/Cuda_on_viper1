#!/usr/bin/env python3
"""
IMX219 RAW8 Camera Tuning Script  (Jetson / Tegra VI edition)
=============================================================
Full software ISP pipeline:
  1. Black level subtraction
  2. Debayer (RGGB bilinear)
  3. AWB gains (linear space)
  4. 3x3 Color Correction Matrix (linear space, from RPi libcamera IMX219 tuning)
  5. Gamma encoding (linear -> sRGB display)
  6. Saturation boost (HSV)
  7. Unsharp mask sharpening
  8. Noise reduction (optional)

All pipeline stages are exposed as trackbar sliders in a separate controls window.

Register map (linux/drivers/media/i2c/imx219.c):
  0x0157  ANA_GAIN_GLOBAL_A   8-bit   0-232
  0x0158  DIG_GAIN_GLOBAL [11:8]
  0x0159  DIG_GAIN_GLOBAL [7:0]  12-bit  0x0100-0x0FFF
  0x015A  COARSE_INTEG_TIME [15:8]
  0x015B  COARSE_INTEG_TIME [7:0]  16-bit  4-65535 lines

Usage:  python3 imx219_tuning.py
Keys (camera window): q/ESC quit | s save snapshot.jpg | r reset | p print state
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

try:
    import smbus2
    HAS_SMBUS = True
except ImportError:
    HAS_SMBUS = False
    print("[WARN] smbus2 not found - I2C writes disabled.")

# ==============================================================================
# Camera capture
# ==============================================================================
class TegraCapture:
    def __init__(self, device_index, width, height, fps=30):
        self._cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open /dev/video{device_index}")
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'RGGB'))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS,          fps)
        self._cap.set(cv2.CAP_PROP_CONVERT_RGB,  0)
        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAP] /dev/video{device_index}  {self.width}x{self.height}  RAW8 RGGB")

    def read(self):
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame.reshape((self.height, self.width))

    def release(self):
        self._cap.release()


# ==============================================================================
# IMX219 I2C
# ==============================================================================
REG_ANA_GAIN     = 0x0157
ANA_GAIN_MIN, ANA_GAIN_MAX, ANA_GAIN_DEFAULT = 0, 232, 0
REG_DIG_GAIN_HI  = 0x0158
REG_DIG_GAIN_LO  = 0x0159
DIG_GAIN_MIN, DIG_GAIN_MAX, DIG_GAIN_DEFAULT = 0x0100, 0x0FFF, 0x0100
REG_EXPOSURE_HI  = 0x015A
REG_EXPOSURE_LO  = 0x015B
EXPOSURE_MIN, EXPOSURE_MAX, EXPOSURE_DEFAULT = 4, 65535, 0x0A00
BAYER_BLACK_LEVEL = 16
BAYER_WHITE_LEVEL = 255

class IMX219I2C:
    def __init__(self, bus, addr):
        self._addr = addr
        if not HAS_SMBUS:
            self._bus = None; return
        self._bus = smbus2.SMBus(bus)
        print(f"[I2C] /dev/i2c-{bus}  addr=0x{addr:02X}")
        try:
            hi = self._read8(0x0000); lo = self._read8(0x0001)
            cid = (hi << 8) | lo
            print(f"[I2C] Chip ID 0x{cid:04X}  {'OK' if cid==0x0219 else 'UNEXPECTED'}")
        except OSError as e:
            print(f"[I2C] Chip ID read failed ({e}) - continuing")

    def _read8(self, reg):
        wb = smbus2.i2c_msg.write(self._addr, [(reg>>8)&0xFF, reg&0xFF])
        rb = smbus2.i2c_msg.read(self._addr, 1)
        self._bus.i2c_rdwr(wb, rb)
        return list(rb)[0]

    def write8(self, reg, val):
        if self._bus is None: return
        try:
            msg = smbus2.i2c_msg.write(self._addr,
                                        [(reg>>8)&0xFF, reg&0xFF, val&0xFF])
            self._bus.i2c_rdwr(msg)
        except OSError as e:
            print(f"[I2C] Write 0x{reg:04X}=0x{val:02X} failed: {e}")

    def close(self):
        if self._bus: self._bus.close()


class IMX219Controller:
    LINE_TIME_US = 3448.0 / 182_400_000.0 * 1e6

    def __init__(self, i2c):
        self._i2c = i2c
        self.ana_code = ANA_GAIN_DEFAULT
        self.dig_code = DIG_GAIN_DEFAULT
        self.exposure_lines = EXPOSURE_DEFAULT
        self._flush()

    @property
    def analog_gain(self):
        return 256.0 / max(1, 256 - self.ana_code)

    @analog_gain.setter
    def analog_gain(self, g):
        self.ana_code = int(np.clip(round(256.0 - 256.0/max(1.0,g)),
                                    ANA_GAIN_MIN, ANA_GAIN_MAX))
        self._i2c.write8(REG_ANA_GAIN, self.ana_code)

    @property
    def digital_gain(self):
        return self.dig_code / 256.0

    @digital_gain.setter
    def digital_gain(self, g):
        self.dig_code = int(np.clip(round(g*256.0), DIG_GAIN_MIN, DIG_GAIN_MAX))
        self._i2c.write8(REG_DIG_GAIN_HI, (self.dig_code>>8)&0x0F)
        self._i2c.write8(REG_DIG_GAIN_LO,  self.dig_code    &0xFF)

    @property
    def exposure_us(self):
        return self.exposure_lines * self.LINE_TIME_US

    @exposure_us.setter
    def exposure_us(self, us):
        self.exposure_lines = int(np.clip(round(us/self.LINE_TIME_US),
                                           EXPOSURE_MIN, EXPOSURE_MAX))
        self._i2c.write8(REG_EXPOSURE_HI, (self.exposure_lines>>8)&0xFF)
        self._i2c.write8(REG_EXPOSURE_LO,  self.exposure_lines    &0xFF)

    def reset(self):
        self.ana_code = ANA_GAIN_DEFAULT
        self.dig_code = DIG_GAIN_DEFAULT
        self.exposure_lines = EXPOSURE_DEFAULT
        self._flush()

    def _flush(self):
        self._i2c.write8(REG_ANA_GAIN,    self.ana_code)
        self._i2c.write8(REG_DIG_GAIN_HI, (self.dig_code>>8)&0x0F)
        self._i2c.write8(REG_DIG_GAIN_LO,  self.dig_code    &0xFF)
        self._i2c.write8(REG_EXPOSURE_HI, (self.exposure_lines>>8)&0xFF)
        self._i2c.write8(REG_EXPOSURE_LO,  self.exposure_lines    &0xFF)

    def print_state(self):
        print(f"  Analog  : code={self.ana_code}  -> {self.analog_gain:.3f}x")
        print(f"  Digital : code=0x{self.dig_code:04X}  -> {self.digital_gain:.3f}x")
        print(f"  Exposure: {self.exposure_lines} lines  ~{self.exposure_us:.0f} us")


# ==============================================================================
# AEC/AGC
# ==============================================================================
class AEController:
    def __init__(self, target=120.0, tol=8.0, max_exp_us=33000.0,
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
        if abs(err) < self.tol: return
        ratio = 1.0 + self.k * err / self.target
        if ratio > 1.0:
            new_exp = ctrl.exposure_us * ratio
            if new_exp <= self.max_exp: ctrl.exposure_us = new_exp; return
            ctrl.exposure_us = self.max_exp
            r2 = new_exp / self.max_exp
            new_ana = ctrl.analog_gain * r2
            if new_ana <= self.max_ana: ctrl.analog_gain = new_ana; return
            ctrl.analog_gain = self.max_ana
            ctrl.digital_gain = min(ctrl.digital_gain * (new_ana/self.max_ana),
                                    self.max_dig)
        else:
            new_dig = ctrl.digital_gain * ratio
            if new_dig >= 1.0: ctrl.digital_gain = new_dig; return
            ctrl.digital_gain = 1.0
            new_ana = ctrl.analog_gain * ratio
            if new_ana >= 1.0: ctrl.analog_gain = new_ana; return
            ctrl.analog_gain = 1.0
            ctrl.exposure_us = max(ctrl.exposure_us * ratio,
                                   EXPOSURE_MIN * IMX219Controller.LINE_TIME_US)


# ==============================================================================
# ISP Pipeline
# ==============================================================================

# ── Color Correction Matrix ───────────────────────────────────────────────────
# From Raspberry Pi libcamera IMX219 tuning file (D65 daylight).
# Applied in LINEAR space (before gamma). Rows = output BGR, Cols = input BGR.
# This maps the sensor's native Bayer color space to sRGB.
CCM_DEFAULT = np.array([
    [ 1.8004, -0.5760, -0.2244],   # output B
    [-0.3566,  1.6925, -0.3359],   # output G
    [-0.0950, -0.6687,  1.7637],   # output R
], dtype=np.float32)

# ── Gamma LUT ─────────────────────────────────────────────────────────────────
def build_gamma_lut(gamma: float) -> np.ndarray:
    """Build a uint8->uint8 LUT for power-law gamma correction."""
    x = np.arange(256, dtype=np.float32) / 255.0
    y = np.power(np.clip(x, 1e-6, 1.0), 1.0 / gamma)
    return (y * 255.0).astype(np.uint8)


# ── AWB neutral-pixel estimate ────────────────────────────────────────────────
def gray_world_gains(bgr: np.ndarray):
    """Neutral-pixel AWB in linear space."""
    b = bgr[:,:,0].astype(np.float32)
    g = bgr[:,:,1].astype(np.float32)
    r = bgr[:,:,2].astype(np.float32)
    gray = (b + g + r) / 3.0
    diff = np.maximum(np.maximum(np.abs(r-gray), np.abs(g-gray)), np.abs(b-gray))
    mask = (diff < 20) & (gray > 20) & (gray < 220)
    if mask.sum() > 500:
        rm = float(r[mask].mean()) + 1e-6
        gm = float(g[mask].mean()) + 1e-6
        bm = float(b[mask].mean()) + 1e-6
    else:
        rm = float(r.mean()) + 1e-6
        gm = float(g.mean()) + 1e-6
        bm = float(b.mean()) + 1e-6
    gr = float(np.clip(gm/rm, 0.5, 4.0))
    gb = float(np.clip(gm/bm, 0.5, 4.0))
    return gr, 1.0, gb


# ── Full ISP pipeline ─────────────────────────────────────────────────────────
def process_frame(raw: np.ndarray, params: dict) -> np.ndarray:
    """
    Full software ISP pipeline.
    params keys:
      black_level  int   [0,64]
      awb          list  [gr, gg, gb]  linear gains
      ccm_strength float [0,1]  blend between identity and CCM_DEFAULT
      gamma        float [0.5, 3.0]
      saturation   float [0, 3.0]  1.0 = unchanged
      sharpness    float [0, 3.0]  0 = off
      denoise      bool
    """
    bl      = params['black_level']
    awb     = params['awb']
    ccm_s   = params['ccm_strength']
    gamma   = params['gamma']
    sat     = params['saturation']
    sharp   = params['sharpness']
    denoise = params['denoise']

    # ── 1. Black level subtraction & normalise to [0,1] ───────────────────
    linear = raw.astype(np.float32) - bl
    np.clip(linear, 0, 255, out=linear)
    linear /= (255.0 - bl)

    # ── 2. Debayer (RGGB -> BGR) ──────────────────────────────────────────
    raw8 = (linear * 255).astype(np.uint8)
    bgr  = cv2.cvtColor(raw8, cv2.COLOR_BayerRG2BGR).astype(np.float32) / 255.0

    # ── 3. AWB gains (linear space) ───────────────────────────────────────
    bgr[:,:,0] *= awb[2]   # B
    bgr[:,:,1] *= awb[1]   # G
    bgr[:,:,2] *= awb[0]   # R
    np.clip(bgr, 0, 1, out=bgr)

    # ── 4. Color Correction Matrix (linear space) ─────────────────────────
    if ccm_s > 0:
        ccm = ccm_s * CCM_DEFAULT + (1.0 - ccm_s) * np.eye(3, dtype=np.float32)
        # Reshape to (H*W, 3) for matrix multiply then back
        flat  = bgr.reshape(-1, 3)
        flat  = flat @ ccm.T          # (N,3) @ (3,3) -> (N,3)
        bgr   = np.clip(flat, 0, 1).reshape(bgr.shape)

    # ── 5. Gamma encoding (linear -> display) ────────────────────────────
    lut = build_gamma_lut(gamma)
    bgr8 = (np.clip(bgr, 0, 1) * 255).astype(np.uint8)
    bgr8 = cv2.LUT(bgr8, lut)

    # ── 6. Saturation (HSV) ───────────────────────────────────────────────
    if abs(sat - 1.0) > 0.01:
        hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * sat, 0, 255)
        bgr8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ── 7. Noise reduction ────────────────────────────────────────────────
    if denoise:
        bgr8 = cv2.fastNlMeansDenoisingColored(bgr8, None, 5, 5, 7, 21)

    # ── 8. Unsharp mask sharpening ────────────────────────────────────────
    if sharp > 0:
        blur   = cv2.GaussianBlur(bgr8, (0, 0), 2.0)
        bgr8   = cv2.addWeighted(bgr8, 1.0 + sharp, blur, -sharp, 0)

    return bgr8


# ==============================================================================
# Controls GUI
# ==============================================================================
CTRL_WIN = "IMX219 Controls"

# Slider scale factors (all sliders are integers)
# gain   x100,  gamma x100,  sat x100,  sharp x100

def create_controls():
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WIN, 520, 580)

    cv2.createTrackbar("── AWB ──────────────",  CTRL_WIN, 0, 0, lambda x: None)
    cv2.createTrackbar("Auto WB  (1=on)",        CTRL_WIN, 1, 1, lambda x: None)
    cv2.createTrackbar("AWB R x100",             CTRL_WIN, 110, 400, lambda x: None)
    cv2.createTrackbar("AWB G x100",             CTRL_WIN, 100, 400, lambda x: None)
    cv2.createTrackbar("AWB B x100",             CTRL_WIN,  85, 400, lambda x: None)

    cv2.createTrackbar("── Pipeline ─────────", CTRL_WIN, 0, 0, lambda x: None)
    cv2.createTrackbar("Black Level",            CTRL_WIN, BAYER_BLACK_LEVEL, 64, lambda x: None)
    cv2.createTrackbar("CCM Strength x100",      CTRL_WIN, 100, 100, lambda x: None)
    cv2.createTrackbar("Gamma x100",             CTRL_WIN, 220, 400, lambda x: None)
    cv2.createTrackbar("Saturation x100",        CTRL_WIN, 130, 300, lambda x: None)
    cv2.createTrackbar("Sharpness x100",         CTRL_WIN,  50, 200, lambda x: None)
    cv2.createTrackbar("Denoise (1=on)",         CTRL_WIN,   0,   1, lambda x: None)

    cv2.createTrackbar("── AEC ──────────────", CTRL_WIN, 0, 0, lambda x: None)
    cv2.createTrackbar("Auto Exp (1=on)",        CTRL_WIN, 1, 1, lambda x: None)
    cv2.createTrackbar("AE Target",              CTRL_WIN, 120, 255, lambda x: None)


def get_controls():
    def tb(name): return cv2.getTrackbarPos(name, CTRL_WIN)
    return {
        'auto_wb':      tb("Auto WB  (1=on)") == 1,
        'man_r':        max(tb("AWB R x100"), 1) / 100.0,
        'man_g':        max(tb("AWB G x100"), 1) / 100.0,
        'man_b':        max(tb("AWB B x100"), 1) / 100.0,
        'black_level':  tb("Black Level"),
        'ccm_strength': tb("CCM Strength x100") / 100.0,
        'gamma':        max(tb("Gamma x100"), 10) / 100.0,
        'saturation':   max(tb("Saturation x100"), 1) / 100.0,
        'sharpness':    tb("Sharpness x100") / 100.0,
        'denoise':      tb("Denoise (1=on)") == 1,
        'auto_aec':     tb("Auto Exp (1=on)") == 1,
        'ae_target':    max(tb("AE Target"), 1),
    }


def sync_awb_sliders(gr, gg, gb):
    cv2.setTrackbarPos("AWB R x100", CTRL_WIN, int(np.clip(gr*100, 1, 400)))
    cv2.setTrackbarPos("AWB G x100", CTRL_WIN, int(np.clip(gg*100, 1, 400)))
    cv2.setTrackbarPos("AWB B x100", CTRL_WIN, int(np.clip(gb*100, 1, 400)))


# ==============================================================================
# Main loop
# ==============================================================================
def run(args):
    dev_idx = int(''.join(filter(str.isdigit, args.device)) or 0)
    cap  = TegraCapture(dev_idx, args.width, args.height)
    i2c  = IMX219I2C(args.i2c_bus, args.i2c_addr)
    ctrl = IMX219Controller(i2c)
    ae   = AEController(target=120.0)

    create_controls()

    awb       = [1.10, 1.0, 0.85]   # [gr, gg, gb] initial estimate
    alpha     = 0.05
    frame_n   = 0
    save_next = False

    print("\nKeys (camera window must have focus):")
    print("  q/ESC quit  |  s save snapshot.jpg  |  r reset  |  p print state\n")

    while True:
        raw = cap.read()
        if raw is None:
            continue
        frame_n += 1

        c = get_controls()
        ae.target = c['ae_target']

        # ── AEC/AGC ──────────────────────────────────────────────────────
        if c['auto_aec']:
            # Quick brightness estimate on raw (before full pipeline)
            raw_bl = raw.astype(np.int16) - c['black_level']
            np.clip(raw_bl, 0, 255, out=raw_bl)
            rough_bgr = cv2.cvtColor(raw_bl.astype(np.uint8),
                                      cv2.COLOR_BayerRG2BGR)
            brt = ae.measure(rough_bgr)
            ae.step(brt, ctrl)

        # ── AWB ───────────────────────────────────────────────────────────
        if c['auto_wb'] and frame_n % 5 == 0:
            raw_bl = raw.astype(np.int16) - c['black_level']
            np.clip(raw_bl, 0, 255, out=raw_bl)
            rough_bgr = cv2.cvtColor(raw_bl.astype(np.uint8),
                                      cv2.COLOR_BayerRG2BGR)
            gr, gg, gb = gray_world_gains(rough_bgr)
            awb[0] = alpha * gr + (1-alpha) * awb[0]
            awb[1] = 1.0
            awb[2] = alpha * gb + (1-alpha) * awb[2]
            sync_awb_sliders(*awb)
        elif not c['auto_wb']:
            awb = [c['man_r'], c['man_g'], c['man_b']]

        # ── Full ISP pipeline ─────────────────────────────────────────────
        params = {
            'black_level': c['black_level'],
            'awb':         awb,
            'ccm_strength':c['ccm_strength'],
            'gamma':       c['gamma'],
            'saturation':  c['saturation'],
            'sharpness':   c['sharpness'],
            'denoise':     c['denoise'],
        }
        bgr_out = process_frame(raw, params)

        # ── Save ─────────────────────────────────────────────────────────
        if save_next:
            cv2.imwrite("snapshot.jpg", bgr_out,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            print("[SAVE] snapshot.jpg")
            save_next = False

        # ── OSD ──────────────────────────────────────────────────────────
        disp = bgr_out.copy()
        brt  = ae.measure(disp)
        wb_mode = "AUTO" if c['auto_wb'] else "MAN"
        osd = [
            f"Frame {frame_n}",
            f"Brightness: {brt:.0f} / target {c['ae_target']}",
            f"AnaGain : {ctrl.analog_gain:.2f}x (code {ctrl.ana_code})",
            f"DigGain : {ctrl.digital_gain:.2f}x",
            f"Exposure: {ctrl.exposure_us:.0f} us ({ctrl.exposure_lines} lines)",
            f"AWB[{wb_mode}] R={awb[0]:.2f} G={awb[1]:.2f} B={awb[2]:.2f}",
            f"CCM:{c['ccm_strength']:.2f}  Gamma:{c['gamma']:.2f}  "
            f"Sat:{c['saturation']:.2f}  Sharp:{c['sharpness']:.2f}",
        ]
        for i, t in enumerate(osd):
            cv2.putText(disp, t, (10, 22 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1,
                        cv2.LINE_AA)

        dh, dw = disp.shape[:2]
        if dw > 1280:
            s = 1280 / dw
            disp = cv2.resize(disp, (int(dw*s), int(dh*s)))

        cv2.imshow("IMX219 Tuning", disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            save_next = True
        elif key == ord('r'):
            ctrl.reset()
            awb[:] = [1.10, 1.0, 0.85]
            sync_awb_sliders(*awb)
            print("[RESET]")
        elif key == ord('p'):
            ctrl.print_state()
            print(f"  AWB [{wb_mode}]: R={awb[0]:.3f} G={awb[1]:.3f} B={awb[2]:.3f}")
            print(f"  CCM:{c['ccm_strength']:.2f}  Gamma:{c['gamma']:.2f}  "
                  f"Sat:{c['saturation']:.2f}  Sharp:{c['sharpness']:.2f}")
            print(f"  Brightness: {brt:.1f}")

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
    p.add_argument("--i2c-addr", type=lambda x: int(x,0), default=0x10,
                   help="Sensor I2C address (default 0x10)")
    p.add_argument("--width",    type=int, default=1920)
    p.add_argument("--height",   type=int, default=1080)
    return p.parse_args()

if __name__ == "__main__":
    run(_parse())
