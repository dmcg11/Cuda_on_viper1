#!/usr/bin/env python3
"""
IMX219 RAW8 Camera Tuning Script
=================================
Performs software-side AWB, AEC/AGC, and black level correction on RAW8 Bayer
frames captured from an IMX219 sensor.  Sensor parameters (analog gain,
digital gain, exposure / coarse integration time) are written back to the
sensor over I2C so that the loop closes on-hardware, not just in software.

Register map (sourced from linux/drivers/media/i2c/imx219.c and the Sony
IMX219PQH5-C datasheet):

  0x0100          MODE_SELECT      – 0x00 = standby, 0x01 = streaming
  0x0157          ANA_GAIN_GLOBAL  – 8-bit, range 0–232
                                     Gain = 256 / (256 - code)
  0x0158–0x0159   DIG_GAIN_GLOBAL  – 12-bit [11:8 in 0x0158, 7:0 in 0x0159]
                                     Range 0x0100 (1×) – 0x0FFF (~16×)
                                     Applied as gain/256
  0x015A–0x015B   COARSE_INTEG_TIME – 16-bit exposure in lines
                                     Min 4, max FRM_LENGTH - 4
  0x0160–0x0161   FRM_LENGTH_A     – 16-bit total frame lines (VTS)

Usage
-----
  python3 imx219_tuning.py [--device /dev/video0] [--i2c-bus 10]
                           [--width 3280] [--height 2464]
                           [--target-brightness 100]
                           [--no-awb] [--no-aec]

Dependencies
------------
  pip install opencv-python smbus2 numpy
"""

import argparse
import sys
import time
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Optional smbus2 import – gracefully degrade if not present
# ---------------------------------------------------------------------------
try:
    import smbus2
    HAS_SMBUS = True
except ImportError:
    HAS_SMBUS = False
    print("[WARN] smbus2 not found – I2C register writes disabled. "
          "Install with: pip install smbus2")

# ---------------------------------------------------------------------------
# IMX219 register definitions  (from torvalds/linux drivers/media/i2c/imx219.c)
# ---------------------------------------------------------------------------
IMX219_I2C_ADDR         = 0x10          # 7-bit I2C address

REG_MODE_SELECT         = 0x0100        # 0x00=standby, 0x01=streaming
REG_CHIP_ID_HI          = 0x0000        # should read 0x02
REG_CHIP_ID_LO          = 0x0001        # should read 0x19

REG_ANA_GAIN            = 0x0157        # 8-bit  analogue_gain_code_global
ANA_GAIN_MIN            = 0
ANA_GAIN_MAX            = 232
ANA_GAIN_DEFAULT        = 0

REG_DIG_GAIN_HI         = 0x0158        # bits [11:8] of 12-bit digital gain
REG_DIG_GAIN_LO         = 0x0159        # bits  [7:0]
DIG_GAIN_MIN            = 0x0100        # 1.0×
DIG_GAIN_MAX            = 0x0FFF        # ~16×
DIG_GAIN_DEFAULT        = 0x0100

REG_COARSE_INTEG_HI     = 0x015A        # exposure [15:8]
REG_COARSE_INTEG_LO     = 0x015B        # exposure [7:0]
EXPOSURE_MIN            = 4
EXPOSURE_MAX            = 65535
EXPOSURE_DEFAULT        = 0x0640        # 1600 lines

REG_FRM_LENGTH_HI       = 0x0160        # VTS [15:8]
REG_FRM_LENGTH_LO       = 0x0161        # VTS [7:0]

# Black level: RAW8 pedestal = 64>>2 = 16  (10-bit native / 4)
BAYER_BLACK_LEVEL_RAW8  = 16
BAYER_WHITE_LEVEL_RAW8  = 255

# IMX219 Bayer pattern (RGGB native)
BAYER_PATTERN           = cv2.COLOR_BayerRG2BGR        # RAW8 RGGB → BGR

# ---------------------------------------------------------------------------
# I2C helper
# ---------------------------------------------------------------------------
class IMX219I2C:
    """Thin wrapper around smbus2 for 16-bit register address writes."""

    def __init__(self, bus_num: int):
        if not HAS_SMBUS:
            self._bus = None
            return
        self._bus = smbus2.SMBus(bus_num)
        print(f"[I2C] Opened /dev/i2c-{bus_num}")

    def _reg16_to_bytes(self, reg: int):
        return [(reg >> 8) & 0xFF, reg & 0xFF]

    def write8(self, reg: int, value: int):
        """Write one byte to a 16-bit addressed register."""
        if self._bus is None:
            return
        msg = smbus2.i2c_msg.write(IMX219_I2C_ADDR,
                                   self._reg16_to_bytes(reg) + [value & 0xFF])
        self._bus.i2c_rdwr(msg)

    def read8(self, reg: int) -> int:
        """Read one byte from a 16-bit addressed register."""
        if self._bus is None:
            return 0
        write_msg = smbus2.i2c_msg.write(IMX219_I2C_ADDR,
                                         self._reg16_to_bytes(reg))
        read_msg  = smbus2.i2c_msg.read(IMX219_I2C_ADDR, 1)
        self._bus.i2c_rdwr(write_msg, read_msg)
        return list(read_msg)[0]

    def verify_chip_id(self) -> bool:
        hi = self.read8(REG_CHIP_ID_HI)
        lo = self.read8(REG_CHIP_ID_LO)
        chip_id = (hi << 8) | lo
        if chip_id == 0x0219:
            print(f"[I2C] Chip ID OK: 0x{chip_id:04X}")
            return True
        print(f"[WARN] Unexpected chip ID: 0x{chip_id:04X} (expected 0x0219)")
        return False

    def close(self):
        if self._bus is not None:
            self._bus.close()


# ---------------------------------------------------------------------------
# Sensor parameter controller
# ---------------------------------------------------------------------------
class IMX219Controller:
    """
    Manages sensor state and translates logical values (real gain, exposure µs)
    to register codes, then writes them over I2C.

    Analog gain formula (from Sony datasheet):
        Gain = 256 / (256 – ANA_GAIN_CODE)
        ANA_GAIN_CODE = 256 – 256/Gain   → clamped to [0, 232]

    Digital gain:
        Register value = desired_multiplier × 256  (12-bit, range 256–4095)

    Exposure:
        The register stores coarse integration time in *lines*.
        lines = exposure_us × pixel_clock_per_line / 1_000_000
        For default 1080p30 mode: line_length = 3448 px, pixel_rate ≈ 182.4 MHz
            → line_time ≈ 18.9 µs
        We expose this as lines directly for precision; callers convert if needed.
    """

    # Approximate line time for IMX219 in 1080p (30fps) mode
    # line_length=3448, pixel_rate=182,400,000 → ~18.9 µs/line
    LINE_TIME_US = 3448.0 / 182_400_000.0 * 1e6   # ≈ 18.90 µs

    def __init__(self, i2c: IMX219I2C):
        self._i2c = i2c
        self.ana_gain_code  = ANA_GAIN_DEFAULT      # sensor register value
        self.dig_gain_code  = DIG_GAIN_DEFAULT       # sensor register value
        self.exposure_lines = EXPOSURE_DEFAULT
        self._apply()

    # ------------------------------------------------------------------
    # Public accessors (work in physical units)
    # ------------------------------------------------------------------
    @property
    def analog_gain(self) -> float:
        """Real analog gain (1.0 – ~8.0)."""
        return 256.0 / max(1, 256 - self.ana_gain_code)

    @analog_gain.setter
    def analog_gain(self, gain: float):
        gain = max(1.0, gain)
        code = int(round(256.0 - 256.0 / gain))
        self.ana_gain_code = int(np.clip(code, ANA_GAIN_MIN, ANA_GAIN_MAX))
        self._write_analog_gain()

    @property
    def digital_gain(self) -> float:
        """Real digital gain (1.0 – ~16.0)."""
        return self.dig_gain_code / 256.0

    @digital_gain.setter
    def digital_gain(self, gain: float):
        code = int(round(gain * 256.0))
        self.dig_gain_code = int(np.clip(code, DIG_GAIN_MIN, DIG_GAIN_MAX))
        self._write_digital_gain()

    @property
    def exposure_us(self) -> float:
        return self.exposure_lines * self.LINE_TIME_US

    @exposure_us.setter
    def exposure_us(self, us: float):
        lines = int(round(us / self.LINE_TIME_US))
        self.exposure_lines = int(np.clip(lines, EXPOSURE_MIN, EXPOSURE_MAX))
        self._write_exposure()

    # ------------------------------------------------------------------
    # I2C writes
    # ------------------------------------------------------------------
    def _write_analog_gain(self):
        self._i2c.write8(REG_ANA_GAIN, self.ana_gain_code)

    def _write_digital_gain(self):
        self._i2c.write8(REG_DIG_GAIN_HI, (self.dig_gain_code >> 8) & 0x0F)
        self._i2c.write8(REG_DIG_GAIN_LO,  self.dig_gain_code        & 0xFF)

    def _write_exposure(self):
        self._i2c.write8(REG_COARSE_INTEG_HI, (self.exposure_lines >> 8) & 0xFF)
        self._i2c.write8(REG_COARSE_INTEG_LO,  self.exposure_lines        & 0xFF)

    def _apply(self):
        self._write_analog_gain()
        self._write_digital_gain()
        self._write_exposure()

    def print_state(self):
        print(f"  Analog gain  : code={self.ana_gain_code:3d}  "
              f"→ {self.analog_gain:.3f}×")
        print(f"  Digital gain : code=0x{self.dig_gain_code:04X}  "
              f"→ {self.digital_gain:.3f}×")
        print(f"  Exposure     : {self.exposure_lines} lines  "
              f"≈ {self.exposure_us:.1f} µs")


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------
def debayer(raw8: np.ndarray) -> np.ndarray:
    """Debayer a RAW8 RGGB frame into a BGR uint8 image."""
    return cv2.cvtColor(raw8, BAYER_PATTERN)


def subtract_black_level(raw8: np.ndarray,
                          black: int = BAYER_BLACK_LEVEL_RAW8) -> np.ndarray:
    """
    Subtract the sensor pedestal and rescale to [0, 255].
    The IMX219 RAW8 black level is nominally 16 (64>>2).
    """
    clamped = raw8.astype(np.int16) - black
    np.clip(clamped, 0, 255, out=clamped)
    # Rescale so that white_level - black maps to 255
    scale = 255.0 / (BAYER_WHITE_LEVEL_RAW8 - black)
    return np.clip(clamped * scale, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# AWB – Gray-World
# ---------------------------------------------------------------------------
def gray_world_awb(bgr: np.ndarray):
    """
    Gray-world AWB.  Returns (gain_r, gain_g, gain_b) where gains normalise
    each channel to the global mean.  gain_g is fixed at 1.0; r and b are
    adjusted relative to green.
    """
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)

    mean_b = np.mean(b) + 1e-6
    mean_g = np.mean(g) + 1e-6
    mean_r = np.mean(r) + 1e-6

    gain_b = mean_g / mean_b
    gain_g = 1.0
    gain_r = mean_g / mean_r

    return gain_r, gain_g, gain_b


def apply_awb_gains(bgr: np.ndarray, gr: float, gg: float, gb: float
                    ) -> np.ndarray:
    """Apply per-channel gains; clip to [0, 255]."""
    out = bgr.astype(np.float32)
    out[:, :, 0] *= gb
    out[:, :, 1] *= gg
    out[:, :, 2] *= gr
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# AEC/AGC – simple proportional controller
# ---------------------------------------------------------------------------
class AEController:
    """
    Simple proportional AE controller.

    Strategy:
      1. First fill exposure up to a soft ceiling (max_exposure_us).
      2. Then ramp analog gain [1×, max_analog_gain].
      3. Finally, if still too dark, apply digital gain [1×, max_digital_gain].
      4. When brightening scene: reverse order (reduce digital → reduce analog
         → reduce exposure).
    """

    def __init__(self,
                 target_brightness: float = 100.0,  # out of 255
                 tolerance: float = 8.0,
                 max_exposure_us: float = 33_000.0,   # ~1 frame @ 30fps
                 max_analog_gain: float = 8.0,
                 max_digital_gain: float = 4.0,
                 proportional_k: float = 0.3):
        self.target      = target_brightness
        self.tolerance   = tolerance
        self.max_exp_us  = max_exposure_us
        self.max_ana     = max_analog_gain
        self.max_dig     = max_digital_gain
        self.k           = proportional_k

    def measure_brightness(self, bgr: np.ndarray) -> float:
        """Average luminance of the central 60% of the frame."""
        h, w = bgr.shape[:2]
        y0, y1 = int(h * 0.2), int(h * 0.8)
        x0, x1 = int(w * 0.2), int(w * 0.8)
        roi = bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def update(self, brightness: float, ctrl: IMX219Controller):
        """Adjust sensor registers to drive brightness toward target."""
        error = self.target - brightness
        if abs(error) < self.tolerance:
            return   # within dead-band, no action

        ratio = 1.0 + self.k * error / self.target   # multiplicative step

        if ratio > 1.0:   # need more light
            # 1) increase exposure first
            new_exp = ctrl.exposure_us * ratio
            if new_exp <= self.max_exp_us:
                ctrl.exposure_us = new_exp
                return
            ctrl.exposure_us = self.max_exp_us
            leftover = new_exp / self.max_exp_us
            # 2) then analog gain
            new_ana = ctrl.analog_gain * leftover
            if new_ana <= self.max_ana:
                ctrl.analog_gain = new_ana
                return
            ctrl.analog_gain = self.max_ana
            leftover2 = new_ana / self.max_ana
            # 3) finally digital gain
            ctrl.digital_gain = min(ctrl.digital_gain * leftover2,
                                    self.max_dig)
        else:             # reduce light
            # reverse: reduce digital first
            new_dig = ctrl.digital_gain * ratio
            if new_dig >= 1.0:
                ctrl.digital_gain = new_dig
                return
            ctrl.digital_gain = 1.0
            leftover = ctrl.digital_gain * ratio   # extra reduction needed
            # then analog
            new_ana = ctrl.analog_gain * ratio
            if new_ana >= 1.0:
                ctrl.analog_gain = new_ana
                return
            ctrl.analog_gain = 1.0
            # finally shorten exposure
            ctrl.exposure_us = max(ctrl.exposure_us * ratio,
                                   EXPOSURE_MIN * IMX219Controller.LINE_TIME_US)


# ---------------------------------------------------------------------------
# Main capture + tuning loop
# ---------------------------------------------------------------------------
def run(args):
    # -----------------------------------------------------------------------
    # Open V4L2 device
    # -----------------------------------------------------------------------
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {args.device}")
        sys.exit(1)

    # Request RAW / uncompressed format; the sensor is already streaming RAW8
    # so we just set the resolution.  If your pipeline delivers already-debayered
    # frames via the ISP you can skip the manual debayer step below.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # Disable any built-in auto-exposure / auto-gain that OpenCV or V4L2 may
    # have enabled by default (exposure_auto = 1 means manual on V4L2)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CAP] Opened {args.device} at {actual_w}×{actual_h}")

    # -----------------------------------------------------------------------
    # I2C + sensor controller
    # -----------------------------------------------------------------------
    i2c  = IMX219I2C(args.i2c_bus)
    if HAS_SMBUS:
        i2c.verify_chip_id()
    ctrl = IMX219Controller(i2c)

    ae   = AEController(target_brightness=args.target_brightness)

    # Persistent AWB gains (smoothed over frames)
    awb_gr, awb_gg, awb_gb = 1.0, 1.0, 1.0
    awb_alpha = 0.05     # EMA smoothing – lower = more stable but slower

    print("\nControls while running:")
    print("  q / ESC  – quit")
    print("  r        – reset gains to default")
    print("  s        – save current frame as PNG")
    print("  p        – print current sensor state\n")

    frame_count = 0
    save_next   = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame")
            time.sleep(0.01)
            continue

        frame_count += 1

        # -------------------------------------------------------------------
        # If the driver hands us a single-channel (RAW8 Bayer) frame we need
        # to process it ourselves.  If it already arrives as BGR (ISP demosaiced)
        # we skip debayer.
        # -------------------------------------------------------------------
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            # True RAW8 path
            raw = frame if frame.ndim == 2 else frame[:, :, 0]
            raw = subtract_black_level(raw, BAYER_BLACK_LEVEL_RAW8)
            bgr = debayer(raw)
            is_raw = True
        else:
            # Demosaiced path (ISP active, or camera already outputs BGR)
            bgr = frame
            is_raw = False

        # -------------------------------------------------------------------
        # AEC/AGC  (every frame)
        # -------------------------------------------------------------------
        if not args.no_aec:
            brightness = ae.measure_brightness(bgr)
            ae.update(brightness, ctrl)

        # -------------------------------------------------------------------
        # AWB  (every 5 frames to reduce flicker)
        # -------------------------------------------------------------------
        if not args.no_awb and frame_count % 5 == 0:
            gr, gg, gb = gray_world_awb(bgr)
            # Clamp gains to sane range [0.5, 3.0]
            gr = float(np.clip(gr, 0.5, 3.0))
            gb = float(np.clip(gb, 0.5, 3.0))
            # EMA smoothing
            awb_gr = awb_alpha * gr + (1 - awb_alpha) * awb_gr
            awb_gg = awb_alpha * gg + (1 - awb_alpha) * awb_gg
            awb_gb = awb_alpha * gb + (1 - awb_alpha) * awb_gb

        bgr_wb = apply_awb_gains(bgr, awb_gr, awb_gg, awb_gb)

        # -------------------------------------------------------------------
        # OSD overlay
        # -------------------------------------------------------------------
        disp = bgr_wb.copy()
        brt  = ae.measure_brightness(disp)
        lines = [
            f"Frame: {frame_count}",
            f"Brightness: {brt:.1f} / target {args.target_brightness}",
            f"AnaGain: {ctrl.analog_gain:.2f}x (code {ctrl.ana_gain_code})",
            f"DigGain: {ctrl.digital_gain:.2f}x",
            f"Exposure: {ctrl.exposure_us:.0f} us ({ctrl.exposure_lines} lines)",
            f"AWB  R={awb_gr:.3f} G={awb_gg:.3f} B={awb_gb:.3f}",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(disp, txt, (10, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)

        # Resize for display if very large
        disp_h, disp_w = disp.shape[:2]
        max_display = 1280
        if disp_w > max_display:
            scale = max_display / disp_w
            disp = cv2.resize(disp,
                              (int(disp_w * scale), int(disp_h * scale)))

        cv2.imshow("IMX219 Tuning", disp)

        # -------------------------------------------------------------------
        # Save
        # -------------------------------------------------------------------
        if save_next:
            fname = f"imx219_frame_{frame_count:05d}.png"
            cv2.imwrite(fname, bgr_wb)
            print(f"[SAVE] {fname}")
            save_next = False

        # -------------------------------------------------------------------
        # Key handling
        # -------------------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):      # q or ESC
            break
        elif key == ord('r'):
            ctrl.ana_gain_code  = ANA_GAIN_DEFAULT
            ctrl.dig_gain_code  = DIG_GAIN_DEFAULT
            ctrl.exposure_lines = EXPOSURE_DEFAULT
            ctrl._apply()
            awb_gr = awb_gg = awb_gb = 1.0
            print("[RESET] Gains and exposure reset to defaults")
        elif key == ord('s'):
            save_next = True
        elif key == ord('p'):
            ctrl.print_state()
            print(f"  AWB gains    : R={awb_gr:.3f}  G={awb_gg:.3f}  "
                  f"B={awb_gb:.3f}")
            print(f"  Brightness   : {ae.measure_brightness(bgr):.1f}")

    cap.release()
    cv2.destroyAllWindows()
    i2c.close()
    print("[DONE]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="IMX219 RAW8 camera tuning – AWB / AEC / Black-level")
    p.add_argument("--device",  default="/dev/video0",
                   help="V4L2 device node (default: /dev/video0)")
    p.add_argument("--i2c-bus", type=int, default=10,
                   help="I2C bus number for sensor register access "
                        "(default: 10, adjust for your platform)")
    p.add_argument("--width",   type=int, default=1920,
                   help="Capture width  (default: 1920)")
    p.add_argument("--height",  type=int, default=1080,
                   help="Capture height (default: 1080)")
    p.add_argument("--target-brightness", type=float, default=100.0,
                   help="AE target mean luminance 0–255 (default: 100)")
    p.add_argument("--no-awb",  action="store_true",
                   help="Disable automatic white balance")
    p.add_argument("--no-aec",  action="store_true",
                   help="Disable automatic exposure / gain control")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)

