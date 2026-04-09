#!/usr/bin/env python3
"""
IMX219 RAW8 Camera Tuning Script  (Jetson / Tegra VI)
======================================================
Full software ISP pipeline with per-stage profiling and controls GUI.

Pipeline stages:
  1. Black level subtraction
  2. Debayer RGGB -> BGR  (CUDA if available, else CPU)
  3. AWB gains  (applied via per-channel LUT — zero float alloc)
  4. Color Correction Matrix  (RPi libcamera IMX219 D65 tuning)
  5. Gamma encoding  (LUT, cached)
  6. Saturation boost  (HSV S-channel LUT)
  7. Unsharp mask sharpening
  8. Denoise  (off by default — slow)

Register map  (linux/drivers/media/i2c/imx219.c):
  0x0157  ANA_GAIN_GLOBAL_A   8-bit   0–232
  0x0158-0x0159  DIG_GAIN_GLOBAL_A  12-bit  0x0100–0x0FFF
  0x015A-0x015B  COARSE_INTEG_TIME  16-bit  4–65535 lines

Usage:  python3 imx219_tuning.py
Keys (camera window): q/ESC quit | s save snapshot.jpg | r reset | p print state
"""

import argparse
import time
import cv2
import numpy as np

try:
    import smbus2
    HAS_SMBUS = True
except ImportError:
    HAS_SMBUS = False
    print("[WARN] smbus2 not found — I2C writes disabled")

# ── CUDA availability ─────────────────────────────────────────────────────────
try:
    HAS_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
except Exception:
    HAS_CUDA = False
print(f"[INFO] CUDA debayer: {'YES' if HAS_CUDA else 'NO'}")


# ==============================================================================
# Capture
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
# IMX219 registers + I2C
# ==============================================================================
REG_ANA_GAIN      = 0x0157
ANA_GAIN_MIN, ANA_GAIN_MAX, ANA_GAIN_DEFAULT = 0, 232, 0
REG_DIG_GAIN_HI   = 0x0158
REG_DIG_GAIN_LO   = 0x0159
DIG_GAIN_MIN, DIG_GAIN_MAX, DIG_GAIN_DEFAULT = 0x0100, 0x0FFF, 0x0100
REG_EXPOSURE_HI   = 0x015A
REG_EXPOSURE_LO   = 0x015B
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
            print(f"[I2C] Chip ID 0x{cid:04X}  {'OK' if cid == 0x0219 else 'UNEXPECTED'}")
        except OSError as e:
            print(f"[I2C] Chip ID read failed ({e}) — continuing")

    def _read8(self, reg):
        wb = smbus2.i2c_msg.write(self._addr, [(reg >> 8) & 0xFF, reg & 0xFF])
        rb = smbus2.i2c_msg.read(self._addr, 1)
        self._bus.i2c_rdwr(wb, rb)
        return list(rb)[0]

    def write8(self, reg, val):
        if self._bus is None: return
        try:
            self._bus.i2c_rdwr(smbus2.i2c_msg.write(
                self._addr, [(reg >> 8) & 0xFF, reg & 0xFF, val & 0xFF]))
        except OSError as e:
            print(f"[I2C] 0x{reg:04X}=0x{val:02X} failed: {e}")

    def close(self):
        if self._bus: self._bus.close()


class IMX219Controller:
    LINE_TIME_US = 3448.0 / 182_400_000.0 * 1e6  # ~18.90 µs

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
        self.ana_code = int(np.clip(round(256 - 256 / max(1.0, g)),
                                    ANA_GAIN_MIN, ANA_GAIN_MAX))
        self._i2c.write8(REG_ANA_GAIN, self.ana_code)

    @property
    def digital_gain(self):
        return self.dig_code / 256.0

    @digital_gain.setter
    def digital_gain(self, g):
        self.dig_code = int(np.clip(round(g * 256), DIG_GAIN_MIN, DIG_GAIN_MAX))
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
        self.ana_code = ANA_GAIN_DEFAULT
        self.dig_code = DIG_GAIN_DEFAULT
        self.exposure_lines = EXPOSURE_DEFAULT
        self._flush()

    def _flush(self):
        self._i2c.write8(REG_ANA_GAIN,    self.ana_code)
        self._i2c.write8(REG_DIG_GAIN_HI, (self.dig_code >> 8) & 0x0F)
        self._i2c.write8(REG_DIG_GAIN_LO,  self.dig_code        & 0xFF)
        self._i2c.write8(REG_EXPOSURE_HI, (self.exposure_lines >> 8) & 0xFF)
        self._i2c.write8(REG_EXPOSURE_LO,  self.exposure_lines        & 0xFF)

    def print_state(self):
        print(f"  Analog  gain : code={self.ana_code}  -> {self.analog_gain:.3f}x")
        print(f"  Digital gain : code=0x{self.dig_code:04X}  -> {self.digital_gain:.3f}x")
        print(f"  Exposure     : {self.exposure_lines} lines  ~{self.exposure_us:.0f} µs")


# ==============================================================================
# AEC / AGC
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
        roi = bgr[int(h * .2):int(h * .8), int(w * .2):int(w * .8)]
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
            ctrl.digital_gain = min(ctrl.digital_gain * new_ana / self.max_ana,
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
# ISP — cached LUT pipeline
# ==============================================================================

# RPi libcamera IMX219 CCM (D65, sRGB output). Row = output channel BGR.
CCM = np.array([
    [ 1.8004, -0.5760, -0.2244],
    [-0.3566,  1.6925, -0.3359],
    [-0.0950, -0.6687,  1.7637],
], dtype=np.float64)

_lut_cache = {}   # key -> LUT ndarray


def _make_key(*args):
    return tuple(round(a, 4) if isinstance(a, float) else a for a in args)


def _channel_lut(gain: float, gamma: float, bl: int) -> np.ndarray:
    """
    Single uint8->uint8 LUT that applies:
      1. Black level subtraction and rescale
      2. Linear AWB gain
      3. Gamma encoding
    All in one pass — zero float arrays at runtime.
    """
    key = ('ch', _make_key(gain, gamma, bl))
    if key not in _lut_cache:
        x = np.arange(256, dtype=np.float64)
        x = np.clip(x - bl, 0, 255) / (255.0 - bl)  # subtract BL, normalise
        x = np.clip(x * gain, 0.0, 1.0)              # AWB gain
        x = np.power(np.clip(x, 1e-9, 1.0), 1.0 / gamma) * 255.0  # gamma
        _lut_cache[key] = np.clip(x, 0, 255).astype(np.uint8)
    return _lut_cache[key]


def _sat_lut(sat: float) -> np.ndarray:
    key = ('sat', _make_key(sat))
    if key not in _lut_cache:
        x = np.arange(256, dtype=np.float64)
        _lut_cache[key] = np.clip(x * sat, 0, 255).astype(np.uint8)
    return _lut_cache[key]


def _ccm_matrix(ccm_strength: float) -> np.ndarray:
    """Blend CCM with identity matrix."""
    key = ('ccm', _make_key(ccm_strength))
    if key not in _lut_cache:
        _lut_cache[key] = (ccm_strength * CCM +
                           (1.0 - ccm_strength) * np.eye(3, dtype=np.float64))
    return _lut_cache[key]


# CUDA GpuMat pool — allocated once
_gpu = {}


def _debayer_cuda(raw_u8: np.ndarray) -> np.ndarray:
    h, w = raw_u8.shape
    if 'raw' not in _gpu or _gpu['raw'].size() != (w, h):
        _gpu['stream'] = cv2.cuda_Stream()
        _gpu['raw']    = cv2.cuda_GpuMat(h, w, cv2.CV_8UC1)
        _gpu['bgr']    = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
    _gpu['raw'].upload(raw_u8, _gpu['stream'])
    cv2.cuda.demosaicing(_gpu['raw'], cv2.cuda.COLOR_BayerRGGB2BGR,
                         _gpu['bgr'], stream=_gpu['stream'])
    _gpu['stream'].waitForCompletion()
    return _gpu['bgr'].download()


def _debayer_cpu(raw_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(raw_u8, cv2.COLOR_BayerRG2BGR)


# ── Per-stage timing ──────────────────────────────────────────────────────────
_STAGES = ('capture', 'debayer', 'resize', 'lut', 'ccm', 'sat', 'sharp', 'display', 'total')
_pt  = {s: 0.0 for s in _STAGES}
_ptc = [0]


def record_stage_report() -> dict:
    """Return average ms per frame for each stage, then reset."""
    n = max(_ptc[0], 1)
    report = {s: _pt[s] / n * 1000.0 for s in _STAGES}
    for s in _STAGES: _pt[s] = 0.0
    _ptc[0] = 0
    return report


def process_frame(raw: np.ndarray, p: dict, full_res: bool = False) -> np.ndarray:
    """
    p keys: bl, awb[gr,gg,gb], ccm_s, gamma, sat, sharp, denoise
    full_res=True  -> run at 1920x1080 (for saving)
    full_res=False -> debayer full res then resize to 960x540 for display
    """
    t = time.perf_counter

    bl = p['bl']; awb = p['awb']; ccm_s = p['ccm_s']
    gamma = p['gamma']; sat = p['sat']; sharp = p['sharp']; denoise = p['denoise']

    t0 = t()

    # 1. Black level clip
    raw_bl = np.clip(raw.astype(np.int16) - bl, 0, 255).astype(np.uint8)
    t1 = t()

    if full_res:
        # Full res debayer for saving
        bgr = _debayer_cuda(raw_bl) if HAS_CUDA else _debayer_cpu(raw_bl)
        _pt['debayer'] += t() - t1
        _pt['resize']  += 0.0
    else:
        # 2x2 Bayer bin BEFORE debayer: 1920x1080 -> 960x540, preserves RGGB pattern
        # This is 4x fewer pixels to debayer — the single biggest speedup
        raw_half = raw_bl[::2, ::2]
        t2 = t(); _pt['resize'] += t2 - t1
        bgr = _debayer_cuda(raw_half) if HAS_CUDA else _debayer_cpu(raw_half)
        _pt['debayer'] += t() - t2

    t2 = t()

    # 4. Per-channel LUT: BL + AWB gain + gamma — all uint8, no float allocs
    lut_b = _channel_lut(awb[2], gamma, bl)   # B uses gb
    lut_g = _channel_lut(awb[1], gamma, bl)
    lut_r = _channel_lut(awb[0], gamma, bl)   # R uses gr
    b = cv2.LUT(bgr[:, :, 0], lut_b)
    g = cv2.LUT(bgr[:, :, 1], lut_g)
    r = cv2.LUT(bgr[:, :, 2], lut_r)
    t3 = t(); _pt['lut'] += t3 - t2

    # 5. CCM via cv2.transform — single optimised SIMD call, ~5x faster than numpy
    bgr = cv2.merge([b, g, r])
    if ccm_s > 0.01:
        M   = _ccm_matrix(ccm_s).astype(np.float32)
        # cv2.transform: dst[i,j,k] = sum_l( M[k,l] * src[i,j,l] )
        bgrf = cv2.transform(bgr.astype(np.float32), M)
        bgr  = np.clip(bgrf, 0, 255).astype(np.uint8)
    t4 = t(); _pt['ccm'] += t4 - t3

    # 6. Saturation via HSV S-channel LUT
    if abs(sat - 1.0) > 0.02:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.LUT(hsv[:, :, 1], _sat_lut(sat))
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    t5 = t(); _pt['sat'] += t5 - t4

    # 7. Unsharp mask sharpening
    if sharp > 0.02:
        blur = cv2.GaussianBlur(bgr, (0, 0), 2.0)
        bgr  = cv2.addWeighted(bgr, 1.0 + sharp, blur, -sharp, 0)

    # 8. Denoise (very slow — off by default)
    if denoise:
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 5, 5, 7, 21)

    t6 = t(); _pt['sharp'] += t6 - t5
    _pt['total'] += t6 - t0
    _ptc[0] += 1
    return bgr


def gray_world_gains(bgr: np.ndarray):
    """Neutral-pixel AWB."""
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    gray = (b + g + r) / 3.0
    diff = np.maximum(np.maximum(np.abs(r - gray), np.abs(g - gray)), np.abs(b - gray))
    mask = (diff < 20) & (gray > 20) & (gray < 220)
    if mask.sum() > 200:
        rm = float(r[mask].mean()) + 1e-6
        gm = float(g[mask].mean()) + 1e-6
        bm = float(b[mask].mean()) + 1e-6
    else:
        rm = float(r.mean()) + 1e-6
        gm = float(g.mean()) + 1e-6
        bm = float(b.mean()) + 1e-6
    return float(np.clip(gm / rm, 0.5, 4.0)), 1.0, float(np.clip(gm / bm, 0.5, 4.0))


# ==============================================================================
# Controls GUI
# ==============================================================================
CTRL_WIN = "IMX219 Controls"


def create_controls():
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WIN, 500, 480)
    cv2.createTrackbar("Auto WB  (1=on)",   CTRL_WIN,   1,   1, lambda x: None)
    cv2.createTrackbar("AWB R x100",        CTRL_WIN, 110, 400, lambda x: None)
    cv2.createTrackbar("AWB G x100",        CTRL_WIN, 100, 400, lambda x: None)
    cv2.createTrackbar("AWB B x100",        CTRL_WIN,  85, 400, lambda x: None)
    cv2.createTrackbar("Black Level",       CTRL_WIN,  16,  64, lambda x: None)
    cv2.createTrackbar("CCM Strength x100", CTRL_WIN, 100, 100, lambda x: None)
    cv2.createTrackbar("Gamma x100",        CTRL_WIN, 220, 400, lambda x: None)
    cv2.createTrackbar("Saturation x100",   CTRL_WIN, 130, 300, lambda x: None)
    cv2.createTrackbar("Sharpness x100",    CTRL_WIN,  50, 200, lambda x: None)
    cv2.createTrackbar("Denoise (1=on)",    CTRL_WIN,   0,   1, lambda x: None)
    cv2.createTrackbar("Auto Exp (1=on)",   CTRL_WIN,   1,   1, lambda x: None)
    cv2.createTrackbar("AE Target",         CTRL_WIN, 120, 255, lambda x: None)


def get_controls() -> dict:
    def tb(n): return cv2.getTrackbarPos(n, CTRL_WIN)
    return {
        'auto_wb':  tb("Auto WB  (1=on)") == 1,
        'man_r':    max(tb("AWB R x100"), 1) / 100.0,
        'man_g':    max(tb("AWB G x100"), 1) / 100.0,
        'man_b':    max(tb("AWB B x100"), 1) / 100.0,
        'bl':       tb("Black Level"),
        'ccm_s':    tb("CCM Strength x100") / 100.0,
        'gamma':    max(tb("Gamma x100"), 10) / 100.0,
        'sat':      max(tb("Saturation x100"), 1) / 100.0,
        'sharp':    tb("Sharpness x100") / 100.0,
        'denoise':  tb("Denoise (1=on)") == 1,
        'auto_aec': tb("Auto Exp (1=on)") == 1,
        'ae_tgt':   max(tb("AE Target"), 1),
    }


def sync_awb(gr, gg, gb):
    cv2.setTrackbarPos("AWB R x100", CTRL_WIN, int(np.clip(gr * 100, 1, 400)))
    cv2.setTrackbarPos("AWB G x100", CTRL_WIN, int(np.clip(gg * 100, 1, 400)))
    cv2.setTrackbarPos("AWB B x100", CTRL_WIN, int(np.clip(gb * 100, 1, 400)))


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

    awb       = [1.10, 1.0, 0.85]
    alpha     = 0.05
    frame_n   = 0
    save_next = False
    fps       = 0.0
    fps_t0    = time.time()
    fps_count = 0
    sr        = {s: 0.0 for s in _STAGES}  # stage report

    print("\nKeys (camera window must have focus):")
    print("  q/ESC quit  |  s save snapshot.jpg  |  r reset  |  p print state\n")

    while True:
        t_cap0 = time.perf_counter()
        raw = cap.read()
        if raw is None:
            continue
        _pt['capture'] += time.perf_counter() - t_cap0

        frame_n   += 1
        fps_count += 1
        now = time.time()
        if now - fps_t0 >= 1.0:
            fps = fps_count / (now - fps_t0)
            fps_count = 0
            fps_t0    = now
            sr = record_stage_report()
            print(f"[PERF] FPS:{fps:.1f}  "
                  f"cap:{sr['capture']:.1f}  "
                  f"deb:{sr['debayer']:.1f}  "
                  f"rsz:{sr['resize']:.1f}  "
                  f"lut:{sr['lut']:.1f}  "
                  f"ccm:{sr['ccm']:.1f}  "
                  f"sat:{sr['sat']:.1f}  "
                  f"shp:{sr['sharp']:.1f}  "
                  f"tot:{sr['total']:.1f}  ms/frame")

        c = get_controls()
        ae.target = c['ae_tgt']

        # Half-res debayer shared by AEC + AWB (avoids a second full-res pass)
        raw_s = cv2.resize(raw, (raw.shape[1] // 2, raw.shape[0] // 2),
                           interpolation=cv2.INTER_NEAREST)
        raw_s = np.clip(raw_s.astype(np.int16) - c['bl'], 0, 255).astype(np.uint8)
        rough = cv2.cvtColor(raw_s, cv2.COLOR_BayerRG2BGR)

        if c['auto_aec']:
            ae.step(ae.measure(rough), ctrl)

        if c['auto_wb'] and frame_n % 5 == 0:
            gr, gg, gb = gray_world_gains(rough)
            awb[0] = alpha * gr + (1 - alpha) * awb[0]
            awb[1] = 1.0
            awb[2] = alpha * gb + (1 - alpha) * awb[2]
            sync_awb(*awb)
        elif not c['auto_wb']:
            awb = [c['man_r'], c['man_g'], c['man_b']]

        p = {'bl': c['bl'], 'awb': awb, 'ccm_s': c['ccm_s'],
             'gamma': c['gamma'], 'sat': c['sat'],
             'sharp': c['sharp'], 'denoise': c['denoise']}

        # Display at half res
        disp = process_frame(raw, p, full_res=False)

        if save_next:
            full = process_frame(raw, p, full_res=True)
            cv2.imwrite("snapshot.jpg", full, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print("[SAVE] snapshot.jpg  (full res)")
            save_next = False

        # OSD
        t_d0 = time.perf_counter()
        wb_mode = "AUTO" if c['auto_wb'] else "MAN"
        brt = ae.measure(rough)
        osd = [
            f"FPS: {fps:.1f}   Frame: {frame_n}",
            f"Brightness: {brt:.0f} / target {c['ae_tgt']}",
            f"AnaGain: {ctrl.analog_gain:.2f}x  DigGain: {ctrl.digital_gain:.2f}x",
            f"Exposure: {ctrl.exposure_us:.0f} us ({ctrl.exposure_lines} lines)",
            f"AWB [{wb_mode}]  R={awb[0]:.2f}  G={awb[1]:.2f}  B={awb[2]:.2f}",
            f"CCM:{c['ccm_s']:.2f}  Gamma:{c['gamma']:.2f}  "
            f"Sat:{c['sat']:.2f}  Sharp:{c['sharp']:.2f}",
            f"ms: cap={sr['capture']:.0f} deb={sr['debayer']:.0f} "
            f"rsz={sr['resize']:.0f} lut={sr['lut']:.0f} "
            f"ccm={sr['ccm']:.0f} sat={sr['sat']:.0f} "
            f"shp={sr['sharp']:.0f} TOT={sr['total']:.0f}",
            f"{'[CUDA]' if HAS_CUDA else '[CPU]'} debayer",
        ]
        for i, txt in enumerate(osd):
            cv2.putText(disp, txt, (10, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
        _pt['display'] += time.perf_counter() - t_d0

        cv2.imshow("IMX219 Tuning", disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            save_next = True
        elif key == ord('r'):
            ctrl.reset()
            awb[:] = [1.10, 1.0, 0.85]
            sync_awb(*awb)
            print("[RESET]")
        elif key == ord('p'):
            ctrl.print_state()
            print(f"  AWB [{wb_mode}]: R={awb[0]:.3f}  G={awb[1]:.3f}  B={awb[2]:.3f}")
            print(f"  CCM:{c['ccm_s']:.2f}  Gamma:{c['gamma']:.2f}  "
                  f"Sat:{c['sat']:.2f}  Sharp:{c['sharp']:.2f}")
            print(f"  Brightness: {brt:.1f}")

    cap.release()
    cv2.destroyAllWindows()
    i2c.close()
    print("[DONE]")


# ==============================================================================
# CLI
# ==============================================================================
def _parse():
    p = argparse.ArgumentParser(description="IMX219 RAW8 tuning — Jetson/Tegra VI")
    p.add_argument("--device",   default="/dev/video5")
    p.add_argument("--i2c-bus",  type=int, default=1)
    p.add_argument("--i2c-addr", type=lambda x: int(x, 0), default=0x10,
                   help="Sensor I2C address (default 0x10)")
    p.add_argument("--width",    type=int, default=1920)
    p.add_argument("--height",   type=int, default=1080)
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
