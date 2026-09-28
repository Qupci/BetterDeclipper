"""Frequency-dependent fusion on the example: x = base + M * (h * (other - base)), h = crossover filter,
M = clipped-sample mask (keeps consistency; reliable samples stay = input). Then clamp to the bounds."""
import sys
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from scipy import signal
from betterdeclipper.metrics import sdr
from betterdeclipper.detect import detect_clip_levels, clip_masks

D = "F:/BetterDeclipper/research/outputs/fusion/"
y, sr = sf.read("F:/BetterDeclipper/ex_sample/sample_-12dB_clipped.wav", dtype="float64", always_2d=True)
gt, _ = sf.read("F:/BetterDeclipper/ex_sample/sample_ground_truth.wav", dtype="float64", always_2d=True)
L = {k: np.load(D + k + ".npy").astype(np.float64) for k in ["fast", "nmf400", "nmf800", "spade", "normal", "pew400"]}
lv = detect_clip_levels(y)
m_hi, m_lo, th_hi, th_lo = clip_masks(y, lv)
M = m_hi | m_lo

def lp(d, fc, order=4):
    sos = signal.butter(order, fc, "lp", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, d, axis=0)

def fuse(base, other, w_lo, w_hi, fc):
    d = other - base
    dl = lp(d, fc)
    corr = w_lo * dl + w_hi * (d - dl)
    x = base + M * corr
    x = np.where(m_hi, np.maximum(x, th_hi[None, :]), x)
    x = np.where(m_lo, np.minimum(x, th_lo[None, :]), x)
    return x

print("reference: nmf400 %.3f  normal(0.65/0.35) %.3f  fast %.3f" % (sdr(gt, L["nmf400"]), sdr(gt, L["normal"]), sdr(gt, L["fast"])))
for fc in (1000, 2000, 4000, 8000):
    row = []
    for wl, wh in ((0.35, 0.0), (0.35, 0.15), (0.45, 0.0), (0.45, 0.15), (0.55, 0.1)):
        row.append(f"{wl:.2f}/{wh:.2f}: {sdr(gt, fuse(L['nmf400'], L['spade'], wl, wh, fc)):.3f}")
    print(f"nmf400+spade fc {fc:5d}: " + "  ".join(row))
# the user's idea: low band from normal, high band from fast
for fc in (1000, 2000, 4000, 8000):
    print(f"fast(HF)+normal(LF) fc {fc:5d}: {sdr(gt, fuse(L['fast'], L['normal'], 1.0, 0.0, fc)):.3f}   nmf400(HF)+normal(LF): {sdr(gt, fuse(L['nmf400'], L['normal'], 1.0, 0.0, fc)):.3f}")
