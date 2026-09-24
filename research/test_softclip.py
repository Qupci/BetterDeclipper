"""Validate soft-clip mode on a synthetic soft-clipped version of the example ground truth.

Soft clipper: |y| = t + (c - t) * tanh((|x| - t) / (c - t)) for |x| > t (slope 1 at the knee t,
saturating at ceiling c), then 16-bit TPDF dither.
python test_softclip.py [preset] [knee] [ceiling]
"""
import sys; sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.engine import declip
from betterdeclipper.metrics import sdr

preset = sys.argv[1] if len(sys.argv) > 1 else "fast"
t = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
c = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
gt, sr = sf.read("F:/BetterDeclipper/ex_sample/sample_ground_truth.wav", dtype="float64")
a = np.abs(gt)
ys = np.where(a > t, np.sign(gt) * (t + (c - t) * np.tanh((a - t) / (c - t))), gt)
rng = np.random.default_rng(0)
ys = np.round(ys * 32768 + rng.random(ys.shape) - rng.random(ys.shape)) / 32768
m = a > t
print(f"soft clip knee {t} ceiling {c}: {m.mean()*100:.1f}% samples above knee, input SDR {sdr(gt, ys):.2f} dB")
for mode in ("auto", "hard"):
    x, info = declip(ys, sr, preset=preset, mode=mode)
    lv = [tuple(None if v is None else round(abs(v), 4) for v in l) for l in info["levels"]]
    print(f"  mode={mode:4s} -> used {info['mode']}, levels {lv}, flagged {info['clipped_frac']*100:.1f}%:"
          f"  SDR {sdr(gt, x):.2f} dB  (compressed samples {sdr(gt[m], x[m]):.2f} dB)  {info['time']:.0f}s", flush=True)
