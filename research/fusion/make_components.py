"""Render component models and presets on the full example (GPU) for error analysis."""
import sys
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.engine import declip
from betterdeclipper.metrics import sdr

y, sr = sf.read("F:/BetterDeclipper/ex_sample/sample_-12dB_clipped.wav", dtype="float64", always_2d=True)
gt, _ = sf.read("F:/BetterDeclipper/ex_sample/sample_ground_truth.wav", dtype="float64", always_2d=True)
runs = {
    "fast": dict(preset="fast"),
    "nmf400": dict(models=[("nmf", 93, {})]),
    "nmf800": dict(models=[("nmf", 93, dict(n_iter=800))]),
    "pew400": dict(models=[("pnp", 93, {})]),
    "spade": dict(models=[("spade", 93, {})]),
    "normal": dict(preset="normal"),
}
for k, kw in runs.items():
    x, info = declip(y, sr, verbose=False, **kw)
    np.save(f"F:/BetterDeclipper/research/outputs/fusion/{k}.npy", x.astype(np.float32))
    print(f"{k}: SDR {sdr(gt, x):.3f} dB  ({info['time']:.1f}s)", flush=True)
