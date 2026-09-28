"""Timings on the example with the new auto analysis (device from argv: cuda/cpu)."""
import sys, time
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.engine import declip
from betterdeclipper.metrics import sdr
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
presets = sys.argv[2].split(",") if len(sys.argv) > 2 else ["fast", "normal", "high", "best"]
y, sr = sf.read("F:/BetterDeclipper/ex_sample/sample_-12dB_clipped.wav", dtype="float64", always_2d=True)
gt, _ = sf.read("F:/BetterDeclipper/ex_sample/sample_ground_truth.wav", dtype="float64", always_2d=True)
declip(y[: sr * 3], sr, preset="fast", device=dev, verbose=False)  # warm-up (CUDA context, plans)
for p in presets:
    t = time.time()
    x, info = declip(y, sr, preset=p, device=dev, verbose=False)
    tot = time.time() - t
    t = time.time()
    x2, _ = declip(y, sr, preset=p, device=dev, mode="legacy", verbose=False)
    leg = time.time() - t
    print(f"{p}: auto {tot:.1f}s (SDR {sdr(gt, x):.3f})   legacy/no analysis {leg:.1f}s (SDR {sdr(gt, x2):.3f})", flush=True)
