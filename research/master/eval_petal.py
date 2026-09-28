"""Baseline: current modes on the user's Petal Dance -6 dB masters (ground truth = source flac)."""
import sys, time
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.engine import declip
from betterdeclipper.metrics import sdr

gt, sr = sf.read(r"F:/deltarune/ch5/17 - Petal Dance.flac", dtype="float64", always_2d=True)
preset = sys.argv[1] if len(sys.argv) > 1 else "fast"
for name in ["HARD", "AL-1"]:
    y, _ = sf.read(f"F:/BetterDeclipper/ex_master/AL-1/17 - Petal Dance -6dB {name}.wav", dtype="float64", always_2d=True)
    print(f"{name}: input SDR {sdr(gt, y):.2f} dB", flush=True)
    for mode in ["auto", "hard", "soft"]:
        x, info = declip(y, sr, preset=preset, mode=mode, verbose=False)
        lv = [tuple(None if v is None else round(20 * np.log10(abs(v)), 2) for v in l) for l in info["levels"]]
        print(f"  {mode:5s} -> {info['mode']:4s} levels {lv} flagged {info['clipped_frac']*100:.2f}%  SDR {sdr(gt, x):.2f} dB  ({info['time']:.1f}s)", flush=True)
