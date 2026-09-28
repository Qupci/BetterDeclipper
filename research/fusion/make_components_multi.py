"""Component estimates (nmf400, spade, pew400, fast) for fusion validation on hard-clip cases:
the 48 kHz test set (8 cases) and the master benchmark hard6/hard9 cases (14)."""
import sys, os, json
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.engine import declip
from betterdeclipper.metrics import sdr

OUT = "F:/BetterDeclipper/research/outputs/fusion/multi/"
os.makedirs(OUT, exist_ok=True)
cases = []
for t in ["gris48", "merlon48", "lofi48", "ghostpage48"]:
    for lv in ["-12dB", "-6dB"]:
        cases.append((f"{t}_{lv}", f"F:/BetterDeclipper/testset/{t}/clipped_{lv}.wav", f"F:/BetterDeclipper/testset/{t}/ground_truth.wav"))
B = "F:/BetterDeclipper/research/outputs/mbench/"
for s in ["ex", "petal", "ruder", "thrash", "violet", "bigshot", "knife"]:
    for d in ["hard6", "hard9"]:
        if s == "ex" and d == "hard6":
            pass
        cases.append((f"{s}_{d}", B + f"{s}__{d}.wav", B + f"{s}__{d}__gt.wav"))
runs = {"fast": dict(preset="fast"), "nmf400": dict(models=[("nmf", 93, {})]), "spade": dict(models=[("spade", 93, {})]),
        "pew400": dict(models=[("pnp", 93, {})])}
only = sys.argv[1:]
for name, yp, gp in cases:
    if only and name not in only:
        continue
    y, sr = sf.read(yp, dtype="float64", always_2d=True)
    gt, _ = sf.read(gp, dtype="float64", always_2d=True)
    line = []
    for k, kw in runs.items():
        fn = OUT + f"{name}__{k}.npy"
        if os.path.exists(fn):
            continue
        x, info = declip(y, sr, mode="hard", verbose=False, **kw)
        np.save(fn, x.astype(np.float32))
        line.append(f"{k} {sdr(gt, x):.2f}")
    print(name, " ".join(line), flush=True)
