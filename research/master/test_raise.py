"""Two-pass knee idea: run soft mode with a low knee, measure how much the declipper raises samples per level
(median |x_hat|/|y| in level bins) and compare with the true compression of the degradation."""
import sys
sys.path.insert(0, "F:/BetterDeclipper"); sys.path.insert(0, "F:/BetterDeclipper/research/master")
import numpy as np, soundfile as sf
from betterdeclipper.engine import declip, default_knees
from test_compress import true_comp
B = "F:/BetterDeclipper/research/outputs/mbench/"
LV = np.array([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
k0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
for src in ["ex", "petal", "ruder", "bigshot"]:
    for deg, db in [("tanh6", -6), ("tanh9k4", -9), ("cubic6", -6), ("softhard6", -6), ("softhard9", -9), ("hard6", -6), ("oshard6", -6), ("mp3_128", -6), ("al1_6", -6)]:
        y, sr = sf.read(B + f"{src}__{deg}.wav", dtype="float64", always_2d=True)
        gt, _ = sf.read(B + f"{src}__{deg}__gt.wav", dtype="float64", always_2d=True)
        c = np.abs(gt).max() * 10 ** (db / 20)
        x, info = declip(y, sr, preset="fast", mode="soft", knees=default_knees(y, None, frac=k0), verbose=False)
        pk = np.quantile(np.abs(y), 1 - 1e-4)
        a = np.abs(y).ravel(); r = (np.abs(x) / np.maximum(np.abs(y), 1e-9)).ravel(); tr = (np.abs(gt) / np.maximum(np.abs(y), 1e-9)).ravel()
        est, act = [], []
        for l in LV:
            m = (a >= (l - 0.025) * pk) & (a < (l + 0.025) * pk)
            est.append(np.median(r[m]) - 1 if m.sum() > 50 else np.nan)
            act.append(np.median(tr[m]) - 1 if m.sum() > 50 else np.nan)
        print(f"{src:7s} {deg:9s} raise%: " + " ".join(f"{v*100:5.1f}" for v in est) + "  | true%: " + " ".join(f"{v*100:5.1f}" for v in act), flush=True)
