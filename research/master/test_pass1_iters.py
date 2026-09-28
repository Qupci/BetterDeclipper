"""Does the first (analysis) pass need the full fast preset? Compare knee decisions with fewer iterations."""
import sys, json
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
import betterdeclipper.auto as A
from betterdeclipper.engine import declip
from betterdeclipper.detect import estimate_lsb
B = "F:/BetterDeclipper/research/outputs/mbench/"
cases = [f"{s}__{d}" for s in ["ex", "petal", "ruder", "bigshot"] for d in ["hard6", "tanh6", "tanh9k4", "cubic6", "softhard6", "softhard9", "mp3_128", "al1_6"]]
for key in cases:
    y, sr = sf.read(B + key + ".wav", dtype="float64", always_2d=True)
    lsb = estimate_lsb(y)
    out = []
    for it in (150, 80, 40):
        def run_fast(yy, cons, it=it):
            xx, _ = declip(yy, sr, models=[("nmf", 93, dict(n_iter=it))], constraints=cons, verbose=False)
            return xx
        _, _, _, _, rep = A.auto_constraints(y, lsb, run_fast, mode="auto", sr=sr)
        out.append(f"it{it}: {rep['mode']}")
    print(f"{key:20s} " + " | ".join(out), flush=True)
