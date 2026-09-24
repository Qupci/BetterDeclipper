import sys, itertools, glob, os; sys.path.insert(0, "F:/BetterDeclipper"); sys.path.insert(0, "F:/BetterDeclipper/research")
import numpy as np, soundfile as sf
from bench_example import load
from betterdeclipper.metrics import sdr
gt, cl, pad, sr = load((5, 10))
O = "F:/BetterDeclipper/research/outputs/"
names = {"pnp4096": "pnp4096_cg25_5_10.wav", "pnp2048": "pnp2048_cg25_5_10.wav", "pnp8192": "pnp8192_cg25_5_10.wav",
         "sp_a4096": "spade_a_4096_5_10.wav", "sp_a2048": "spade_a_2048_5_10.wav", "sp_a8192": "spade_a_8192_5_10.wav",
         "sp_s4096": "spade_s_4096_5_10.wav"}
outs = {k: sf.read(O + v, dtype="float64")[0] for k, v in names.items() if os.path.exists(O + v)}
for k, v in outs.items(): print(f"{k:10s} {sdr(gt, v):.3f}")
keys = list(outs)
res = []
for r in range(2, len(keys) + 1):
    for combo in itertools.combinations(keys, r):
        res.append((sdr(gt, np.mean([outs[c] for c in combo], 0)), combo))
res.sort(reverse=True)
print("best uniform averages:")
for s_, c in res[:8]: print(f"  {s_:.3f}  {' + '.join(c)}")
# oracle LS weights on clipped-region residuals (upper bound for linear fusion)
m = np.abs(cl) >= 0.2499
X = np.stack([outs[k][m] - cl[m] for k in keys], 1)
w, *_ = np.linalg.lstsq(X, gt[m] - cl[m], rcond=None)
est = cl.copy(); est[m] = cl[m] + X @ w
print("oracle LS weights:", dict(zip(keys, np.round(w, 3))), f"-> {sdr(gt, est):.3f}")
