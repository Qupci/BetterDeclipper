import sys, warnings; sys.path.insert(0, "F:/BetterDeclipper"); sys.path.insert(0, "F:/BetterDeclipper/research")
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf
from bench_testset import list_cases
from betterdeclipper.engine import declip
from betterdeclipper.metrics import sdr
cases = list_cases()
for name in sys.argv[1].split(","):
    gt_p, cl_p, th, _ = cases[name]
    gt, sr = sf.read(gt_p, dtype="float64", always_2d=True); cl, _ = sf.read(cl_p, dtype="float64", always_2d=True)
    a, b = 5 * sr, 10 * sr
    gt, cl = gt[a:b], cl[a:b]
    n, _ = declip(cl, sr, models=[("nmf", 93, {})])
    s, _ = declip(cl, sr, models=[("spade", 93, {})])
    p, _ = declip(cl, sr, models=[("pnp", 93, {})])
    w = [(sdr(gt, a_ * n + (1 - a_) * s), a_) for a_ in np.linspace(0, 1, 21)]
    print(f"{name}: nmf {sdr(gt, n):.2f} spade {sdr(gt, s):.2f} pew {sdr(gt, p):.2f} | equal n+s {sdr(gt, (n+s)/2):.2f}  0.65/0.35 {sdr(gt, 0.65*n+0.35*s):.2f}  best {max(w)[0]:.2f}@{max(w)[1]:.2f} | equal n+p+s {sdr(gt, (n+p+s)/3):.2f}  0.45/0.2/0.35 {sdr(gt, 0.45*n+0.2*p+0.35*s):.2f}", flush=True)
