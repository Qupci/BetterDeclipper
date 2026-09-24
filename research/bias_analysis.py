"""Systematic bias of an estimate inside clipped runs: signed residual (gt-est)*sign vs run length/position."""
import sys; sys.path.insert(0, "F:/BetterDeclipper"); sys.path.insert(0, "F:/BetterDeclipper/research")
import numpy as np, soundfile as sf
from bench_example import load
from betterdeclipper.metrics import sdr
gt, cl, pad, sr = load((5, 10))
est, _ = sf.read(sys.argv[1], dtype="float64")
th = 0.25
for name, e in [("ours", est), ("PAD", pad)]:
    rows = []
    corr = e.copy()
    bins = [(1, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 400)]
    stats = {b: [] for b in bins}
    for ch in range(2):
        m = np.abs(gt[:, ch]) > th
        d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
        s, en = np.where(d == 1)[0], np.where(d == -1)[0]
        for a, b in zip(s, en):
            L = b - a
            sg = np.sign(cl[a, ch])
            r = (gt[a:b, ch] - e[a:b, ch]) * sg
            ex = (e[a:b, ch] * sg - th)  # how far estimate is above threshold
            exg = (gt[a:b, ch] * sg - th)
            for bb in bins:
                if bb[0] <= L < bb[1]:
                    stats[bb].append((r.mean(), ex.mean(), exg.mean(), np.mean(r ** 2)))
    print(f"== {name}: SDR {sdr(gt, e):.2f}")
    for bb in bins:
        st = np.array(stats[bb])
        if len(st):
            print(f"  run len {bb[0]:3d}-{bb[1]:<3d} n={len(st):4d}  mean signed resid {st[:,0].mean():+.4f}  est excess {st[:,1].mean():.4f}  true excess {st[:,2].mean():.4f}  mse {st[:,3].mean():.5f}")
