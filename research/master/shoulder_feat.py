"""Shoulder (soft-saturation pile-up below the ceiling) features for all benchmark cases and the user's files."""
import sys, os, glob
sys.path.insert(0, "F:/BetterDeclipper"); sys.path.insert(0, "F:/BetterDeclipper/research/master")
import numpy as np, soundfile as sf
from smear_fit import fit_smeared
from betterdeclipper.detect import estimate_lsb


def shoulder_stats(s, theta, sigma, fit=(0.35, 0.6), top_c=6.0, nb=200):
    """log-ratio of observed density to a log-linear trend (fit on fit*theta), over [0.62 theta, theta - top_c sigma]."""
    s = s[s > 0]
    hi = theta - top_c * sigma
    lo = fit[0] * theta
    if hi <= 0.65 * theta:
        return dict(max=np.nan, mean=np.nan, start=np.nan)
    h, e = np.histogram(s, bins=nb, range=(lo, theta))
    u = 0.5 * (e[1:] + e[:-1]) / theta
    ok = (u <= fit[1]) & (h > 0)
    b, a = np.polyfit(u[ok], np.log(h[ok]), 1, w=np.sqrt(h[ok]))
    b = min(b, 0.0)
    lr = np.log(np.maximum(np.convolve(h, np.ones(5) / 5, "same"), 0.5)) - (a + b * u)
    t = (u >= 0.62) & (u * theta <= hi)
    over = np.flatnonzero(t & (lr > np.log(1.25)))
    return dict(max=float(lr[t].max()), mean=float(lr[t].mean()), start=float(u[over[0]]) if over.size else np.nan)


if __name__ == "__main__":
    B = "F:/BetterDeclipper/research/outputs/mbench/"
    rows = []
    for fn in sorted(glob.glob(B + "*.wav")):
        if fn.endswith("__gt.wav"):
            continue
        key = os.path.basename(fn)[:-4]
        rows.append((key, fn))
    for s in ["ex", "petal", "ruder", "thrash"]:
        rows.append((f"{s}__GT", B + f"{s}__hard6__gt.wav"))
    rows += [("USER Rome", "F:/BetterDeclipper/ex_master/soft_works/17_Rome_Search.wav"),
             ("USER Tarzan", "F:/BetterDeclipper/ex_master/soft_works/tarzan taz normal LOOP.wav"),
             ("USER Bangarang", "F:/BetterDeclipper/ex_master/blind/bad/Bangarang.flac"),
             ("USER GoinHard", "F:/BetterDeclipper/ex_master/blind/bad/Goin' In (Skrillex Goin Hard Remix).flac"),
             ("USER AllIsFair", "F:/BetterDeclipper/ex_master/blind/bad/All Is Fair In Love And Brostep.flac"),
             ("USER Purple", "F:/BetterDeclipper/ex_master/blind/bad/Purple Lamborghini.wav"),
             ("USER OWSLA", "F:/BetterDeclipper/research/outputs/master/OWSLA 3.wav"),
             ("USER Mothership", "F:/BetterDeclipper/research/outputs/master/Mothership Teaser.wav"),
             ("USER PetalAL1", "F:/BetterDeclipper/ex_master/AL-1/17 - Petal Dance -6dB AL-1.wav"),
             ("USER metallica", "F:/BetterDeclipper/research/outputs/blind/metallica_in.wav")]
    for key, fn in rows:
        if not os.path.exists(fn):
            continue
        y, sr = sf.read(fn, dtype="float64", always_2d=True)
        lsb = estimate_lsb(y)
        out = []
        for c in range(min(2, y.shape[1])):
            s = y[:, c]
            f = fit_smeared(s, lsb)
            P = np.quantile(s[s > 0], 1 - 1e-4)
            th = min(f["theta"], P)
            sig = f["sigma"] if f["theta"] <= P else 0.0
            st = shoulder_stats(s, th, sig)
            out.append(f"max {st['max']:5.2f} mean {st['mean']:5.2f} start {st['start']:.2f} (bump {f['bump_ratio']:7.1f} s/t {f['sigma']/f['theta']:.4f})")
        print(f"{key:26s} " + " | ".join(out), flush=True)
