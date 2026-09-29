"""Histogram-top features that could reveal a 'rounded' ceiling (clipping followed by smoothing that never
overshoots, e.g. linear-interpolation resampling): a flat shelf of density right up to a sharp cutoff.
Per polarity: top1/next4 density ratio (cutoff sharpness), shelf excess over the exp trend, peak ratios.
usage: python shelf_feat.py files..."""
import sys, os
import numpy as np, soundfile as sf


def feats(s):
    s = s[s > 0]
    smax = s.max()
    u = s / smax
    top1 = np.count_nonzero(u >= 0.99)
    nxt4 = np.count_nonzero((u >= 0.95) & (u < 0.99))
    sharp = top1 / max(nxt4 / 4, 1)
    h, e = np.histogram(u, bins=110, range=(0.45, 1.0))
    c = 0.5 * (e[1:] + e[:-1])
    fit = (c < 0.75) & (h > 0)
    b, a = np.polyfit(c[fit], np.log(h[fit]), 1)
    band = (c >= 0.85) & (c < 0.99)
    excess = h[band].sum() / np.exp(a + b * c[band]).sum()
    p4 = np.quantile(s, 1 - 1e-4) / smax
    return sharp, excess, p4


if __name__ == "__main__":
    for f in sys.argv[1:]:
        try:
            y, sr = sf.read(f, dtype="float64", always_2d=True)
        except Exception as ex:
            print(f, "ERR", ex); continue
        row = []
        for c in range(y.shape[1]):
            for sg in (1, -1):
                sh, ex, p4 = feats(sg * y[:, c])
                row.append(f"{sh:5.2f}/{ex:5.2f}/{p4:.3f}")
        print(f"{os.path.basename(f)[:38]:38s} " + "  ".join(row), flush=True)
