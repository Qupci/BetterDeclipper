"""Profile the user's mastered examples: detection results, top-of-histogram shape, plateau runs."""
import sys, os, glob
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from betterdeclipper.detect import detect_clip_levels, estimate_lsb, detect_knee, clip_masks

FILES = sorted(glob.glob("F:/BetterDeclipper/ex_master/soft_works/*.wav")) + \
        sorted(glob.glob("F:/BetterDeclipper/ex_master/blind/bad/*")) + \
        ["F:/BetterDeclipper/research/outputs/master/OWSLA 3.wav",
         "F:/BetterDeclipper/research/outputs/master/Mothership Teaser.wav",
         "F:/BetterDeclipper/ex_master/blind/wierd/signal (dan sena remix).wav",
         "F:/BetterDeclipper/ex_master/AL-1/17 - Petal Dance -6dB AL-1.wav",
         "F:/BetterDeclipper/ex_master/AL-1/17 - Petal Dance -6dB HARD.wav"]


def runs_at(mask):
    """lengths of runs of True"""
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


fig, axes = plt.subplots(4, 4, figsize=(20, 16)); axes = axes.ravel()
for fi, f in enumerate(FILES):
    y, sr = sf.read(f, dtype="float64", always_2d=True)
    name = os.path.basename(f)
    lsb = estimate_lsb(y)
    lv = detect_clip_levels(y, lsb)
    kn = detect_knee(y)
    m_hi, m_lo, _, _ = clip_masks(y, lv)
    print(f"== {name}: {y.shape[1]} ch, {len(y)/sr:.0f} s, lsb {lsb:.2e}")
    for c in range(y.shape[1]):
        for sgn, pol in ((1, "+"), (-1, "-")):
            s = sgn * y[:, c]
            pk = s.max()
            lvl = lv[c][0 if sgn > 0 else 1]
            k = kn[c][0 if sgn > 0 else 1]
            r = runs_at(s >= pk - 1.5 * max(lsb, 1e-6))
            r2 = runs_at(s >= 0.99 * pk)
            print(f"  ch{c}{pol}: peak {20*np.log10(pk):6.2f} dBFS  plateau {('-' if lvl is None else f'{20*np.log10(abs(lvl)):6.2f} dB ({abs(lvl)/pk:.4f} pk)')}"
                  f"  knee {('-' if k is None else f'{abs(k)/pk:.3f} pk')}  n@peak {int((s >= pk - 1.5*max(lsb,1e-6)).sum())} (runs>=3: {int((r>=3).sum())}, max run {r.max() if r.size else 0})"
                  f"  n>=.99pk {int((s>=.99*pk).sum())} ({(s>=.99*pk).mean()*100:.3f}%) runs>=3 {int((r2>=3).sum())}")
    print(f"  flagged hard/auto: {(m_hi | m_lo).mean()*100:.3f}%")
    ax = axes[fi]
    for c in range(y.shape[1]):
        for sgn, ls in ((1, "-"), (-1, "--")):
            s = sgn * y[:, c]; s = s[s > 0]; pk = s.max()
            h, e = np.histogram(s / pk, bins=400, range=(0, 1.0))
            ax.semilogy(0.5 * (e[1:] + e[:-1]), np.maximum(h, 0.5), ls, lw=0.8, label=f"ch{c}{'+' if sgn > 0 else '-'}")
    ax.set_xlim(0.3, 1.0); ax.set_title(name[:40], fontsize=9); ax.grid(alpha=0.3)
axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("F:/BetterDeclipper/research/plots/master_hist.png", dpi=80)
