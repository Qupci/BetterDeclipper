"""Gain envelope g = y/x of AL-1 around ceiling touches (user's Petal Dance -6 dB render vs source)."""
import numpy as np, soundfile as sf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
src, sr = sf.read(r"F:/deltarune/ch5/17 - Petal Dance.flac", dtype="float64", always_2d=True)
al, _ = sf.read("F:/BetterDeclipper/ex_master/AL-1/17 - Petal Dance -6dB AL-1.wav", dtype="float64", always_2d=True)
x, y = src[:, 0], al[:, 0]
pk = np.abs(y).max()
touch = np.flatnonzero(np.abs(y) >= 0.995 * pk)
# pick isolated touch events (>= 10 ms apart), strongest source peaks
ev = [touch[0]]
for t in touch[1:]:
    if t - ev[-1] > int(0.01 * sr):
        ev.append(t)
ev = sorted(ev, key=lambda t: -abs(x[t]))[:12]
W = int(0.004 * sr)
fig, axes = plt.subplots(3, 4, figsize=(20, 11)); axes = axes.ravel()
for ax, t in zip(axes, ev):
    sl = slice(t - W, t + W)
    tt = (np.arange(sl.start, sl.stop) - t) / sr * 1000
    g = y[sl] / np.where(np.abs(x[sl]) > 0.02, x[sl], np.nan)
    ax.plot(tt, g, ".", ms=3, label="g = y/x")
    ax.plot(tt, x[sl] / abs(x[t]), "k-", lw=0.6, alpha=0.5, label="x (norm)")
    ax.set_ylim(-0.1, 1.3); ax.grid(alpha=0.3); ax.set_title(f"t={t/sr:.3f}s |x|={abs(x[t]):.3f}")
axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("F:/BetterDeclipper/research/plots/al1_gain_shape.png", dpi=65)
