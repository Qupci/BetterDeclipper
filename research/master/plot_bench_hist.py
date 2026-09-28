import sys
import numpy as np, soundfile as sf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
B = "F:/BetterDeclipper/research/outputs/mbench/"
src = sys.argv[1] if len(sys.argv) > 1 else "ex"
degs = ["hard6", "tanh6", "tanh9k4", "cubic6", "softhard6", "softhard9", "oshard6", "resamp6", "mp3_128", "aac_256", "vorbis_192", "al1_6", "osal1"]
fig, axes = plt.subplots(4, 4, figsize=(20, 16)); axes = axes.ravel()
for i, d in enumerate(degs):
    y, sr = sf.read(B + f"{src}__{d}.wav", dtype="float64", always_2d=True)
    gt, _ = sf.read(B + f"{src}__{d}__gt.wav", dtype="float64", always_2d=True)
    ax = axes[i]
    for c in range(y.shape[1]):
        s = y[:, c]; s = s[s > 0]; pk = s.max()
        h, e = np.histogram(s / pk, bins=400, range=(0, 1.0))
        ax.semilogy(0.5 * (e[1:] + e[:-1]), np.maximum(h, 0.5), lw=0.8, label=f"ch{c}+ degraded")
        g = gt[:, c]; g = g[g > 0]
        h, e = np.histogram(g / pk, bins=400, range=(0, 1.0))
        ax.semilogy(0.5 * (e[1:] + e[:-1]), np.maximum(h, 0.5), ":", lw=0.8, label=f"ch{c}+ source")
    ax.set_xlim(0.3, 1.0); ax.set_title(f"{src} {d}", fontsize=10); ax.grid(alpha=0.3)
axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"F:/BetterDeclipper/research/plots/mbench_hist_{src}.png", dpi=75)
