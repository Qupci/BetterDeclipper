import numpy as np, soundfile as sf, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = "F:/BetterDeclipper/research/outputs/fusion/"
y, sr = sf.read("F:/BetterDeclipper/ex_sample/sample_-12dB_clipped.wav", dtype="float64", always_2d=True)
gt, _ = sf.read("F:/BetterDeclipper/ex_sample/sample_ground_truth.wav", dtype="float64", always_2d=True)
sp = np.load(D + "spade.npy").astype(np.float64); nm = np.load(D + "nmf400.npy").astype(np.float64)
ts = [float(v) for v in sys.argv[1:]] or [20.017, 4.725, 14.929, 1.661]
fig, axes = plt.subplots(len(ts), 2, figsize=(18, 4 * len(ts)))
th = np.abs(y).max()
for r, t0 in enumerate(ts):
    i = int(t0 * sr); sl = slice(i - int(0.004 * sr), i + int(0.004 * sr))
    tt = (np.arange(sl.start, sl.stop) - i) / sr * 1000
    for c in range(2):
        ax = axes[r, c]
        ax.plot(tt, gt[sl, c], "k-", lw=1.2, label="ground truth")
        ax.plot(tt, y[sl, c], "c-", lw=0.8, label="clipped")
        ax.plot(tt, sp[sl, c], "r.-", lw=0.8, ms=2, label="SPADE")
        ax.plot(tt, nm[sl, c], "b.-", lw=0.8, ms=2, label="NMF")
        ax.axhline(th, color="gray", lw=0.4); ax.axhline(-th, color="gray", lw=0.4)
        ax.set_title(f"t={t0}s ch{c}"); ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=8)
plt.tight_layout(); plt.savefig("F:/BetterDeclipper/research/plots/spade_clicks.png", dpi=65)
