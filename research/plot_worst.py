"""Plot the clipped runs with the largest restoration error (gt vs clipped vs estimates)."""
import sys; sys.path.insert(0, "F:/BetterDeclipper"); sys.path.insert(0, "F:/BetterDeclipper/research")
import numpy as np, soundfile as sf, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bench_example import load
gt, cl, pad, sr = load((5, 10))
est, _ = sf.read(sys.argv[1], dtype="float64")
ch = 0
m = np.abs(gt[:, ch]) > 0.25
d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
s, e = np.where(d == 1)[0], np.where(d == -1)[0]
err = np.array([np.sum((gt[a:b, ch] - est[a:b, ch]) ** 2) for a, b in zip(s, e)])
order = np.argsort(err)[::-1][:6]
fig, axs = plt.subplots(3, 2, figsize=(16, 11))
for ax, i in zip(axs.ravel(), order):
    a, b = s[i], e[i]
    c = (a + b) // 2
    w = max(300, (b - a) * 2)
    sl = slice(max(0, c - w), c + w)
    t = np.arange(sl.start, sl.stop)
    ax.plot(t, gt[sl, ch], "k", lw=2, label="ground truth")
    ax.plot(t, cl[sl, ch], "c", lw=1, label="clipped")
    ax.plot(t, pad[sl, ch], "r--", lw=1.2, label="PAD")
    ax.plot(t, est[sl, ch], "b", lw=1.2, label="ours")
    ax.set_title(f"run len {b-a}, err ours {10*np.log10(err[i]):.1f} dB, PAD {10*np.log10(np.sum((gt[a:b,ch]-pad[a:b,ch])**2)):.1f} dB")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(sys.argv[2], dpi=70)
print("run length stats of worst:", [(int(e[i] - s[i])) for i in order])
