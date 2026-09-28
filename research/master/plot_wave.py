import sys
import numpy as np, soundfile as sf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fn, out = sys.argv[1], sys.argv[2]
th = float(sys.argv[3]) if len(sys.argv) > 3 else None
y, sr = sf.read(fn, dtype="float64", always_2d=True)
s = y[:, 0]
# pick 6 peaks spread over the file
idx = []
for k in range(6):
    a, b = int(len(s) * (0.1 + 0.13 * k)), int(len(s) * (0.1 + 0.13 * k) + 10 * sr)
    idx.append(a + int(np.argmax(np.abs(s[a:b]))))
fig, axes = plt.subplots(3, 2, figsize=(16, 10)); axes = axes.ravel()
for ax, i in zip(axes, idx):
    sl = slice(i - int(0.006 * sr), i + int(0.006 * sr))
    t = (np.arange(sl.start, sl.stop) - i) / sr * 1000
    ax.plot(t, s[sl], ".-", ms=2, lw=0.7)
    if th:
        for v in (th, -th):
            ax.axhline(v, color="r", lw=0.5)
    ax.set_title(f"t = {i/sr:.2f} s"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(out, dpi=70)
