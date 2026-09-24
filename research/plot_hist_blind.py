import numpy as np, soundfile as sf, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
B = "F:/BetterDeclipper/ProAudioDeclipper/blind_examples/"
fig, axs = plt.subplots(3, 2, figsize=(15, 11))
for row, f in enumerate(["greenday-cd.wav", "metallica-cd.wav", "scar_tissue-cd.wav"]):
    x, sr = sf.read(B + f, dtype="float64", always_2d=True)
    for c in range(2):
        ax = axs[row, c]
        v = np.abs(x[:, c])
        pk = v.max()
        h, e = np.histogram(v, bins=400, range=(0, pk))
        ax.semilogy(e[:-1] / pk, h + 1, lw=1)
        ax.set_title(f"{f} ch{c}: |x| histogram (x / peak {pk:.4f})")
        ax.set_xlim(0.3, 1.01)
        ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("F:/BetterDeclipper/research/plots/blind_hist.png", dpi=65)
