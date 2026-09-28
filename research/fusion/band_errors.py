"""Per-band error of each model on the example + short high-frequency error bursts ("clicks")."""
import numpy as np, soundfile as sf
from scipy import signal

D = "F:/BetterDeclipper/research/outputs/fusion/"
y, sr = sf.read("F:/BetterDeclipper/ex_sample/sample_-12dB_clipped.wav", dtype="float64", always_2d=True)
gt, _ = sf.read("F:/BetterDeclipper/ex_sample/sample_ground_truth.wav", dtype="float64", always_2d=True)
pad, _ = sf.read("F:/BetterDeclipper/ex_sample/sample_-12dB_declipped_by_proaudiodeclipper.wav", dtype="float64", always_2d=True)
est = {"input": y, "PAD": pad}
for k in ["fast", "nmf400", "nmf800", "pew400", "spade", "normal"]:
    est[k] = np.load(D + k + ".npy").astype(np.float64)
bands = [(0, 150), (150, 500), (500, 1500), (1500, 4000), (4000, 8000), (8000, 16000), (16000, 22050)]
f, t, G = signal.stft(gt.T, sr, nperseg=2048, noverlap=1536)
Pg = (np.abs(G) ** 2).sum((0, 2))
print("band error (dB relative to ground-truth band energy); lower is better")
print("model    " + " ".join(f"{a:>5d}-{b:<5d}" for a, b in bands) + "   total")
E = {}
for k, x in est.items():
    _, _, X = signal.stft((x - gt).T, sr, nperseg=2048, noverlap=1536)
    Pe = (np.abs(X) ** 2).sum((0, 2))
    E[k] = Pe
    row = [10 * np.log10(Pe[(f >= a) & (f < b)].sum() / Pg[(f >= a) & (f < b)].sum()) for a, b in bands]
    print(f"{k:8s} " + " ".join(f"{v:11.2f}" for v in row) + f"   {10*np.log10(Pe.sum()/Pg.sum()):6.2f}")
print("\nshare of total error energy per band (%)")
for k in ["fast", "nmf400", "spade", "normal"]:
    tot = E[k].sum()
    print(f"{k:8s} " + " ".join(f"{100*E[k][(f >= a) & (f < b)].sum()/tot:11.1f}" for a, b in bands))

# clicks: 1 ms RMS of the >4 kHz error
sos = signal.butter(6, 4000, "hp", fs=sr, output="sos")
w = int(0.001 * sr)
def env(x):
    e = signal.sosfiltfilt(sos, (x - gt), axis=0)
    return np.sqrt(np.convolve((e ** 2).sum(1), np.ones(w) / w, "same"))
envs = {k: env(est[k]) for k in ["fast", "nmf400", "spade", "normal", "PAD"]}
gth = signal.sosfiltfilt(sos, gt, axis=0)
genv = np.sqrt(np.convolve((gth ** 2).sum(1), np.ones(20 * w) / (20 * w), "same"))  # local HF level of the music
clipped = (np.abs(y) >= np.abs(y).max() * 0.999).any(1)
for k in ["normal", "nmf400", "spade"]:
    r = envs[k] / np.maximum(envs["fast"], 1e-6)
    ev = (r > 3.0) & (envs[k] > 0.3 * genv) & (envs[k] > 3e-3)
    # count separate events (>= 5 ms apart)
    idx = np.flatnonzero(ev)
    n_ev = 0 if idx.size == 0 else 1 + int(np.sum(np.diff(idx) > int(0.005 * sr)))
    print(f"\n{k}: HF error bursts >3x fast and above 0.3x local HF music level: {n_ev} events")
    if idx.size:
        starts = np.concatenate([[idx[0]], idx[1:][np.diff(idx) > int(0.005 * sr)]])
        top = sorted(starts, key=lambda i: -envs[k][i])[:8]
        for i in sorted(top):
            j = slice(max(0, i - 2 * w), i + 2 * w)
            print(f"   t={i/sr:7.3f}s  env {k} {envs[k][j].max():.4f} fast {envs['fast'][j].max():.4f} nmf400 {envs['nmf400'][j].max():.4f} spade {envs['spade'][j].max():.4f} PAD {envs['PAD'][j].max():.4f}  music HF {genv[i]:.4f}")
