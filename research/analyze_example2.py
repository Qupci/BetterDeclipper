import numpy as np, soundfile as sf
D = "F:/BetterDeclipper/ex_sample/"
gt, sr = sf.read(D + "sample_ground_truth.wav", dtype="float64")
cl, _ = sf.read(D + "sample_-12dB_clipped.wav", dtype="float64")
pad, _ = sf.read(D + "sample_-12dB_declipped_by_proaudiodeclipper.wav", dtype="float64")
clq = np.round(cl * 32768).astype(int)
vals, cnts = np.unique(clq, return_counts=True)
sel = np.abs(vals) >= 8180
print("value counts near clip level:")
for v, c in zip(vals[sel], cnts[sel]): print(v, c)
# Where gt exceeds the 0.25 threshold
th = 0.25
truly_clipped = np.abs(gt) > th
print("fraction of samples with |gt|>0.25:", truly_clipped.mean() * 100, "%")
# distribution of clipped-signal values where gt is clipped
print("clq values where gt clipped (pos):", np.unique(clq[gt > th], return_counts=True))
print("clq values where gt clipped (neg):", np.unique(clq[gt < -th], return_counts=True))
# is gt quantized? check gt*32768 fractional part
fr = gt * 32768 - np.round(gt * 32768)
print("gt fractional part std (0 if 16-bit):", fr.std())
# noise between clipped and gt on unclipped samples (dither?)
u = ~truly_clipped
e = (cl - gt)[u] * 32768
print("unclipped diff in LSB: mean", e.mean(), "std", e.std(), "min", e.min(), "max", e.max())
def sdr(ref, est): return 10 * np.log10(np.sum(ref**2) / np.sum((ref - est) ** 2))
m = truly_clipped
print(f"SDR full: clipped {sdr(gt, cl):.3f}  pad {sdr(gt, pad):.3f}")
print(f"SDR on truly-clipped samples: clipped {sdr(gt[m], cl[m]):.3f}  pad {sdr(gt[m], pad[m]):.3f}")
print("PAD change on unclipped samples (LSB): rms", np.sqrt(np.mean(((pad - cl)[u] * 32768) ** 2)), "max", np.abs((pad - cl)[u] * 32768).max())
print("PAD error on unclipped samples vs gt (LSB): rms", np.sqrt(np.mean(((pad - gt)[u] * 32768) ** 2)))
# per-channel
for ch in range(2):
    mm = m[:, ch]
    print(f"ch{ch} clipped frac {mm.mean()*100:.2f}%  SDR clipped {sdr(gt[:,ch], cl[:,ch]):.2f} pad {sdr(gt[:,ch], pad[:,ch]):.2f}")
# Error energy share: clipped region vs unclipped
err = pad - gt
print("PAD error energy fraction in clipped region:", np.sum(err[m]**2) / np.sum(err**2))
# does PAD produce values within the clip region that are below threshold?
print("PAD |x| < 0.25 at clipped positions:", np.mean(np.abs(pad[m]) < th) * 100, "%")
# run lengths of clipped segments
lens = []
for ch in range(2):
    mm = m[:, ch].astype(int)
    d = np.diff(np.concatenate([[0], mm, [0]]))
    s = np.where(d == 1)[0]; e_ = np.where(d == -1)[0]
    lens += list(e_ - s)
lens = np.array(lens)
print("clipped runs:", len(lens), "mean len", lens.mean(), "median", np.median(lens), "max", lens.max(), "p90", np.percentile(lens, 90), "p99", np.percentile(lens, 99))
