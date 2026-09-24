"""Analyze the provided -12 dB example: clip level, clipped fraction, and PAD's restoration quality."""
import numpy as np, soundfile as sf

D = "F:/BetterDeclipper/ex_sample/"
gt, sr = sf.read(D + "sample_ground_truth.wav", dtype="float64")
cl, _ = sf.read(D + "sample_-12dB_clipped.wav", dtype="float64")
pad, _ = sf.read(D + "sample_-12dB_declipped_by_proaudiodeclipper.wav", dtype="float64")
dgt, _ = sf.read(D + "sample_-12dB_clipped_delta_ground_truth.wav", dtype="float64")
dpad, _ = sf.read(D + "sample_-12dB_clipped_delta_proaudiodeclipper.wav", dtype="float64")

def db(x): return 20 * np.log10(np.maximum(x, 1e-12))
def sdr(ref, est): return 10 * np.log10(np.sum(ref**2) / np.sum((ref - est) ** 2))

print("sr", sr, "shape", gt.shape)
for name, x in [("gt", gt), ("clipped", cl), ("pad", pad), ("delta_gt", dgt), ("delta_pad", dpad)]:
    print(f"{name:10s} peak {db(np.abs(x).max()):7.2f} dBFS  rms {db(np.sqrt(np.mean(x**2))):7.2f} dBFS  max+ {x.max(axis=0)} min- {x.min(axis=0)}")

# clip level detection
for ch in range(2):
    c = cl[:, ch]
    mx, mn = c.max(), c.min()
    print(f"ch{ch}: clip+ {mx:.6f} ({db(mx):.3f} dB) clip- {mn:.6f}  n+ {np.sum(c >= mx)} n- {np.sum(c <= mn)}  frac {(np.sum(c >= mx) + np.sum(c <= mn)) / len(c) * 100:.2f}%")

# Relationship between gt and clipped on the reliable part
th = np.abs(cl).max()
mask = np.abs(cl) < th - 1e-9
print("reliable-part max abs diff clipped vs gt:", np.abs(cl[mask] - gt[mask]).max())
print("clip threshold vs gt: is clipped == clip(gt, th)?", np.abs(np.clip(gt, -th, th) - cl).max())

# SDR metrics
print(f"SDR clipped vs gt : {sdr(gt, cl):.3f} dB")
print(f"SDR pad vs gt     : {sdr(gt, pad):.3f} dB")
m = ~mask
print(f"SDR on clipped samples only: clipped {sdr(gt[m], cl[m]):.3f} dB  pad {sdr(gt[m], pad[m]):.3f} dB")
print("PAD changes on reliable samples: max abs diff", np.abs(pad[mask] - cl[mask]).max(), " rms", np.sqrt(np.mean((pad[mask] - cl[mask]) ** 2)))
# check deltas are what they say
print("delta_gt == gt - clipped?", np.abs(dgt - (gt - cl)).max(), " or clipped-gt?", np.abs(dgt - (cl - gt)).max())
print("delta_pad == pad - clipped?", np.abs(dpad - (pad - cl)).max(), " or clipped-pad?", np.abs(dpad - (cl - pad)).max())
# gain/delay check between pad and gt
for lag in range(-3, 4):
    a = gt[10000:-10000, 0]; b = np.roll(pad[:, 0], lag)[10000:-10000]
    print("lag", lag, "corr", np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))
# is PAD output consistent (clipped samples >= threshold)?
over = np.abs(pad[m]) >= th - 1e-4
print("PAD clipped-sample estimates with |x|>=th:", over.mean() * 100, "%")
