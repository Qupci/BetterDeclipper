"""Where is the remaining error? Band-wise error energy of an estimate vs ground truth.

python error_analysis.py est.wav [--seg a b]   (compares with ex_sample ground truth + PAD)
"""
import sys, argparse
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from scipy.signal import butter, sosfiltfilt

D = "F:/BetterDeclipper/ex_sample/"
BANDS = [(0, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 22050)]


def band(x, sr, lo, hi):
    if lo <= 0:
        sos = butter(6, hi, "low", fs=sr, output="sos")
    elif hi >= sr / 2:
        sos = butter(6, lo, "high", fs=sr, output="sos")
    else:
        sos = butter(6, [lo, hi], "band", fs=sr, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def analyze(gt, sr, ests):
    tot = {k: np.sum((gt - e) ** 2) for k, e in ests.items()}
    print(f"{'band':>12s} " + " ".join(f"{k:>18s}" for k in ests) + "   (error energy dB rel. gt energy; share of total)")
    E = np.sum(gt ** 2)
    for lo, hi in BANDS:
        row = []
        for k, e in ests.items():
            eb = np.sum(band(gt - e, sr, lo, hi) ** 2)
            row.append(f"{10*np.log10(eb/E):7.1f} ({eb/tot[k]*100:4.1f}%)")
        print(f"{lo:5d}-{hi:<6d} " + " ".join(f"{r:>18s}" for r in row))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("est", nargs="*")
    ap.add_argument("--seg", type=float, nargs=2, default=[5, 10])
    args = ap.parse_args()
    gt, sr = sf.read(D + "sample_ground_truth.wav", dtype="float64")
    cl, _ = sf.read(D + "sample_-12dB_clipped.wav", dtype="float64")
    pad, _ = sf.read(D + "sample_-12dB_declipped_by_proaudiodeclipper.wav", dtype="float64")
    a, b = int(args.seg[0] * sr), int(args.seg[1] * sr)
    ests = {"clipped": cl[a:b], "PAD": pad[a:b]}
    for p in args.est:
        e, _ = sf.read(p, dtype="float64")
        ests[p.split("/")[-1][:18]] = e if len(e) == b - a else e[a:b]
    analyze(gt[a:b], sr, ests)
