"""Benchmark declipping methods on the provided -12 dB example (ground truth available)."""
import sys, time, json, argparse
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.detect import detect_clip_levels, clip_masks, estimate_lsb
from betterdeclipper.metrics import sdr

D = "F:/BetterDeclipper/ex_sample/"


def load(seg=None):
    gt, sr = sf.read(D + "sample_ground_truth.wav", dtype="float64")
    cl, _ = sf.read(D + "sample_-12dB_clipped.wav", dtype="float64")
    pad, _ = sf.read(D + "sample_-12dB_declipped_by_proaudiodeclipper.wav", dtype="float64")
    if seg is not None:
        a, b = int(seg[0] * sr), int(seg[1] * sr)
        gt, cl, pad = gt[a:b], cl[a:b], pad[a:b]
    return gt, cl, pad, sr


def evaluate(gt, cl, est, tmask):
    return dict(sdr=sdr(gt, est), sdr_c=sdr(gt[tmask], est[tmask]), dsdr=sdr(gt, est) - sdr(gt, cl))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=float, nargs=2, default=None)
    ap.add_argument("--method", default="social")
    ap.add_argument("--kw", default="{}")
    args = ap.parse_args()
    gt, cl, pad, sr = load(args.seg)
    levels = detect_clip_levels(cl, estimate_lsb(cl))
    print("detected levels:", [(round(a * 32768, 1), round(b * 32768, 1)) for a, b in levels])
    m_hi, m_lo, th_hi, th_lo = clip_masks(cl, levels)
    tmask = np.abs(gt) > 0.25
    print(f"detected clipped: {np.mean(m_hi | m_lo)*100:.2f}%  true clipped: {tmask.mean()*100:.2f}%")
    print("PAD:", evaluate(gt, cl, pad, tmask))
    kw = json.loads(args.kw)
    t0 = time.time()
    if args.method == "social":
        from betterdeclipper.methods.social import declip_social
        cb = lambda it, xr: print(f"  it {it+1}: sdr {sdr(gt, xr):.3f}")
        est = declip_social(cl, m_hi, m_lo, th_hi, th_lo, callback=cb, **kw)
    print(f"{args.method} {kw}: {evaluate(gt, cl, est, tmask)}  time {time.time()-t0:.1f}s")
