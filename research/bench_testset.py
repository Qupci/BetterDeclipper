"""Evaluate a declipping method on the example + the external 48 kHz test set.

Usage: python bench_testset.py --method social --kw '{"win_ms":93, ...}' [--cases example,gris48_-12,...] [--tag name]
Results are appended to research/results/testset.jsonl.
"""
import sys, os, time, json, glob, argparse
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.detect import detect_clip_levels, clip_masks, estimate_lsb
from betterdeclipper.metrics import sdr

ROOT = "F:/BetterDeclipper"


def list_cases():
    cases = {"example": (ROOT + "/ex_sample/sample_ground_truth.wav", ROOT + "/ex_sample/sample_-12dB_clipped.wav", 10 ** (-12 / 20),
                         ROOT + "/ex_sample/sample_-12dB_declipped_by_proaudiodeclipper.wav")}
    for d in sorted(glob.glob(ROOT + "/testset/*/")):
        name = os.path.basename(d[:-1])
        for cdb in (-12, -6):
            pad = d + f"clipped_{cdb}dB_pad.wav"
            cases[f"{name}_{cdb}"] = (d + "ground_truth.wav", d + f"clipped_{cdb}dB.wav", 10 ** (cdb / 20),
                                      pad if os.path.exists(pad) else None)
    return cases


def nice_len(n):
    """Nearest FFT-friendly length (2^a * 3^b, even) to n."""
    best = None
    for a in range(1, 20):
        for b in range(0, 4):
            v = 2 ** a * 3 ** b
            if best is None or abs(v - n) < abs(best - n):
                best = v
    return best


def run_method(method, cl, sr, kw):
    levels = detect_clip_levels(cl, estimate_lsb(cl))
    m_hi, m_lo, th_hi, th_lo = clip_masks(cl, levels)
    kw = dict(kw)
    if "win_ms" in kw:
        wl = nice_len(kw.pop("win_ms") * 1e-3 * sr)
        kw["win_len"] = wl
        kw["hop"] = wl // kw.pop("hop_div", 4)
    if method == "social":
        from betterdeclipper.methods.social import declip_social
        return declip_social(cl, m_hi, m_lo, th_hi, th_lo, **kw), (m_hi | m_lo)
    raise ValueError(method)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="social")
    ap.add_argument("--kw", default="{}")
    ap.add_argument("--cases", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    kw = json.loads(args.kw)
    cases = list_cases()
    sel = args.cases.split(",") if args.cases else list(cases)
    rows = []
    for name in sel:
        gt_p, cl_p, th, pad_p = cases[name]
        gt, sr = sf.read(gt_p, dtype="float64", always_2d=True)
        cl, _ = sf.read(cl_p, dtype="float64", always_2d=True)
        t0 = time.time()
        est, dm = run_method(args.method, cl, sr, kw)
        el = time.time() - t0
        tm = np.abs(gt) > th
        r = dict(case=name, sr=sr, sdr_in=sdr(gt, cl), sdr=sdr(gt, est), sdr_c=sdr(gt[tm], est[tm]), time=el,
                 det_frac=float(dm.mean()), true_frac=float(tm.mean()))
        r["dsdr"] = r["sdr"] - r["sdr_in"]
        if pad_p:
            pad, _ = sf.read(pad_p, dtype="float64", always_2d=True)
            r["pad_sdr"] = sdr(gt, pad)
        r.update(method=args.method, kw=kw, tag=args.tag)
        rows.append(r)
        print(f"{name:16s} in {r['sdr_in']:6.2f}  out {r['sdr']:6.2f}  d {r['dsdr']:6.2f}  c {r['sdr_c']:6.2f}"
              + (f"  PAD {r['pad_sdr']:6.2f}" if 'pad_sdr' in r else "") + f"  ({el:.0f}s)", flush=True)
        with open(ROOT + "/research/results/testset.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        if args.save:
            od = ROOT + "/research/outputs/" + (args.tag or args.method)
            os.makedirs(od, exist_ok=True)
            sf.write(f"{od}/{name}.wav", est.astype(np.float32), sr, subtype="FLOAT")
    print(f"MEAN dSDR {np.mean([r['dsdr'] for r in rows]):.3f}  MEAN SDR {np.mean([r['sdr'] for r in rows]):.3f}")
