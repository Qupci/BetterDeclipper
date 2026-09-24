"""Parameter sweeps on a small dev set (5 s excerpts of 4 cases, 44.1 + 48 kHz).

python devsweep.py --method social --name NAME --grid '{"param":[...], ...}'
Configs may use "win_ms" (window in ms, converted per sample rate) and "hop_div".
Results appended to research/results/dev_<name>.jsonl.
"""
import sys, time, json, itertools, argparse
sys.path.insert(0, "F:/BetterDeclipper")
sys.path.insert(0, "F:/BetterDeclipper/research")
import numpy as np, soundfile as sf
from bench_testset import list_cases, nice_len
from betterdeclipper.detect import detect_clip_levels, clip_masks, estimate_lsb
from betterdeclipper.metrics import sdr

DEV = ["example", "merlon48_-12", "lofi48_-12", "ghostpage48_-6"]
SEG = (5.0, 10.0)


def load_dev():
    cases = list_cases()
    out = []
    for name in DEV:
        gt_p, cl_p, th, pad_p = cases[name]
        gt, sr = sf.read(gt_p, dtype="float64", always_2d=True)
        cl, _ = sf.read(cl_p, dtype="float64", always_2d=True)
        a, b = int(SEG[0] * sr), int(SEG[1] * sr)
        pad = sf.read(pad_p, dtype="float64", always_2d=True)[0][a:b] if pad_p else None
        out.append((name, gt[a:b], cl[a:b], th, sr, pad))
    return out


def run_config(method, kw, cl, sr):
    levels = detect_clip_levels(cl, estimate_lsb(cl))
    m_hi, m_lo, th_hi, th_lo = clip_masks(cl, levels)
    kw = dict(kw)
    if "win_ms" in kw:
        wl = nice_len(kw.pop("win_ms") * 1e-3 * sr)
        kw["win_len"] = wl
        kw["hop"] = wl // kw.pop("hop_div", 4)
    if method == "social":
        from betterdeclipper.methods.social import declip_social
        return declip_social(cl, m_hi, m_lo, th_hi, th_lo, **kw)
    if method == "pnp":
        from betterdeclipper.methods.pnp import declip_pnp
        return declip_pnp(cl, m_hi, m_lo, th_hi, th_lo, sr=sr, **kw)
    raise ValueError(method)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="social")
    ap.add_argument("--grid", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    grid = json.loads(args.grid)
    keys = list(grid)
    configs = [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]
    dev = load_dev()
    for kw in configs:
        t0 = time.time()
        res = {}
        for name, gt, cl, th, sr, pad in dev:
            est = run_config(args.method, kw, cl, sr)
            res[name] = (sdr(gt, est), sdr(gt, cl))
        md = np.mean([a - b for a, b in res.values()])
        r = dict(kw=kw, method=args.method, mean_dsdr=md, per_case={k: round(v[0], 3) for k, v in res.items()},
                 time=time.time() - t0)
        print(f"{json.dumps(kw)}  mean dSDR {md:.3f}  " + "  ".join(f"{k} {v[0]:.2f}" for k, v in res.items())
              + f"  ({r['time']:.0f}s)", flush=True)
        with open(f"F:/BetterDeclipper/research/results/dev_{args.name}.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
