"""Parameter sweeps on the example excerpt; results appended to research/results/<name>.jsonl."""
import sys, time, json, itertools, argparse
sys.path.insert(0, "F:/BetterDeclipper")
sys.path.insert(0, "F:/BetterDeclipper/research")
import numpy as np
from bench_example import load, evaluate
from betterdeclipper.detect import detect_clip_levels, clip_masks, estimate_lsb
from betterdeclipper.metrics import sdr


def run(method, configs, seg, name):
    gt, cl, pad, sr = load(seg)
    levels = detect_clip_levels(cl, estimate_lsb(cl))
    m_hi, m_lo, th_hi, th_lo = clip_masks(cl, levels)
    tmask = np.abs(gt) > 0.25
    print("PAD:", evaluate(gt, cl, pad, tmask), flush=True)
    for kw in configs:
        t0 = time.time()
        best = [-1e9, -1]
        def cb(it, xr):
            s = sdr(gt, xr)
            if s > best[0]:
                best[0], best[1] = s, it + 1
        if method in ("social",):
            from betterdeclipper.methods.social import declip_social
            est = declip_social(cl, m_hi, m_lo, th_hi, th_lo, callback=cb, **kw)
        elif method == "multires":
            from betterdeclipper.methods.multires import declip_multires
            kw2 = dict(kw)
            kw2["frames"] = [(f[0], f[1], tuple(f[2])) for f in kw2["frames"]]
            est = declip_multires(cl, m_hi, m_lo, th_hi, th_lo, callback=cb, **kw2)
        elif method == "pnp":
            from betterdeclipper.methods.pnp import declip_pnp
            est = declip_pnp(cl, m_hi, m_lo, th_hi, th_lo, callback=cb, **kw)
        elif method == "spade":
            from betterdeclipper.methods.spade import declip_spade
            est = declip_spade(cl, m_hi, m_lo, th_hi, th_lo, verbose=True, **kw)
        r = evaluate(gt, cl, est, tmask)
        r.update(kw=kw, best_sdr=best[0], best_it=best[1], time=time.time() - t0, seg=seg)
        print(json.dumps(r), flush=True)
        with open(f"F:/BetterDeclipper/research/results/{name}.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--method", default="social")
    ap.add_argument("--seg", type=float, nargs=2, default=[5, 10])
    args = ap.parse_args()
    grid = json.loads(args.grid)
    keys = list(grid.keys())
    configs = [dict(zip(keys, v)) for v in itertools.product(*[grid[k] for k in keys])]
    # derived params
    for c in configs:
        if "hop_div" in c:
            c["hop"] = c["win_len"] // c.pop("hop_div")
        if "neigh" in c:
            c["neigh"] = tuple(c["neigh"])
    run(args.method, configs, args.seg, args.name)
