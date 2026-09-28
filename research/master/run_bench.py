"""Run declipping strategies on the master benchmark and store SDRs.

python run_bench.py OUTNAME strategy [strategy ...] [--preset fast] [--degs a,b] [--srcs a,b]
strategy: auto | hard | soft | soft:<frac of peak> | any name registered in STRATEGIES below
Results are merged into research/results/mbench_<OUTNAME>.json as {case: {strategy: sdr}}.
"""
import sys, os, json, time, argparse
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.engine import declip, default_knees
from betterdeclipper.metrics import sdr

B = "F:/BetterDeclipper/research/outputs/mbench/"
STRATEGIES = {}
sys.path.insert(0, "F:/BetterDeclipper/research/master")
import strategies as _st
_st.register(STRATEGIES); _st.register_prox(STRATEGIES); _st.register_env(STRATEGIES); _st.register_raise(STRATEGIES); _st.register_dip(STRATEGIES)


def run_strategy(name, y, sr, preset):
    if name in STRATEGIES:
        return STRATEGIES[name](y, sr, preset)
    if name.startswith("soft:"):
        fr = float(name.split(":")[1])
        return declip(y, sr, preset=preset, mode="soft", knees=default_knees(y, None, frac=fr), verbose=False)
    return declip(y, sr, preset=preset, mode=name, verbose=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("strategies", nargs="+")
    ap.add_argument("--preset", default="fast"); ap.add_argument("--degs", default=None); ap.add_argument("--srcs", default=None)
    a = ap.parse_args(argv)
    meta = json.load(open(B + "meta.json"))
    res_path = f"F:/BetterDeclipper/research/results/mbench_{a.out}.json"
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    degs = a.degs.split(",") if a.degs else None
    srcs = a.srcs.split(",") if a.srcs else None
    for key, m in meta.items():
        if (degs and m["deg"] not in degs) or (srcs and m["src"] not in srcs):
            continue
        y, sr = sf.read(B + key + ".wav", dtype="float64", always_2d=True)
        gt, _ = sf.read(B + key + "__gt.wav", dtype="float64", always_2d=True)
        r = res.setdefault(key, {"in": m["sdr_in"]})
        line = []
        for s in a.strategies:
            t = time.time()
            x, info = run_strategy(s, y, sr, a.preset)
            r[s] = float(sdr(gt, x))
            r[s + "#flag"] = float(info["clipped_frac"])
            r[s + "#mode"] = str(info.get("mode"))
            if "raise" in info:
                r[s + "#raise"] = info["raise"]
            line.append(f"{s} {r[s]:.2f} ({info['mode']},{info['clipped_frac']*100:.1f}%)")
        print(f"{key:24s} in {m['sdr_in']:6.2f} | " + " | ".join(line), flush=True)
        json.dump(res, open(res_path, "w"), indent=1)


if __name__ == "__main__":
    main()
