"""Headline: new auto vs previous auto (legacy) vs oracle (best of the knee sweep) on the master benchmark."""
import sys, json, os, time, argparse
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.engine import declip
from betterdeclipper.metrics import sdr

B = "F:/BetterDeclipper/research/outputs/mbench/"
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--preset", default="fast"); ap.add_argument("--modes", default="auto")
ap.add_argument("--degs", default=None); ap.add_argument("--srcs", default=None)
a = ap.parse_args()
meta = json.load(open(B + "meta.json"))
path = f"F:/BetterDeclipper/research/results/mbench_{a.out}.json"
res = json.load(open(path)) if os.path.exists(path) else {}
for key, m in meta.items():
    if (a.degs and m["deg"] not in a.degs.split(",")) or (a.srcs and m["src"] not in a.srcs.split(",")):
        continue
    y, sr = sf.read(B + key + ".wav", dtype="float64", always_2d=True)
    gt, _ = sf.read(B + key + "__gt.wav", dtype="float64", always_2d=True)
    r = res.setdefault(key, {"in": m["sdr_in"]})
    line = []
    for mode in a.modes.split(","):
        x, info = declip(y, sr, preset=a.preset, mode=mode, verbose=False)
        r[mode] = float(sdr(gt, x)); r[mode + "#mode"] = info["mode"]; r[mode + "#flag"] = info["clipped_frac"]
        line.append(f"{mode} {r[mode]:.2f} [{info['mode']}] {info['clipped_frac']*100:.1f}% {info['time']:.0f}s")
    print(f"{key:24s} in {m['sdr_in']:6.2f} | " + " | ".join(line), flush=True)
    json.dump(res, open(path, "w"), indent=1)
