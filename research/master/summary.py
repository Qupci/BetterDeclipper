"""Headline table: per degradation mean SDR - input, old auto, new auto, oracle (best fixed strategy)."""
import json, sys
import numpy as np
R = "F:/BetterDeclipper/research/results/"
base = json.load(open(R + "mbench_base.json"))
new = json.load(open(R + f"mbench_{sys.argv[1] if len(sys.argv) > 1 else 'newauto'}.json"))
extra = {}
for f in ("mbench_smear.json", "mbench_raise.json"):
    try:
        for k, v in json.load(open(R + f)).items():
            extra.setdefault(k, {}).update({s: x for s, x in v.items() if "#" not in s and s != "in"})
    except FileNotFoundError:
        pass
rows = {}
for k, v in new.items():
    if "auto" not in v or k not in base:
        continue
    b = base[k]
    cands = [b[s] for s in ["hard", "soft:0.5", "soft:0.6", "soft:0.7", "soft", "soft:0.9"] if s in b] + list(extra.get(k, {}).values())
    rows.setdefault(k.split("__")[1], []).append((v["in"], b["auto"], v["auto"], max(cands + [v["auto"]])))
print(f"{'degradation':12s} {'n':>2s} {'input':>7s} {'old auto':>9s} {'new auto':>9s} {'oracle':>7s} {'new-old':>8s}")
allr = []
for d, rs in sorted(rows.items()):
    a = np.array(rs); allr += rs
    print(f"{d:12s} {len(rs):2d} {a[:,0].mean():7.2f} {a[:,1].mean():9.2f} {a[:,2].mean():9.2f} {a[:,3].mean():7.2f} {a[:,2].mean()-a[:,1].mean():+8.2f}")
a = np.array(allr)
print(f"{'ALL':12s} {len(allr):2d} {a[:,0].mean():7.2f} {a[:,1].mean():9.2f} {a[:,2].mean():9.2f} {a[:,3].mean():7.2f} {a[:,2].mean()-a[:,1].mean():+8.2f}")
