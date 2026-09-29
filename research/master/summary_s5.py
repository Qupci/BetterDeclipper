"""Session 5 table: per degradation mean SDR - input, old auto (before session 4), session-4 auto (limiter knee),
session-5 auto (faithful limiter handling), --mode limiter (experimental), and the oracle of summary.py."""
import json
import numpy as np
R = "F:/BetterDeclipper/research/results/"
base = json.load(open(R + "mbench_base.json"))
s4 = json.load(open(R + "mbench_final2.json"))
s5 = json.load(open(R + "mbench_s5.json"))
extra = {}
for f in ("mbench_smear.json", "mbench_raise.json"):
    for k, v in json.load(open(R + f)).items():
        extra.setdefault(k, {}).update({s: x for s, x in v.items() if "#" not in s and s != "in"})
rows = {}
worse = []
for k, v in s5.items():
    if "auto" not in v or k not in base or k not in s4:
        continue
    b = base[k]
    cands = [b[s] for s in ["hard", "soft:0.5", "soft:0.6", "soft:0.7", "soft", "soft:0.9"] if s in b] + \
        list(extra.get(k, {}).values()) + [s4[k]["auto"], v["auto"]] + ([v["limiter"]] if "limiter" in v else [])
    rows.setdefault(k.split("__")[1], []).append((v["in"], b["auto"], s4[k]["auto"], v["auto"],
                                                  v.get("limiter", np.nan), max(cands)))
    if v["auto"] < b["auto"] - 0.01:
        worse.append((k, b["auto"], v["auto"]))
print(f"{'degradation':12s} {'n':>2s} {'input':>7s} {'old':>7s} {'s4 auto':>8s} {'s5 auto':>8s} {'limiter':>8s} {'oracle':>7s}")
allr = []
for d, rs in sorted(rows.items()):
    a = np.array(rs); allr += rs
    lim = f"{np.nanmean(a[:,4]):8.2f}" if np.isfinite(a[:, 4]).any() else f"{'-':>8s}"
    print(f"{d:12s} {len(rs):2d} {a[:,0].mean():7.2f} {a[:,1].mean():7.2f} {a[:,2].mean():8.2f} {a[:,3].mean():8.2f} {lim} {a[:,5].mean():7.2f}")
a = np.array(allr)
print(f"{'ALL':12s} {len(allr):2d} {a[:,0].mean():7.2f} {a[:,1].mean():7.2f} {a[:,2].mean():8.2f} {a[:,3].mean():8.2f} {'':8s} {a[:,5].mean():7.2f}")
print("s5 auto worse than old auto:", worse or "none")
diff = [(k, s4[k]["auto"], s5[k]["auto"], s4[k].get("auto#mode"), s5[k].get("auto#mode")) for k in s5
        if k in s4 and "auto" in s5[k] and abs(s5[k]["auto"] - s4[k]["auto"]) > 0.005]
print("changed vs session 4:")
for r in diff:
    print(f"  {r[0]:22s} {r[1]:6.2f} -> {r[2]:6.2f}   {r[3]} -> {r[4]}")
