"""Offline evaluation of knee rules from the saved first-pass raise curves.

SDR at an arbitrary knee is interpolated from the base sweep (hard ~ knee 1.0, soft:0.9 ... soft:0.5).
Prints per-degradation mean SDR of each rule vs the oracle (best sweep point) and vs current auto.
"""
import json, sys
import numpy as np

sys.path.insert(0, "F:/BetterDeclipper/research/master")
R = "F:/BetterDeclipper/research/results/"
base = json.load(open(R + "mbench_base.json"))
rai = json.load(open(R + "mbench_raise.json"))
LV = np.arange(0.45, 1.0001, 0.025)


def sdr_at(b, k):
    """interpolate SDR at knee k (fraction of peak) from the sweep; k >= 0.995 -> hard"""
    pts = [(0.5, b.get("soft:0.5")), (0.6, b.get("soft:0.6")), (0.7, b.get("soft:0.7")), (0.8, b.get("soft")),
           (0.9, b.get("soft:0.9")), (1.0, b.get("hard"))]
    pts = [(x, v) for x, v in pts if v is not None]
    xs, vs = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    return float(np.interp(min(max(k, 0.5), 1.0), xs, vs))


def rule_excess(cur, delta=0.01, base_rng=(0.45, 0.6)):
    ok = np.isfinite(cur)
    bm = ok & (LV >= base_rng[0] - 1e-9) & (LV <= base_rng[1] + 1e-9)
    b1, b0 = np.polyfit(LV[bm], cur[bm], 1)
    exc = cur - (b0 + b1 * LV)
    knee = None
    for i in np.flatnonzero(ok)[::-1]:
        if exc[i] > delta:
            knee = LV[i]
        else:
            break
    return 1.0 if knee is None or knee >= 0.975 else knee


def rule_convex(cur, frac=0.25, acc_min=0.1, top=0.95):
    """knee where the slope of the (smoothed) raise curve has risen by `frac` of its total rise from the
    minimum slope; no knee if the top slope exceeds the minimum by less than acc_min (per unit level)."""
    ok = np.isfinite(cur) & (LV <= top + 1e-9)
    c = np.array(cur)[ok]; lv = LV[ok]
    if c.size < 6:
        return 1.0
    cs = np.convolve(np.pad(c, 1, mode="edge"), np.ones(3) / 3, "valid")
    sl = np.gradient(cs, lv)
    i0 = int(np.argmin(sl[: max(2, len(sl) - 3)]))
    s_top = sl[-2:].mean()
    if s_top - sl[i0] < acc_min:
        return 1.0
    thr = sl[i0] + frac * (s_top - sl[i0])
    after = np.flatnonzero((np.arange(len(sl)) >= i0) & (sl >= thr))
    return float(lv[after[0]]) if after.size else 1.0


RULES = {f"excess{d}": (lambda d: lambda c: rule_excess(c, d))(d) for d in (0.005, 0.01, 0.02, 0.03)}
for fr in (0.1, 0.2, 0.3, 0.5):
    for am in (0.05, 0.1, 0.2):
        RULES[f"convex{fr}/{am}"] = (lambda fr, am: lambda c: rule_convex(c, fr, am))(fr, am)

if __name__ == "__main__":
    rows = {}
    for key, v in rai.items():
        cur = v.get("raise:0.01#raise")
        b = base.get(key)
        if cur is None or b is None:
            continue
        cur = np.array([np.nan if x is None else x for x in cur], float)
        deg = key.split("__")[1]
        oracle = max(b[s] for s in ["hard", "soft:0.5", "soft:0.6", "soft:0.7", "soft", "soft:0.9"])
        r = dict(oracle=oracle, auto=b["auto"], inp=b["in"])
        for name, fn in RULES.items():
            k = fn(cur)
            r[name] = sdr_at(b, k)
            r[name + "#k"] = k
        for s in ("raise:0.005", "raise:0.01", "raise:0.02", "smear6"):
            if s in v:
                r["RUN " + s] = v[s]
        rows.setdefault(deg, []).append(r)
    names = ["inp", "auto", "oracle", "RUN raise:0.01", "RUN smear6"] + list(RULES)
    print(f"{'deg':11s} " + " ".join(f"{n[:13]:>13s}" for n in names))
    allv = {n: [] for n in names}
    for deg, rs in sorted(rows.items()):
        line = []
        for n in names:
            vals = [r[n] for r in rs if n in r]
            allv[n] += [r[n] - r["oracle"] for r in rs if n in r]
            line.append(f"{np.mean(vals):13.2f}" if vals else f"{'-':>13s}")
        print(f"{deg:11s} " + " ".join(line))
    print(f"{'mean-oracle':11s} " + " ".join(f"{np.mean(allv[n]):13.2f}" if allv[n] else f"{'-':>13s}" for n in names))
