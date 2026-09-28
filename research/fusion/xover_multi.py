"""Two-band fusion validation on 22 hard-clip cases (48 kHz test set + master-benchmark hard6/hard9)."""
import sys, os, glob
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.engine import fuse
from betterdeclipper.metrics import sdr
from betterdeclipper.detect import detect_clip_levels, clip_masks

D = "F:/BetterDeclipper/research/outputs/fusion/multi/"
cases = []
for t in ["gris48", "merlon48", "lofi48", "ghostpage48"]:
    for lv in ["-12dB", "-6dB"]:
        cases.append((f"{t}_{lv}", f"F:/BetterDeclipper/testset/{t}/clipped_{lv}.wav", f"F:/BetterDeclipper/testset/{t}/ground_truth.wav"))
B = "F:/BetterDeclipper/research/outputs/mbench/"
for s in ["ex", "petal", "ruder", "thrash", "violet", "bigshot", "knife"]:
    for d in ["hard6", "hard9"]:
        cases.append((f"{s}_{d}", B + f"{s}__{d}.wav", B + f"{s}__{d}__gt.wav"))
variants = {
    "normal plain": ([0.65, 0.35], [0.65, 0.35], None),
}
for fc in (3000, 4000, 6000, 8000):
    for wl in (0.35, 0.45):
        variants[f"normal 2band fc{fc} lo{wl}"] = ([1 - wl, wl], [1.0, 0.0], fc)
variants3 = {"high plain": ([1, 1, 1], [1, 1, 1], None)}
for fc in (4000, 6000, 8000):
    variants3[f"high 2band fc{fc} hf nmf"] = ([1, 1, 1], [1, 0, 0], fc)
    variants3[f"high 2band fc{fc} hf nmf+pew"] = ([1, 1, 1], [1, 1, 0], fc)
res = {k: [] for k in list(variants) + list(variants3)}
for name, yp, gp in cases:
    if not os.path.exists(D + f"{name}__nmf400.npy"):
        continue
    y, sr = sf.read(yp, dtype="float64", always_2d=True)
    gt, _ = sf.read(gp, dtype="float64", always_2d=True)
    E = {k: np.load(D + f"{name}__{k}.npy").astype(np.float64) for k in ["nmf400", "spade", "pew400"]}
    lv = detect_clip_levels(y)
    m_hi, m_lo, th_hi, th_lo = clip_masks(y, lv)
    cl = m_hi | m_lo

    def proj(x):
        x = x.copy(); x[~cl] = y[~cl]
        x = np.where(m_hi, np.maximum(x, th_hi[None, :]), x)
        return np.where(m_lo, np.minimum(x, th_lo[None, :]), x)
    for k, (wl, wh, fc) in variants.items():
        res[k].append(sdr(gt, proj(fuse(y, [E["nmf400"], E["spade"]], wl, wh, sr, fc or 6000))))
    for k, (wl, wh, fc) in variants3.items():
        res[k].append(sdr(gt, proj(fuse(y, [E["nmf400"], E["pew400"], E["spade"]], wl, wh, sr, fc or 6000))))
    print(name, f"normal plain {res['normal plain'][-1]:.3f}  2band 6k {res['normal 2band fc6000 lo0.35'][-1]:.3f}", flush=True)
n = len(res["normal plain"])
print(f"\n{n} cases; mean SDR and mean delta vs plain")
for group, ref in ((variants, "normal plain"), (variants3, "high plain")):
    for k in group:
        d = np.array(res[k]) - np.array(res[ref])
        print(f"{k:32s} {np.mean(res[k]):7.3f}  delta {d.mean():+.3f}  min {d.min():+.3f}  max {d.max():+.3f}")
