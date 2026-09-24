import sys, itertools; sys.path.insert(0, "F:/BetterDeclipper"); sys.path.insert(0, "F:/BetterDeclipper/research")
import numpy as np, soundfile as sf
from bench_example import load
from betterdeclipper.detect import detect_clip_levels, clip_masks, estimate_lsb
from betterdeclipper.methods.pnp import declip_pnp
from betterdeclipper.metrics import sdr
gt, cl, pad, sr = load((5, 10))
lv = detect_clip_levels(cl, estimate_lsb(cl)); m_hi, m_lo, th_hi, th_lo = clip_masks(cl, lv)
outs = {}
for w in (2048, 4096, 8192):
    outs[f"pnp{w}"] = declip_pnp(cl, m_hi, m_lo, th_hi, th_lo, win_len=w, hop=w // 4, neigh=(3, 7), n_iter=400, stereo="pca", chan_gain=[1, 2.5])
    sf.write(f"F:/BetterDeclipper/research/outputs/pnp{w}_cg25_5_10.wav", outs[f"pnp{w}"].astype(np.float32), sr, subtype="FLOAT")
outs["PAD"] = pad
for k, v in outs.items():
    print(f"{k:10s} {sdr(gt, v):.3f}")
for r in (2, 3):
    for combo in itertools.combinations([k for k in outs if k != "PAD"], r):
        print(" + ".join(combo), f"{sdr(gt, np.mean([outs[c] for c in combo], 0)):.3f}")
print("pnp4096 + PAD", f"{sdr(gt, 0.5 * (outs['pnp4096'] + pad)):.3f}")
e1 = (gt - outs["pnp4096"]).ravel(); e2 = (gt - pad).ravel(); e3 = (gt - outs["pnp8192"]).ravel()
print("error corr ours4096-PAD", np.corrcoef(e1, e2)[0, 1], " ours4096-ours8192", np.corrcoef(e1, e3)[0, 1])
