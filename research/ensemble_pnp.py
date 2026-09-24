"""Run PnP variants on the excerpt; report SDR, error correlation with pnp4096 & SPADE, and fusion."""
import sys, time, json; sys.path.insert(0, "F:/BetterDeclipper"); sys.path.insert(0, "F:/BetterDeclipper/research")
import numpy as np, soundfile as sf
from bench_example import load
from betterdeclipper.detect import detect_clip_levels, clip_masks, estimate_lsb
from betterdeclipper.methods.pnp import declip_pnp
from betterdeclipper.metrics import sdr
gt, cl, pad, sr = load((5, 10))
lv = detect_clip_levels(cl, estimate_lsb(cl)); m_hi, m_lo, th_hi, th_lo = clip_masks(cl, lv)
O = "F:/BetterDeclipper/research/outputs/"
ours = sf.read(O + "pnp4096_cg25_5_10.wav", dtype="float64")[0]
spd = sf.read(O + "spade_a_4096_5_10.wav", dtype="float64")[0]
ens6 = np.mean([sf.read(O + f, dtype="float64")[0] for f in ["pnp4096_cg25_5_10.wav", "pnp2048_cg25_5_10.wav", "pnp8192_cg25_5_10.wav", "spade_a_4096_5_10.wav", "spade_a_2048_5_10.wav", "spade_a_8192_5_10.wav"]], 0)
for kw in json.loads(sys.argv[1]):
    tag = kw.pop("tag", None)
    t0 = time.time()
    best = [-99, 0]
    def cb(it, xr):
        s_ = sdr(gt, xr)
        if s_ > best[0]: best[:] = [s_, it + 1]
    est = declip_pnp(cl, m_hi, m_lo, th_hi, th_lo, callback=cb, **kw)
    el = time.time() - t0
    e = (gt - est).ravel()
    print(json.dumps(kw), f"sdr {sdr(gt, est):.3f} (best {best[0]:.3f}@{best[1]})  corr pnp {np.corrcoef(e, (gt-ours).ravel())[0,1]:.3f} spade {np.corrcoef(e, (gt-spd).ravel())[0,1]:.3f}"
          f"  +pnp {sdr(gt, (est+ours)/2):.3f}  +pnp+spade {sdr(gt, (est+ours+spd)/3):.3f}  ens6+this {sdr(gt, (6*ens6+est)/7):.3f}  {el:.0f}s", flush=True)
    if tag:
        sf.write(O + tag + "_5_10.wav", est.astype(np.float32), sr, subtype="FLOAT")
