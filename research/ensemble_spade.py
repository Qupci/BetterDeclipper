import sys, time, json; sys.path.insert(0, "F:/BetterDeclipper"); sys.path.insert(0, "F:/BetterDeclipper/research")
import numpy as np, soundfile as sf
from bench_example import load
from betterdeclipper.detect import detect_clip_levels, clip_masks, estimate_lsb
from betterdeclipper.methods.spade import declip_spade
from betterdeclipper.metrics import sdr
gt, cl, pad, sr = load((5, 10))
lv = detect_clip_levels(cl, estimate_lsb(cl)); m_hi, m_lo, th_hi, th_lo = clip_masks(cl, lv)
ours = sf.read("F:/BetterDeclipper/research/outputs/pnp4096_cg25_5_10.wav", dtype="float64")[0]
for kw in json.loads(sys.argv[1]):
    t0 = time.time()
    est = declip_spade(cl, m_hi, m_lo, th_hi, th_lo, verbose=True, **kw)
    el = time.time() - t0
    e1 = (gt - ours).ravel(); e2 = (gt - est).ravel()
    best = max((sdr(gt, a * ours + (1 - a) * est), a) for a in np.linspace(0, 1, 11))
    print(kw, f"sdr {sdr(gt, est):.3f}  err-corr w/ ours {np.corrcoef(e1, e2)[0,1]:.3f}  avg {sdr(gt, 0.5*(ours+est)):.3f}  best mix {best[0]:.3f} (a={best[1]:.1f})  {el:.0f}s", flush=True)
    sf.write(f"F:/BetterDeclipper/research/outputs/spade_{kw.get('variant','a')}_{kw['win_len']}_5_10.wav", est.astype(np.float32), sr, subtype="FLOAT")
