import sys, time, json; sys.path.insert(0, "F:/BetterDeclipper"); sys.path.insert(0, "F:/BetterDeclipper/research")
import numpy as np, soundfile as sf
from bench_example import load
from betterdeclipper.detect import detect_clip_levels, clip_masks, estimate_lsb
from betterdeclipper.methods.janssen import ar_refine
from betterdeclipper.metrics import sdr
gt, cl, pad, sr = load((5, 10))
lv = detect_clip_levels(cl, estimate_lsb(cl)); m_hi, m_lo, th_hi, th_lo = clip_masks(cl, lv)
pilot = sf.read(sys.argv[1], dtype="float64")[0]
print("pilot sdr", sdr(gt, pilot))
for kw in json.loads(sys.argv[2]):
    t0 = time.time()
    cb = lambda o, x: print(f"   outer {o}: {sdr(gt, x):.3f}", flush=True)
    est = ar_refine(cl, pilot, m_hi, m_lo, th_hi, th_lo, callback=cb, **kw)
    print(kw, f"sdr {sdr(gt, est):.3f}  ({time.time()-t0:.0f}s)", flush=True)
