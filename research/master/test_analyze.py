import sys, os, glob
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.auto import analyze_ceilings
from betterdeclipper.detect import estimate_lsb
B = "F:/BetterDeclipper/research/outputs/mbench/"
files = [B + f"{s}__{d}.wav" for s in ["ex", "petal", "ruder", "bigshot"] for d in
         ["hard6", "hard9", "tanh6", "tanh9k4", "cubic6", "softhard6", "softhard9", "oshard6", "resamp6", "mp3_128", "aac_256", "vorbis_192", "al1_6", "osal1"]]
files += [B + f"{s}__hard6__gt.wav" for s in ["ex", "petal", "ruder", "bigshot"]]
files += ["F:/BetterDeclipper/ex_master/soft_works/17_Rome_Search.wav", "F:/BetterDeclipper/ex_master/soft_works/tarzan taz normal LOOP.wav"] + \
         sorted(glob.glob("F:/BetterDeclipper/ex_master/blind/bad/*")) + \
         ["F:/BetterDeclipper/research/outputs/master/OWSLA 3.wav", "F:/BetterDeclipper/research/outputs/master/Mothership Teaser.wav",
          "F:/BetterDeclipper/ex_master/blind/wierd/signal (dan sena remix).wav", "F:/BetterDeclipper/ex_master/AL-1/17 - Petal Dance -6dB AL-1.wav",
          "F:/BetterDeclipper/ex_master/AL-1/17 - Petal Dance -6dB HARD.wav", "F:/BetterDeclipper/ex_sample/sample_-12dB_clipped.wav"]
for fn in files:
    y, sr = sf.read(fn, dtype="float64", always_2d=True)
    lsb = estimate_lsb(y)
    refs = analyze_ceilings(y, lsb)
    desc = " | ".join(f"{r['kind']:7s} th {r['theta'] if r['theta'] else 0:.4f} s/t {r['sigma']/max(r['theta'] or 1,1e-9):.4f} fl {r['flat']:.2f}" for row in refs for r in row)
    print(f"{os.path.basename(fn)[:30]:30s} {desc}", flush=True)
