import sys
sys.path.insert(0, "F:/BetterDeclipper"); sys.path.insert(0, "F:/BetterDeclipper/research/al1"); sys.path.insert(0, "F:/BetterDeclipper/research/_vendor")
import numpy as np, soundfile as sf
from al1 import load
from betterdeclipper.metrics import sdr
src, sr = sf.read(r"F:/deltarune/ch5/17 - Petal Dance.flac", dtype="float64", always_2d=True)
usr, _ = sf.read("F:/BetterDeclipper/ex_master/AL-1/17 - Petal Dance -6dB AL-1.wav", dtype="float64", always_2d=True)
x, u = src[:30 * sr], usr[:30 * sr]
def run(bs=4096, **t):
    p = load(**t)
    y = p.process(x.T.astype(np.float32), sr, buffer_size=bs, reset=True).T.astype(np.float64)
    return y
for bs in (512, 1024, 256):
    y = run(bs, ceiling=-6.0)
    print(f"bs {bs}: vs user {sdr(u, y):.2f}  vs src {sdr(x, y):.2f}", flush=True)
for t in [dict(ceiling=-6.0, tp=True, os=True), dict(ceiling=-6.0, detector="Slow"), dict(ceiling=-6.0, voicing="Dense"),
          dict(ceiling=-6.0, voicing="Balanced"), dict(ceiling=-6.0, link=True), dict(ceiling=-6.0, optimize=True)]:
    y = run(4096, **t)
    print(f"{t}: vs user {sdr(u, y):.2f}  vs src {sdr(x, y):.2f}", flush=True)
