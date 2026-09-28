"""Find AL-1 settings reproducing the user's '-6dB AL-1' render of Petal Dance."""
import sys
sys.path.insert(0, "F:/BetterDeclipper"); sys.path.insert(0, "F:/BetterDeclipper/research/al1")
import numpy as np, soundfile as sf
from al1 import render
from betterdeclipper.metrics import sdr

src, sr = sf.read(r"F:/deltarune/ch5/17 - Petal Dance.flac", dtype="float64", always_2d=True)
usr, _ = sf.read("F:/BetterDeclipper/ex_master/AL-1/17 - Petal Dance -6dB AL-1.wav", dtype="float64", always_2d=True)
seg = slice(0, 30 * sr)
x, u = src[seg], usr[seg]
trials = [
    dict(ceiling=-6.0),
    dict(ceiling=-6.0, makeup=0.0, lift=0.0),
    dict(ceiling=-6.0, makeup=0.0, lift=0.0, os=True),
    dict(ceiling=-6.0, tp=True),
    dict(input=6.0, ceiling=-0.1),
    dict(ceiling=-6.0, mode="Continuous"),
    dict(ceiling=-6.0, mode="Hybrid"),
    dict(ceiling=-6.0, engine="Gen 1"),
]
for t in trials:
    y, lag, p = render(x, sr, **t)
    print(f"{t}: latency {lag}, SDR vs user render {sdr(u, y):.2f} dB, vs source {sdr(x, y):.2f} (user vs source {sdr(x, u):.2f}), peak {20*np.log10(np.abs(y).max()):.2f}", flush=True)
