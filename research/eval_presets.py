import sys, time; sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.engine import declip
from betterdeclipper.metrics import sdr
gt, sr = sf.read("F:/BetterDeclipper/ex_sample/sample_ground_truth.wav", dtype="float64")
cl, _ = sf.read("F:/BetterDeclipper/ex_sample/sample_-12dB_clipped.wav", dtype="float64")
m = np.abs(gt) > 0.25
for p in sys.argv[1].split(","):
    x, info = declip(cl, sr, preset=p)
    print(f"{p:7s} SDR {sdr(gt, x):.3f} dB  clipped-samples {sdr(gt[m], x[m]):.3f} dB  time {info['time']:.0f}s", flush=True)
    sf.write(f"F:/BetterDeclipper/research/outputs/preset_{p}_full.wav", x.astype(np.float32), sr, subtype="FLOAT")
