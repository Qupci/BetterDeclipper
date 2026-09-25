import sys, time, warnings; sys.path.insert(0, "F:/BetterDeclipper"); warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, torch
from betterdeclipper.engine import declip
from betterdeclipper.detect import detect_clip_levels, estimate_lsb, clip_masks
cl, sr = sf.read("F:/BetterDeclipper/ex_sample/sample_-12dB_clipped.wav", dtype="float64")
declip(cl[: sr * 3], sr, preset="fast", device="cuda")  # warm-up
t = time.time(); lv = detect_clip_levels(cl, estimate_lsb(cl)); m = clip_masks(cl, lv); print(f"detection+masks (CPU numpy): {time.time()-t:.2f}s")
for name, models in [("nmf 400it", [("nmf", 93, {})]), ("nmf 150it", [("nmf", 93, dict(n_iter=150))]), ("pew 400it", [("pnp", 93, {})]), ("spade", [("spade", 93, {})])]:
    torch.cuda.synchronize(); t = time.time()
    x, info = declip(cl, sr, models=models, device="cuda")
    torch.cuda.synchronize(); print(f"{name:10s}: {time.time()-t:.2f}s")
