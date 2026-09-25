"""Long-file speed test: the example repeated 8x (174 s, 26.8 % clipped) through the engine."""
import sys, time, warnings; sys.path.insert(0, "F:/BetterDeclipper"); warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, torch
from betterdeclipper.engine import declip
from betterdeclipper.metrics import sdr
dev = sys.argv[1]; presets = sys.argv[2].split(",")
gt, sr = sf.read("F:/BetterDeclipper/ex_sample/sample_ground_truth.wav", dtype="float64")
cl, _ = sf.read("F:/BetterDeclipper/ex_sample/sample_-12dB_clipped.wav", dtype="float64")
gt8, cl8 = np.tile(gt, (8, 1)), np.tile(cl, (8, 1))
declip(cl[: sr * 3], sr, preset="fast", device=dev)  # warm-up
for p in presets:
    if dev == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = time.time(); x, info = declip(cl8, sr, preset=p, device=dev); el = time.time() - t
    mem = torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0
    print(f"{p:7s} {dev}: {len(cl8)/sr:.0f} s audio in {el:.1f} s ({len(cl8)/sr/el:.2f}x real time)  SDR {sdr(gt8, x):.3f} dB  peak GPU mem {mem:.0f} MiB", flush=True)
