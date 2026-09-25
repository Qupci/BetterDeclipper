import sys, warnings, os; sys.path.insert(0, "F:/BetterDeclipper"); warnings.filterwarnings("ignore")
os.environ["BD_CUDA_GRAPHS"] = "0"
import numpy as np, soundfile as sf, torch
from torch.profiler import profile, ProfilerActivity
from betterdeclipper.engine import declip
cl, sr = sf.read("F:/BetterDeclipper/ex_sample/sample_-12dB_clipped.wav", dtype="float64")
declip(cl[: sr * 2], sr, models=[("nmf", 93, dict(n_iter=10))], device="cuda")
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    declip(cl, sr, models=[("nmf", 93, dict(n_iter=100))], device="cuda"); torch.cuda.synchronize()
rows = sorted(prof.key_averages(), key=lambda e: -e.device_time_total)
tot = sum(e.self_device_time_total for e in rows)
print(f"total self CUDA time {tot/1e3:.0f} ms for 100 iterations ({tot/1e3/100:.2f} ms/iteration)")
for e in rows[:40]:
    if e.self_device_time_total > 0.01 * tot:
        print(f"{e.self_device_time_total/1e3:8.1f} ms  {100*e.self_device_time_total/tot:5.1f}%  n={e.count:5d}  {e.key[:90]}")
