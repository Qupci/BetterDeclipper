"""Benchmark presets on the full example on a given device; compare with CPU reference SDRs.
python bench_gpu.py [device] [presets]"""
import sys, time, warnings; sys.path.insert(0, "F:/BetterDeclipper")
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, torch
from betterdeclipper.engine import declip
from betterdeclipper.metrics import sdr
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
presets = (sys.argv[2] if len(sys.argv) > 2 else "fast,normal,high,best").split(",")
CPU_REF = {"fast": (24.166, 35), "normal": (25.208, 142), "high": (25.299, 181), "best": (25.369, 313)}
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", "archs", torch.cuda.get_arch_list() if torch.cuda.is_available() else "")
gt, sr = sf.read("F:/BetterDeclipper/ex_sample/sample_ground_truth.wav", dtype="float64")
cl, _ = sf.read("F:/BetterDeclipper/ex_sample/sample_-12dB_clipped.wav", dtype="float64")
m = np.abs(gt) > 0.25
declip(cl[: sr * 3], sr, preset="fast", device=dev)  # warm-up (CUDA context, cuFFT plans)
for p in presets:
    if dev.startswith("cuda"):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    x, info = declip(cl, sr, preset=p, device=dev)
    mem = torch.cuda.max_memory_allocated() / 2 ** 20 if dev.startswith("cuda") else 0
    ref_sdr, ref_t = CPU_REF[p]
    print(f"{p:7s} {info['device']:6s} SDR {sdr(gt, x):.3f} dB (CPU {ref_sdr:.3f})  time {info['time']:6.1f}s (CPU {ref_t}s, "
          f"speedup {ref_t / info['time']:.1f}x)  peak GPU mem {mem:.0f} MiB", flush=True)
