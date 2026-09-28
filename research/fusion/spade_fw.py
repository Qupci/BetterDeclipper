"""SPADE with frequency-weighted sparsity on the full example (and fusion with the stored NMF-400 estimate)."""
import sys
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.engine import declip
from betterdeclipper.metrics import sdr
y, sr = sf.read("F:/BetterDeclipper/ex_sample/sample_-12dB_clipped.wav", dtype="float64", always_2d=True)
gt, _ = sf.read("F:/BetterDeclipper/ex_sample/sample_ground_truth.wav", dtype="float64", always_2d=True)
nm = np.load("F:/BetterDeclipper/research/outputs/fusion/nmf400.npy").astype(np.float64)
M = np.abs(y) >= np.abs(y).max() * 0.999
for fw in [None, (8000, 1), (4000, 1), (2000, 1), (4000, 2), (2000, 2), (1000, 1)]:
    x, info = declip(y, sr, mode="hard", models=[("spade", 93, dict(freq_weight=fw, sr=sr))], verbose=False)
    tag = "none" if fw is None else f"{fw[0]}/{fw[1]}"
    np.save(f"F:/BetterDeclipper/research/outputs/fusion/spade_fw_{tag.replace('/', '_')}.npy", x.astype(np.float32))
    f = 0.65 * nm + 0.35 * x
    print(f"freq_weight {tag:10s}: SPADE {sdr(gt, x):.3f}  fused 0.65/0.35 {sdr(gt, f):.3f}  ({info['time']:.1f}s)", flush=True)
