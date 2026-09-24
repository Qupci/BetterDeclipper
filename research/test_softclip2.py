import sys, warnings; sys.path.insert(0, "F:/BetterDeclipper")
warnings.filterwarnings("ignore", message=".*smallest subnormal.*")
import numpy as np, soundfile as sf
from betterdeclipper.engine import declip
from betterdeclipper.metrics import sdr
preset = sys.argv[1]; t = float(sys.argv[2]); c = float(sys.argv[3]); knees = [float(k) for k in sys.argv[4].split(",")]
gt, sr = sf.read("F:/BetterDeclipper/ex_sample/sample_ground_truth.wav", dtype="float64")
a = np.abs(gt)
ys = np.where(a > t, np.sign(gt) * (t + (c - t) * np.tanh((a - t) / (c - t))), gt)
rng = np.random.default_rng(0)
ys = np.round(ys * 32768 + rng.random(ys.shape) - rng.random(ys.shape)) / 32768
m = a > t
print(f"soft clip knee {t} ceiling {c}: {m.mean()*100:.1f}% above knee, input SDR {sdr(gt, ys):.2f} dB (compressed samples {sdr(gt[m], ys[m]):.2f})")
for k in knees:
    x, info = declip(ys, sr, preset=preset, mode="soft", knees=[(k, -k)] * 2)
    print(f"  forced knee {k}: flagged {info['clipped_frac']*100:.1f}%  SDR {sdr(gt, x):.2f} dB (compressed samples {sdr(gt[m], x[m]):.2f})  {info['time']:.0f}s", flush=True)
