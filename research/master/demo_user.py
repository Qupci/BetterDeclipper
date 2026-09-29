"""Process the user's example masters (full files) for listening.
usage: python demo_user.py [preset] [mode] [outdir-suffix] [name filters...]
Outputs: research/outputs/master_demo<suffix>/<name> [<mode> <preset>].wav (32-bit float) + a summary line per
file (flagged %, peaks, RMS change, 99th percentile / max gain of the changed samples)."""
import sys, os, glob
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.engine import declip

B = "F:/BetterDeclipper/"
preset = sys.argv[1] if len(sys.argv) > 1 else "normal"
mode = sys.argv[2] if len(sys.argv) > 2 else "auto"
OUT = B + "research/outputs/master_demo" + (sys.argv[3] if len(sys.argv) > 3 else "") + "/"
os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob(B + "ex_master/soft_works/*.wav")) + sorted(glob.glob(B + "ex_master/blind/bad/*")) + \
    [B + "research/outputs/master/OWSLA 3.wav", B + "research/outputs/master/Mothership Teaser.wav"] + \
    sorted(glob.glob(B + "ex_master/AL-1/*.wav")) + sorted(glob.glob(B + "ex_strange/*.wav")) + \
    [B + "ProAudioDeclipper/blind_examples/scar_tissue-cd.wav"]
only = sys.argv[4:]
for fn in files:
    name = os.path.splitext(os.path.basename(fn))[0]
    if only and not any(o.lower() in name.lower() for o in only):
        continue
    y, sr = sf.read(fn, dtype="float64", always_2d=True)
    x, info = declip(y, sr, preset=preset, mode=mode, verbose=False)
    sf.write(OUT + f"{name} [{mode} {preset}].wav", x.astype(np.float32), sr, subtype="FLOAT")
    ch = np.abs(x - y) > 1e-7
    g = 20 * np.log10(np.abs(x[ch]) / np.maximum(np.abs(y[ch]), 1e-9)) if ch.any() else np.zeros(1)
    print(f"{name[:40]:40s} {info['mode'][:44]:44s} flagged {info['clipped_frac']*100:5.2f}%  peak in "
          f"{20*np.log10(np.abs(y).max()):+.2f} out {20*np.log10(np.abs(x).max()):+.2f} dBFS  rms change "
          f"{20*np.log10(np.sqrt(np.mean(x**2))/np.sqrt(np.mean(y**2))):+.3f} dB  gain p99/max {np.quantile(g, 0.99):+.1f}/"
          f"{g.max():+.1f} dB  {info['time']:.0f}s", flush=True)
