"""Process the user's ex_master examples with the new auto mode (full files) for listening.
Outputs: research/outputs/master_demo/<name> [<preset>].wav (32-bit float) + a summary line per file."""
import sys, os, glob, time
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.engine import declip

OUT = "F:/BetterDeclipper/research/outputs/master_demo/"
os.makedirs(OUT, exist_ok=True)
preset = sys.argv[1] if len(sys.argv) > 1 else "normal"
mode = sys.argv[2] if len(sys.argv) > 2 else "auto"
files = sorted(glob.glob("F:/BetterDeclipper/ex_master/soft_works/*.wav")) + sorted(glob.glob("F:/BetterDeclipper/ex_master/blind/bad/*")) + \
        ["F:/BetterDeclipper/research/outputs/master/OWSLA 3.wav", "F:/BetterDeclipper/research/outputs/master/Mothership Teaser.wav",
         "F:/BetterDeclipper/ex_master/blind/wierd/signal (dan sena remix).wav"] + sorted(glob.glob("F:/BetterDeclipper/ex_master/AL-1/*.wav"))
only = sys.argv[3:]
for fn in files:
    name = os.path.splitext(os.path.basename(fn))[0]
    if only and not any(o.lower() in name.lower() for o in only):
        continue
    y, sr = sf.read(fn, dtype="float64", always_2d=True)
    x, info = declip(y, sr, preset=preset, mode=mode, verbose=False)
    sf.write(OUT + f"{name} [{mode} {preset}].wav", x.astype(np.float32), sr, subtype="FLOAT")
    an = info.get("analysis")
    kinds = an["kinds"] if an else "-"
    print(f"{name[:40]:40s} {info['mode']:34s} flagged {info['clipped_frac']*100:5.2f}%  peak in {20*np.log10(np.abs(y).max()):+.2f} out {20*np.log10(np.abs(x).max()):+.2f} dBFS"
          f"  rms change {20*np.log10(np.sqrt(np.mean(x**2))/np.sqrt(np.mean(y**2))):+.2f} dB  {info['time']:.0f}s", flush=True)
