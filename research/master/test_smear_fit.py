import sys, json, os
sys.path.insert(0, "F:/BetterDeclipper"); sys.path.insert(0, "F:/BetterDeclipper/research/master")
import numpy as np, soundfile as sf
from smear_fit import fit_smeared
from betterdeclipper.detect import estimate_lsb
B = "F:/BetterDeclipper/research/outputs/mbench/"
DB = {"hard6": -6, "hard9": -9, "tanh6": -6, "tanh9k4": -9, "cubic6": -6, "softhard6": -6, "softhard9": -9, "oshard6": -6, "resamp6": -6,
      "mp3_128": -6, "aac_256": -6, "vorbis_192": -6, "al1_6": -6, "osal1": -7}
srcs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["ex", "ruder", "bigshot"]
for src in srcs:
    for d, db in DB.items():
        y, sr = sf.read(B + f"{src}__{d}.wav", dtype="float64", always_2d=True)
        gt, _ = sf.read(B + f"{src}__{d}__gt.wav", dtype="float64", always_2d=True)
        c = np.abs(gt).max() * 10 ** (db / 20)
        lsb = estimate_lsb(y)
        out = []
        for ch in range(y.shape[1]):
            f = fit_smeared(y[:, ch], lsb)
            out.append(f"th/c {f['theta']/c:.4f} sg/th {f['sigma']/f['theta']:.4f} bump {f['bump_ratio']:8.1f} pk/th {f['peak']/f['theta']:.3f}")
        print(f"{src:7s} {d:10s} | " + " | ".join(out), flush=True)
files = ["F:/BetterDeclipper/research/outputs/master/OWSLA 3.wav", "F:/BetterDeclipper/research/outputs/master/Mothership Teaser.wav",
         "F:/BetterDeclipper/ex_master/blind/wierd/signal (dan sena remix).wav", "F:/BetterDeclipper/ex_master/soft_works/17_Rome_Search.wav",
         "F:/BetterDeclipper/ex_master/soft_works/tarzan taz normal LOOP.wav", "F:/BetterDeclipper/ex_master/blind/bad/Bangarang.flac",
         "F:/BetterDeclipper/ex_master/blind/bad/Goin' In (Skrillex Goin Hard Remix).flac", "F:/BetterDeclipper/ex_master/blind/bad/All Is Fair In Love And Brostep.flac",
         "F:/BetterDeclipper/ex_master/blind/bad/Purple Lamborghini.wav", "F:/BetterDeclipper/ex_sample/sample_-12dB_clipped.wav", "F:/BetterDeclipper/ex_sample/sample_ground_truth.wav"]
if len(sys.argv) <= 2:
    for fn in files:
        y, sr = sf.read(fn, dtype="float64", always_2d=True)
        lsb = estimate_lsb(y)
        out = []
        for ch in range(y.shape[1]):
            for sg in (1, -1):
                f = fit_smeared(sg * y[:, ch], lsb)
                out.append(f"th {20*np.log10(f['theta']):+.2f}dB sg/th {f['sigma']/f['theta']:.4f} bump {f['bump_ratio']:7.1f} pk/th {f['peak']/f['theta']:.3f}")
        print(f"{os.path.basename(fn)[:28]:28s} | " + " | ".join(out), flush=True)
