"""Build the master benchmark: 30 s excerpts of unclipped sources x mastering-style degradations.
Output: research/outputs/mbench/<src>__<deg>.wav (degraded) and <src>__<deg>__gt.wav (ground truth)."""
import sys, os, json, time
sys.path.insert(0, "F:/BetterDeclipper/research/master")
import numpy as np, soundfile as sf
from degrade import DEGRADATIONS

OUT = "F:/BetterDeclipper/research/outputs/mbench/"
os.makedirs(OUT, exist_ok=True)
SOURCES = {
    "ex": ("F:/BetterDeclipper/ex_sample/sample_ground_truth.wav", 0),
    "petal": ("F:/deltarune/ch5/17 - Petal Dance.flac", 0),
    "ruder": ("F:/deltarune/ch3n4/11 Ruder Buster.flac", 66),
    "thrash": ("F:/deltarune/ch1/20 Thrash Machine.flac", 20),
    "violet": ("F:/deltarune/ch5/21 - Violet Tactics.flac", 9),
    "bigshot": ("F:/deltarune/ch2/39 BIG SHOT.flac", 107),
    "knife": ("F:/deltarune/ch3n4/30 Black Knife.flac", 85),
}
only = sys.argv[1:]  # optional list of degradations to (re)build
meta = json.load(open(OUT + "meta.json")) if os.path.exists(OUT + "meta.json") else {}
for sname, (path, t0) in SOURCES.items():
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = x[int(t0 * sr): int((t0 + 30) * sr)]
    for dname, fn in DEGRADATIONS.items():
        if only and dname not in only:
            continue
        key = f"{sname}__{dname}"
        t = time.time()
        y, gt, sr_o = fn(x, sr)
        sf.write(OUT + key + ".wav", y.astype(np.float32), sr_o, subtype="FLOAT")
        sf.write(OUT + key + "__gt.wav", gt.astype(np.float32), sr_o, subtype="FLOAT")
        e = y - gt
        meta[key] = dict(src=sname, deg=dname, sr=sr_o, sdr_in=float(10 * np.log10(np.sum(gt ** 2) / np.sum(e ** 2))))
        print(f"{key:24s} sr {sr_o}  input SDR {meta[key]['sdr_in']:6.2f} dB  ({time.time()-t:.1f}s)", flush=True)
json.dump(meta, open(OUT + "meta.json", "w"), indent=1)
