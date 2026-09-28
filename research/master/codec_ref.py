"""Reference: SDR of the lossy codec alone (unclipped source through the codec) = rough ceiling for lossy cases."""
import sys
sys.path.insert(0, "F:/BetterDeclipper/research/master"); sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf, json
from degrade import lossy
from betterdeclipper.metrics import sdr
B = "F:/BetterDeclipper/research/outputs/mbench/"
out = {}
for src in ["ex", "petal", "ruder", "thrash", "violet", "bigshot", "knife"]:
    gt, sr = sf.read(B + f"{src}__mp3_128__gt.wav", dtype="float64", always_2d=True)
    for name, codec, br in [("mp3_128", "mp3", "128k"), ("aac_256", "aac", "256k"), ("vorbis_192", "vorbis", "192k")]:
        # same level handling as the degradation (codec sees the signal scaled so the clip level hits 0 dBFS)
        c = np.abs(gt).max() * 10 ** (-6 / 20)
        y, _, _ = lossy(gt * (0.999 / c) * c, sr, db=0.0, codec=codec, bitrate=br, clip=False)
        out[f"{src}__{name}"] = sdr(gt, y)
    print(src, {k.split('__')[1]: round(v, 2) for k, v in out.items() if k.startswith(src)}, flush=True)
json.dump(out, open("F:/BetterDeclipper/research/results/codec_ref.json", "w"), indent=1)
