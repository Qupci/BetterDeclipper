"""Evaluate engine model lists on the FULL example. python eval_full.py '[[["nmf",93,{"nmf_rank":128}]], ...]' [--save tag]"""
import sys, json, time; sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.engine import declip
from betterdeclipper.metrics import sdr
gt, sr = sf.read("F:/BetterDeclipper/ex_sample/sample_ground_truth.wav", dtype="float64")
cl, _ = sf.read("F:/BetterDeclipper/ex_sample/sample_-12dB_clipped.wav", dtype="float64")
m = np.abs(gt) > 0.25
for models in json.loads(sys.argv[1]):
    models = [tuple(x) for x in models]
    x, info = declip(cl, sr, models=models)
    tag = "+".join(f"{k}{w}{json.dumps(e, separators=(',', ':'))}" for k, w, e in models)
    print(f"{sdr(gt, x):.3f} dB (clipped {sdr(gt[m], x[m]):.3f})  {info['time']:.0f}s  {tag}", flush=True)
    if len(sys.argv) > 2:
        sf.write(f"F:/BetterDeclipper/research/outputs/full_{sys.argv[2]}_{len(models)}m_{abs(hash(tag))%10000}.wav", x.astype(np.float32), sr, subtype="FLOAT")
        print("  saved", f"full_{sys.argv[2]}_{len(models)}m_{abs(hash(tag))%10000}.wav")
