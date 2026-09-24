"""Run the user's ProAudioDeclipper VST3 via pedalboard (for benchmarking only).

python run_pad.py in.wav out.wav [--svq 6] [--look 2] [--limiter 0] [--seg a b]
Output is written as 32-bit float to avoid re-quantization.
"""
import sys, time, argparse
sys.path.insert(0, "F:/BetterDeclipper/research/_vendor")
import numpy as np, soundfile as sf
import pedalboard

PLUG = "F:/BetterDeclipper/ProAudioDeclipper/bin/ProAudioDeclipper_x64.vst3"


def run_pad(x, sr, svq=6, look=2, limiter=0, processors=4, buffer_size=8192):
    p = pedalboard.load_plugin(PLUG)
    p.speedvsquality = float(svq)
    p.lookaheadsamples = float(look)
    p.processors = float(processors)
    p.limitermode = float(limiter)
    lat_pad = 16384  # flush tail: append silence and trim by measured delay afterwards
    xin = np.concatenate([x, np.zeros((lat_pad, x.shape[1]))], axis=0).T.astype(np.float32)
    t0 = time.time()
    y = p.process(xin, sr, buffer_size=buffer_size, reset=True).T.astype(np.float64)
    el = time.time() - t0
    # estimate delay by cross-correlating on unclipped content
    best, bd = -1, 0
    a = x[:, 0]
    for d in range(0, lat_pad):
        if d > 12000:
            break
        seg = y[d:d + len(a), 0]
        if len(seg) < len(a):
            break
        c = np.dot(a[: min(len(a), 200000)], seg[: min(len(a), 200000)])
        if c > best:
            best, bd = c, d
    return y[bd:bd + len(x)], bd, el


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--svq", type=int, default=6)
    ap.add_argument("--look", type=int, default=2)
    ap.add_argument("--limiter", type=int, default=0)
    ap.add_argument("--seg", type=float, nargs=2, default=None)
    args = ap.parse_args()
    x, sr = sf.read(args.inp, dtype="float64", always_2d=True)
    if args.seg:
        x = x[int(args.seg[0] * sr):int(args.seg[1] * sr)]
    y, delay, el = run_pad(x, sr, args.svq, args.look, args.limiter)
    print(f"delay {delay} samples, processing time {el:.1f}s for {len(x)/sr:.1f}s audio")
    sf.write(args.out, y.astype(np.float32), sr, subtype="FLOAT")
