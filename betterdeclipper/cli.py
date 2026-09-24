"""Command-line interface: python -m betterdeclipper input.wav output.wav [options]"""
import argparse
import sys
import numpy as np
import soundfile as sf

from . import __version__
from .engine import declip, PRESETS


def _parse_level(s):
    """'-12' / '-12dB' -> dBFS; '0.25' (no unit, 0<v<=1) -> linear."""
    s = s.strip().lower().replace("dbfs", "").replace("db", "")
    v = float(s)
    return 10 ** (v / 20) if v <= 0 else v


def main(argv=None):
    ap = argparse.ArgumentParser(prog="betterdeclipper", description="High-accuracy offline audio declipper.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--preset", choices=list(PRESETS), default="normal",
                    help="speed/quality trade-off (default: normal)")
    ap.add_argument("--clip-level", default=None,
                    help="override automatic detection: clip level in dBFS (e.g. -12) or linear (e.g. 0.25), "
                         "applied to both polarities of all channels")
    ap.add_argument("--format", choices=["float", "pcm24", "pcm16"], default="float",
                    help="output sample format (default: 32-bit float, keeps restored peaks above 0 dBFS)")
    ap.add_argument("--normalize", type=float, default=None, metavar="DBFS",
                    help="scale output so its peak is at this level (e.g. -0.1); recommended with PCM output")
    ap.add_argument("--gain", type=float, default=0.0, metavar="DB", help="output gain in dB")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)

    y, sr = sf.read(args.input, dtype="float64", always_2d=True)
    C = y.shape[1]
    levels = None
    if args.clip_level is not None:
        lv = _parse_level(args.clip_level)
        levels = [(lv, -lv)] * C
    print(f"input: {args.input}  {sr} Hz, {C} ch, {len(y)/sr:.1f} s")

    def progress(i, n, el):
        print(f"  chunk {i}/{n}  elapsed {el:.0f}s", flush=True)

    x, info = declip(y, sr, preset=args.preset, levels=levels, threads=args.threads, progress=progress)
    lv_str = ", ".join(
        f"ch{c}: " + "/".join("-" if v is None else f"{20*np.log10(abs(v)):.2f} dBFS" for v in lvl)
        for c, lvl in enumerate(info["levels"]))
    print(f"clip levels: {lv_str}")
    print(f"clipped samples: {info['clipped_frac']*100:.2f}%   preset: {args.preset}   time: {info['time']:.1f}s")
    if info["clipped_frac"] == 0:
        print("no clipping detected; output equals input (use --clip-level to force a level)")
    x = x * 10 ** (args.gain / 20)
    peak = np.abs(x).max()
    if args.normalize is not None and peak > 0:
        x = x * (10 ** (args.normalize / 20) / peak)
        peak = np.abs(x).max()
    subtype = {"float": "FLOAT", "pcm24": "PCM_24", "pcm16": "PCM_16"}[args.format]
    if subtype != "FLOAT" and peak > 1.0:
        print(f"warning: output peak {20*np.log10(peak):+.2f} dBFS exceeds 0 dBFS and will clip in {args.format}; "
              f"use --normalize or --format float", file=sys.stderr)
    sf.write(args.output, x.astype(np.float32 if subtype == "FLOAT" else np.float64), sr, subtype=subtype)
    print(f"output: {args.output}  peak {20*np.log10(max(peak, 1e-12)):+.2f} dBFS ({args.format})")
    return 0
