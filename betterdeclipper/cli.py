"""Command-line interface: python -m betterdeclipper input.wav output.wav [options]"""
import argparse
import os
import sys
import numpy as np
import soundfile as sf

from . import __version__
from .engine import declip, PRESETS
from .auto import ACC_MIN_FREE, ACC_HINT_FREE, load_analysis, save_analysis


def _parse_level(s):
    """'-12' / '-12dB' -> dBFS; '0.25' (no unit, 0<v<=1) -> linear."""
    s = s.strip().lower().replace("dbfs", "").replace("db", "")
    v = float(s)
    return 10 ** (v / 20) if v <= 0 else v


KIND_TEXT = {
    "clip": "flat plateaus at the ceiling (hard clipping)",
    "smeared": "clipped, then a lossy codec / resampling / oversampling blurred the plateaus",
    "limiter": "isolated peaks touch the ceiling (brickwall limiter)",
}


def _db(v):
    return f"{20 * np.log10(abs(v)):+.2f} dBFS"


def analysis_lines(info, C):
    """Human-readable account of what the automatic analysis found and what gets restored."""
    an = info["analysis"]
    refs, knees, req = an["refs"], an["knees"], an.get("request", "auto")
    names = ["L", "R"] if C == 2 else [f"ch{c + 1}" for c in range(C)]
    kinds = [k for k in ("clip", "smeared", "limiter") if k in an["kinds"]]
    out = [f"analysis ({req}):"]
    src = an.get("reused_from")
    if src is not None:
        what = f"'{src['file']}'" if src.get("file") else "another input"
        out.append(f"  reused from {what} ({src.get('sr', '?')} Hz, {src.get('samples', 0) / max(src.get('sr', 1), 1):.1f} s):"
                   f" levels and kinds below were measured there, not on this input")
    for c in range(C):
        for pi, pol in enumerate("+-"):
            r, k = refs[c][pi], knees[c][pi]
            fl = info["flagged"][c][pi] * 100
            lab = f"{names[c]}{pol}".ljust(5)
            if r["theta"] is None:
                out.append(f"  {lab}too few samples to analyze")
                continue
            if r["kind"] == "none":
                txt = f"no ceiling       (peak {_db(r['peak'])})"
            else:
                blur = f", blur {100 * r['sigma'] / r['theta']:.1f}%" if r["kind"] == "smeared" else ""
                txt = f"{r['kind']:8s} ceiling {_db(r['theta'])}{blur}"
            if k is None:
                act = "not restored"
            elif k >= 0.99 * r["theta"]:
                act = "2+ sample runs at ceiling restored" if r["kind"] == "limiter" else "ceiling samples restored"
            else:
                act = f"restored above {_db(k)} ({20 * np.log10(k / r['theta']):+.1f} dB)"
            out.append(f"  {lab}{txt:44s} {act:38s} {fl:5.2f}% of samples")
    for k in kinds:
        out.append(f"  {k} = {KIND_TEXT[k]}")
    acc, amin, kr = an.get("acc"), an.get("acc_min"), an.get("knee_rel")
    if req == "hard":
        out.append("  soft shoulder: not searched (--mode hard)")
    elif acc is None:
        out.append("  soft shoulder: nothing to search")
    elif an.get("knee_found"):
        free = amin >= ACC_MIN_FREE
        out.append(f"  soft shoulder: yes - " + ("soft saturation without a ceiling" if free else
                                                "peaks were compressed below the ceiling") +
                   f" (first-pass lift accelerates {acc:.3f} towards the top, threshold {amin:.2f}) -> knee "
                   f"{kr:.2f} x {'robust peak' if free else 'ceiling'} ({20 * np.log10(kr):+.1f} dB)")
    elif kr:
        out.append(f"  soft shoulder: none found (first-pass lift acceleration {acc:.3f} < {amin:.2f}); "
                   f"--mode soft: default knee {kr:.2f} x ceiling ({20 * np.log10(kr):+.1f} dB)")
    elif amin >= ACC_MIN_FREE and acc >= ACC_HINT_FREE:
        out.append(f"  soft shoulder: weak sign of saturation without a ceiling (first-pass lift acceleration "
                   f"{acc:.3f}; auto needs {amin:.2f} here, unclipped music reaches ~0.3) - left untouched; "
                   f"if it sounds squashed, try --mode soft")
    else:
        extra = "" if set(an["kinds"]) == {"limiter"} else "; --mode soft forces one"
        out.append(f"  soft shoulder: none found (first-pass lift acceleration {acc:.3f} < {amin:.2f}{extra})")
    if "limiter" in an["kinds"]:
        lk = an.get("limiter_knee")
        if lk is None:
            txt = "only flat plateaus (2+ samples at the ceiling) are restored (--mode hard)"
        elif req == "limiter":
            txt = f"soft region from {lk:.2f} x ceiling (experimental --mode limiter), capped at +9 dB"
        else:
            why = "the knee found above" if an.get("knee_found") else "default, no shoulder found"
            txt = (f"soft region from {lk:.2f} x ceiling ({20 * np.log10(lk):+.1f} dB; {why}), capped at +9 dB "
                   f"(--mode hard: flat plateaus only)")
        out.append("  limiter ceilings: " + txt)
    return out


def restoration_label(mode, info, forced=None):
    """Short name of the restoration for default output names: 'auto <what the analysis found>' (clip,
    smeared, limiter, + soft when a soft shoulder was restored; none if nothing was), or the chosen mode,
    followed by a forced level if one was given (e.g. 'hard -12dB', 'soft knee -9dB')."""
    if forced:
        return f"{mode} {forced}"
    an = info.get("analysis")
    if mode != "auto" or an is None:
        return mode
    parts = [k for k in ("clip", "smeared", "limiter") if k in an["kinds"]]
    if an.get("knee_found"):
        parts.append("soft")
    return "auto " + ("+".join(parts) if parts and info["clipped_frac"] > 0 else "none")


def default_output(path, label, preset):
    """'<folder>/<name> [<label> <preset>].wav' next to the input"""
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(os.path.dirname(os.path.abspath(path)), f"{stem} [{label} {preset}].wav")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="betterdeclipper", description="High-accuracy offline audio declipper.")
    ap.add_argument("input")
    ap.add_argument("output", nargs="?", default=None,
                    help="output file (WAV); default: '<input name> [<restoration> <preset>].wav' in the input's "
                         "folder, e.g. 'song [auto clip+soft normal].wav' or 'song [hard best].wav'")
    ap.add_argument("--preset", choices=list(PRESETS), default="normal",
                    help="speed/quality trade-off (default: normal)")
    ap.add_argument("--clip-level", default=None,
                    help="override automatic detection: clip level in dBFS (e.g. -12) or linear (e.g. 0.25), "
                         "applied to both polarities of all channels")
    ap.add_argument("--mode", choices=["auto", "hard", "soft", "limiter", "legacy"], default="auto",
                    help="auto (default): analyzes the master - clip plateau, limiter ceiling, smeared ceiling "
                         "(lossy codec / resampling / oversampled clipper) and soft saturation below it - and "
                         "restores accordingly: the soft region's knee comes from a first restoration pass "
                         "(below limiter ceilings always a soft region, 0.8 x ceiling if no knee is found); "
                         "hard: ceilings only, no soft region; soft: the same analysis and knee, but always a "
                         "soft region below every ceiling (0.8 x ceiling if no knee is found); limiter: "
                         "EXPERIMENTAL, a deeper soft region (0.5 x ceiling) below limiter ceilings (raises the "
                         "measured accuracy on synthetic limiter tests but can add audible distortion); "
                         "legacy: the previous plateau-or-knee detection")
    ap.add_argument("--knee", default=None,
                    help="soft mode: force the knee level in dBFS (e.g. -9) or linear (e.g. 0.35)")
    ap.add_argument("--max-gain", type=float, default=None, metavar="DB",
                    help="optional safety cap: restored samples may exceed the clip level by at most this "
                         "many dB (built into the constraints, so restored peaks stay smooth)")
    ap.add_argument("--format", choices=["float", "pcm24", "pcm16"], default="float",
                    help="output sample format (default: 32-bit float, keeps restored peaks above 0 dBFS)")
    ap.add_argument("--normalize", type=float, default=None, metavar="DBFS",
                    help="scale output so its peak is at this level (e.g. -0.1); recommended with PCM output")
    ap.add_argument("--gain", type=float, default=0.0, metavar="DB", help="output gain in dB")
    ap.add_argument("--device", default="auto",
                    help="auto (default: CUDA GPU if available, else CPU), cpu, cuda, or cuda:N")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (CPU processing only)")
    ap.add_argument("--save-analysis", nargs="?", const="", default=None, metavar="FILE",
                    help="also save the analysis as JSON (default: '<input name>.analysis.json' next to the "
                         "output), to skip it next time with --load-analysis")
    ap.add_argument("--load-analysis", default=None, metavar="FILE",
                    help="use a saved analysis instead of analyzing the input: the same input with another "
                         "preset or mode, or another track clipped the same way (e.g. the same album; "
                         "check by ear that it fits)")
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)
    if (args.save_analysis is not None or args.load_analysis) and (
            args.clip_level is not None or args.knee is not None or args.mode == "legacy"):
        ap.error("--save-analysis / --load-analysis work with --mode auto, hard, soft or limiter, "
                 "not with --clip-level, --knee or --mode legacy")
    analysis = load_analysis(args.load_analysis) if args.load_analysis else None

    y, sr = sf.read(args.input, dtype="float64", always_2d=True)
    C = y.shape[1]
    levels = None
    knees = None
    mode = args.mode
    forced = None
    tag = lambda v: f"{round(20 * np.log10(v), 1):g}dB"
    if args.clip_level is not None:
        lv = _parse_level(args.clip_level)
        levels = [(lv, -lv)] * C
        mode, forced = "hard", tag(lv)
    if args.knee is not None:
        kv = _parse_level(args.knee)
        knees = [(kv, -kv)] * C
        mode, forced = "soft", "knee " + tag(kv)
    print(f"input: {args.input}  {sr} Hz, {C} ch, {len(y)/sr:.1f} s")
    if analysis is not None:
        print(f"analysis loaded from {args.load_analysis}", flush=True)

    def progress(i, n, el):
        print(f"  step {i}/{n}  elapsed {el:.0f}s", flush=True)

    try:
        x, info = declip(y, sr, preset=args.preset, levels=levels, threads=args.threads, progress=progress,
                         mode=mode, knees=knees, max_gain_db=args.max_gain, device=args.device, analysis=analysis,
                         full_analysis=args.save_analysis is not None)
    except ValueError as e:
        if analysis is None:
            raise
        print(f"error: {e}", file=sys.stderr)
        return 2
    lv_str = ", ".join(
        f"ch{c}: " + "/".join("-" if v is None else f"{20*np.log10(abs(v)):.2f} dBFS" for v in lvl)
        for c, lvl in enumerate(info["levels"] or []))
    if info.get("analysis") is not None:
        print("\n".join(analysis_lines(info, C)))
    else:
        print(f"mode: {info['mode']}   {'knees' if info['mode'] == 'soft' else 'clip levels'}: {lv_str}")
    print(f"flagged samples: {info['clipped_frac']*100:.2f}%   preset: {args.preset}   device: {info.get('device', 'cpu')}"
          f"   time: {info['time']:.1f}s")
    if info["clipped_frac"] == 0:
        print("no clipping or limiting detected; output equals input (use --mode soft, --knee or --clip-level to force it)")
    x = x * 10 ** (args.gain / 20)
    peak = np.abs(x).max()
    if args.normalize is not None and peak > 0:
        x = x * (10 ** (args.normalize / 20) / peak)
        peak = np.abs(x).max()
    subtype = {"float": "FLOAT", "pcm24": "PCM_24", "pcm16": "PCM_16"}[args.format]
    if subtype != "FLOAT" and peak > 1.0:
        print(f"warning: output peak {20*np.log10(peak):+.2f} dBFS exceeds 0 dBFS and will clip in {args.format}; "
              f"use --normalize or --format float", file=sys.stderr)
    out = args.output or default_output(args.input, restoration_label(mode, info, forced), args.preset)
    sf.write(out, x.astype(np.float32 if subtype == "FLOAT" else np.float64), sr, subtype=subtype)
    print(f"output: {out}  peak {20*np.log10(max(peak, 1e-12)):+.2f} dBFS ({args.format})")
    if args.save_analysis is not None:
        an = info["analysis"]
        path = args.save_analysis or os.path.join(os.path.dirname(os.path.abspath(out)),
                                                  os.path.splitext(os.path.basename(args.input))[0] + ".analysis.json")
        extra = {} if an.get("reused_from") is not None else {"file": os.path.basename(args.input)}
        save_analysis(path, an["data"], **extra)
        print(f"analysis saved: {path}")
    return 0
