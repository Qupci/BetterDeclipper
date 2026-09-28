"""Mastering-style degradations with known ground truth (for the master benchmark).

All functions take x (T, C) float64 at sample rate sr and return (y, gt, sr_out): the degraded signal,
the matching ground truth (the input, or its resampled version) and the output sample rate.
Levels are in dB relative to the source peak. 16-bit TPDF dither is applied to PCM-type outputs.
"""
import os, subprocess, tempfile, sys
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, "F:/BetterDeclipper/research/al1")


def dither16(y, seed=0):
    rng = np.random.default_rng(seed)
    q = 32768.0
    return np.round(y * q + rng.random(y.shape) - rng.random(y.shape)) / q


def _lvl(x, db):
    return np.abs(x).max() * 10 ** (db / 20)


def hard(x, sr, db=-6.0):
    c = _lvl(x, db)
    return dither16(np.clip(x, -c, c)), x, sr


def tanh_soft(x, sr, db=-6.0, knee=0.6):
    """Slope-1 below knee*c, tanh saturation towards the ceiling c above (asymptotic, no plateau)."""
    c = _lvl(x, db); t = knee * c
    a = np.abs(x)
    y = np.where(a > t, np.sign(x) * (t + (c - t) * np.tanh((a - t) / (c - t))), x)
    return dither16(y), x, sr


def cubic_soft(x, sr, db=-6.0, drive=1.0):
    """Classic cubic soft clipper on x/(drive*c): f(u) = 1.5u - 0.5u^3 for |u|<1, sign(u) beyond;
    output scaled so the ceiling is c. Has no linear region (bends from zero)."""
    c = _lvl(x, db)
    u = np.clip(x / (drive * c * 1.5), -1, 1)  # choose scale so small-signal gain is ~1/drive
    y = c * (1.5 * u - 0.5 * u ** 3)
    return dither16(y), x, sr


def soft_hard(x, sr, db=-6.0, knee=0.6, soft_over=1.15):
    """Tanh shoulder above knee*c towards soft_over*c, then a hard ceiling at c (plateau + shoulder)."""
    c = _lvl(x, db); t = knee * c; cs = soft_over * c
    a = np.abs(x)
    y = np.where(a > t, np.sign(x) * (t + (cs - t) * np.tanh((a - t) / (cs - t))), x)
    return dither16(np.clip(y, -c, c)), x, sr


def os_hard(x, sr, db=-6.0, factor=4):
    """Oversampled hard clipper: upsample, clip, downsample (rounded plateaus with ripple/overshoot)."""
    c = _lvl(x, db)
    u = resample_poly(x, factor, 1, axis=0)
    y = resample_poly(np.clip(u, -c, c), 1, factor, axis=0)[: len(x)]
    return dither16(y), x, sr


def hard_resample(x, sr, db=-6.0, sr_out=48000):
    """Hard clip at sr, then resample to sr_out (the plateau is no longer on the sample grid)."""
    c = _lvl(x, db)
    from math import gcd
    g = gcd(sr, sr_out)
    up, dn = sr_out // g, sr // g
    y = resample_poly(np.clip(x, -c, c), up, dn, axis=0)
    gt = resample_poly(x, up, dn, axis=0)
    return dither16(y), gt, sr_out


def lossy(x, sr, db=-6.0, codec="mp3", bitrate="192k", clip=True, pre_gain_db=None):
    """Hard clip, then lossy encode/decode with ffmpeg. The clip is at 0 dBFS after a gain so the
    codec sees full-scale material (like a real release); the output is scaled back to the source level."""
    c = _lvl(x, db)
    xc = np.clip(x, -c, c) if clip else x
    g = 0.999 / c if clip else 1.0
    enc = {"mp3": ["-c:a", "libmp3lame", "-b:a", bitrate, "tmp.mp3"],
           "aac": ["-c:a", "aac", "-b:a", bitrate, "tmp.m4a"],
           "vorbis": ["-c:a", "libvorbis", "-b:a", bitrate, "tmp.ogg"],
           "opus": ["-c:a", "libopus", "-b:a", bitrate, "tmp.opus"]}[codec]
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.wav")
        sf.write(src, (xc * g).astype(np.float32), sr, subtype="FLOAT")
        out = os.path.join(td, enc[-1])
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src] + enc[:-1] + [out], check=True)
        dec = os.path.join(td, "dec.wav")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", out, "-ar", str(sr), "-c:a", "pcm_f32le", dec], check=True)
        y, _ = sf.read(dec, dtype="float64", always_2d=True)
    # codec delay: align by cross-correlation (mp3/aac/vorbis priming)
    a, b = xc[:, 0] * g, y[:, 0]
    N = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    cc = np.fft.irfft(np.conj(np.fft.rfft(a, N)) * np.fft.rfft(b, N), N)
    lag = int(np.argmax(cc[:8192]))
    y = y[lag:lag + len(x)]
    if len(y) < len(x):
        y = np.concatenate([y, np.zeros((len(x) - len(y), y.shape[1]))])
    return y / g, x, sr


def al1(x, sr, db=-6.0, **params):
    """AL-1 limiter (the user's VST3) with the ceiling at `db` below the source peak."""
    from al1 import render
    pk = np.abs(x).max()
    p = dict(ceiling=-6.0, detector="Slow")
    p.update(params)
    # AL-1 ceiling is absolute: normalize the source to 0 dBFS peak, render, scale back
    y, lag, _ = render(x / pk * 10 ** (-0.0 / 20), sr, **dict(p, ceiling=db))
    return dither16(y * pk), x, sr


def os_hard_al1(x, sr, db_clip=-4.0, db=-6.0, factor=4):
    """EDM-style chain: oversampled hard clip, then AL-1 limiting to the ceiling."""
    yc, _, _ = os_hard(x, sr, db_clip, factor)
    y, _, _ = al1(yc, sr, db)
    return y, x, sr


DEGRADATIONS = {
    "hard6": lambda x, sr: hard(x, sr, -6),
    "hard9": lambda x, sr: hard(x, sr, -9),
    "tanh6": lambda x, sr: tanh_soft(x, sr, -6, 0.6),
    "tanh9k4": lambda x, sr: tanh_soft(x, sr, -9, 0.4),
    "cubic6": lambda x, sr: cubic_soft(x, sr, -6, 1.0),
    "softhard6": lambda x, sr: soft_hard(x, sr, -6, 0.6, 1.15),
    "softhard9": lambda x, sr: soft_hard(x, sr, -9, 0.7, 1.1),
    "oshard6": lambda x, sr: os_hard(x, sr, -6, 4),
    "resamp6": lambda x, sr: hard_resample(x, sr, -6, 48000),
    "mp3_128": lambda x, sr: lossy(x, sr, -6, "mp3", "128k"),
    "aac_256": lambda x, sr: lossy(x, sr, -6, "aac", "256k"),
    "vorbis_192": lambda x, sr: lossy(x, sr, -6, "vorbis", "192k"),
    "al1_6": lambda x, sr: al1(x, sr, -6),
    "osal1": lambda x, sr: os_hard_al1(x, sr, -4, -7),
    "al1slow": lambda x, sr: al1(x, sr, -6, engine="Gen 1", mode="Continuous", attack=5.0, release=123.0),
}
