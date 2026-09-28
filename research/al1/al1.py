"""Render audio through the user's AL-1 VST3 (Naturl Audio) with pedalboard; latency-compensated."""
import sys
sys.path.insert(0, "F:/BetterDeclipper/research/_vendor")
import numpy as np
import pedalboard

PLUG = r"D:/SOFTWARE/Audio/VST3/AL-1.vst3/AL-1.vst3"


def load(**params):
    p = pedalboard.load_plugin(PLUG)
    for k, v in params.items():
        setattr(p, k, v)
    return p


def render(x, sr, tail=32768, **params):
    """x: (T, C). Returns (y (T, C), latency in samples, plugin)."""
    p = load(**params)
    xin = np.concatenate([x, np.zeros((tail, x.shape[1]))], 0).T.astype(np.float32)
    y = p.process(xin, sr, buffer_size=4096, reset=True).T.astype(np.float64)
    # latency: cross-correlate a low-level band (limiter does not act there) -> use full signal xcorr
    a, b = x[:, 0], y[:, 0]
    N = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    c = np.fft.irfft(np.conj(np.fft.rfft(a, N)) * np.fft.rfft(b, N), N)
    lag = int(np.argmax(c[: tail]))
    return y[lag:lag + len(x)], lag, p
