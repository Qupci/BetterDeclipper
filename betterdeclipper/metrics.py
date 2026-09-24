"""Objective quality metrics for declipping evaluation."""
import numpy as np


def sdr(ref, est):
    """Signal-to-distortion ratio in dB (plain, no scale invariance)."""
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    num = np.sum(ref ** 2)
    den = np.sum((ref - est) ** 2)
    return 10.0 * np.log10(num / max(den, 1e-30))


def sdr_masked(ref, est, mask):
    """SDR computed only on the samples selected by `mask` (e.g. the clipped samples)."""
    return sdr(np.asarray(ref)[mask], np.asarray(est)[mask])


def report(ref, clipped, est, mask=None):
    """Return a dict with SDR, delta SDR and (optionally) clipped-sample SDR."""
    out = {
        "sdr_in": sdr(ref, clipped),
        "sdr": sdr(ref, est),
    }
    out["dsdr"] = out["sdr"] - out["sdr_in"]
    if mask is not None and np.any(mask):
        out["sdr_c"] = sdr_masked(ref, est, mask)
    return out
