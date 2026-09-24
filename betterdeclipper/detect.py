"""Clip-level detection.

Clipped samples form a dense "plateau" in the amplitude histogram near the signal peak.
When the clipped file has been dithered/requantized, the plateau is smeared over a few LSBs,
so we locate the plateau by its histogram density relative to the background density just
below it, and use the plateau's lower edge as the clip bound.
"""
import numpy as np


def _plateau_lower_edge(s, lsb, min_run=4, ratio=8.0, search=0.15, peak_discard=1e-4):
    """Find the clip bound for one polarity. `s` holds the non-negative (polarity-flipped) samples.

    Returns (level, n_clipped) or (None, 0) if no plateau was found.
    """
    s = s[s > 0]
    if s.size < 100:
        return None, 0
    # robust peak: discard a tiny fraction of the largest values (glitches)
    k = int(np.floor(s.size * peak_discard))
    top = np.partition(s, s.size - 1 - k)[s.size - 1 - k] if k > 0 else s.max()
    peak = s.max()
    bw = max(lsb, top * 1e-5)
    lo = top * (1.0 - search)
    v = s[s >= lo]
    nb = int(np.ceil((peak - lo) / bw)) + 1
    hist = np.bincount(np.minimum(((v - lo) / bw).astype(np.int64), nb - 1), minlength=nb)
    # background density: median bin count in the lower half of the search region
    bg = max(np.median(hist[: nb // 2]) if nb >= 4 else 1.0, 1.0)
    dense = hist > ratio * bg
    # walk down from the robust peak bin while bins are dense (allow tiny gaps)
    i = min(int((top - lo) / bw), nb - 1)
    # move up to the densest bin near the top (plateau may sit slightly below `top`)
    while i >= 0 and not dense[i]:
        i -= 1
        if (nb - 1 - i) * bw > 0.02 * top:
            return None, 0
    if i < 0:
        return None, 0
    j = i
    gap = 0
    while j - 1 >= 0 and (dense[j - 1] or (gap < 1 and j - 2 >= 0 and dense[j - 2])):
        if not dense[j - 1]:
            gap += 1
        j -= 1
    level = lo + j * bw
    n = int(np.sum(s >= level))
    if n < min_run:
        return None, 0
    return level, n


def detect_clip_levels(y, lsb=None):
    """Detect positive/negative clip levels per channel.

    y: (T,) or (T, C) float array in [-1, 1].
    lsb: quantization step (e.g. 1/32768 for 16-bit). If None it is estimated.
    Returns list of (theta_pos, theta_neg) per channel, where theta_neg is negative;
    entries are None when no clipping was detected for that polarity.
    """
    y = np.atleast_2d(np.asarray(y, dtype=np.float64).T).T if y.ndim == 1 else np.asarray(y, dtype=np.float64)
    if lsb is None:
        lsb = estimate_lsb(y)
    out = []
    for c in range(y.shape[1]):
        yc = y[:, c]
        tp, _ = _plateau_lower_edge(yc, lsb)
        tn, _ = _plateau_lower_edge(-yc, lsb)
        out.append((tp, -tn if tn is not None else None))
    return out


def estimate_lsb(y):
    """Estimate the quantization step of y (returns a tiny value for true float audio)."""
    v = np.abs(np.asarray(y, dtype=np.float64).ravel()[:200000])
    for bits in (8, 16, 24):
        q = 2.0 ** (bits - 1)
        if np.all(np.abs(v * q - np.round(v * q)) < 1e-6):
            return 1.0 / q
    return 1e-7


def clip_masks(y, levels):
    """Build boolean masks (reliable, high, low) and bound arrays for each channel.

    Returns mask_hi, mask_lo (T, C) and theta_hi, theta_lo (C,) arrays. Channels/polarities
    without detected clipping get infinite thresholds (never clipped).
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    C = y.shape[1]
    th_hi = np.array([lv[0] if lv[0] is not None else np.inf for lv in levels])
    th_lo = np.array([lv[1] if lv[1] is not None else -np.inf for lv in levels])
    m_hi = y >= th_hi[None, :]
    m_lo = y <= th_lo[None, :]
    return m_hi, m_lo, th_hi, th_lo


def detect_knee(y, fit_range=(0.25, 0.5), excess=1.6, run=6, bins=200, peak_discard=1e-4):
    """Soft-clipping knee per channel and polarity.

    Fits the natural (log-linear) decay of the amplitude histogram on `fit_range` x peak and returns the
    lowest level above the fit range where the observed density exceeds the fitted trend by `excess`
    for `run` consecutive bins (compressed samples pile up below the ceiling). Returns a list of
    (knee_pos, knee_neg) per channel (knee_neg negative), None when no pile-up is found.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    out = []
    for c in range(y.shape[1]):
        res = []
        for sgn in (1.0, -1.0):
            s = sgn * y[:, c]
            s = s[s > 0]
            if s.size < 1000:
                res.append(None)
                continue
            k = int(np.floor(s.size * peak_discard))
            peak = np.partition(s, s.size - 1 - k)[s.size - 1 - k] if k > 0 else s.max()
            h, e = np.histogram(np.minimum(s / peak, 1.0), bins=bins, range=(0.0, 1.0))
            u = 0.5 * (e[:-1] + e[1:])
            sel = (u >= fit_range[0]) & (u <= fit_range[1]) & (h > 0)
            if sel.sum() < 5:
                res.append(None)
                continue
            b, a = np.polyfit(u[sel], np.log(h[sel]), 1)
            b = min(b, 0.0)  # natural density does not grow with amplitude
            pred = np.exp(a + b * u)
            hs = np.convolve(h, np.ones(3) / 3, mode="same")
            over = hs > excess * pred
            knee = None
            start = int(np.searchsorted(u, fit_range[1]))
            for i in range(start, bins - run + 1):
                if over[i:i + run].all():
                    knee = e[i] * peak
                    break
            res.append(None if knee is None else sgn * knee)
        out.append(tuple(res))
    return out
