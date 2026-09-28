"""Blind estimate of a static compression curve from the amplitude histogram (quantile matching).

The natural amplitude density of the (unknown) original is modelled on a low range where the
signal is assumed untouched, as log-linear (exponential tail); its survival function is extrapolated
upward. For an observed level y, the original level is x = S_x^-1(S_y(y)) (monotone memoryless map),
so the compression at y is x/y - 1.
"""
import numpy as np


def compression_curve(s, fit=(0.2, 0.45), peak_discard=1e-4, levels=None, quad=False):
    """s: positive samples (one polarity). Returns (levels (fraction of robust peak), est. compression x/y - 1, peak)."""
    s = np.sort(s[s > 0])
    n = s.size
    if n < 5000:
        return None
    k = int(np.floor(n * peak_discard))
    peak = s[n - 1 - k]
    a0, a1 = fit[0] * peak, fit[1] * peak
    sel = s[(s >= a0) & (s < a1)]
    h, e = np.histogram(sel, bins=60)
    u = 0.5 * (e[1:] + e[:-1]); bw = e[1] - e[0]
    ok = h > 0
    deg = 2 if quad else 1
    coef = np.polyfit(u[ok], np.log(h[ok] / bw), deg, w=np.sqrt(h[ok]))
    if not quad:
        beta, alpha = coef
        beta = min(beta, -1e-6 / peak)
        surv = lambda a: np.exp(alpha + beta * a) / (-beta)          # expected count above a
        inv = lambda S: (np.log(S * (-beta)) - alpha) / beta
    else:
        # numeric survival for the quadratic log-density (upper tail integral on a fine grid)
        grid = np.linspace(a0, 20 * peak, 200000)
        dens = np.exp(np.clip(np.polyval(coef, grid), -700, 700))
        dens[np.polyval(np.polyder(coef), grid) > 0] = 0  # ignore any rising part of the fitted curve
        cs = np.cumsum(dens[::-1])[::-1] * (grid[1] - grid[0])
        surv = lambda a: np.interp(a, grid, cs)
        inv = lambda S: np.interp(-S, -cs, grid)
    # anchor: expected count above a1 must equal the observed count above a1 (normalization)
    obs_a1 = n - np.searchsorted(s, a1)
    scale = obs_a1 / max(surv(a1), 1e-30)
    if levels is None:
        levels = np.linspace(fit[1], 1.0, 56)
    ys = levels * peak
    Sy = n - np.searchsorted(s, ys)                    # observed count above y
    x = inv(np.maximum(Sy, 0.5) / scale)
    return levels, x / ys - 1.0, peak


def knee_from_compression(s, thr=0.025, **kw):
    r = compression_curve(s, **kw)
    if r is None:
        return None
    lv, comp, peak = r
    over = np.flatnonzero(comp > thr)
    return None if over.size == 0 else lv[over[0]] * peak
