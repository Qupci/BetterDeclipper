"""Smeared-ceiling model fit on the top of the amplitude histogram.

Observed density of |y| near the top (one channel, one polarity):
    p(a) = [n(a) 1(a < theta) + M delta(a - theta)] * N(0, sigma^2)
with the natural density n(a) = exp(alpha + beta a) (log-linear locally). Closed form:
    p(a) = exp(alpha + beta a + beta^2 sigma^2 / 2) Phi((theta - a - beta sigma^2) / sigma) + M phi((a - theta)/sigma)/sigma
theta: clip level, sigma: smearing (codec noise, resampling/oversampling ripple, dither), M: clipped mass.
Fitted by Poisson maximum likelihood on histogram bins.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr, log_ndtr


def _model(par, u):
    alpha, beta, theta, lsig, lM = par
    sig = np.exp(lsig)
    cont = np.exp(np.clip(alpha + beta * u + 0.5 * beta ** 2 * sig ** 2 + log_ndtr((theta - u - beta * sig ** 2) / sig), -700, 700))
    bump = np.exp(lM) * np.exp(-0.5 * ((u - theta) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
    return cont + bump


def fit_smeared(s, lsb=1e-7, lo_frac=0.45, nbins=600, peak_discard=1e-5):
    """s: positive samples of one polarity. Returns dict(theta, sigma, M, alpha, beta, nll, peak, bg_at_theta)
    in absolute amplitude units, or None if too few samples."""
    s = s[s > 0]
    if s.size < 2000:
        return None
    k = int(np.floor(s.size * peak_discard))
    peak = np.partition(s, s.size - 1 - k)[s.size - 1 - k] if k > 0 else s.max()
    lo = lo_frac * peak
    v = s[(s >= lo) & (s <= peak)]
    bw = max((peak - lo) / nbins, lsb)
    edges = np.arange(lo, peak + bw, bw)
    h, e = np.histogram(v, bins=edges)
    u = 0.5 * (e[1:] + e[:-1])
    # normalized coordinates (peak = 1) for conditioning
    un = u / peak
    bwn = bw / peak
    hs = np.convolve(h, np.ones(5) / 5, "same")
    # initial natural trend: log-linear fit on the lower half of the range
    sel = (un < 0.5 * (lo_frac + 1)) & (h > 0)
    b0, a0 = np.polyfit(un[sel], np.log(h[sel] / bwn), 1) if sel.sum() > 5 else (0.0, np.log(max(h.mean(), 1) / bwn))
    best = None
    # candidate theta: largest excess over the trend and the steepest drop of the smoothed log density
    trend = np.exp(a0 + b0 * un) * bwn
    ratio = hs / np.maximum(trend, 1e-9)
    cands = {float(un[np.argmax(ratio)])}
    ld = np.log(np.maximum(hs, 0.5))
    dld = np.gradient(ld)
    cands.add(float(un[np.argmin(dld[: len(dld) - 3])]))
    cands.add(1.0 - 2 * bwn)
    for th0 in cands:
        for sg0 in (0.003, 0.01, 0.03):
            exc = max(float(np.sum(np.maximum(h - trend, 0)[np.abs(un - th0) < 3 * sg0 + 2 * bwn])), 1.0)
            x0 = np.array([a0, min(b0, 5.0), th0, np.log(sg0), np.log(exc)])

            def nll(p):
                lam = np.maximum(_model(p, un) * bwn, 1e-12)
                return float(np.sum(lam - h * np.log(lam)))

            bounds = [(None, None), (-200, 50), (lo_frac, 1.05), (np.log(max(bwn * 0.25, 1e-6)), np.log(0.2)), (np.log(1e-3), np.log(s.size))]
            r = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best[0]:
                best = (r.fun, r.x)
    alpha, beta, theta, lsig, lM = best[1]
    sig = np.exp(lsig)
    bg = np.exp(alpha + beta * theta)  # natural density at theta (per unit of normalized amplitude)
    return dict(theta=theta * peak, sigma=sig * peak, M=float(np.exp(lM)), alpha=alpha, beta=beta, nll=best[0], peak=peak,
                bg_at_theta=bg, bump_ratio=float(np.exp(lM) / max(bg * sig * np.sqrt(2 * np.pi), 1e-12)), n=int(v.size))
