"""BetterDeclipper engine: detection, chunked processing and model fusion.

The restoration is an average of several consistent estimates from structurally different
sparse models (plug-and-play PEW social sparsity, and stereo A-SPADE) at different time-frequency
resolutions. Every model output satisfies the clipping constraints, and so does their average
(the constraint set is convex); their errors are only partly correlated, so averaging improves
the restoration noticeably (see research/LOG.md).
"""
import math
import time
import numpy as np
import torch

torch.set_flush_denormal(True)  # denormals are extremely slow on older x86 CPUs (NMF updates create them)

from .detect import detect_clip_levels, clip_masks, estimate_lsb
from .methods.pnp import declip_pnp
from .methods.spade import declip_spade
from .methods.social import channel_mixing
from .stft import TightSTFT


def nice_len(n):
    """Nearest FFT-friendly even length (2^a * 3^b) to n."""
    best = None
    for a in range(1, 21):
        for b in range(0, 4):
            v = 2 ** a * 3 ** b
            if best is None or abs(v - n) < abs(best - n):
                best = v
    return best


# Model definitions. Window lengths are in ms (93 ms == 4096 samples @ 44.1 kHz).
# nmf: plug-and-play with an NMF-Wiener denoiser (low-rank spectrogram model, rank ~0.3 x frames);
# pnp: plug-and-play with PEW social shrinkage; spade: stereo A-SPADE (eps/s scale with the window).
PRESETS = {
    "fast": [("nmf", 93, dict(n_iter=150))],
    "normal": [("nmf", 93, {}), ("spade", 93, {})],
    "high": [("nmf", 93, {}), ("pnp", 93, {}), ("spade", 93, {})],
    "best": [("nmf", 93, dict(n_iter=800)), ("pnp", 93, dict(n_iter=800)), ("spade", 93, {})],
}


def _model_kwargs(kind, win_ms, sr, extra):
    W = nice_len(win_ms * 1e-3 * sr)
    if kind == "pnp":
        kw = dict(win_len=W, hop=W // 4, neigh=(3, 7), n_iter=400, stereo="pca", chan_gain=[1.0, 2.5])
    elif kind == "nmf":
        kw = dict(win_len=W, hop=W // 4, n_iter=400, stereo="pca", den_type="nmf", nmf_rank=128, nmf_rank_ratio=0.3, nmf_iter=1,
                  gain_mode="wiener")
    else:
        r = W / 4096.0
        kw = dict(win_len=W, hop=W // 4, variant="a", s=max(1, int(round(8 * r))), eps=3.0 * math.sqrt(r),
                  stereo="pca", max_iter=2000)
    kw.update(extra)
    return kw


def _chunks(T, chunk, ctx, fade):
    """Yield (a, b, ia, ib): processing span [a,b) and kept interior [ia,ib) (with fade overlap).
    A short tail (< chunk/2) is merged into the previous chunk."""
    starts = list(range(0, T, chunk))
    if len(starts) > 1 and T - starts[-1] < chunk // 2:
        starts.pop()
    for i, s in enumerate(starts):
        e = T if i == len(starts) - 1 else min(T, s + chunk)
        ia = max(0, s - fade // 2) if i > 0 else 0
        ib = min(T, e + fade // 2) if e < T else T
        yield max(0, ia - ctx), min(T, ib + ctx), ia, ib


def declip(y, sr, preset="normal", levels=None, chunk_s=20.0, ctx_s=1.5, fade_s=0.05,
           threads=None, verbose=True, progress=None, models=None):
    """Declip y (T, C) float array. Returns (x_hat (T, C), info dict)."""
    t_start = time.time()
    if threads:
        torch.set_num_threads(threads)
    y = np.asarray(y, dtype=np.float64)
    mono = y.ndim == 1
    if mono:
        y = y[:, None]
    T, C = y.shape
    lsb = estimate_lsb(y)
    if levels is None:
        levels = detect_clip_levels(y, lsb)
    m_hi, m_lo, th_hi, th_lo = clip_masks(y, levels)
    clipped = m_hi | m_lo
    info = dict(levels=levels, clipped_frac=float(clipped.mean()), lsb=lsb, preset=preset)
    if not clipped.any():
        info["time"] = time.time() - t_start
        return (y[:, 0] if mono else y), info
    models = models or PRESETS[preset]
    # file-global lambda reference for the PnP models (consistent schedule over chunks)
    scale = float(np.max(np.abs(np.concatenate([th_hi[np.isfinite(th_hi)], th_lo[np.isfinite(th_lo)]]))))
    Q = torch.as_tensor(channel_mixing(y, "pca" if C == 2 else "none"), dtype=torch.float32)
    lam_refs = {}
    for kind, win_ms, extra in models:
        if kind in ("pnp", "nmf") and win_ms not in lam_refs:
            W = nice_len(win_ms * 1e-3 * sr)
            st = TightSTFT(W, W // 4)
            u = torch.einsum("ji,tj->it", Q, torch.as_tensor(y / scale, dtype=torch.float32))
            zmax = 0.0
            for a in range(0, T, 60 * sr):  # blockwise to bound memory
                seg = u[:, a:a + 60 * sr + W]
                if seg.shape[1] >= W:
                    zmax = max(zmax, float(torch.abs(st.analysis(seg)).max()))
            lam_refs[win_ms] = zmax
    chunk, ctx, fade = int(chunk_s * sr), int(ctx_s * sr), max(2, int(fade_s * sr))
    out = y.copy()
    wsum = np.zeros(T)
    acc = np.zeros((T, C))
    spans = list(_chunks(T, chunk, ctx, fade))
    for ci, (a, b, ia, ib) in enumerate(spans):
        # crossfade weights for the kept interior
        wt = np.ones(ib - ia)
        if ia > 0:
            wt[:fade] = np.linspace(0, 1, fade + 2)[1:-1][: min(fade, ib - ia)]
        if ib < T:
            wt[-fade:] = np.minimum(wt[-fade:], np.linspace(1, 0, fade + 2)[1:-1])
        if not clipped[a:b].any():
            acc[ia:ib] += y[ia:ib] * wt[:, None]
            wsum[ia:ib] += wt
            continue
        yc, mh, ml = y[a:b], m_hi[a:b], m_lo[a:b]
        ests = []
        for kind, win_ms, extra in models:
            kw = _model_kwargs(kind, win_ms, sr, extra)
            if kind in ("pnp", "nmf"):
                est = declip_pnp(yc, mh, ml, th_hi, th_lo, sr=sr, lam_ref=lam_refs[win_ms], **kw)
            else:
                est = declip_spade(yc, mh, ml, th_hi, th_lo, **kw)
            ests.append(est)
        est = np.mean(ests, axis=0)
        acc[ia:ib] += est[ia - a:ib - a] * wt[:, None]
        wsum[ia:ib] += wt
        if progress:
            progress(ci + 1, len(spans), time.time() - t_start)
    out = acc / np.maximum(wsum, 1e-12)[:, None]
    # exact consistency: reliable samples untouched, clipped samples beyond the clip level
    out[~clipped] = y[~clipped]
    out = np.where(m_hi, np.maximum(out, th_hi[None, :]), out)
    out = np.where(m_lo, np.minimum(out, th_lo[None, :]), out)
    info["time"] = time.time() - t_start
    return (out[:, 0] if mono else out), info
