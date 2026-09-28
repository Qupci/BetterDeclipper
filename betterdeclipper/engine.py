"""BetterDeclipper engine: detection, chunked processing and model fusion.

The restoration is an average of several consistent estimates from structurally different
sparse models (plug-and-play PEW social sparsity, and stereo A-SPADE) at different time-frequency
resolutions. Every model output satisfies the clipping constraints, and so does their average
(the constraint set is convex); their errors are only partly correlated, so averaging improves
the restoration noticeably (see research/LOG.md).
"""
import math
import time
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*smallest subnormal.*")  # side effect of flush-to-zero below
torch.set_flush_denormal(True)  # denormals are extremely slow on older x86 CPUs (NMF updates create them)

from .detect import detect_clip_levels, clip_masks, estimate_lsb, detect_knee
from .auto import auto_constraints
from .methods.common import full_thresholds, threshold_scale
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


# Model definitions. Window lengths are in ms (93 ms == 4096 samples @ 44.1 kHz). Optional "weight"
# in the model options sets its share in the (convex) fusion; default equal weights. Optional
# "weight_hi" sets a different share above FUSION_XOVER_HZ (two-band fusion, see fuse()).
# nmf: plug-and-play with an NMF-Wiener denoiser (low-rank spectrogram model, rank ~0.3 x frames);
# pnp: plug-and-play with PEW social shrinkage; spade: stereo A-SPADE (eps/s scale with the window).
# Above FUSION_XOVER_HZ only the NMF model is used: SPADE (and PEW) add spiky high-frequency errors in long
# clipped gaps (audible clicks) while their gain is in the low/mid range (research/LOG.md, session 4).
PRESETS = {
    "fast": [("nmf", 93, dict(n_iter=150))],
    "normal": [("nmf", 93, dict(weight=0.65, weight_hi=1.0)), ("spade", 93, dict(weight=0.35, weight_hi=0.0))],
    "high": [("nmf", 93, dict(weight_hi=1.0)), ("pnp", 93, dict(weight_hi=0.0)), ("spade", 93, dict(weight_hi=0.0))],
    "best": [("nmf", 93, dict(n_iter=800, weight_hi=1.0)), ("pnp", 93, dict(n_iter=800, weight_hi=0.0)),
             ("spade", 93, dict(weight_hi=0.0))],
}


FUSION_XOVER_HZ = 4000.0


def fuse(y, ests, w_lo, w_hi, sr, fc=FUSION_XOVER_HZ):
    """Convex fusion of consistent estimates, optionally with different weights above/below fc.

    Each estimate equals y on reliable samples, so its correction e_i = est_i - y lives on the flagged
    samples. Two-band fusion: x = y + sum_i [w_lo_i LP(e_i) + w_hi_i (e_i - LP(e_i))] with a zero-phase
    Butterworth low-pass LP (complementary bands, weights normalized per band). The caller re-applies
    the consistency projection (the filtered corrections leak slightly onto reliable samples)."""
    w_lo = np.asarray(w_lo, float) / np.sum(w_lo)
    w_hi = np.asarray(w_hi, float) / np.sum(w_hi)
    if np.allclose(w_lo, w_hi):
        return np.tensordot(w_lo, np.stack(ests, 0), axes=1)
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, min(fc, 0.45 * sr), "lp", fs=sr, output="sos")
    out = y.copy()
    for e, wl, wh in zip(ests, w_lo, w_hi):
        corr = e - y
        lo = sosfiltfilt(sos, corr, axis=0)
        out += wl * lo + wh * (corr - lo)
    return out


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


def resolve_device(device="auto"):
    """'auto' -> first CUDA GPU if this torch build has CUDA and a GPU is present, else CPU."""
    if device in (None, "auto"):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available (CPU-only torch build or no NVIDIA GPU); use --device cpu")
    return dev


def default_knees(y, knees=None, frac=0.8, peak_discard=1e-4):
    """Fill missing soft-clip knees with frac x robust peak (per channel and polarity).
    On synthetic tanh saturation the best declared knee was ~0.75-0.8 x peak (research/LOG.md)."""
    out = []
    for c in range(y.shape[1]):
        kp, kn = knees[c] if knees is not None else (None, None)
        pos, neg = y[:, c][y[:, c] > 0], -y[:, c][y[:, c] < 0]
        rp = lambda v: np.quantile(v, 1 - peak_discard) if v.size > 100 else (v.max() if v.size else None)
        if kp is None and pos.size:
            kp = frac * rp(pos)
        if kn is None and neg.size:
            kn = -frac * rp(neg)
        out.append((kp, kn))
    return out


def soft_constraints(y, knees, lsb, tol_lsb=2.0):
    """Soft-clip constraints: beyond the knee, the original is at least as large as the observed
    value (minus a small quantization tolerance). Returns masks and per-sample thresholds (T, C)."""
    T, C = y.shape
    m_hi = np.zeros((T, C), bool); m_lo = np.zeros((T, C), bool)
    th_hi = np.full((T, C), np.inf); th_lo = np.full((T, C), -np.inf)
    tol = tol_lsb * max(lsb, 1e-7)
    for c, (kp, kn) in enumerate(knees):
        if kp is not None:
            m_hi[:, c] = y[:, c] >= kp
            th_hi[m_hi[:, c], c] = y[m_hi[:, c], c] - tol
        if kn is not None:
            m_lo[:, c] = y[:, c] <= kn
            th_lo[m_lo[:, c], c] = y[m_lo[:, c], c] + tol
    return m_hi, m_lo, th_hi, th_lo


def snap_levels(y, levels, lsb, tol_db=0.3):
    """User-given clip levels are approximate (-12 dBFS = 0.2512, but a 16-bit clip is often at 0.25):
    use a detected plateau edge within tol_db instead, else the level itself."""
    det = detect_clip_levels(y, lsb)
    out = []
    for (up, lo), (dp, dl) in zip(levels, det):
        f = 10 ** (tol_db / 20)
        up = dp if (up is not None and dp is not None and up / f <= dp <= up * f) else up
        lo = dl if (lo is not None and dl is not None and abs(lo) / f <= abs(dl) <= abs(lo) * f) else lo
        out.append((up, lo))
    return out


def declip(y, sr, preset="normal", levels=None, chunk_s=20.0, ctx_s=1.5, fade_s=0.05,
           threads=None, verbose=True, progress=None, models=None, mode="auto", knees=None, max_gain_db=None,
           device="auto", constraints=None, dips=None):
    """Declip y (T, C) float array. Returns (x_hat (T, C), info dict).

    mode: 'auto' (analyze the ceilings: clip plateau, limiter, smeared (lossy/resampled) ceiling, soft
    shoulder -> matching constraints, see auto.py), 'hard' (ceilings only, no soft shoulder), 'soft' (always
    add a soft region below the ceiling; knee from the first pass or 0.8 x ceiling), 'legacy' (the
    previous plateau-or-knee logic). Explicit `levels` (clip levels) or `knees` override the analysis.
    device: 'auto' (CUDA GPU if available, else CPU), 'cpu', 'cuda' or 'cuda:N'."""
    t_start = time.time()
    if threads:
        torch.set_num_threads(threads)
    device = resolve_device(device)
    y = np.asarray(y, dtype=np.float64)
    mono = y.ndim == 1
    if mono:
        y = y[:, None]
    T, C = y.shape
    lsb = estimate_lsb(y)
    used_mode = mode
    auto_report = None
    gcap = None  # optional per-sample cap on |x| / threshold for flagged samples (T, C), >= 1, inf = none
    if constraints is not None:  # research hook: explicit (m_hi, m_lo, th_hi, th_lo[, gcap])
        m_hi, m_lo, th_hi, th_lo = constraints[:4]
        gcap = constraints[4] if len(constraints) > 4 else None
        used_mode = "custom"
    elif mode in ("auto", "hard", "soft") and levels is None and knees is None:
        # ceiling analysis (clip plateau / limiter / smeared ceiling) + soft-shoulder knee (first pass)
        def run_fast(yy, cons):  # analysis pass: 80 NMF iterations give the same knees as 150
            xx, _ = declip(yy, sr, models=[("nmf", 93, dict(n_iter=80))], constraints=cons, device=device,
                           verbose=False, chunk_s=chunk_s, ctx_s=ctx_s, fade_s=fade_s)
            return xx
        m_hi, m_lo, th_hi, th_lo, rep = auto_constraints(y, lsb, run_fast, mode=mode, sr=sr)
        gcap = rep.pop("gcap", None)
        levels = rep["knees"]
        used_mode = rep["mode"]
        auto_report = rep
    elif mode == "soft":
        knees = default_knees(y, knees or detect_knee(y))
        m_hi, m_lo, th_hi, th_lo = soft_constraints(y, knees, lsb)
        levels = knees
    else:
        if levels is None:
            levels = detect_clip_levels(y, lsb)
        else:
            levels = snap_levels(y, levels, lsb)
        m_hi, m_lo, th_hi, th_lo = clip_masks(y, levels)
        if mode in ("auto", "legacy") and all(a is None and b is None for a, b in levels):
            knees = knees or detect_knee(y)
            if any(a is not None or b is not None for a, b in knees):
                m_hi, m_lo, th_hi, th_lo = soft_constraints(y, knees, lsb)
                levels, used_mode = knees, "soft"
        elif mode in ("auto", "legacy"):
            used_mode = "hard"
    if max_gain_db is not None:
        g = 10 ** (max_gain_db / 20)
        gcap = g if gcap is None else np.minimum(gcap, g)
    clipped = m_hi | m_lo
    info = dict(levels=levels, clipped_frac=float(clipped.mean()), lsb=lsb, preset=preset, mode=used_mode,
                device=str(device), analysis=auto_report)
    if not clipped.any():
        info["time"] = time.time() - t_start
        return (y[:, 0] if mono else y), info
    models = models or PRESETS[preset]
    # file-global lambda reference for the PnP models (consistent schedule over chunks)
    scale = threshold_scale(th_hi, th_lo)
    Q = torch.as_tensor(channel_mixing(y, "pca" if C == 2 else "none"), dtype=torch.float32, device=device)
    lam_refs = {}
    for kind, win_ms, extra in models:
        if kind in ("pnp", "nmf") and win_ms not in lam_refs:
            W = nice_len(win_ms * 1e-3 * sr)
            st = TightSTFT(W, W // 4, device=device)
            u = torch.einsum("ji,tj->it", Q, torch.as_tensor(y / scale, dtype=torch.float32, device=device))
            zmax = 0.0
            for a in range(0, T, 60 * sr):  # blockwise to bound memory
                seg = u[:, a:a + 60 * sr + W]
                if seg.shape[1] >= W:
                    zmax = max(zmax, float(torch.abs(st.analysis(seg)).max()))
            lam_refs[win_ms] = zmax
    chunk, ctx, fade = int(chunk_s * sr), int(ctx_s * sr), max(2, int(fade_s * sr))
    spans = list(_chunks(T, chunk, ctx, fade))
    n_steps = sum(1 if kind == "spade" else len(spans) for kind, _, _ in models)
    step = [0]

    def tick():
        step[0] += 1
        if progress:
            progress(step[0], n_steps, time.time() - t_start)

    ests, wts, wts_hi = [], [], []
    for kind, win_ms, extra in models:
        extra = dict(extra)
        wts.append(extra.pop("weight", 1.0))
        wts_hi.append(extra.pop("weight_hi", wts[-1]))
        kw = _model_kwargs(kind, win_ms, sr, extra)
        if gcap is not None:
            kw["max_gain"] = gcap
        if kind == "spade":
            # frame-wise and independent per frame: one pass over the whole file (internally batched)
            ests.append(declip_spade(y, m_hi, m_lo, th_hi, th_lo, device=device, **kw))
            tick()
            continue
        # global (per-chunk) models: overlapping chunks with context, crossfaded interiors
        acc = np.zeros((T, C))
        wsum = np.zeros(T)
        for a, b, ia, ib in spans:
            wt = np.ones(ib - ia)
            if ia > 0:
                wt[:fade] = np.linspace(0, 1, fade + 2)[1:-1][: min(fade, ib - ia)]
            if ib < T:
                wt[-fade:] = np.minimum(wt[-fade:], np.linspace(1, 0, fade + 2)[1:-1])
            if clipped[a:b].any():
                thh = th_hi if th_hi.ndim == 1 else th_hi[a:b]
                thl = th_lo if th_lo.ndim == 1 else th_lo[a:b]
                kwc = dict(kw)
                if gcap is not None and np.ndim(gcap) == 2:
                    kwc["max_gain"] = gcap[a:b]
                if dips is not None and kind != "spade":
                    kwc["dips"] = (dips[0][a:b], dips[1][a:b], dips[2], dips[3])
                est = declip_pnp(y[a:b], m_hi[a:b], m_lo[a:b], thh, thl, sr=sr, lam_ref=lam_refs[win_ms],
                                 device=device, **kwc)[ia - a:ib - a]
            else:
                est = y[ia:ib]
            acc[ia:ib] += est * wt[:, None]
            wsum[ia:ib] += wt
            tick()
        ests.append(acc / np.maximum(wsum, 1e-12)[:, None])
    out = fuse(y, ests, wts, wts_hi, sr)
    del ests
    # exact consistency: reliable samples untouched, clipped samples beyond the clip level
    out[~clipped] = y[~clipped]
    out = np.where(m_hi, np.maximum(out, full_thresholds(th_hi, out.shape)), out)
    out = np.where(m_lo, np.minimum(out, full_thresholds(th_lo, out.shape)), out)
    if gcap is not None:
        with np.errstate(invalid="ignore"):
            ub = np.where(np.isinf(gcap), np.inf, gcap * full_thresholds(th_hi, out.shape))
            lb = np.where(np.isinf(gcap), -np.inf, gcap * full_thresholds(th_lo, out.shape))
        out = np.where(m_hi, np.minimum(out, ub), out)
        out = np.where(m_lo, np.maximum(out, lb), out)
    info["time"] = time.time() - t_start
    return (out[:, 0] if mono else out), info
