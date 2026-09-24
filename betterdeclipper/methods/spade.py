"""SPADE declipping (A-SPADE / S-SPADE), frame-wise, vectorized over all frames with torch.

References:
 - Kitic, Bertin, Gribonval, "Sparsity and cosparsity for audio declipping: a flexible
   non-convex approach", LVA/ICA 2015 (A-SPADE, S-SPADE).
 - Zaviska, Rajmic, Prusa, Vesely, "Revisiting synthesis model in sparse audio declipper",
   LVA/ICA 2018 (the improved S-SPADE used here).

Each Hann-windowed frame is declipped independently with a redundant DFT frame A (zero-padded
FFT, Parseval) under a hard-sparsity constraint (k largest coefficients) that grows by `s`
every `r` iterations until the frame is (nearly) consistent. Frames are then overlap-added.
"""
import math
import numpy as np
import torch


def _frames(x, win, hop):
    return x.unfold(-1, win, hop)


def _ola(frames, hop, Tp):
    B, F, W = frames.shape
    return torch.nn.functional.fold(frames.transpose(1, 2), output_size=(1, Tp), kernel_size=(1, W),
                                    stride=(1, hop)).reshape(B, Tp)


def _hard_k(c, k):
    """Keep the k largest-magnitude coefficients per row. c: (N, K) complex; k: (N,) long."""
    mag = c.real ** 2 + c.imag ** 2
    srt, _ = torch.sort(mag, dim=-1, descending=True)
    kk = torch.clamp(k, 1, c.shape[-1]) - 1
    thr = torch.gather(srt, 1, kk[:, None])
    return torch.where(mag >= thr, c, torch.zeros_like(c))


def declip_spade(y, m_hi, m_lo, th_hi, th_lo, win_len=4096, hop=None, red=2, variant="a",
                 s=1, r=1, eps=0.1, max_iter=1000, device="cpu", dtype=torch.float32,
                 verbose=False, stereo=None):
    """y: (T, C) clipped signal. Returns (T, C) estimate."""
    T, C = y.shape
    hop = hop or win_len // 4
    scale = float(np.max(np.abs(np.concatenate([th_hi[np.isfinite(th_hi)], th_lo[np.isfinite(th_lo)], [1e-3]]))))
    yy = y.T / scale
    lb = yy.copy(); ub = yy.copy()
    thh = (th_hi / scale)[:, None]; thl = (th_lo / scale)[:, None]
    mh = m_hi.T; ml = m_lo.T
    lb[mh] = np.broadcast_to(thh, yy.shape)[mh]; ub[mh] = np.inf
    lb[ml] = -np.inf; ub[ml] = np.broadcast_to(thl, yy.shape)[ml]
    pad = win_len - hop
    Tp0 = T + 2 * pad
    extra = (hop - (Tp0 - win_len) % hop) % hop
    Tp = Tp0 + extra
    def padz(a, val):
        return np.concatenate([np.full((C, pad), val), a, np.full((C, pad + extra), val)], axis=1)
    lbp = torch.as_tensor(padz(lb, 0.0), dtype=dtype, device=device)
    ubp = torch.as_tensor(padz(ub, 0.0), dtype=dtype, device=device)
    yp = torch.as_tensor(padz(yy, 0.0), dtype=dtype, device=device)
    w = torch.hann_window(win_len, periodic=True, dtype=dtype, device=device)
    # windowed frames and windowed bounds (window >= 0 so bounds scale directly)
    Yf = _frames(yp, win_len, hop) * w           # (C, F, W)
    LB = _frames(lbp, win_len, hop) * w
    UB = _frames(ubp, win_len, hop) * w
    Cn, Fn, W = Yf.shape
    N = Cn * Fn
    Yf = Yf.reshape(N, W); LB = LB.reshape(N, W); UB = UB.reshape(N, W)
    # frames without clipped samples are kept as is
    clipped = torch.isinf(LB).any(-1) | torch.isinf(UB).any(-1)
    # Note: inf*0 at window zeros gives nan -> fix
    LB = torch.nan_to_num(LB, nan=0.0, neginf=-float("inf"))
    UB = torch.nan_to_num(UB, nan=0.0, posinf=float("inf"))
    nfft = red * W
    A = lambda v: torch.fft.rfft(v, n=nfft, dim=-1, norm="ortho")
    As = lambda c: torch.fft.irfft(c, n=nfft, dim=-1, norm="ortho")[..., :W]
    proj = lambda v: torch.maximum(torch.minimum(v, UB_), LB_)

    idx = torch.nonzero(clipped).flatten()
    out = Yf.clone()
    if idx.numel() > 0:
        LB_ = LB[idx]; UB_ = UB[idx]
        yv = Yf[idx]
        n = idx.numel()
        k = torch.full((n,), s, dtype=torch.long, device=device)
        active = torch.ones(n, dtype=torch.bool, device=device)
        if variant == "a":
            x = yv.clone()
            u = torch.zeros_like(A(yv))
            for i in range(max_iter):
                zb = _hard_k(A(x) + u, k)
                xn = proj(As(zb - u))
                res = A(xn) - zb
                nr = torch.sqrt((res.real ** 2 + res.imag ** 2).sum(-1))
                x = torch.where(active[:, None], xn, x)
                u = torch.where(active[:, None], u + res, u)
                active = active & (nr > eps)
                if (i + 1) % r == 0:
                    k = k + s * active.long()
                if not active.any():
                    break
            est = x
        else:
            z = A(yv)
            u = torch.zeros_like(z)
            for i in range(max_iter):
                zb = _hard_k(z - u, k)
                v = zb + u
                dv = As(v)
                zn = v - A(dv - proj(dv))
                res = zn - zb
                nr = torch.sqrt((res.real ** 2 + res.imag ** 2).sum(-1))
                z = torch.where(active[:, None], zn, z)
                u = torch.where(active[:, None], u + zb - zn, u)
                active = active & (nr > eps)
                if (i + 1) % r == 0:
                    k = k + s * active.long()
                if not active.any():
                    break
            est = proj(As(z))
        if verbose:
            print(f"SPADE-{variant}: {i+1} iterations, frames {n}, final k mean {k.float().mean():.0f}")
        out[idx] = est
    # overlap-add: sum of Hann windows at 75% overlap = 2 (window applied once)
    wsum = _ola(w.expand(1, Fn, W).contiguous(), hop, Tp)[0]
    x = _ola(out.reshape(Cn, Fn, W), hop, Tp) / torch.clamp(wsum, min=1e-8)
    x = x[:, pad:pad + T]
    # final consistency on reliable samples (exact) and bounds
    lbt = torch.as_tensor(lb, dtype=dtype, device=device); ubt = torch.as_tensor(ub, dtype=dtype, device=device)
    x = torch.maximum(torch.minimum(x, ubt), lbt)
    return x.T.cpu().numpy().astype(np.float64) * scale
