"""SPADE declipping (A-SPADE / S-SPADE), frame-wise, vectorized over all frames with torch.

References:
 - Kitic, Bertin, Gribonval, "Sparsity and cosparsity for audio declipping: a flexible
   non-convex approach", LVA/ICA 2015 (A-SPADE, S-SPADE).
 - Zaviska, Rajmic, Prusa, Vesely, "Revisiting synthesis model in sparse audio declipper",
   LVA/ICA 2018 (the improved S-SPADE used here).

Each Hann-windowed frame is declipped independently with a redundant DFT frame A (zero-padded
FFT, Parseval) under a hard-sparsity constraint (k largest coefficients) that grows by `s`
every `r` iterations until the frame is (nearly) consistent. Frames are then overlap-added.

Stereo extension: all channels of a frame are processed jointly in a rotated (PCA) channel
basis; the k largest coefficients are selected jointly over channels, with per-channel ranking
weights (chan_weight < 1 makes the minor/side component harder to select).
"""
import math
import numpy as np
import torch

from .social import channel_mixing
from .common import make_bounds, threshold_scale
from ..stft import overlap_add


def _hard_k(c, k, rank_w=None):
    """Keep the k largest-magnitude coefficients per row. c: (N, M) complex; k: (N,) long."""
    mag = c.real ** 2 + c.imag ** 2
    rmag = mag if rank_w is None else mag * rank_w
    srt, _ = torch.sort(rmag, dim=-1, descending=True)
    kk = torch.clamp(k, 1, c.shape[-1]) - 1
    thr = torch.gather(srt, 1, kk[:, None])
    return torch.where(rmag >= thr, c, torch.zeros_like(c))


def _hard_single_k(c, k, rank_w=None):
    """Keep the k largest (weighted) magnitudes per row, same k for all rows (kthvalue)."""
    mag = c.real ** 2 + c.imag ** 2
    rmag = mag if rank_w is None else mag * rank_w
    thr = torch.kthvalue(rmag, rmag.shape[-1] - k + 1, dim=-1, keepdim=True)[0]
    return c * (rmag >= thr)


def declip_spade(y, m_hi, m_lo, th_hi, th_lo, win_len=4096, hop=None, red=2, variant="a",
                 s=8, r=1, eps=0.1, max_iter=1000, stereo="pca", chan_weight=None,
                 device="cpu", dtype=torch.float32, verbose=False, max_gain=None):
    """y: (T, C) clipped signal. Returns (T, C) estimate."""
    T, C = y.shape
    hop = hop or win_len // 4
    scale = threshold_scale(th_hi, th_lo)
    yy = y.T / scale
    lb, ub = make_bounds(y / scale, m_hi, m_lo, np.asarray(th_hi) / scale, np.asarray(th_lo) / scale, max_gain)
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
    fr = lambda a: a.unfold(-1, win_len, hop).permute(1, 0, 2) * w     # (F, C, W)
    Yf = fr(yp)
    # bounds: inf * w -> keep inf where w>0 ; at w==0 the bound is 0 (irrelevant)
    LB = torch.nan_to_num(fr(lbp), nan=0.0, neginf=-float("inf"))
    UB = torch.nan_to_num(fr(ubp), nan=0.0, posinf=float("inf"))
    Fn = Yf.shape[0]
    Q = torch.as_tensor(channel_mixing(y, stereo if C == 2 else "none"), dtype=dtype, device=device)
    rank_w = None
    if chan_weight is not None:
        cw = torch.as_tensor(chan_weight, dtype=dtype, device=device) ** 2
        rank_w = cw[:, None].expand(C, red * win_len // 2 + 1).reshape(1, -1)
    nfft = red * win_len
    K = nfft // 2 + 1
    # analysis on the rotated channels, joint over channels: (N, C*K)
    def A(xf):  # xf: (N, C, W) -> (N, C*K)
        u = torch.einsum("ji,njw->niw", Q, xf)
        return torch.fft.rfft(u, n=nfft, dim=-1, norm="ortho").reshape(xf.shape[0], C * K)
    def As(c):  # (N, C*K) -> (N, C, W)
        u = torch.fft.irfft(c.reshape(c.shape[0], C, K), n=nfft, dim=-1, norm="ortho")[..., :win_len]
        return torch.einsum("ij,njw->niw", Q, u)

    clipped = ((UB - LB) > 0).any(-1).any(-1)  # (F,) frames containing unreliable samples
    idx = torch.nonzero(clipped).flatten()
    out = Yf.clone()
    if idx.numel() > 0:
        LB_ = LB[idx]; UB_ = UB[idx]
        proj = lambda v: torch.maximum(torch.minimum(v, UB_), LB_)
        yv = Yf[idx]
        n = idx.numel()
        # All active frames share the same k (it grows in lockstep), so a single kthvalue per
        # iteration replaces a per-row sort; converged frames are dropped from the computation.
        M = C * K
        kk = s
        if variant == "a":
            x = yv.clone()
            u = torch.zeros_like(A(yv))
            act = torch.arange(n, device=device)
            for i in range(max_iter):
                xa, ua = x[act], u[act]
                c = A(xa) + ua
                zb = _hard_single_k(c, min(kk, M), rank_w)
                xn = torch.maximum(torch.minimum(As(zb - ua), UB_[act]), LB_[act])
                res = A(xn) - zb
                nr = torch.sqrt((res.real ** 2 + res.imag ** 2).sum(-1))
                x[act] = xn
                u[act] = ua + res
                act = act[nr > eps]
                if (i + 1) % r == 0:
                    kk += s
                if act.numel() == 0:
                    break
            est = x
        else:
            z = A(yv)
            u = torch.zeros_like(z)
            act = torch.arange(n, device=device)
            for i in range(max_iter):
                za, ua = z[act], u[act]
                zb = _hard_single_k(za - ua, min(kk, M), rank_w)
                v = zb + ua
                dv = As(v)
                zn = v - A(dv - torch.maximum(torch.minimum(dv, UB_[act]), LB_[act]))
                res = zn - zb
                nr = torch.sqrt((res.real ** 2 + res.imag ** 2).sum(-1))
                z[act] = zn
                u[act] = ua + zb - zn
                act = act[nr > eps]
                if (i + 1) % r == 0:
                    kk += s
                if act.numel() == 0:
                    break
            est = proj(As(z))
        if verbose:
            print(f"SPADE-{variant}: {i+1} iterations, frames {n}, final k {kk}")
        out[idx] = est
    # overlap-add: sum of Hann windows at hop=win/4 is 2 (window applied once)
    frames = out.permute(1, 0, 2).contiguous()                          # (C, F, W)
    wsum = overlap_add(w.expand(1, Fn, win_len).contiguous(), hop, Tp)[0]
    x = overlap_add(frames, hop, Tp) / torch.clamp(wsum, min=1e-8)
    x = x[:, pad:pad + T]
    lbt = torch.as_tensor(lb, dtype=dtype, device=device); ubt = torch.as_tensor(ub, dtype=dtype, device=device)
    x = torch.maximum(torch.minimum(x, ubt), lbt)
    return x.T.cpu().numpy().astype(np.float64) * scale
