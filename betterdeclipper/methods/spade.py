"""SPADE declipping (A-SPADE / S-SPADE), frame-wise, vectorized over batches of frames with torch.

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

Frames are independent, so a whole file is processed in large frame batches (no chunking), and
only frames containing unreliable samples are processed. Frame corrections are accumulated
directly into the output (no full frame buffers), so memory stays bounded for long files.
"""
import math
import numpy as np
import torch

from .social import channel_mixing
from .common import make_bounds, threshold_scale, ChannelMix


def _hard_single_k(c, k, rank_w=None):
    """Keep the k largest (weighted) magnitudes per row, same k for all rows."""
    mag = c.real ** 2 + c.imag ** 2
    rmag = mag if rank_w is None else mag * rank_w
    if rmag.is_cuda:  # same value as kthvalue; topk is faster on GPUs, kthvalue on CPUs
        thr = torch.topk(rmag, k, dim=-1, sorted=False)[0].min(-1, keepdim=True)[0]
    else:
        thr = torch.kthvalue(rmag, rmag.shape[-1] - k + 1, dim=-1, keepdim=True)[0]
    return c.masked_fill_(rmag < thr, 0)  # in place: c is a temporary


def _aspade_batch(Yb, LBb, UBb, A, As, s, r, eps, max_iter, M, rank_w, pad_multiple):
    """A-SPADE on a batch of windowed frames (n, C, W). Returns (estimates, iterations, final k).

    All rows share the same k schedule. Converged rows are written out when they converge and are
    then ignored. The working set is compacted only when it can shrink by `pad_multiple` rows
    (padded with duplicate rows whose output is ignored), so batch shapes, and with them the
    cuFFT plans, change rarely. Per-row arithmetic is identical to an unbatched run."""
    n = Yb.shape[0]
    dev = Yb.device
    x = Yb.clone()
    xa, ua, LBa, UBa = x, torch.zeros_like(A(Yb)), LBb, UBb
    rows = torch.arange(n, device=dev)            # frame index of each working-set row
    live = torch.ones(n, dtype=torch.bool, device=dev)
    kk = s
    n_live = n
    it = 0
    for it in range(max_iter):
        zb = _hard_single_k(A(xa).add_(ua), min(kk, M), rank_w)
        xn = torch.clamp(As(zb - ua), min=LBa, max=UBa)
        res = A(xn).sub_(zb)
        nr = torch.sqrt((res.real ** 2 + res.imag ** 2).sum(-1))
        ua.add_(res)
        xa = xn
        if (it + 1) % r == 0:
            kk += s
        newly = live & (nr <= eps)
        n_new = int(newly.sum())
        if n_new:
            x[rows[newly]] = xa[newly]
            live &= ~newly
            n_live -= n_new
            if n_live == 0:
                break
            m_new = int(math.ceil(n_live / pad_multiple) * pad_multiple)
            if m_new < xa.shape[0]:
                kept = torch.nonzero(live).flatten()
                sel = torch.cat([kept, kept[:1].expand(m_new - n_live)]) if m_new > n_live else kept
                xa, ua, LBa, UBa, rows = xa[sel], ua[sel], LBa[sel], UBa[sel], rows[sel]
                live = torch.zeros(m_new, dtype=torch.bool, device=dev)
                live[:n_live] = True
    if n_live:
        x[rows[live]] = xa[live]
    return x, it + 1, kk


def _sspade_batch(Yb, LBb, UBb, A, As, s, r, eps, max_iter, M, rank_w):
    """S-SPADE (Zaviska et al. 2018) on a batch of frames."""
    n = Yb.shape[0]
    z = A(Yb)
    u = torch.zeros_like(z)
    act = torch.arange(n, device=Yb.device)
    kk = s
    it = 0
    for it in range(max_iter):
        za, ua = z[act], u[act]
        zb = _hard_single_k(za - ua, min(kk, M), rank_w)
        v = zb + ua
        dv = As(v)
        zn = v - A(dv - torch.clamp(dv, min=LBb[act], max=UBb[act]))
        res = zn - zb
        nr = torch.sqrt((res.real ** 2 + res.imag ** 2).sum(-1))
        z[act] = zn
        u[act] = ua + zb - zn
        act = act[nr > eps]
        if (it + 1) % r == 0:
            kk += s
        if act.numel() == 0:
            break
    return torch.clamp(As(z), min=LBb, max=UBb), it + 1, kk


def declip_spade(y, m_hi, m_lo, th_hi, th_lo, win_len=4096, hop=None, red=2, variant="a",
                 s=8, r=1, eps=0.1, max_iter=1000, stereo="pca", chan_weight=None,
                 device="cpu", dtype=torch.float32, verbose=False, max_gain=None,
                 batch_frames=None, pad_multiple=None):
    """y: (T, C) clipped signal. Returns (T, C) estimate."""
    T, C = y.shape
    hop = hop or win_len // 4
    R = win_len // hop
    assert R * hop == win_len, "hop must divide the window length"
    scale = threshold_scale(th_hi, th_lo)
    lb, ub = make_bounds(y / scale, m_hi, m_lo, np.asarray(th_hi) / scale, np.asarray(th_lo) / scale, max_gain)
    pad = win_len - hop
    Tp0 = T + 2 * pad
    extra = (hop - (Tp0 - win_len) % hop) % hop
    Tp = Tp0 + extra

    def padz(a):
        return torch.as_tensor(np.concatenate([np.zeros((C, pad)), a, np.zeros((C, pad + extra))], axis=1),
                               dtype=dtype, device=device)

    yp, lbp, ubp = padz(y.T / scale), padz(lb), padz(ub)
    w = torch.hann_window(win_len, periodic=True, dtype=dtype, device=device)
    cm = ChannelMix(channel_mixing(y, stereo if C == 2 else "none"))
    rank_w = None
    nfft = red * win_len
    K = nfft // 2 + 1
    if chan_weight is not None:
        cw = torch.as_tensor(chan_weight, dtype=dtype, device=device) ** 2
        rank_w = cw[:, None].expand(C, K).reshape(1, -1)
    M = C * K

    def A(xf):  # (N, C, W) -> (N, C*K): analysis on the rotated channels, joint over channels
        u = cm.unmix(xf, dim=1)
        return torch.fft.rfft(u, n=nfft, dim=-1, norm="ortho").reshape(xf.shape[0], C * K)

    def As(c):  # (N, C*K) -> (N, C, W)
        u = torch.fft.irfft(c.reshape(c.shape[0], C, K), n=nfft, dim=-1, norm="ortho")[..., :win_len]
        return cm.mix(u, dim=1)

    def frames(sig, fb):  # windowed frames fb of a padded signal -> (nb, C, W)
        return sig.unfold(-1, win_len, hop)[:, fb].permute(1, 0, 2) * w

    # frames containing unreliable samples (ub > lb); only these are processed
    unrel = (ubp > lbp).any(0).to(dtype)
    idx = torch.nonzero(unrel.unfold(0, win_len, hop).sum(-1) > 0).flatten()
    acc = torch.zeros(C, Tp, dtype=dtype, device=device)
    on_gpu = torch.device(device).type == "cuda"
    if batch_frames:
        B = batch_frames
    elif on_gpu:
        # ~0.9 MB per stereo 4096-sample frame at redundancy 2 (measured); use <= 40 % of free memory
        free = torch.cuda.mem_get_info(torch.device(device))[0]
        per_frame = 0.9e6 * (win_len / 4096) * (red / 2) * (C / 2)
        B = int(min(3072, max(128, 0.4 * free / per_frame)))
    else:
        B = 2048
    # GPU: shrink the working set in steps of 64 rows so cuFFT plans are reused;
    # CPU: no plans to reuse, so drop converged rows immediately (no wasted work)
    pad_multiple = pad_multiple or (64 if on_gpu else 1)
    tot_it = 0
    for b0 in range(0, idx.numel(), B):
        fb = idx[b0:b0 + B]
        Yb = frames(yp, fb)
        # bounds: inf * w -> keep inf where w > 0; at w == 0 the bound is 0 (irrelevant)
        LBb = torch.nan_to_num(frames(lbp, fb), nan=0.0, neginf=-float("inf"))
        UBb = torch.nan_to_num(frames(ubp, fb), nan=0.0, posinf=float("inf"))
        if variant == "a":
            est, its, kk = _aspade_batch(Yb, LBb, UBb, A, As, s, r, eps, max_iter, M, rank_w, pad_multiple)
        else:
            est, its, kk = _sspade_batch(Yb, LBb, UBb, A, As, s, r, eps, max_iter, M, rank_w)
        tot_it += its
        delta = est - Yb                                   # frame corrections (nb, C, W)
        # overlap-add into acc; frames with equal (index mod R) never overlap -> deterministic index_add
        ar = torch.arange(win_len, device=delta.device)
        for ph in range(R):
            sel = (fb % R) == ph
            if bool(sel.any()):
                pos = (fb[sel] * hop)[:, None] + ar
                acc.index_add_(1, pos.reshape(-1), delta[sel].permute(1, 0, 2).reshape(C, -1))
        if verbose:
            print(f"SPADE-{variant}: batch {b0 // B + 1}: {fb.numel()} frames, {its} iterations, final k {kk}")
    # sum of the (shifted) windows is periodic with period hop in the fully covered region
    wsum = w.reshape(R, hop).sum(0)
    x = (yp + acc / wsum.repeat(Tp // hop + 1)[:Tp])[:, pad:pad + T]
    x = torch.clamp(x, min=torch.as_tensor(lb, dtype=dtype, device=device),
                    max=torch.as_tensor(ub, dtype=dtype, device=device))
    return x.T.cpu().numpy().astype(np.float64) * scale
