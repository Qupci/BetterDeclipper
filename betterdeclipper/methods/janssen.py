"""AR (Janssen-style) refinement of a declipped estimate.

For overlapping windows, an AR(p) model is fitted to the current estimate (autocorrelation
method), then the clipped samples of the window are re-estimated by minimizing the AR
prediction-error energy subject to the clipping constraints (box-constrained QP solved with
FISTA, batched over all windows via grouped conv1d). Windows are merged by weighted overlap-add.
Reference: A. Janssen, R. Veldhuis, L. Vries, "Adaptive interpolation of discrete-time signals
that can be modeled as autoregressive processes", IEEE TASSP 1986.
"""
import math
import numpy as np
import torch
import torch.nn.functional as Fnn
from scipy.linalg import solve_toeplitz


def _ar_coeffs(frames, order, lag_win=True, wnc=1e-6):
    """frames: (N, W) numpy (already windowed). Returns (N, order+1) AR polynomials [1, a1..ap]."""
    N, W = frames.shape
    nfft = 1 << int(math.ceil(math.log2(2 * W)))
    F = np.fft.rfft(frames, n=nfft, axis=1)
    r = np.fft.irfft(np.abs(F) ** 2, n=nfft, axis=1)[:, : order + 1]
    if lag_win:
        # Gaussian lag window (bandwidth expansion) for numerical robustness
        k = np.arange(order + 1)
        r = r * np.exp(-0.5 * (2 * np.pi * 5.0 * k / 44100.0) ** 2)[None, :]
    A = np.zeros((N, order + 1))
    A[:, 0] = 1.0
    for i in range(N):
        ri = r[i].copy()
        if ri[0] <= 1e-12:
            continue
        ri[0] *= 1.0 + wnc
        a = solve_toeplitz((ri[:order], ri[:order]), -ri[1: order + 1])
        A[i, 1:] = a
    return A


def ar_refine(y, x0, m_hi, m_lo, th_hi, th_lo, win_len=2048, hop=512, order=256, n_outer=3,
              n_inner=100, device="cpu", dtype=torch.float32, callback=None):
    """y, x0: (T, C). Returns refined (T, C) estimate (consistent with the clipping constraints)."""
    T, C = y.shape
    lb = np.where(m_hi, th_hi[None, :], np.where(m_lo, -np.inf, y))
    ub = np.where(m_lo, th_lo[None, :], np.where(m_hi, np.inf, y))
    miss = (m_hi | m_lo)
    pad = win_len
    Tp = T + 2 * pad
    Tp += (hop - (Tp - win_len) % hop) % hop
    def padarr(a, v):
        out = np.full((Tp, C), v, dtype=np.float64)
        out[pad:pad + T] = a
        return out
    x = padarr(x0, 0.0)
    LB = padarr(lb, 0.0); UB = padarr(ub, 0.0); MS = padarr(miss, False).astype(bool)
    starts = np.arange(0, Tp - win_len + 1, hop)
    idx = starts[:, None] + np.arange(win_len)[None, :]         # (nW, W)
    # only windows containing clipped samples matter
    wsel = [np.where(MS[idx, c].any(axis=1))[0] for c in range(C)]
    wa = np.hanning(win_len + 2)[1:-1]
    for outer in range(n_outer):
        xn = x.copy()
        acc = np.zeros((Tp, C)); wacc = np.zeros((Tp, C))
        for c in range(C):
            ws = wsel[c]
            if ws.size == 0:
                continue
            I = idx[ws]                                      # (n, W)
            V = x[I, c]                                      # current estimate frames
            A = _ar_coeffs(V * wa[None, :], order)           # (n, p+1)
            vt = torch.as_tensor(V, dtype=dtype, device=device)
            lbt = torch.as_tensor(LB[I, c], dtype=dtype, device=device)
            ubt = torch.as_tensor(UB[I, c], dtype=dtype, device=device)
            mt = torch.as_tensor(MS[I, c], dtype=torch.bool, device=device)
            at = torch.as_tensor(A, dtype=dtype, device=device)
            n = at.shape[0]
            # Lipschitz constant of A^T A per window: max |A(w)|^2
            Lw = torch.abs(torch.fft.rfft(at, n=4096, dim=-1)).max(-1).values ** 2  # (n,)
            wconv = at.flip(-1)[:, None, :]                  # conv1d = correlation -> flip for convolution
            def grad(v):
                e = Fnn.conv1d(v[None], wconv, groups=n)     # (1, n, W-p): prediction errors (valid)
                g = Fnn.conv_transpose1d(e, wconv, groups=n)  # (1, n, W)
                return g[0]
            proj = lambda v: torch.where(mt, torch.maximum(torch.minimum(v, ubt), lbt), vt)
            v = vt.clone(); vb = v.clone(); t = 1.0
            step = (1.0 / Lw)[:, None]
            for it in range(n_inner):
                vnew = proj(vb - step * grad(vb))
                tn = 0.5 * (1 + math.sqrt(1 + 4 * t * t))
                vb = vnew + ((t - 1) / tn) * (vnew - v)
                v, t = vnew, tn
            vv = v.cpu().numpy().astype(np.float64)
            np.add.at(acc[:, c], I.ravel(), (vv * wa[None, :]).ravel())
            np.add.at(wacc[:, c], I.ravel(), np.broadcast_to(wa[None, :], vv.shape).ravel())
        upd = wacc > 1e-9
        xn[upd] = acc[upd] / wacc[upd]
        # keep reliable samples exact, clipped samples within bounds
        xn = np.where(MS, np.clip(xn, LB, UB), x)
        x = xn
        if callback is not None:
            callback(outer, x[pad:pad + T])
    return x[pad:pad + T]
