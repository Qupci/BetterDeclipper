"""Consistent social-sparsity declipping over a union of STFT frames (multi-resolution).

D = (1/sqrt(n)) [D_1 ... D_n] is a Parseval tight frame when every D_i is. Each sub-frame has its
own PEW neighborhood, so short windows can model transients and long windows sustained/low-frequency
content. Solved with FISTA + lambda continuation, like methods.social.
"""
import math
import numpy as np
import torch

from ..stft import TightSTFT
from .common import make_bounds, pad_bounds, Box
from .social import _neigh_kernel, shrink, channel_mixing


def declip_multires(y, m_hi, m_lo, th_hi, th_lo, frames=((4096, 1024, (3, 7)),), n_iter=400,
                    lam0=0.1, lam1=1e-4, stereo="pca", kind="pew", frame_gain=None,
                    device="cpu", dtype=torch.float32, callback=None, max_peak=None):
    """frames: sequence of (win_len, hop, (neigh_bins, neigh_frames)).
    frame_gain: optional per-frame multipliers of lambda (default 1).
    max_peak: optional absolute upper bound on |x| (e.g. 1.0 = 0 dBFS) for clipped samples.
    """
    T, C = y.shape
    scale = float(np.max(np.abs(np.concatenate([th_hi[np.isfinite(th_hi)], th_lo[np.isfinite(th_lo)], [1e-3]]))))
    lb, ub = make_bounds(y / scale, m_hi, m_lo, th_hi / scale, th_lo / scale)
    if max_peak is not None:
        ub = np.minimum(ub, max_peak / scale)
        lb = np.maximum(lb, -max_peak / scale)
    stfts = [TightSTFT(w, h, None, device, dtype) for w, h, _ in frames]
    kernels = [_neigh_kernel(nb[0], nb[1], device, dtype) for _, _, nb in frames]
    maxpad = max(w - h for w, h, _ in frames)
    L = 1
    for _, h, _ in frames:
        L = L * h // math.gcd(L, h)
    L = max(L, max(w for w, _, _ in frames))
    Tp = int(math.ceil((T + 2 * maxpad) / L) * L)
    left, right = maxpad, Tp - T - maxpad
    lb, ub = pad_bounds(lb, ub, left, right)
    proj = Box(lb, ub, device, dtype)
    Q = torch.as_tensor(channel_mixing(y, stereo), dtype=dtype, device=device)
    mix = lambda u: torch.einsum("ij,jt->it", Q, u)
    unmix = lambda x: torch.einsum("ji,jt->it", Q, x)
    a = 1.0 / math.sqrt(len(frames))

    def D(zs):
        return mix(a * sum(s.synthesis(z, Tp) for s, z in zip(stfts, zs)))

    def Dt(x):
        u = unmix(x)
        return [a * s.analysis(u) for s in stfts]

    x0 = torch.zeros(C, Tp, dtype=dtype, device=device)
    x0[:, left:left + T] = torch.as_tensor(y.T / scale, dtype=dtype, device=device)
    z = Dt(x0)
    zmax = max(float(torch.abs(zi).max()) for zi in z)
    lams = np.geomspace(lam0 * zmax, lam1 * zmax, n_iter)
    fg = frame_gain or [1.0] * len(frames)
    zbar = [zi.clone() for zi in z]
    t = 1.0
    for it in range(n_iter):
        x = D(zbar)
        g = Dt(x - proj(x))
        znew = [shrink(zb - gi, lams[it] * fgi, kind, k) for zb, gi, fgi, k in zip(zbar, g, fg, kernels)]
        tn = 0.5 * (1 + math.sqrt(1 + 4 * t * t))
        zbar = [zn + ((t - 1) / tn) * (zn - zo) for zn, zo in zip(znew, z)]
        z, t = znew, tn
        if callback is not None and (it % 50 == 49 or it == n_iter - 1):
            xr = proj(D(z))[:, left:left + T].T.cpu().numpy().astype(np.float64) * scale
            callback(it, xr)
    x = proj(D(z))
    return x[:, left:left + T].T.cpu().numpy().astype(np.float64) * scale
