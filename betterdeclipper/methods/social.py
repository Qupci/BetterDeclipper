"""Consistent declipping with (social) sparsity shrinkage, solved by FISTA.

Model: min_z  1/2 dist(Q D z, Gamma)^2 + lambda * R(z), with D a Parseval tight STFT frame,
Q an orthogonal inter-channel mixing (identity, mid/side or PCA rotation) and Gamma the
consistency set (reliable samples fixed, clipped samples beyond the clip level).
Shrinkage operators: 'l1' (soft), 'ew' (empirical Wiener), 'pew' (persistent EW: neighborhood
energy over time/frequency, optionally pooled across channels). Reference: Siedenburg,
Kowalski, Doerfler, "Audio declipping with social sparsity", ICASSP 2014.
"""
import numpy as np
import torch
import torch.nn.functional as Fnn

from ..stft import TightSTFT
from .common import make_bounds, pad_bounds, Box


def _neigh_kernel(nf, nt, device, dtype):
    """Normalized separable Hann-like neighborhood kernel of size (nt frames, nf bins)."""
    def w(n):
        if n == 1:
            return torch.ones(1, dtype=torch.float64)
        return torch.hann_window(n + 2, periodic=False, dtype=torch.float64)[1:-1]
    k = torch.outer(w(nt), w(nf))
    k = k / k.sum()
    return k.to(device=device, dtype=dtype)[None, None]


def shrink(z, lam, kind, kernel=None, joint=0.0, alpha=1.0, beta=1.0):
    """z: (C, F, K) complex. lam: scalar or (K,) tensor (frequency-dependent threshold).

    PEW gain generalized as max(0, 1 - (lam^2/E)^alpha)^beta; alpha=beta=1 is the classic PEW,
    large alpha approaches hard social thresholding (no amplitude bias for kept coefficients).
    """
    a2 = z.real ** 2 + z.imag ** 2
    if kind == "l1":
        g = torch.clamp(1.0 - lam / torch.sqrt(a2 + 1e-30), min=0.0)
    elif kind == "ew":
        g = torch.clamp(1.0 - lam ** 2 / (a2 + 1e-30), min=0.0)
    elif kind == "pew":
        kt, kf = kernel.shape[-2:]
        e = Fnn.conv2d(a2[:, None], kernel, padding=(kt // 2, kf // 2))[:, 0]
        if joint > 0 and e.shape[0] > 1:
            e = (1 - joint) * e + joint * e.mean(dim=0, keepdim=True)
        ratio = lam ** 2 / (e + 1e-30)
        if alpha != 1.0:
            ratio = ratio ** alpha
        g = torch.clamp(1.0 - ratio, min=0.0)
        if beta != 1.0:
            g = g ** beta
    else:
        raise ValueError(kind)
    return z * g


def channel_mixing(y, mode):
    """Orthogonal matrix Q (C x C) such that x = Q @ u, with u the modeled components."""
    C = y.shape[1]
    if C != 2 or mode in (None, "none"):
        return np.eye(C)
    if mode == "ms":
        return np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
    if mode == "pca":
        # principal axes of the reliable (unclipped) part of the signal
        cov = np.cov(y.T)
        w, V = np.linalg.eigh(cov)
        return V[:, ::-1].copy()
    raise ValueError(mode)


def bin_rotations(z, smooth_bins=0):
    """Per-frequency-bin 2x2 real rotations aligning (L, R) coefficients with their principal axes.

    z: (2, F, K) complex (channels, frames, bins). Returns R: (K, 2, 2) with columns = principal axes.
    """
    zl, zr = z[0], z[1]
    a = (zl.real ** 2 + zl.imag ** 2).sum(0)
    b = (zr.real ** 2 + zr.imag ** 2).sum(0)
    c = (zl.real * zr.real + zl.imag * zr.imag).sum(0)
    if smooth_bins > 1:
        k = torch.ones(1, 1, smooth_bins, dtype=a.dtype, device=a.device) / smooth_bins
        sm = lambda v: Fnn.conv1d(v[None, None], k, padding=smooth_bins // 2)[0, 0][: v.shape[0]]
        a, b, c = sm(a), sm(b), sm(c)
    theta = 0.5 * torch.atan2(2 * c, a - b)  # principal-axis angle
    ct, st = torch.cos(theta), torch.sin(theta)
    R = torch.stack([torch.stack([ct, -st], -1), torch.stack([st, ct], -1)], -2)  # (K, 2, 2)
    return R


def rot_apply(R, z):
    """u = R^T z per bin. z: (2, F, K) complex."""
    return torch.einsum("kji,jfk->ifk", R.to(z.dtype), z)


def rot_unapply(R, u):
    """z = R u per bin."""
    return torch.einsum("kij,jfk->ifk", R.to(u.dtype), u)


def declip_social(y, m_hi, m_lo, th_hi, th_lo, win_len=2048, hop=512, nfft=None,
                  kind="pew", neigh=(3, 7), n_iter=500, lam0=None, lam1=None,
                  stereo="none", joint=0.0, fweight=None, alpha=1.0, beta=1.0,
                  device="cpu", dtype=torch.float32, verbose=False, callback=None):
    """y: (T, C) clipped signal; masks (T, C); thresholds (C,). Returns (T, C) estimate.

    stereo: 'none' | 'ms' | 'pca'  orthogonal channel mixing of the sparse model.
    joint: 0..1 pooling of PEW neighborhood energy across (mixed) channels.
    fweight: None or exponent a; threshold scales as (f / f_ref)^a (a>0 penalizes HF more).
    """
    T, C = y.shape
    scale = float(np.max(np.abs(np.concatenate([th_hi[np.isfinite(th_hi)], th_lo[np.isfinite(th_lo)], [1e-3]]))))
    lb, ub = make_bounds(y / scale, m_hi, m_lo, th_hi / scale, th_lo / scale)
    stft = TightSTFT(win_len, hop, nfft, device, dtype)
    left, right = stft.pad_len(T)
    lb, ub = pad_bounds(lb, ub, left, right)
    Tp = lb.shape[1]
    proj = Box(lb, ub, device, dtype)
    per_bin = stereo in ("bin", "bin_adapt") and C == 2
    Q = torch.as_tensor(channel_mixing(y, "none" if per_bin else stereo), dtype=dtype, device=device)
    mix = lambda u: torch.einsum("ij,jt->it", Q, u)      # model -> channels
    unmix = lambda x: torch.einsum("ji,jt->it", Q, x)    # channels -> model (Q^T)

    x0 = torch.zeros(C, Tp, dtype=dtype, device=device)
    x0[:, left:left + T] = torch.as_tensor(y.T / scale, dtype=dtype, device=device)
    kernel = _neigh_kernel(neigh[0], neigh[1], device, dtype) if kind == "pew" else None

    # coefficient-domain operators (optionally with per-bin stereo rotation)
    R = None
    if per_bin:
        R = bin_rotations(stft.analysis(x0), smooth_bins=9)
    D = lambda zz: mix(stft.synthesis(zz if R is None else rot_unapply(R, zz), Tp))
    Dt = lambda xx: (lambda c: c if R is None else rot_apply(R, c))(stft.analysis(unmix(xx)))

    z = Dt(x0)
    zmax = float(torch.abs(z).max())
    lam0 = lam0 if lam0 is not None else 0.1
    lam1 = lam1 if lam1 is not None else 1e-4
    lams = np.geomspace(lam0 * zmax, lam1 * zmax, n_iter)
    K = z.shape[-1]
    if fweight:
        f = torch.arange(K, dtype=dtype, device=device).clamp(min=1.0) / (K * 0.05)
        fw = f ** float(fweight)
        fw = fw / fw.mean()
    else:
        fw = None
    zbar = z.clone()
    t = 1.0
    for it in range(n_iter):
        if stereo == "bin_adapt" and R is not None and it > 0 and it % 50 == 0:
            # re-estimate the per-bin rotations from the current (consistent) estimate
            xc = proj(D(z))
            zc = rot_unapply(R, z)
            R = bin_rotations(stft.analysis(xc), smooth_bins=9)
            z = rot_apply(R, zc)
            zbar = z.clone()
            t = 1.0
        x = D(zbar)
        g = Dt(x - proj(x))
        lam = lams[it] if fw is None else lams[it] * fw
        znew = shrink(zbar - g, lam, kind, kernel, joint, alpha, beta)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        zbar = znew + ((t - 1) / tn) * (znew - z)
        z, t = znew, tn
        if callback is not None and (it % 50 == 49 or it == n_iter - 1):
            xr = proj(D(z))[:, left:left + T].T.cpu().numpy().astype(np.float64) * scale
            callback(it, xr)
    x = proj(D(z))
    return x[:, left:left + T].T.cpu().numpy().astype(np.float64) * scale
