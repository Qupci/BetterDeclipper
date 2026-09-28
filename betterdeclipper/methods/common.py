"""Shared helpers for declipping methods: constraint sets and projections."""
import numpy as np
import torch


def make_bounds(y, m_hi, m_lo, th_hi, th_lo, max_gain=None):
    """Per-sample lower/upper bounds of the consistency set.

    y, m_hi, m_lo: (T, C) numpy arrays. th_hi/th_lo: (C,).
    Reliable samples: lb = ub = y. Clipped high: lb = th_hi, ub = +inf. Clipped low: lb = -inf, ub = th_lo.
    max_gain (scalar or (T, C) array, may hold inf) caps flagged samples at max_gain x threshold.
    Returns lb, ub as (C, T) float64 arrays.
    """
    y = np.asarray(y, dtype=np.float64)
    lb = y.copy()
    ub = y.copy()
    th_hi_b = full_thresholds(th_hi, y.shape)
    th_lo_b = full_thresholds(th_lo, y.shape)
    lb[m_hi] = th_hi_b[m_hi]
    ub[m_lo] = th_lo_b[m_lo]
    if max_gain is None:
        ub[m_hi] = np.inf
        lb[m_lo] = -np.inf
    else:  # scalar or per-sample (T, C) cap on |x| / threshold
        mg = np.broadcast_to(np.asarray(max_gain, dtype=np.float64), y.shape)
        with np.errstate(invalid="ignore"):
            ub[m_hi] = np.where(np.isinf(mg[m_hi]), np.inf, mg[m_hi] * th_hi_b[m_hi])
            lb[m_lo] = np.where(np.isinf(mg[m_lo]), -np.inf, mg[m_lo] * th_lo_b[m_lo])
    return lb.T.copy(), ub.T.copy()


def full_thresholds(th, shape):
    """Thresholds may be per channel (C,) or per sample (T, C) (soft-clip mode: the observed value)."""
    th = np.asarray(th, dtype=np.float64)
    return np.broadcast_to(th[None, :], shape) if th.ndim == 1 else th


def threshold_scale(th_hi, th_lo):
    """Normalization scale: the largest finite clip threshold magnitude."""
    v = np.concatenate([np.ravel(th_hi)[np.isfinite(np.ravel(th_hi))], np.ravel(th_lo)[np.isfinite(np.ravel(th_lo))], [1e-3]])
    return float(np.max(np.abs(v)))


def pad_bounds(lb, ub, left, right):
    """Pad bounds with unconstrained samples (the padded region is free)."""
    C = lb.shape[0]
    fl = np.full((C, left), -np.inf)
    fr = np.full((C, right), -np.inf)
    lb = np.concatenate([fl, lb, fr], axis=1)
    ub = np.concatenate([-fl, ub, -fr], axis=1)
    return lb, ub


class Box:
    """Projection onto {x : lb <= x <= ub} with torch tensors (inf-safe)."""

    def __init__(self, lb, ub, device="cpu", dtype=torch.float32):
        self.lb = torch.as_tensor(lb, dtype=dtype, device=device)
        self.ub = torch.as_tensor(ub, dtype=dtype, device=device)

    def __call__(self, x):
        return torch.clamp(x, min=self.lb, max=self.ub)  # lb <= ub everywhere


class ChannelMix:
    """Orthogonal channel mixing x = Q u (and u = Q^T x) along a channel axis.

    For 2 channels this is 4 scalar multiply-adds per sample; einsum would dispatch a batched
    matmul with an inner dimension of 2, which is very slow on GPUs."""

    def __init__(self, Q):
        self.Qn = np.asarray(Q, dtype=np.float64)
        self.C = self.Qn.shape[0]
        self.identity = bool(np.allclose(self.Qn, np.eye(self.C)))

    def _apply(self, M, x, dim):
        if self.identity:
            return x
        if self.C == 2:
            x0, x1 = x.select(dim, 0), x.select(dim, 1)
            y0 = x0 * float(M[0, 0]) + x1 * float(M[0, 1])
            y1 = x0 * float(M[1, 0]) + x1 * float(M[1, 1])
            return torch.stack([y0, y1], dim)
        Mt = torch.as_tensor(M, dtype=x.dtype if not x.is_complex() else x.real.dtype, device=x.device)
        return torch.movedim(torch.tensordot(Mt, torch.movedim(x, dim, 0), dims=1), 0, dim)

    def mix(self, u, dim=0):
        """x = Q u"""
        return self._apply(self.Qn, u, dim)

    def unmix(self, x, dim=0):
        """u = Q^T x"""
        return self._apply(self.Qn.T, x, dim)


class DipProj:
    """Projection for limiter restoration ("gain dips"): inside event windows the restoration is
    x = y * (1 + b_k v(t)), one non-negative depth b_k per limiter event k with a fixed dip shape v;
    elsewhere the usual box projection. idx (C, T) holds the event id per sample (-1: no event)."""

    def __init__(self, box, y, idx, v, n_events, bmax=1.0):
        self.box = box
        m = idx >= 0
        self.pos = torch.nonzero(m.reshape(-1)).flatten()          # flat positions inside windows
        self.ev = idx.reshape(-1)[self.pos]
        self.yw = y.reshape(-1)[self.pos]
        self.vw = v.reshape(-1)[self.pos]
        self.n = int(n_events)
        self.bmax = float(bmax)
        den = torch.zeros(self.n, dtype=y.dtype, device=y.device).index_add_(0, self.ev, (self.yw * self.vw) ** 2)
        self.den = den.clamp(min=1e-12)

    def __call__(self, z):
        x = self.box(z).reshape(-1)
        zw = z.reshape(-1)[self.pos]
        num = torch.zeros(self.n, dtype=z.dtype, device=z.device).index_add_(0, self.ev, (zw - self.yw) * self.yw * self.vw)
        b = (num / self.den).clamp(min=0.0, max=self.bmax)
        x = x.index_copy(0, self.pos, self.yw * (1.0 + self.vw * b[self.ev]))
        return x.reshape(z.shape)
