"""Shared helpers for declipping methods: constraint sets and projections."""
import numpy as np
import torch


def make_bounds(y, m_hi, m_lo, th_hi, th_lo, max_gain=None):
    """Per-sample lower/upper bounds of the consistency set.

    y, m_hi, m_lo: (T, C) numpy arrays. th_hi/th_lo: (C,).
    Reliable samples: lb = ub = y. Clipped high: lb = th_hi, ub = +inf. Clipped low: lb = -inf, ub = th_lo.
    Returns lb, ub as (C, T) float64 arrays.
    """
    y = np.asarray(y, dtype=np.float64)
    lb = y.copy()
    ub = y.copy()
    th_hi_b = full_thresholds(th_hi, y.shape)
    th_lo_b = full_thresholds(th_lo, y.shape)
    lb[m_hi] = th_hi_b[m_hi]
    ub[m_hi] = np.inf if max_gain is None else max_gain * th_hi_b[m_hi]
    lb[m_lo] = -np.inf if max_gain is None else max_gain * th_lo_b[m_lo]
    ub[m_lo] = th_lo_b[m_lo]
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
