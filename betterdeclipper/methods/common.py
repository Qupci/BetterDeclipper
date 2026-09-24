"""Shared helpers for declipping methods: constraint sets and projections."""
import numpy as np
import torch


def make_bounds(y, m_hi, m_lo, th_hi, th_lo):
    """Per-sample lower/upper bounds of the consistency set.

    y, m_hi, m_lo: (T, C) numpy arrays. th_hi/th_lo: (C,).
    Reliable samples: lb = ub = y. Clipped high: lb = th_hi, ub = +inf. Clipped low: lb = -inf, ub = th_lo.
    Returns lb, ub as (C, T) float64 arrays.
    """
    y = np.asarray(y, dtype=np.float64)
    lb = y.copy()
    ub = y.copy()
    th_hi_b = np.broadcast_to(th_hi[None, :], y.shape)
    th_lo_b = np.broadcast_to(th_lo[None, :], y.shape)
    lb[m_hi] = th_hi_b[m_hi]
    ub[m_hi] = np.inf
    lb[m_lo] = -np.inf
    ub[m_lo] = th_lo_b[m_lo]
    return lb.T.copy(), ub.T.copy()


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
        return torch.maximum(torch.minimum(x, self.ub), self.lb)
