"""Signal-domain plug-and-play declipping: x <- Denoise_lambda(P_Gamma(x)) with FISTA momentum.

The denoiser is a time-frequency shrinkage (PEW) in a Parseval STFT; averaging it over several
time shifts of the STFT grid ("cycle spinning") makes it translation invariant. lambda is
annealed geometrically (continuation).

Options:
 - chan_gain: lambda multiplier per (PCA-rotated) channel, e.g. [1, 2] shrinks the minor
   (side-like) component harder.
 - fweight: frequency-dependent lambda ~ (f/f0)^fweight.
 - pilot: a previous estimate whose smoothed TF energy is used (mixed with the current energy
   by `pilot_mix`) in the Wiener gain -> BM3D-like second stage.
"""
import math
import numpy as np
import torch

torch.set_flush_denormal(True)  # denormals are very slow on older x86 CPUs
import torch.nn.functional as Fnn

from ..stft import TightSTFT
from .common import make_bounds, pad_bounds, Box
from .social import _neigh_kernel, channel_mixing


class PEWDenoiser:
    def __init__(self, stft, neighs, device, dtype, shifts=1, combine="max", chan_gain=None, fw=None,
                 pilot_e=None, pilot_mix=0.0, gain_mode="pew"):
        self.stft = stft
        self.kernels = [_neigh_kernel(nb[0], nb[1], device, dtype) for nb in neighs]
        self.shifts = [int(round(i * stft.hop / shifts)) for i in range(shifts)]
        self.combine = combine
        self.chan_gain = None if chan_gain is None else torch.as_tensor(chan_gain, dtype=dtype, device=device)[:, None, None]
        self.fw = fw
        self.pilot_e = pilot_e  # list per shift of (C, F, K) energies
        self.pilot_mix = pilot_mix
        self.gain_mode = gain_mode

    def energy(self, a2):
        es = []
        for k in self.kernels:
            kt, kf = k.shape[-2:]
            es.append(Fnn.conv2d(a2[:, None], k, padding=(kt // 2, kf // 2))[:, 0])
        if len(es) == 1:
            return es[0]
        e = torch.stack(es, 0)
        return e.max(0).values if self.combine == "max" else e.mean(0)

    def __call__(self, x, lam):
        Tp = x.shape[-1]
        out = torch.zeros_like(x)
        lam2 = lam ** 2
        if self.chan_gain is not None:
            lam2 = lam2 * self.chan_gain ** 2
        if self.fw is not None:
            lam2 = lam2 * self.fw ** 2
        for si, s in enumerate(self.shifts):
            xs = torch.roll(x, -s, dims=-1) if s else x
            z = self.stft.analysis(xs)
            a2 = z.real ** 2 + z.imag ** 2
            e = self.energy(a2)
            if self.pilot_e is not None and self.pilot_mix > 0:
                e = (1 - self.pilot_mix) * e + self.pilot_mix * self.pilot_e[si]
            if self.gain_mode == "wiener":
                g = e / (e + lam2)
            else:
                g = torch.clamp(1.0 - lam2 / (e + 1e-30), min=0.0)
            xd = self.stft.synthesis(z * g, Tp)
            out += torch.roll(xd, s, dims=-1) if s else xd
        return out / len(self.shifts)


class NMFDenoiser:
    """Wiener-type shrinkage whose signal power comes from a low-rank NMF model of the current
    iterate's power spectrogram (templates shared by all channels, warm-started across calls).
    gain_mode 'wiener': V/(V+lam^2); 'pew': max(0, 1 - lam^2/V)."""

    def __init__(self, stft, rank=32, nmf_iter=2, beta=1.0, gain_mode="wiener", smooth_mix=0.0,
                 pew_kernel=None, device="cpu", dtype=torch.float32, chan_gain=None):
        self.stft, self.rank, self.nmf_iter, self.beta = stft, rank, nmf_iter, beta
        self.gain_mode, self.smooth_mix, self.pew_kernel = gain_mode, smooth_mix, pew_kernel
        self.W = None
        self.H = None
        self.device, self.dtype = device, dtype
        self.shifts = [0]
        self.chan_gain = None if chan_gain is None else torch.as_tensor(chan_gain, dtype=dtype, device=device)[:, None, None]

    def energy(self, a2):  # used for pilot energies (same interface as PEWDenoiser)
        return a2

    def _fit(self, P, n_iter):
        eps = 1e-12
        if self.W is None:
            g = torch.Generator().manual_seed(0)
            N, K = P.shape
            m = P.mean()
            self.W = (torch.rand(K, self.rank, generator=g, dtype=self.dtype) + 0.1).to(self.device) * torch.sqrt(m)
            self.H = (torch.rand(N, self.rank, generator=g, dtype=self.dtype) + 0.1).to(self.device) * torch.sqrt(m)
        W, H, b = self.W, self.H, self.beta
        for _ in range(n_iter):
            V = H @ W.T + eps
            if b == 1.0:  # KL: denominators are column sums (ones @ W == W.sum(0))
                H = H * ((P / V) @ W) / (W.sum(0, keepdim=True) + eps)
                V = H @ W.T + eps
                W = W * ((P / V).T @ H) / (H.sum(0, keepdim=True) + eps)
            else:
                H = H * ((V ** (b - 2) * P) @ W) / ((V ** (b - 1)) @ W + eps)
                V = H @ W.T + eps
                W = W * ((V ** (b - 2) * P).T @ H) / ((V ** (b - 1)).T @ H + eps)
            # normalize templates (scale into activations)
            s = W.sum(0, keepdim=True) + eps
            W = W / s
            H = H * s
        self.W, self.H = W, H
        return H @ W.T

    def __call__(self, x, lam):
        Tp = x.shape[-1]
        z = self.stft.analysis(x)
        C, F, K = z.shape
        a2 = z.real ** 2 + z.imag ** 2
        n_it = self.nmf_iter if self.W is not None else 50
        V = self._fit(a2.reshape(C * F, K), n_it).reshape(C, F, K)
        if self.smooth_mix > 0 and self.pew_kernel is not None:
            kt, kf = self.pew_kernel.shape[-2:]
            E = Fnn.conv2d(a2[:, None], self.pew_kernel, padding=(kt // 2, kf // 2))[:, 0]
            V = (1 - self.smooth_mix) * V + self.smooth_mix * E
        lam2 = lam ** 2
        if self.chan_gain is not None:
            lam2 = lam2 * self.chan_gain ** 2
        if self.gain_mode == "wiener":
            g = V / (V + lam2)
        else:
            g = torch.clamp(1.0 - lam2 / (V + 1e-30), min=0.0)
        return self.stft.synthesis(z * g, Tp)


def declip_pnp(y, m_hi, m_lo, th_hi, th_lo, sr=44100, win_len=4096, hop=1024, neigh=(3, 7), neighs=None,
               combine="max", shifts=1, n_iter=400, lam0=0.1, lam1=1e-4, stereo="pca", momentum=True,
               chan_gain=None, fweight=None, pilot=None, pilot_mix=0.0, gain_mode="pew", relax=1.0,
               device="cpu", dtype=torch.float32, callback=None, x_init=None, lam_ref=None,
               den_type="pew", nmf_rank=32, nmf_iter=2, nmf_beta=1.0, nmf_smooth=0.0, nmf_rank_ratio=None):
    T, C = y.shape
    scale = float(np.max(np.abs(np.concatenate([th_hi[np.isfinite(th_hi)], th_lo[np.isfinite(th_lo)], [1e-3]]))))
    lb, ub = make_bounds(y / scale, m_hi, m_lo, th_hi / scale, th_lo / scale)
    stft = TightSTFT(win_len, hop, None, device, dtype)
    left, right = stft.pad_len(T)
    # extra right padding so that circular shifts only wrap free (padded) samples
    right += win_len
    lb, ub = pad_bounds(lb, ub, left, right)
    Tp = lb.shape[1]
    rem = (Tp - win_len) % hop
    if rem:
        extra = hop - rem
        lb = np.concatenate([lb, np.full((C, extra), -np.inf)], 1)
        ub = np.concatenate([ub, np.full((C, extra), np.inf)], 1)
        Tp += extra
    proj = Box(lb, ub, device, dtype)
    Q = torch.as_tensor(channel_mixing(y, stereo), dtype=dtype, device=device)
    mix = lambda u: torch.einsum("ij,jt->it", Q, u)
    unmix = lambda x: torch.einsum("ji,jt->it", Q, x)

    def to_pad(a):
        v = torch.zeros(C, Tp, dtype=dtype, device=device)
        v[:, left:left + T] = torch.as_tensor(a.T / scale, dtype=dtype, device=device)
        return v

    fw = None
    if fweight:
        K = (stft.nfft // 2 + 1)
        f = torch.arange(K, dtype=dtype, device=device).clamp(min=1.0) / (K * 0.05)
        fw = f ** float(fweight)
        fw = fw / fw.mean()
    if den_type == "nmf":
        if nmf_rank_ratio is not None:
            # rank proportional to the number of spectrogram rows (channels x frames) of this chunk
            n_rows = C * ((Tp - win_len) // hop + 1)
            nmf_rank = int(max(16, min(nmf_rank, round(nmf_rank_ratio * n_rows))))
        den = NMFDenoiser(stft, nmf_rank, nmf_iter, nmf_beta, gain_mode, nmf_smooth,
                          _neigh_kernel(neigh[0], neigh[1], device, dtype), device, dtype, chan_gain)
    else:
        den = PEWDenoiser(stft, neighs or [neigh], device, dtype, shifts, combine, chan_gain, fw, None, pilot_mix, gain_mode)
    if pilot is not None:
        pu = unmix(to_pad(pilot))
        pe = []
        for s in den.shifts:
            zp = stft.analysis(torch.roll(pu, -s, dims=-1) if s else pu)
            pe.append(den.energy(zp.real ** 2 + zp.imag ** 2))
        den.pilot_e = pe

    x = to_pad(y if x_init is None else x_init)
    # lambda reference: max |coefficient| of the (normalized) clipped signal; pass lam_ref to use a
    # file-global value so that chunks of a long file share the same schedule
    zmax = lam_ref if lam_ref is not None else float(torch.abs(stft.analysis(unmix(to_pad(y)))).max())
    lams = np.geomspace(lam0 * zmax, lam1 * zmax, n_iter)
    xbar = x.clone()
    t = 1.0
    for it in range(n_iter):
        px = proj(xbar)
        if relax != 1.0:
            px = xbar + relax * (px - xbar)
        xn = mix(den(unmix(px), lams[it]))
        if momentum:
            tn = 0.5 * (1 + math.sqrt(1 + 4 * t * t))
            xbar = xn + ((t - 1) / tn) * (xn - x)
            t = tn
        else:
            xbar = xn
        x = xn
        if callback is not None and (it % 50 == 49 or it == n_iter - 1):
            callback(it, proj(x)[:, left:left + T].T.cpu().numpy().astype(np.float64) * scale)
    x = proj(x)
    return x[:, left:left + T].T.cpu().numpy().astype(np.float64) * scale
