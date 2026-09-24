"""Parseval tight-frame STFT (Hann window, overlap-add) implemented with torch.

analysis(x) -> complex coefficients (B, F, K); synthesis(c) -> time signal (B, T).
synthesis(analysis(x)) == x exactly (up to float error) on the padded signal, and
||analysis(x)||^2 == ||x||^2 when the rfft bins are weighted (interior bins count twice).
"""
import math
import torch
import torch.nn.functional as Fnn


class TightSTFT:
    def __init__(self, win_len=2048, hop=512, nfft=None, device="cpu", dtype=torch.float32):
        assert win_len % hop == 0
        self.win_len = win_len
        self.hop = hop
        self.nfft = nfft or win_len
        self.device = device
        self.dtype = dtype
        w = torch.hann_window(win_len, periodic=True, dtype=torch.float64)
        # sum_k w^2(n - k*hop) is constant = (win_len/hop) * mean(w^2)
        const = (win_len / hop) * torch.mean(w ** 2)
        self.win = (w / torch.sqrt(const)).to(device=device, dtype=dtype)
        # rfft bin weights for inner products (interior bins represent 2 complex bins)
        wt = torch.full((self.nfft // 2 + 1,), 2.0, dtype=dtype, device=device)
        wt[0] = 1.0
        if self.nfft % 2 == 0:
            wt[-1] = 1.0
        self.bin_weight = wt

    def pad_len(self, T):
        """Padding (left, right) so that every sample is covered by win_len/hop frames."""
        left = self.win_len - self.hop
        total = T + 2 * left
        rem = (total - self.win_len) % self.hop
        right = left + ((self.hop - rem) % self.hop)
        return left, right

    def analysis(self, x):
        """x: (B, Tp) padded signal -> (B, F, K) complex."""
        frames = x.unfold(-1, self.win_len, self.hop) * self.win
        return torch.fft.rfft(frames, n=self.nfft, dim=-1, norm="ortho")

    def synthesis(self, c, Tp):
        """c: (B, F, K) complex -> (B, Tp) real (overlap-add)."""
        frames = torch.fft.irfft(c, n=self.nfft, dim=-1, norm="ortho")[..., : self.win_len] * self.win
        return overlap_add(frames, self.hop, Tp)


def overlap_add(frames, hop, Tp):
    """frames: (B, F, W) with W a multiple of hop -> (B, Tp). Much faster than F.fold on CPU."""
    B, F, W = frames.shape
    R = W // hop
    fr = frames.reshape(B, F, R, hop)
    out = torch.zeros(B, F + R - 1, hop, dtype=frames.dtype, device=frames.device)
    for j in range(R):
        out[:, j:j + F] += fr[:, :, j]
    out = out.reshape(B, (F + R - 1) * hop)
    if out.shape[1] < Tp:
        out = Fnn.pad(out, (0, Tp - out.shape[1]))
    return out[:, :Tp]
