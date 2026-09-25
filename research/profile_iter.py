"""Time the parts of one NMF-PnP iteration at full-example scale (21.7 s stereo, 4096 window)."""
import sys, time; sys.path.insert(0, "F:/BetterDeclipper")
import torch, numpy as np
torch.set_flush_denormal(True)
import torch.nn.functional as Fnn
from betterdeclipper.stft import TightSTFT
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
T = 44100 * 22; C = 2
st = TightSTFT(4096, 1024, device=dev)
Tp = T + 2 * 3072 + 4096; Tp += (1024 - (Tp - 4096) % 1024) % 1024
x = torch.randn(C, Tp, device=dev) * 0.3
Q = torch.tensor([[0.7, 0.7], [0.7, -0.7]], device=dev)
lb = torch.full((C, Tp), -1.0, device=dev); ub = torch.full((C, Tp), 1.0, device=dev)
z = st.analysis(x); Cn, F, K = z.shape
N, r = C * F, 128
W = torch.rand(K, r, device=dev) + 0.1; H = torch.rand(N, r, device=dev) + 0.1
def sync():
    if dev != "cpu": torch.cuda.synchronize()
def tm(name, f, n=5):
    f(); sync(); t = time.time()
    for _ in range(n): out = f()
    sync(); dt = (time.time() - t) / n * 1000
    print(f"  {name:28s} {dt:8.2f} ms"); return out, dt
tot = 0
_, d = tm("mix/unmix einsum", lambda: torch.einsum("ji,jt->it", Q, x)); tot += 2 * d
_, d = tm("projection (box)", lambda: torch.maximum(torch.minimum(x, ub), lb)); tot += d
_, d = tm("stft analysis", lambda: st.analysis(x)); tot += d
a2, d = tm("abs2 (complex)", lambda: z.real ** 2 + z.imag ** 2); tot += d
P = a2.reshape(N, K)
def mu():
    V = H @ W.T + 1e-12
    H2 = H * ((P / V) @ W) / (W.sum(0, keepdim=True) + 1e-12)
    V = H2 @ W.T + 1e-12
    W2 = W * ((P / V).T @ H2) / (H2.sum(0, keepdim=True) + 1e-12)
    return H2 @ W2.T
V, d = tm("NMF MU step (rank 128)", mu); tot += d
V = V.reshape(C, F, K)
g, d = tm("wiener gain", lambda: V / (V + 0.01)); tot += d
zg, d = tm("complex * gain", lambda: z * g); tot += d
_, d = tm("stft synthesis", lambda: st.synthesis(zg, Tp)); tot += d
_, d = tm("fista momentum", lambda: x + 0.5 * (x - x)); tot += d
print(f"  {'TOTAL per iteration':28s} {tot:8.2f} ms   (frames {F}, bins {K})")
