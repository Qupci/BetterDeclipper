"""Check the A-SPADE "cache A(xn)" optimization (user report): the transform A(xn) computed for the residual
at the end of an iteration is exactly A(xa) at the start of the next one (xa = xn), so caching it saves one
of the three FFTs per iteration. Compares the current methods.spade._aspade_batch with the previous
implementation (copied below): bit-identical output and SPADE time.
usage: python spade_cache.py [cpu] [file ...]"""
import sys, os, time, math
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf, torch
from betterdeclipper.methods import spade as S
from betterdeclipper.engine import declip, _model_kwargs
from betterdeclipper.detect import estimate_lsb
from betterdeclipper import auto as A_


def aspade_ref(Yb, LBb, UBb, A, As, s, r, eps, max_iter, M, rank_w, pad_multiple):
    """previous implementation (A(xa) recomputed at the start of every iteration)"""
    n = Yb.shape[0]
    dev = Yb.device
    x = Yb.clone()
    xa, ua, LBa, UBa = x, torch.zeros_like(A(Yb)), LBb, UBb
    rows = torch.arange(n, device=dev)
    live = torch.ones(n, dtype=torch.bool, device=dev)
    kk = s
    n_live = n
    it = 0
    for it in range(max_iter):
        zb = S._hard_single_k(A(xa).add_(ua), min(kk, M), rank_w)
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


def constraints(y, sr, device):
    lsb = estimate_lsb(y)

    def run_fast(yy, cons):
        return declip(yy, sr, models=[("nmf", 93, dict(n_iter=80))], constraints=cons, verbose=False, device=device)[0]
    m_hi, m_lo, th_hi, th_lo, rep = A_.auto_constraints(y, lsb, run_fast, mode="auto", sr=sr)
    return m_hi, m_lo, th_hi, th_lo, rep["mode"]


def timed(fn, dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    out = fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    return out, time.time() - t


if __name__ == "__main__":
    args = sys.argv[1:]
    dev = "cpu" if args and args[0] == "cpu" else "cuda"
    files = [a for a in args if a != "cpu"] or ["F:/BetterDeclipper/ex_sample/sample_-12dB_clipped.wav"]
    new_fn = S._aspade_batch
    for fn in files:
        y, sr = sf.read(fn, dtype="float64", always_2d=True)
        if dev == "cpu":
            y = y[: 8 * sr]                   # CPU: 8 s excerpt
        m_hi, m_lo, th_hi, th_lo, mode = constraints(y, sr, dev)
        kw = _model_kwargs("spade", 93, sr, {})
        run = lambda: S.declip_spade(y, m_hi, m_lo, th_hi, th_lo, device=dev, **kw)
        run()                                 # warm-up (cuFFT plans, allocator)
        res = {}
        for name, f in (("ref", aspade_ref), ("new", new_fn), ("ref2", aspade_ref), ("new2", new_fn)):
            S._aspade_batch = f
            res[name] = timed(run, dev)
        S._aspade_batch = new_fn
        same = np.array_equal(res["ref"][0], res["new"][0])
        maxd = float(np.abs(res["ref"][0] - res["new"][0]).max())
        t_ref = min(res["ref"][1], res["ref2"][1]); t_new = min(res["new"][1], res["new2"][1])
        print(f"{os.path.basename(fn)[:34]:34s} {len(y)/sr:6.1f}s {dev} {mode[:26]:26s} flagged {(m_hi|m_lo).mean()*100:5.2f}%"
              f"  SPADE ref {t_ref:6.2f}s  new {t_new:6.2f}s  ({(1 - t_new/t_ref)*100:4.1f}% faster)"
              f"  bit-identical {same} (max diff {maxd:.2e})", flush=True)
