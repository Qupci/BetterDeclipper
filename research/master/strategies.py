"""Experimental constraint strategies for the master benchmark (registered into run_bench.STRATEGIES)."""
import sys
sys.path.insert(0, "F:/BetterDeclipper"); sys.path.insert(0, "F:/BetterDeclipper/research/master")
import numpy as np
from betterdeclipper.engine import declip, default_knees
from betterdeclipper.detect import estimate_lsb, detect_clip_levels
from smear_fit import fit_smeared


def ceiling_constraints(y, params, lsb):
    """params[c] = (pos, neg), each None or dict(knee, cap, tol) in positive amplitude units.
    Flag sgn*y >= knee; lower bound sgn*x >= min(sgn*y, cap) - tol."""
    T, C = y.shape
    m_hi = np.zeros((T, C), bool); m_lo = np.zeros((T, C), bool)
    th_hi = np.full((T, C), np.inf); th_lo = np.full((T, C), -np.inf)
    for c, pr in enumerate(params):
        for sgn, p, m, th in ((1.0, pr[0], m_hi, th_hi), (-1.0, pr[1], m_lo, th_lo)):
            if p is None:
                continue
            s = sgn * y[:, c]
            f = s >= p["knee"]
            lbv = np.minimum(s, p.get("cap", np.inf)) - p.get("tol", 2 * lsb)
            m[:, c] = f
            th[f, c] = sgn * lbv[f]
    return m_hi, m_lo, th_hi, th_lo


_fit_cache = {}


def smear_params(y, lsb):
    key = (y.shape, float(y[:1000].sum()), float(np.abs(y).max()))
    if key not in _fit_cache:
        out = []
        for c in range(y.shape[1]):
            out.append(tuple(fit_smeared(sg * y[:, c], lsb) for sg in (1.0, -1.0)))
        _fit_cache[key] = out
    return _fit_cache[key]


def smear(c1, c2):
    def run(y, sr, preset):
        lsb = estimate_lsb(y)
        fits = smear_params(y, lsb)
        params = [tuple(None if f is None else dict(knee=f["theta"] - c1 * f["sigma"], cap=f["theta"], tol=c2 * f["sigma"] + 2 * lsb)
                        for f in pr) for pr in fits]
        cons = ceiling_constraints(y, params, lsb)
        return declip(y, sr, preset=preset, constraints=cons, verbose=False)
    return run


def softcap(kfrac, c2):
    """knee = kfrac * theta (smeared fit), cap theta, tol c2*sigma"""
    def run(y, sr, preset):
        lsb = estimate_lsb(y)
        fits = smear_params(y, lsb)
        params = [tuple(None if f is None else dict(knee=kfrac * f["theta"], cap=f["theta"], tol=c2 * f["sigma"] + 2 * lsb)
                        for f in pr) for pr in fits]
        return declip(y, sr, preset=preset, constraints=ceiling_constraints(y, params, lsb), verbose=False)
    return run


def register(S):
    for c1 in (1, 2, 4, 8):
        for c2 in (0, 1, 2):
            S[f"smear:{c1}:{c2}"] = smear(c1, c2)
    for k in (0.6, 0.7, 0.8, 0.9, 0.95):
        for c2 in (0, 1):
            S[f"softcap:{k}:{c2}"] = softcap(k, c2)


def prox_constraints(y, tau, lsb, union=False, touch_rel=0.003, knee_frac=None):
    """Limiter model: within +-tau samples of a ceiling touch, |x| >= |y| (gain <= 1 preserves sign).
    Touch: |y| >= theta*(1 - touch_rel) with theta from the smeared fit per channel/polarity."""
    T, C = y.shape
    fits = smear_params(y, lsb)
    touch = np.zeros((T, C), bool)
    for c in range(C):
        for sg, f in zip((1.0, -1.0), fits[c]):
            if f is not None:
                touch[:, c] |= sg * y[:, c] >= f["theta"] * (1 - touch_rel)
    if union:
        touch[:] = touch.any(1, keepdims=True)
    # dilate by tau (box filter via cumulative sum)
    near = np.zeros_like(touch)
    for c in range(C):
        cs = np.concatenate([[0], np.cumsum(touch[:, c])])
        lo = np.clip(np.arange(T) - tau, 0, T); hi = np.clip(np.arange(T) + tau + 1, 0, T)
        near[:, c] = (cs[hi] - cs[lo]) > 0
    if knee_frac is not None:  # also flag everything above a level knee
        pk = np.quantile(np.abs(y), 1 - 1e-4, axis=0)
        near |= np.abs(y) >= knee_frac * pk[None, :]
    tol = 2 * lsb
    m_hi = near & (y > 0); m_lo = near & (y < 0)
    th_hi = np.where(m_hi, y - tol, np.inf); th_lo = np.where(m_lo, y + tol, -np.inf)
    return m_hi, m_lo, th_hi, th_lo


def prox(tau_ms, union=False, knee_frac=None):
    def run(y, sr, preset):
        lsb = estimate_lsb(y)
        cons = prox_constraints(y, int(round(tau_ms * 1e-3 * sr)), lsb, union=union, knee_frac=knee_frac)
        return declip(y, sr, preset=preset, constraints=cons, verbose=False)
    return run


def register_prox(S):
    for t in (0.25, 0.5, 1.0, 2.0):
        S[f"prox:{t}"] = prox(t)
        S[f"proxu:{t}"] = prox(t, union=True)
    for t in (0.5, 1.0):
        for k in (0.7, 0.8):
            S[f"proxk:{t}:{k}"] = prox(t, knee_frac=k)


def envelope_constraints(y, lsb, k0=0.5, R=0.25, p=2.0, eps=0.002):
    """Soft knee with a level-dependent cap: flagged |y| >= k0*theta; lb = min(|y|, theta);
    ub = gcap * lb with gcap = 1 + R*((|y| - k0 theta)/(theta - k0 theta))^p below the ceiling, inf at the ceiling."""
    T, C = y.shape
    fits = smear_params(y, lsb)
    m_hi = np.zeros((T, C), bool); m_lo = np.zeros((T, C), bool)
    th_hi = np.full((T, C), np.inf); th_lo = np.full((T, C), -np.inf)
    gcap = np.full((T, C), np.inf)
    for c in range(C):
        for sg, f in zip((1.0, -1.0), fits[c]):
            s = sg * y[:, c]
            P = np.quantile(s[s > 0], 1 - 1e-4) if (s > 0).sum() > 100 else s.max()
            th = min(f["theta"], P) if f is not None else P
            fl = s >= k0 * th
            lbv = np.minimum(s, th)
            rel = np.clip((s - k0 * th) / max(th - k0 * th, 1e-9), 0, 1)
            g = np.where(s >= th * (1 - eps), np.inf, 1 + R * rel ** p)
            m = m_hi if sg > 0 else m_lo
            m[:, c] = fl
            (th_hi if sg > 0 else th_lo)[fl, c] = sg * lbv[fl]
            gcap[fl, c] = g[fl]
    return m_hi, m_lo, th_hi, th_lo, gcap


def envelope(k0, R, p):
    def run(y, sr, preset):
        lsb = estimate_lsb(y)
        return declip(y, sr, preset=preset, constraints=envelope_constraints(y, lsb, k0, R, p), verbose=False)
    return run


def register_env(S):
    for k0 in (0.5, 0.6):
        for R in (0.1, 0.25, 0.5):
            for p in (1, 2):
                S[f"env:{k0}:{R}:{p}"] = envelope(k0, R, p)


def ref_levels(y, lsb):
    """per channel/polarity (theta_ref, sigma, P): ceiling from the smeared fit if it lies below the robust peak"""
    fits = smear_params(y, lsb)
    out = []
    for c in range(y.shape[1]):
        row = []
        for sg, f in zip((1.0, -1.0), fits[c]):
            s = sg * y[:, c]
            P = np.quantile(s[s > 0], 1 - 1e-4) if (s > 0).sum() > 100 else max(s.max(), 1e-9)
            if f is not None and f["theta"] <= P * 1.001 and f["bump_ratio"] > 1.0:
                row.append((f["theta"], f["sigma"], P))
            else:
                row.append((P, 0.0, P))
        out.append(row)
    return out


def knee_constraints(y, lsb, knee_rel, refs, c1=6.0):
    """flag |y| >= min(knee_rel*theta, theta - c1*sigma); lb = min(|y|, theta)"""
    params = []
    for c in range(y.shape[1]):
        row = []
        for (th, sig, P) in refs[c]:
            k = min(knee_rel * th, th - c1 * sig) if knee_rel is not None else th - c1 * sig
            row.append(dict(knee=k, cap=th, tol=2 * lsb))
        params.append(tuple(row))
    return ceiling_constraints(y, params, lsb)


LV = np.arange(0.45, 1.0001, 0.025)
_pass1_cache = {}
_pass2_cache = {}


def raise_curve(y, x, refs, lv=LV, w=0.025):
    a_all, r_all = [], []
    for c in range(y.shape[1]):
        for sg, (th, sig, P) in zip((1.0, -1.0), refs[c]):
            s = sg * y[:, c]; xs = sg * x[:, c]
            m = s > 0.4 * th
            a_all.append(s[m] / th); r_all.append(xs[m] / s[m])
    a = np.concatenate(a_all); r = np.concatenate(r_all)
    cur = np.full(len(lv), np.nan)
    for i, l in enumerate(lv):
        m = (a >= l - w) & (a < l + w)
        if m.sum() > 30:
            cur[i] = np.median(r[m]) - 1
    return cur


def knee_from_raise(cur, lv=LV, delta=0.01, base=(0.45, 0.6)):
    ok = np.isfinite(cur)
    bm = ok & (lv >= base[0] - 1e-9) & (lv <= base[1] + 1e-9)
    if bm.sum() < 3:
        return None
    b1, b0 = np.polyfit(lv[bm], cur[bm], 1)
    exc = cur - (b0 + b1 * lv)
    # lowest level from which the excess stays above delta up to the top valid bin
    idx = np.flatnonzero(ok)
    knee = None
    for i in idx[::-1]:
        if exc[i] > delta:
            knee = lv[i]
        else:
            break
    return knee


def raise2(delta=0.01, k0=0.4, c1=6.0, preset1="fast"):
    def run(y, sr, preset):
        lsb = estimate_lsb(y)
        refs = ref_levels(y, lsb)
        key = (y.shape, float(np.abs(y).sum()), k0, c1, preset1)
        if key not in _pass1_cache:
            x1, _ = declip(y, sr, preset=preset1, constraints=knee_constraints(y, lsb, k0, refs, c1), verbose=False)
            _pass1_cache.clear()
            _pass1_cache[key] = raise_curve(y, x1, refs)
        cur = _pass1_cache[key]
        kr = knee_from_raise(cur, delta=delta)
        # a knee within the top bins is indistinguishable from the ceiling itself -> hard/smeared
        kr = None if (kr is not None and kr >= 0.975) else kr
        k2 = (key, kr, preset)
        if k2 in _pass2_cache:
            x, info = _pass2_cache[k2]
        else:
            x, info = declip(y, sr, preset=preset, constraints=knee_constraints(y, lsb, kr, refs, c1), verbose=False)
            if len(_pass2_cache) > 4:
                _pass2_cache.clear()
            _pass2_cache[k2] = (x, dict(info))
        info = dict(info)
        info["mode"] = "hard" if kr is None else f"k{kr:.3f}"
        info["raise"] = cur.tolist()
        return x, info
    return run


def register_raise(S):
    for d in (0.005, 0.01, 0.02):
        S[f"raise:{d}"] = raise2(d)
    S["smear6"] = smear(6.0, 0.0)


def dip_setup(y, sr, lsb, tau_ms=3.0, merge_ms=0.5, touch_rel=0.003):
    """Limiter events (clusters of ceiling touches) per channel and a triangular dip window around each."""
    T, C = y.shape
    refs = ref_levels(y, lsb)
    tau = max(2, int(round(tau_ms * 1e-3 * sr)))
    merge = max(1, int(round(merge_ms * 1e-3 * sr)))
    idx = np.full((T, C), -1, dtype=np.int64)
    v = np.zeros((T, C))
    n_ev = 0
    for c in range(C):
        tp = np.zeros(T, bool)
        for sg, (th, sig, P) in zip((1.0, -1.0), refs[c]):
            tp |= sg * y[:, c] >= th * (1 - touch_rel)
        t = np.flatnonzero(tp)
        if t.size == 0:
            continue
        # clusters of touches -> event centre at the largest |y|
        brk = np.flatnonzero(np.diff(t) > merge) + 1
        centers = np.array([seg[np.argmax(np.abs(y[seg, c]))] for seg in np.split(t, brk)])
        pos = np.arange(T)
        j = np.clip(np.searchsorted(centers, pos), 1, len(centers) - 1) if len(centers) > 1 else np.zeros(T, int)
        if len(centers) > 1:
            left, right = centers[j - 1], centers[j]
            near = np.where(pos - left <= right - pos, j - 1, j)
        else:
            near = np.zeros(T, int)
        d = np.abs(pos - centers[near])
        w = np.maximum(0.0, 1.0 - d / tau)
        inwin = w > 0
        idx[inwin, c] = n_ev + near[inwin]
        v[inwin, c] = w[inwin]
        n_ev += len(centers)
    return idx, v, n_ev


def dip(tau_ms=3.0, bmax=1.0):
    def run(y, sr, preset):
        lsb = estimate_lsb(y)
        idx, v, n_ev = dip_setup(y, sr, lsb, tau_ms)
        fl = idx >= 0
        tol = 2 * lsb
        m_hi = fl & (y > 0); m_lo = fl & (y < 0)
        th_hi = np.where(m_hi, y - tol, np.inf); th_lo = np.where(m_lo, y + tol, -np.inf)
        models = [("nmf", 93, dict(n_iter=150))] if preset == "fast" else [("nmf", 93, {})]
        x, info = declip(y, sr, preset=preset, constraints=(m_hi, m_lo, th_hi, th_lo), dips=(idx, v, n_ev, bmax),
                         models=models, verbose=False)
        info["mode"] = f"dip{n_ev}"
        return x, info
    return run


def register_dip(S):
    for t in (1.0, 2.0, 3.0, 5.0):
        S[f"dip:{t}"] = dip(t)
    S["dip:3.0:b0.5"] = dip(3.0, 0.5)
