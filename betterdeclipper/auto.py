"""Automatic analysis of how a master was clipped/limited, and the matching constraint sets.

Every mode uses one constraint family: a sample is flagged (unreliable) when |y| >= knee, and then the
restoration must satisfy |x| >= min(|y|, theta) with the same sign (theta: the ceiling). Unflagged samples
are kept exactly. Special cases:
  hard clip       knee = theta (plateau)                       -> |x| >= theta
  soft clip       knee < theta, no overshoot above theta         -> |x| >= |y| above the knee
  smeared ceiling knee = theta - c*sigma (lossy codec, resampling, oversampled clipper: the plateau is
                  blurred by sigma and codec/filter overshoots above theta must not be forced upward)
The knee of a soft shoulder is found with a first restoration pass (see knee_from_raise).
"""
import numpy as np

from .detect import fit_ceiling, detect_clip_levels

SMEAR_C = 6.0          # smeared ceilings: knee = theta - SMEAR_C * sigma
SHARP_SIGMA = 1e-3     # sigma/theta below this: sharp ceiling (dither only)
MIN_EVENTS = 20          # limiter / smeared ceilings must be hit at least this often ...
MIN_EVENT_RATE = 8.0     # ... and this often per second (real ones: 10-390/s; unclipped excerpts: <= 6.2/s)
SMEAR_MAX = 0.07       # sigma/theta above this: no ceiling (natural density tail; codecs measured 0.008-0.06)


def _run_stats(s, level, lsb, min_run=4):
    """Samples at/above `level`: fraction lying in runs of >= min_run consecutive samples, and the fraction
    of within-run steps that are flat (<= 3 LSB / 1e-4 x level). Hard clipping: long flat runs;
    oversampled / resampled clipping: long rippled runs; limiters: isolated touches."""
    at = s >= level
    n = int(at.sum())
    if n == 0:
        return 0.0, 0.0
    m = np.concatenate([[0], at.astype(np.int8), [0]])
    d = np.diff(m)
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    L = en - st
    long_frac = float(L[L >= min_run].sum() / n)
    inrun = at[1:] & at[:-1]
    if not inrun.any():
        return long_frac, 0.0
    steps = np.abs(np.diff(s))[inrun]
    flat = float(np.mean(steps <= max(3 * max(lsb, 1e-7), 1e-4 * level)))
    return long_frac, flat


def _touch_events(s, level):
    """number of separate runs of samples at/above level"""
    at = s >= level
    return int(np.count_nonzero(np.diff(at.astype(np.int8)) == 1) + (1 if at[0] else 0))


def analyze_ceilings(y, lsb, sr=44100):
    """Per channel and polarity: dict(theta, sigma, peak, kind, flat, bump).

    kind: 'clip'     sharp ceiling made of flat runs (hard clipping)
          'limiter'  sharp ceiling touched by isolated samples (brickwall limiter / true-peak stage)
          'smeared'  blurred ceiling with overshoots (lossy codec, resampling, oversampled clipping)
          'none'     no ceiling found (unclipped, or soft saturation without a ceiling)"""
    T, C = y.shape
    plateau = detect_clip_levels(y, lsb)
    out = []
    for c in range(C):
        row = []
        for pi, sg in enumerate((1.0, -1.0)):
            s = sg * y[:, c]
            pos = s[s > 0]
            if pos.size < 2000:
                row.append(dict(theta=None, sigma=0.0, peak=None, kind="none", flat=0.0, bump=0.0))
                continue
            smax = float(pos.max())
            P = float(np.quantile(pos, 1 - 1e-4))
            f = fit_ceiling(pos, lsb)
            pl = plateau[c][pi]
            pl = abs(pl) if pl is not None else None
            info = dict(theta=P, sigma=0.0, peak=P, kind="none", flat=0.0, bump=0.0 if f is None else f["bump_ratio"])
            sharp = None
            if pl is not None:
                sharp = pl
            elif (f is not None and f["bump_ratio"] > 10.0 and f["sigma"] < SHARP_SIGMA * f["theta"]
                  and f["theta"] <= smax * 1.0005):
                sharp = min(f["theta"], smax)
            if sharp is not None:
                sig = f["sigma"] if (f is not None and abs(f["theta"] - sharp) < 0.01 * sharp) else 0.0
                sig = min(sig, SHARP_SIGMA * sharp)
                lvl = sharp - max(3 * sig, 1.5 * max(lsb, 1e-7))
                long_frac, flat = _run_stats(s, lvl, lsb)
                few = _touch_events(s, lvl) < max(MIN_EVENTS, MIN_EVENT_RATE * T / sr)
                if long_frac < 0.3:
                    kind = "none" if few else "limiter"   # isolated touches: limiter, or just the maximum
                elif flat >= 0.5:
                    kind = "clip"             # flat plateaus (only plateau samples get flagged: always safe)
                elif few:
                    kind = "none"
                else:
                    kind = "smeared"          # long but rippled plateaus (oversampled / resampled clipping)
                    sig = max(sig, 0.5 * SHARP_SIGMA * sharp)
                if kind == "none":
                    info.update(flat=flat, long=long_frac)
                else:
                    info.update(theta=sharp, sigma=sig, flat=flat, long=long_frac, kind=kind)
            elif (f is not None and SHARP_SIGMA * f["theta"] <= f["sigma"] < SMEAR_MAX * f["theta"]
                  and smax > f["theta"] + 1.5 * f["sigma"]
                  and _touch_events(s, f["theta"] - 2 * f["sigma"]) >= max(MIN_EVENTS, MIN_EVENT_RATE * T / sr)):
                info.update(theta=f["theta"], sigma=f["sigma"], kind="smeared")
            row.append(info)
        out.append(row)
    _share_limiter_ceiling(y, out, lsb)
    return out


def _share_limiter_ceiling(y, refs, lsb):
    """One limiter acts on all channels and polarities with one ceiling, but some of them may touch it too
    rarely for the hit-rate test (SPACELLEX: 37 touches/s on L+, ~4/s on the others). A polarity without a
    ceiling that reaches a limiter ceiling of the same file, never exceeds it and touches it at least
    MIN_EVENTS times joins that limiter, so all sides are restored alike."""
    lims = [r for row in refs for r in row if r["kind"] == "limiter"]
    for c, row in enumerate(refs):
        for pi, r in enumerate(row):
            if r["kind"] != "none" or r["theta"] is None:
                continue
            s = (1.0 if pi == 0 else -1.0) * y[:, c]
            for L in lims:
                lvl = L["theta"] - max(3 * L["sigma"], 1.5 * max(lsb, 1e-7))
                if s.max() <= L["theta"] * 1.001 and _touch_events(s, lvl) >= MIN_EVENTS:
                    r.update(theta=L["theta"], sigma=L["sigma"], kind="limiter", shared=True)
                    break


def ceiling_constraints(y, refs, knees, lsb, tol_lsb=2.0):
    """refs from analyze_ceilings; knees[c][p]: knee level (amplitude) or None (polarity not flagged).
    Returns m_hi, m_lo, th_hi, th_lo (T, C): flagged masks and per-sample lower bounds on |x| (signed).
    At a limiter ceiling (knee at the ceiling) only flat plateaus of >= 2 samples are flagged (the evidence
    of clipping): single samples touching it are what a limiter's smooth gain produces anyway, and freeing
    them only invites spikes."""
    T, C = y.shape
    m_hi = np.zeros((T, C), bool); m_lo = np.zeros((T, C), bool)
    th_hi = np.full((T, C), np.inf); th_lo = np.full((T, C), -np.inf)
    tol = tol_lsb * max(lsb, 1e-7)
    for c in range(C):
        for pi, sg in enumerate((1.0, -1.0)):
            k = knees[c][pi]
            r = refs[c][pi]
            if k is None or r["theta"] is None:
                continue
            s = sg * y[:, c]
            fl = s >= k
            if r["kind"] == "limiter" and k >= 0.99 * r["theta"]:
                pair = fl[1:] & fl[:-1] & (np.abs(np.diff(s)) <= max(3 * max(lsb, 1e-7), 1e-4 * r["theta"]))
                fl = np.concatenate([pair, [False]]) | np.concatenate([[False], pair])
            lbv = np.minimum(s, r["theta"])
            if r["kind"] != "clip":
                lbv = lbv - tol          # quantization tolerance where |x| >= |y| is used
            m, th = (m_hi, th_hi) if sg > 0 else (m_lo, th_lo)
            m[:, c] = fl
            th[fl, c] = sg * lbv[fl]
    return m_hi, m_lo, th_hi, th_lo


def base_knees(refs, smear_c=SMEAR_C):
    """Knees without a soft shoulder: plateau edge for clip plateaus, theta - c*sigma for smeared and limiter
    ceilings, None where there is no ceiling."""
    out = []
    for row in refs:
        kr = []
        for r in row:
            if r["kind"] == "clip":
                kr.append(r["theta"])
            elif r["kind"] in ("smeared", "limiter"):
                kr.append(r["theta"] - smear_c * r["sigma"])
            else:
                kr.append(None)
        out.append(kr)
    return out


# ---- soft-shoulder knee from a first restoration pass ------------------------------------------------
RAISE_K0 = 0.4                                   # first pass: flag everything above 0.4 x ceiling
LIMITER_KNEE = 0.5     # experimental mode 'limiter': soft region below limiter ceilings (fraction of the ceiling)
LIMITER_MAX_GAIN_DB = 9.0  # restoration cap under limiter ceilings (x observed value / ceiling)
KNEE_MIN = 0.5         # lowest soft knee (the first pass cannot see compression below ~0.45 x ceiling)
ACC_MIN_CEIL = 0.1     # min. acceleration of the lift curve for a shoulder below a detected ceiling
ACC_MIN_FREE = 0.45    # ... when no ceiling was found (unclipped sources reached 0.31, tanh >= 1.16)
ACC_HINT_FREE = 0.25   # no ceiling, acceleration between this and ACC_MIN_FREE: reported as a weak sign only
RAISE_LV = np.arange(0.45, 1.0001, 0.025)        # level grid (fraction of the ceiling)


def raise_curve(y, x, refs, lv=RAISE_LV, w=0.025, use=None):
    """Median lift |x|/|y| - 1 of the first-pass restoration per level bin (levels relative to theta).
    use[c][p]: optional mask of the polarities to include."""
    a_all, r_all = [], []
    for c in range(y.shape[1]):
        for pi, sg in enumerate((1.0, -1.0)):
            th = refs[c][pi]["theta"]
            if th is None or (use is not None and not use[c][pi]):
                continue
            s = sg * y[:, c]; xs = sg * x[:, c]
            m = s > 0.4 * th
            a_all.append(s[m] / th); r_all.append(xs[m] / s[m])
    if not a_all:
        return np.full(len(lv), np.nan)
    a = np.concatenate(a_all); r = np.concatenate(r_all)
    cur = np.full(len(lv), np.nan)
    for i, l in enumerate(lv):
        m = (a >= l - w) & (a < l + w)
        if m.sum() > 30:
            cur[i] = np.median(r[m]) - 1
    return cur


def _lift_slopes(cur, lv=RAISE_LV, top=0.95):
    """slopes of the smoothed lift curve up to `top`, their levels, the index of the minimum slope and the
    acceleration (mean slope of the top two levels minus the minimum slope), or None if too few points"""
    ok = np.isfinite(cur) & (lv <= top + 1e-9)
    c = np.asarray(cur)[ok]; l = lv[ok]
    if c.size < 6:
        return None
    cs = np.convolve(np.pad(c, 1, mode="edge"), np.ones(3) / 3, "valid")
    sl = np.gradient(cs, l)
    i0 = int(np.argmin(sl[: max(2, len(sl) - 3)]))
    return sl, l, i0, float(sl[-2:].mean() - sl[i0])


def lift_acceleration(cur, lv=RAISE_LV, top=0.95):
    """How much the first-pass lift curve accelerates towards the ceiling (NaN if undetermined)."""
    r = _lift_slopes(cur, lv, top)
    return float("nan") if r is None else r[3]


def knee_from_raise(cur, lv=RAISE_LV, frac=0.1, acc_min=0.1, top=0.95):
    """Soft-shoulder knee (fraction of theta) or None from the first-pass lift curve.

    Freed but uncompressed samples are lifted by a bias that grows with level and then flattens; real
    compression makes the curve accelerate towards the ceiling. The knee is where the slope of the
    (smoothed) curve has risen by `frac` of its total rise above its minimum; no knee when the slope near
    the top exceeds the minimum by less than acc_min (lift per unit level)."""
    r = _lift_slopes(cur, lv, top)
    if r is None:
        return None
    sl, l, i0, acc = r
    if acc < acc_min:
        return None
    s_top = sl[i0] + acc
    thr = sl[i0] + frac * (s_top - sl[i0])
    after = np.flatnonzero((np.arange(len(sl)) >= i0) & (sl >= thr))
    if after.size == 0 or l[after[0]] >= 0.975:
        return None
    return max(float(l[after[0]]), KNEE_MIN)


SOFT_DEFAULT_KNEE = 0.8   # knee when the first pass finds no shoulder (fraction of the ceiling): used below
                          # limiter ceilings by auto, and below every ceiling by --mode soft


def _loud_windows(y, refs, sr, win_s=10.0, total_s=60.0, use=None):
    """Non-overlapping windows (start, end) with the most samples near the ceilings, up to total_s."""
    T = y.shape[0]
    W = int(win_s * sr)
    if T <= int(1.25 * total_s * sr):
        return [(0, T)]
    near = np.zeros(T)
    for c in range(y.shape[1]):
        for pi, sg in enumerate((1.0, -1.0)):
            th = refs[c][pi]["theta"]
            if th is not None and (use is None or use[c][pi]):
                near += sg * y[:, c] >= 0.7 * th
    n = T // W
    score = near[: n * W].reshape(n, W).sum(1)
    pick = sorted(np.argsort(-score)[: int(total_s // win_s)])
    return [(int(i) * W, int(i + 1) * W) for i in pick]


MODES = ("auto", "hard", "soft", "limiter")


def auto_constraints(y, lsb, run_fast, mode="auto", sr=44100):
    """Analyze the ceilings and build the constraint set. run_fast(y, constraints) -> x runs a quick
    restoration (used for the soft-shoulder knee). Returns (m_hi, m_lo, th_hi, th_lo, report dict).

    mode 'auto':    clip plateaus and smeared ceilings are restored, plus the soft shoulder below them when
                    the first pass finds one (also without any ceiling, on stronger evidence). Below limiter
                    ceilings a soft region is always restored: from the knee the first pass finds, else from
                    0.8 x ceiling (moderate; the 0.5 knee of session 4 was heard to add distortion).
    mode 'hard':    ceilings only, no soft region (limiter ceilings: flat plateaus of >= 2 samples only).
    mode 'soft':    like auto, with a soft region below every ceiling (first-pass knee, else 0.8 x ceiling).
    mode 'limiter': experimental - like auto, with a soft region from 0.5 x ceiling below limiter ceilings
                    (raises SDR on synthetic limiter tests but reshapes cleanly limited peaks)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    refs = analyze_ceilings(y, lsb, sr)
    knees = base_knees(refs)
    kinds = sorted({r["kind"] for row in refs for r in row})
    has_ceiling = any(r["kind"] != "none" and r["theta"] is not None for row in refs for r in row)
    rep = dict(refs=refs, kinds=kinds, knee_rel=None, knee_found=False, acc=None, limiter_knee=None,
               acc_min=ACC_MIN_CEIL if has_ceiling else ACC_MIN_FREE, request=mode)
    if mode != "hard" and any(r["theta"] is not None for row in refs for r in row):
        k0 = [[RAISE_K0 * r["theta"] if r["theta"] else None for r in row] for row in refs]
        ys, xs = [], []
        for a, b in _loud_windows(y, refs, sr):   # long files: first pass on the loudest 60 s only
            yw = y[a:b]
            ys.append(yw)
            xs.append(run_fast(yw, ceiling_constraints(yw, refs, k0, lsb)))
        cur = raise_curve(np.concatenate(ys), np.concatenate(xs), refs)
        # without any ceiling the evidence must be stronger (unclipped music can bend the curve a little)
        kr = knee_from_raise(cur, acc_min=rep["acc_min"])
        rep.update(acc=lift_acceleration(cur), raise_curve=cur, knee_found=kr is not None, knee_rel=kr)
    kr = rep["knee_rel"]
    if mode == "soft" and kr is None:
        rep["knee_rel"] = kr = SOFT_DEFAULT_KNEE
    if "limiter" in kinds and mode != "hard":
        rep["limiter_knee"] = LIMITER_KNEE if mode == "limiter" else (kr or SOFT_DEFAULT_KNEE)

    def rel(r):  # knee (fraction of the ceiling) of one polarity, or None
        if r["kind"] == "limiter" and rep["limiter_knee"] is not None:
            return rep["limiter_knee"]
        return kr

    knees = [[(k if (r["theta"] is None or rel(r) is None) else
               rel(r) * r["theta"] if k is None else min(k, rel(r) * r["theta"]))
              for k, r in zip(krow, row)] for krow, row in zip(knees, refs)]
    rep["knees"] = knees
    main = [k for k in ("clip", "limiter", "smeared") if k in kinds]
    desc = "+".join(main or ["none"])
    if rep["limiter_knee"] is not None:
        desc += f", limiter knee {rep['limiter_knee']:.2f}" + (" (experimental)" if mode == "limiter" else "")
    if kr is not None and set(kinds) != {"limiter"}:
        desc += f", soft knee {kr:.2f}"
    rep["mode"] = desc
    m_hi, m_lo, th_hi, th_lo = ceiling_constraints(y, refs, knees, lsb)
    # safety cap below limiter ceilings: limiters rarely take more than 6-9 dB, and fully flagged
    # stretches (e.g. deliberately square-clipped bass under a limiter) would otherwise be extrapolated
    # far beyond any plausible original
    rep["gcap"] = None
    if "limiter" in kinds:
        g = np.full(y.shape, np.inf)
        for c in range(y.shape[1]):
            for pi, (m, sg) in enumerate(((m_hi, 1.0), (m_lo, -1.0))):
                if refs[c][pi]["kind"] == "limiter":
                    g[m[:, c], c] = 10 ** (LIMITER_MAX_GAIN_DB / 20)
        rep["gcap"] = g
    return m_hi, m_lo, th_hi, th_lo, rep
