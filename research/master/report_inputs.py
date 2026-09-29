"""Per-file analysis report of the user's examples: ceiling kind per channel/polarity with the statistics
behind it, the first-pass lift curve and the resulting knee (no full restoration).
usage: python report_inputs.py [name filters...]"""
import sys, os, glob, time
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.engine import declip
from betterdeclipper.detect import estimate_lsb
from betterdeclipper import auto as A

B = "F:/BetterDeclipper/"
FILES = sorted(glob.glob(B + "ex_master/soft_works/*.wav")) + sorted(glob.glob(B + "ex_master/blind/bad/*")) + \
    [B + "research/outputs/master/OWSLA 3.wav", B + "research/outputs/master/Mothership Teaser.wav"] + \
    sorted(glob.glob(B + "ex_master/AL-1/*.wav")) + sorted(glob.glob(B + "ex_strange/*.wav")) + \
    [B + "ProAudioDeclipper/blind_examples/" + f for f in ("scar_tissue-cd.wav", "greenday-cd.wav", "metallica-cd.wav")]


def lift_acc(cur, lv=A.RAISE_LV, top=0.95):
    """slope rise of the smoothed lift curve (same quantity knee_from_raise thresholds)"""
    ok = np.isfinite(cur) & (lv <= top + 1e-9)
    c = np.asarray(cur)[ok]; l = lv[ok]
    if c.size < 6:
        return float("nan")
    cs = np.convolve(np.pad(c, 1, mode="edge"), np.ones(3) / 3, "valid")
    sl = np.gradient(cs, l)
    i0 = int(np.argmin(sl[: max(2, len(sl) - 3)]))
    return float(sl[-2:].mean() - sl[i0])


def report(fn):
    y, sr = sf.read(fn, dtype="float64", always_2d=True)
    T, C = y.shape
    lsb = estimate_lsb(y)
    t0 = time.time()

    def run_fast(yy, cons):
        xx, _ = declip(yy, sr, models=[("nmf", 93, dict(n_iter=80))], constraints=cons, verbose=False)
        return xx
    m_hi, m_lo, th_hi, th_lo, rep = A.auto_constraints(y, lsb, run_fast, mode="auto", sr=sr)
    print(f"== {os.path.basename(fn)}  {sr} Hz {C} ch {T/sr:.0f} s lsb {lsb:.1e}  -> {rep['mode']}  "
          f"flagged {(m_hi | m_lo).mean()*100:.2f}%  ({time.time()-t0:.0f}s)")
    for c in range(C):
        for pi, pol in enumerate("+-"):
            r = rep["refs"][c][pi]
            s = (1 if pi == 0 else -1) * y[:, c]
            th = r["theta"]
            if th is None:
                print(f"   ch{c}{pol}: -")
                continue
            ev = A._touch_events(s, th - max(3 * r["sigma"], 1.5 * max(lsb, 1e-7)))
            k = rep["knees"][c][pi]
            print(f"   ch{c}{pol}: {r['kind']:8s} theta {20*np.log10(th):6.2f} dBFS  peak {20*np.log10(s.max()):6.2f}"
                  f"  sig/th {r['sigma']/th:.4f}  long {r.get('long', 0):.2f} flat {r.get('flat', 0):.2f}"
                  f"  bump {r.get('bump', 0):7.1f}  touches {ev} ({ev/(T/sr):.1f}/s)"
                  f"  knee {('-' if k is None else f'{20*np.log10(k/th):+.2f} dB')}"
                  f"  flagged {((m_hi if pi == 0 else m_lo)[:, c]).mean()*100:.2f}%")
    cur = rep.get("raise_curve")
    if cur is not None:
        print(f"   lift acc {lift_acc(cur):.3f}  knee_rel {rep['knee_rel']}  curve: " +
              " ".join("nan" if not np.isfinite(v) else f"{v:.3f}" for v in cur[::2]))


if __name__ == "__main__":
    only = sys.argv[1:]
    for fn in FILES:
        if only and not any(o.lower() in os.path.basename(fn).lower() for o in only):
            continue
        report(fn)
