"""Analysis-only scan (no restoration written): ceiling kinds, first-pass lift acceleration and knee per file.
Used to measure how far unclipped material bends the lift curve (false-positive margin of the 'none' path).
usage: python acc_scan.py files..."""
import sys, os, time
sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.engine import declip
from betterdeclipper.detect import estimate_lsb
from betterdeclipper import auto as A

if __name__ == "__main__":
    for fn in sys.argv[1:]:
        y, sr = sf.read(fn, dtype="float64", always_2d=True)
        lsb = estimate_lsb(y)
        t0 = time.time()

        def run_fast(yy, cons):
            return declip(yy, sr, models=[("nmf", 93, dict(n_iter=80))], constraints=cons, verbose=False)[0]
        m_hi, m_lo, _, _, rep = A.auto_constraints(y, lsb, run_fast, mode="auto", sr=sr)
        kinds = "".join(r["kind"][0] for row in rep["refs"] for r in row)
        acc = rep.get("acc")
        print(f"{os.path.basename(fn)[:40]:40s} kinds {kinds}  acc {('nan' if acc is None else f'{acc:6.3f}')}"
              f" (min {rep['acc_min']:.2f})  knee {rep['knee_rel']}  flagged {(m_hi | m_lo).mean()*100:.2f}%"
              f"  {time.time()-t0:.0f}s", flush=True)
