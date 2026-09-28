"""Look for flat runs (>= n equal consecutive samples near local extrema) at any level: hidden clip plateaus
that a later gain stage/limiter moved off the file's peak value."""
import sys, glob, os
import numpy as np, soundfile as sf

FILES = sys.argv[1:] or (sorted(glob.glob("F:/BetterDeclipper/ex_master/soft_works/*.wav")) + sorted(glob.glob("F:/BetterDeclipper/ex_master/blind/bad/*")) +
                         ["F:/BetterDeclipper/ex_master/AL-1/17 - Petal Dance -6dB AL-1.wav", "F:/BetterDeclipper/ex_master/AL-1/17 - Petal Dance -6dB HARD.wav",
                          r"F:/deltarune/ch5/17 - Petal Dance.flac"])
for f in FILES:
    y, sr = sf.read(f, dtype="float64", always_2d=True)
    tot = {}
    lv = []
    for c in range(y.shape[1]):
        s = y[:, c]
        pk = np.abs(s).max()
        eq = np.concatenate([[False], np.abs(np.diff(s)) <= 1e-9])  # equal to previous sample
        # run starts/ends of equal-value runs (run length = number of equal diffs + 1)
        m = np.concatenate([[0], eq[1:].astype(np.int8), [0]])
        d = np.diff(m)
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)  # diff-index runs
        L = en - st + 1
        val = np.abs(s[st])
        for n in (3, 5, 8):
            sel = (L >= n) & (val > 0.3 * pk)
            tot[n] = tot.get(n, 0) + int(sel.sum())
        sel = (L >= 3) & (val > 0.3 * pk)
        lv.append(val[sel] / pk)
    lv = np.concatenate(lv)
    q = np.percentile(lv, [5, 25, 50, 75, 95]) if lv.size else []
    print(f"{os.path.basename(f)[:45]:45s} flat runs >=3: {tot[3]:6d}  >=5: {tot[5]:6d}  >=8: {tot[8]:6d} per minute(>=3): {tot[3]/(len(y)/sr/60):7.0f}"
          f"  level/peak pct 5/25/50/75/95: {np.round(q, 3)}")
