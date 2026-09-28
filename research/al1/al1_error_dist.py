"""Where is the AL-1 error? By output level and by distance to the nearest ceiling touch."""
import numpy as np, soundfile as sf
B = "F:/BetterDeclipper/research/outputs/mbench/"
for key in ["ex__al1_6", "petal__al1_6", "ruder__al1_6", "ex__osal1"]:
    y, sr = sf.read(B + key + ".wav", dtype="float64", always_2d=True)
    gt, _ = sf.read(B + key + "__gt.wav", dtype="float64", always_2d=True)
    e = (y - gt) ** 2
    pk = np.quantile(np.abs(y), 1 - 1e-4)
    a = np.abs(y)
    tot = e.sum()
    print(f"== {key}: input SDR {10*np.log10((gt**2).sum()/tot):.2f}")
    print("  |y|/peak bins: " + "  ".join(f"[{lo:.1f},{hi:.1f}) {100*e[(a>=lo*pk)&(a<hi*pk)].sum()/tot:5.1f}% g{np.median(np.abs(gt[(a>=lo*pk)&(a<hi*pk)])/np.maximum(a[(a>=lo*pk)&(a<hi*pk)],1e-9)):.3f}"
                              for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, .85), (.85, .95), (.95, 1.01)]))
    # distance to nearest ceiling touch (|y| >= 0.97 pk) in ms
    touch = (a >= 0.97 * pk).any(1)
    idx = np.flatnonzero(touch)
    pos = np.arange(len(y))
    j = np.searchsorted(idx, pos)
    d = np.minimum(np.abs(pos - idx[np.clip(j - 1, 0, len(idx) - 1)]), np.abs(idx[np.clip(j, 0, len(idx) - 1)] - pos)) / sr * 1000
    es = e.sum(1)
    print("  dist to touch: " + "  ".join(f"<{hi}ms {100*es[d<hi].sum()/tot:5.1f}% (time {100*(d<hi).mean():4.1f}%)" for hi in [0.5, 1, 2, 5, 10, 20, 50, 100]))
