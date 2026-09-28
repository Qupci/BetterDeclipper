import sys
sys.path.insert(0, "F:/BetterDeclipper/research/master")
import numpy as np, soundfile as sf
from compress_est import compression_curve
B = "F:/BetterDeclipper/research/outputs/mbench/"
LV = np.array([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])

def true_comp(deg, ys, c):
    """true x/y - 1 at output levels ys (absolute)"""
    if deg in ("tanh6", "tanh9k4", "softhard6", "softhard9"):
        knee, cs = {"tanh6": (0.6, 1.0), "tanh9k4": (0.4, 1.0), "softhard6": (0.6, 1.15), "softhard9": (0.7, 1.1)}[deg]
        t, cc = knee * c, cs * c
        z = np.clip((ys - t) / (cc - t), -0.999999, 0.999999)
        return np.where(ys > t, (t + (cc - t) * np.arctanh(z)) / ys - 1, 0.0)
    if deg == "cubic6":
        out = []
        for yv in ys:
            r = np.roots([-0.5 / 1.0, 0, 1.5, -yv / c])  # 1.5u - 0.5u^3 = y/c
            u = min([v.real for v in r if abs(v.imag) < 1e-9 and 0 <= v.real <= 1] or [1.0])
            out.append(u * 1.5 * c / yv - 1)
        return np.array(out)
    return np.zeros_like(ys)

quad = len(sys.argv) > 1 and sys.argv[1] == "quad"
for src in ["ex", "petal", "ruder", "bigshot"]:
    for deg, db in [("tanh6", -6), ("tanh9k4", -9), ("cubic6", -6), ("softhard6", -6), ("softhard9", -9), ("hard6", -6), ("gt", 0)]:
        fn = B + f"{src}__{'hard6__gt' if deg == 'gt' else deg}.wav"
        y, sr = sf.read(fn, dtype="float64", always_2d=True)
        gt, _ = sf.read(B + f"{src}__hard6__gt.wav", dtype="float64", always_2d=True)
        c = np.abs(gt).max() * 10 ** (db / 20)
        lv, comp, peak = compression_curve(y[:, 0][y[:, 0] > 0], levels=LV, quad=quad)
        tc = true_comp(deg, LV * peak, c)
        print(f"{src:7s} {deg:9s} est: " + " ".join(f"{v*100:6.1f}" for v in comp) + "   true: " + " ".join(f"{v*100:6.1f}" for v in tc))
