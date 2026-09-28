"""Check candidate ground-truth sources: peak, flat runs at peak (pre-existing clipping), loudest 30 s window."""
import sys, os
import numpy as np, soundfile as sf
for f in sys.argv[1:]:
    try:
        y, sr = sf.read(f, dtype="float64", always_2d=True)
    except Exception as ex:
        print(f, "ERR", ex); continue
    pk = np.abs(y).max()
    eq = np.abs(np.diff(y, axis=0)) <= 1e-9
    near = np.abs(y[1:]) > 0.98 * pk
    flat3 = int(np.sum(eq[1:] & eq[:-1] & near[1:]))
    # loudest 30 s window (1 s hop)
    p = (y ** 2).mean(1); n = len(p) // sr; e = p[: n * sr].reshape(n, sr).sum(1)
    w = np.convolve(e, np.ones(30), "valid")
    i = int(np.argmax(w)) if w.size else 0
    rms = np.sqrt(w[i] / (30 * sr)) if w.size else np.sqrt(np.mean(y ** 2))
    print(f"{os.path.basename(f)[:40]:40s} {sr} Hz {len(y)/sr:6.1f}s  peak {20*np.log10(pk):6.2f} dBFS  flat3@peak {flat3:6d}  loudest30s @{i:4d}s rms {20*np.log10(rms):6.2f} dBFS  crest {20*np.log10(pk/rms):5.2f} dB")
