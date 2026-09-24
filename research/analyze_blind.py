import sys; sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.detect import detect_clip_levels
B = "F:/BetterDeclipper/ProAudioDeclipper/blind_examples/"
pairs = [("greenday-cd.wav", "greenday-cd_declipped_by_proaudiodeclipper.wav"),
         ("metallica-cd.wav", "metallica-cd_declipped_by_by_proaudiodeclipper.wav"),
         ("scar_tissue-cd.wav", "scar_tissue-cd_declipped_by_by_proaudiodeclipper.wav")]
for a, b in pairs:
    x, sr = sf.read(B + a, dtype="float64", always_2d=True)
    y, _ = sf.read(B + b, dtype="float64", always_2d=True)
    q = np.round(x * 32768).astype(int)
    print(f"== {a}: peak {x.max(0)} {x.min(0)} rms {20*np.log10(np.sqrt(np.mean(x**2))):.2f} dBFS")
    for c in range(2):
        v, n = np.unique(q[:, c], return_counts=True)
        top = np.argsort(v)[-6:]; bot = np.argsort(v)[:6]
        print(f"  ch{c} top values/counts: {list(zip(v[top], n[top]))}")
        print(f"  ch{c} bottom values/counts: {list(zip(v[bot], n[bot]))}")
    lv = detect_clip_levels(x, 1 / 32768)
    print("  detected:", [(None if p is None else round(p * 32768), None if m is None else round(m * 32768)) for p, m in lv])
    d = y - x
    changed = np.abs(d) > 2 / 32768
    print(f"  PAD: out peak {np.abs(y).max():.3f}, changed samples {changed.mean()*100:.2f}%  min|x| where changed {np.abs(x[changed]).min() if changed.any() else 0:.3f}")
    # distribution of |x| where PAD changed samples
    if changed.any():
        print("   |x| percentiles at changed samples (5,25,50):", np.percentile(np.abs(x[changed]), [5, 25, 50]).round(3))
    # PAD applies gain? compare on unchanged
    print("   rms diff on unchanged samples (LSB):", np.sqrt(np.mean((d[~changed] * 32768) ** 2)))
