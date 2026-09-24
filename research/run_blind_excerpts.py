"""Run BetterDeclipper on 30 s excerpts of the blind CD examples (for listening vs PAD)."""
import sys, warnings; sys.path.insert(0, "F:/BetterDeclipper")
import numpy as np, soundfile as sf
from betterdeclipper.engine import declip
B = "F:/BetterDeclipper/ProAudioDeclipper/blind_examples/"
O = "F:/BetterDeclipper/research/outputs/blind/"
preset = sys.argv[1] if len(sys.argv) > 1 else "fast"
for f, pf, start in [("greenday-cd.wav", "greenday-cd_declipped_by_proaudiodeclipper.wav", 60),
                     ("metallica-cd.wav", "metallica-cd_declipped_by_by_proaudiodeclipper.wav", 120),
                     ("scar_tissue-cd.wav", "scar_tissue-cd_declipped_by_by_proaudiodeclipper.wav", 40)]:
    y, sr = sf.read(B + f, dtype="float64", always_2d=True)
    p, _ = sf.read(B + pf, dtype="float64", always_2d=True)
    a, b = start * sr, (start + 30) * sr
    x, info = declip(y[a:b], sr, preset=preset, mode="auto")
    name = f.replace("-cd.wav", "")
    sf.write(O + f"{name}_30s_input.wav", y[a:b].astype(np.float32), sr, subtype="FLOAT")
    sf.write(O + f"{name}_30s_pad.wav", p[a:b].astype(np.float32), sr, subtype="FLOAT")
    sf.write(O + f"{name}_30s_betterdeclipper_{preset}.wav", x.astype(np.float32), sr, subtype="FLOAT")
    lv = [tuple(None if v is None else round(20 * np.log10(abs(v)), 2) for v in l) for l in info["levels"]]
    ch = np.abs(x - y[a:b]) > 2 / 32768
    print(f"{name:12s} mode {info['mode']}, levels(dBFS) {lv}, flagged {info['clipped_frac']*100:.2f}%, "
          f"changed {ch.mean()*100:.2f}%, peak in {20*np.log10(np.abs(y[a:b]).max()):.2f} -> ours {20*np.log10(np.abs(x).max()):.2f} / PAD {20*np.log10(np.abs(p[a:b]).max()):.2f} dBFS, {info['time']:.0f}s", flush=True)
