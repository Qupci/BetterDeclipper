"""Build a small ground-truth test set from a few external tracks (48 kHz focus).

For each track: pick the loudest ~15 s window that contains no pre-existing clipping,
normalize its peak to -0.2 dBFS, then hard-clip at -12 dBFS and -6 dBFS (like the provided
example), dither to 16-bit (TPDF), and store ground truth (float32) + clipped (PCM_16).
"""
import os, sys, json
import numpy as np, soundfile as sf

OUT = "F:/BetterDeclipper/testset"
TRACKS = {
    "gris48": "D:/MUSIC/SORTED/Albums/Berlinist/2018 Gris (Original Soundtrack)/02 Gris pt. 1.flac",
    "merlon48": "D:/MUSIC/SORTED/Albums/Shane Mesa/2021 Paper Dream/06 Merlon's Myth.flac",
    "ghostpage48": "D:/MUSIC/SORTED/Albums/Red Vox/2016 What Could Go Wrong/11 Ghost Page.flac",
    "lofi48": "D:/MUSIC/SORTED/Albums/Dj Cutman, James Landino, GameChops/2018 The Legend of LoFi/01 The Legend of LoFi.flac",
}
SEG_S = 15.0
CLIP_DB = [-12.0, -6.0]


def has_clipping(x, frac=0.999, run=3):
    """True if any run of >= `run` consecutive samples sits at >= frac * peak (flat tops)."""
    pk = np.abs(x).max()
    m = np.abs(x) >= frac * pk
    for c in range(x.shape[1]):
        mc = m[:, c].astype(np.int8)
        d = np.diff(np.concatenate([[0], mc, [0]]))
        s, e = np.where(d == 1)[0], np.where(d == -1)[0]
        if np.any(e - s >= run):
            return True
    return False


def pick_segment(x, sr):
    L = int(SEG_S * sr)
    hop = sr  # 1 s steps
    best, best_rms = None, -1
    for a in range(0, x.shape[0] - L, hop):
        seg = x[a:a + L]
        rms = np.sqrt(np.mean(seg ** 2))
        if rms > best_rms and not has_clipping(seg):
            best, best_rms = a, rms
    return best


def tpdf_dither_16(x, rng):
    q = 32768.0
    d = (rng.random(x.shape) - rng.random(x.shape))
    return np.clip(np.round(x * q + d), -32768, 32767) / q


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    meta = {}
    for name, path in TRACKS.items():
        x, sr = sf.read(path, dtype="float64", always_2d=True)
        a = pick_segment(x, sr)
        if a is None:
            print(name, "no clean segment found"); continue
        seg = x[a:a + int(SEG_S * sr)]
        seg = seg * (10 ** (-0.2 / 20) / np.abs(seg).max())
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        sf.write(os.path.join(d, "ground_truth.wav"), seg.astype(np.float32), sr, subtype="FLOAT")
        info = dict(src=path, start_s=a / sr, sr=sr, rms_db=20 * np.log10(np.sqrt(np.mean(seg ** 2))))
        for cdb in CLIP_DB:
            th = 10 ** (cdb / 20)
            y = tpdf_dither_16(np.clip(seg, -th, th), rng)
            sf.write(os.path.join(d, f"clipped_{int(cdb)}dB.wav"), y, sr, subtype="PCM_16")
            frac = np.mean(np.abs(seg) > th)
            sdr_in = 10 * np.log10(np.sum(seg ** 2) / np.sum((seg - y) ** 2))
            info[f"clip_{int(cdb)}dB"] = dict(frac_clipped=float(frac), sdr_in=float(sdr_in))
        meta[name] = info
        print(name, json.dumps(info))
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
