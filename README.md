# BetterDeclipper

Offline audio declipper that aims to restore clipped audio as closely as possible to the original
(unclipped) signal. It is not real-time and not a plugin: it trades CPU time for accuracy.

## Results

On the provided example (`ex_sample/`: 21.7 s, 44.1 kHz stereo, hard-clipped at -12 dBFS, so 26.8 %
of samples are clipped, then dithered to 16 bit). Score is SDR against the ground truth; higher is better.

| restoration | SDR (whole file) | SDR (clipped samples only) | time on i5-2320 |
|-------------|------------------|----------------------------|-----------------|
| clipped input | 10.47 dB | 9.04 dB | - |
| ProAudioDeclipper (provided output) | 21.82 dB | 20.41 dB | - |
| BetterDeclipper `--preset fast` | **24.17 dB** | 22.73 dB | 38 s |
| BetterDeclipper `--preset normal` | **25.21 dB** | 23.78 dB | 2.6 min |
| BetterDeclipper `--preset high` | **25.30 dB** | 23.87 dB | 3.3 min |

Even the `fast` preset (1.75x real time on a 2011 quad-core CPU) beats ProAudioDeclipper by 2.3 dB.

## Usage

Requires Python 3 with `numpy`, `scipy`, `soundfile` and `torch` (the CPU build is enough).

```
python -m betterdeclipper input.wav output.wav                 # auto-detect clip level, "normal" preset
python -m betterdeclipper input.wav output.wav --preset best   # slowest, most accurate
python -m betterdeclipper input.flac output.wav --clip-level -12   # force the clip level (dBFS)
python -m betterdeclipper in.wav out.wav --format pcm24 --normalize -0.1
```

- Output is 32-bit float by default: restored peaks can exceed the clip level (and even 0 dBFS
  when the input was clipped at full scale). For PCM output, use `--normalize` or `--gain`.
- Any sample rate works (window lengths are defined in milliseconds).
- `--clip-level` forces a hard-clip level (e.g. when auto-detection finds nothing).
- **Clipping modes** (`--mode`, default `auto`):
  - `hard`: a flat clipping plateau (digital clipping, possibly dithered or requantized). Restored
    samples must lie beyond the clip level.
  - `soft`: soft clipping or heavy limiting (e.g. loudness-war masters without a flat top). Above a
    knee, the original is assumed to be at least as large as the observed sample. The knee comes from
    the pile-up of the amplitude histogram, or `0.8 x peak` if there is none (`--knee` overrides it).
    This is experimental: on a synthetic tanh-saturated test it improved SDR from 24.1 to 34.0 dB.
  - `auto`: `hard` if a plateau is found, `soft` if only a histogram pile-up is found, otherwise
    the input is returned unchanged.
- `--max-gain DB` is an optional safety cap. Restored samples may exceed the clip level by at most
  DB decibels, and the cap is part of the constraints, so peaks stay smooth. It is off by default: in
  the example, the true peaks are 11.8 dB above the clip level.

Presets (each averages structurally different models):

| preset | models averaged | time on the example (21.7 s audio) |
|--------|-----------------|---------------|
| fast   | NMF-PnP (150 it) | 38 s |
| normal | NMF-PnP (weight 0.65) + stereo A-SPADE (0.35) | 158 s |
| high   | NMF-PnP + PEW-PnP + stereo A-SPADE | 199 s |
| best   | like `high` with twice the iterations | ~6 min |

## How it works

1. **Clip detection** (`detect.py`). Clipped samples form a dense plateau in the amplitude histogram.
   The plateau's lower edge becomes the clip level, separately per channel and polarity. This
   tolerates dither and requantization noise, which smears the plateau over a few LSBs. For
   soft clipping, a knee is detected where the amplitude density rises above its natural decay.
2. **Consistency**. Unclipped samples are kept exactly. Clipped samples are only allowed to lie
   beyond the clip level, with the sign of the clipped sample.
3. **Restoration models**. Each one finds a consistent signal that fits a prior of the time-frequency (TF) coefficients:
   - *NMF-PnP* (`methods/pnp.py`, strongest single model): plug-and-play iterations
     `x <- Wiener_V(P_consistent(x))`, where the Wiener gain `V/(V+lambda^2)` uses a low-rank
     non-negative matrix factorization `V = W H` of the current power spectrogram. The spectral
     templates `W` are shared by both channels and warm-started across iterations, and lambda is annealed.
     Repeating sounds (drum hits, notes) are explained by a few templates, and clipping distortion is not.
   - *PnP-PEW* (`methods/pnp.py`): plug-and-play iterations `x <- PEW(P_consistent(x))` with
     "persistent empirical Wiener" social shrinkage (Siedenburg et al. 2014) in a Parseval STFT,
     FISTA momentum, and a geometrically annealed threshold.
   - *Stereo A-SPADE* (`methods/spade.py`): frame-wise hard-sparsity ADMM (Kitić et al. 2015,
     Záviška et al. 2018), vectorized over all frames, with the k largest coefficients selected
     jointly over both channels.
   - *Stereo coupling*: both models work on PCA-rotated channels (a mid/side-like basis), so
     unclipped samples in one channel inform the other. The minor component is regularized harder.
4. **Fusion** (`engine.py`). The averaged models are structurally different: NMF/PEW tend to
   slightly undershoot peaks, while SPADE overshoots (PAD behaves like SPADE). Their errors are only
   weakly correlated (~0.5), so averaging adds up to about 1 dB. The average of consistent signals is
   still consistent.
5. **Speed**. torch float32 FFTs, vectorized frames, slice-add overlap-add, and flush-to-zero
   for denormal floats. Without flush-to-zero, the NMF updates run 10x slower on older CPUs.
6. **Chunking**. Processing runs in 20 s chunks with 1.5 s of context and a short crossfade, so
   memory stays bounded. Chunks without clipping are copied through.

The research history, all experiments and their numbers are in `research/LOG.md`.
