# BetterDeclipper

Offline audio declipper that aims to restore clipped audio as closely as possible to the original
(unclipped) signal. It is not real-time and not a plugin: it trades CPU time for accuracy.

## Results

On the provided example (`ex_sample/`: 21.7 s, 44.1 kHz stereo, hard-clipped at -12 dBFS, so 26.8 %
of samples are clipped, then dithered to 16 bit). Score is SDR against the ground truth; higher is better.
The example audio and the ProAudioDeclipper files are not included in this repository.

| restoration | SDR (whole file) | SDR (clipped samples only) | CPU (i5-2320) | GPU (GTX 1660 Ti) |
|-------------|------------------|----------------------------|---------------|-------------------|
| clipped input | 10.47 dB | 9.04 dB | - | - |
| ProAudioDeclipper (provided output) | 21.82 dB | 20.41 dB | - | - |
| BetterDeclipper `--preset fast` | **24.16 dB** | 22.73 dB | 86 s | 5.4 s |
| BetterDeclipper `--preset normal` | **25.23 dB** | 23.80 dB | ~180 s | 13.0 s |
| BetterDeclipper `--preset high` | **25.39 dB** | 23.96 dB | ~220 s | 14.5 s |
| BetterDeclipper `--preset best` | **25.47 dB** | 24.04 dB | ~350 s | 18.2 s |

Even the `fast` preset beats ProAudioDeclipper by 2.3 dB. GPU and CPU give the same quality (within 0.004 dB).
Times include the automatic analysis (about 4 s on the GPU and 40 s on the CPU for this file; the CPU
times marked ~ are the measured restoration times plus that). `--clip-level`, `--knee` or `--mode legacy`
skip the analysis.

### Mastered material

Real masters are rarely clipped like the example: they are soft-clipped, clipped and then limited,
lossy-encoded or resampled. To measure this, 7 unclipped 30 s excerpts (Deltarune soundtrack pieces
and the example's source, 44.1 and 48 kHz) were processed in 15 mastering-style ways, and the
restorations were compared with the unprocessed originals (`research/master/`). Mean SDR over the 7
excerpts, `fast` preset:

| processing | input | old `auto` | `auto` | best fixed setting* |
|------------|-------|------------|--------|---------------------|
| hard clip -6 / -9 dB | 20.1 / 13.1 | 30.1 / 22.8 | **30.1 / 22.8** | 30.4 / 23.0 |
| soft shoulder + hard ceiling (2 kinds) | 19.9 / 13.1 | 24.8 / 20.2 | **27.2 / 21.9** | 27.8 / 22.1 |
| tanh saturation (2 kinds) | 19.0 / 12.2 | 19.5 / 13.8 | **26.4 / 19.2** | 26.5 / 19.6 |
| cubic saturation | 17.5 | 17.8 | **21.7** | 22.4 |
| 4x oversampled clipper | 20.1 | 20.3 | **29.6** | 30.1 |
| hard clip, then resampled to 48 kHz | 20.1 | 22.1 | **29.9** | 30.0 |
| hard clip, then MP3 128k / AAC 256k / Vorbis 192k | 16.7 / 17.6 / 19.4 | 17.1 / 19.4 / 21.5 | **19.8 / 21.8 / 24.8** | 19.9 / 22.2 / 24.9 |
| AL-1 limiter / OS clip + AL-1 / slow AL-1 | 15.1 / 7.6 / 4.5 | 15.1 / 7.6 / 4.5 | 15.1 / 7.6 / 4.5 | 17.0 / 9.6 / 4.6 |
| ... with `--mode limiter` (experimental) | | | 16.6 / 8.8 / 4.6 | |
| **all 105 cases** | 15.7 | 18.4 | **21.5** | 22.0 |

\* the best of all tried constraint settings for each case, chosen with the ground truth (not available
in practice). `auto` is never worse than the old one (by more than 0.01 dB). On the AL-1 limiter
renders it deliberately restores almost nothing: a limiter's gain riding leaves no clipping in the
waveform. Guessing it back (`--mode limiter`) scores 1-2 dB higher on these renders, but on real
limited masters it was heard to reshape clean peaks and add distortion. The gain dips could only be
undone properly with the right depth per limiter event (+5 to +10 dB), and no blind estimate of the
depths has been found yet (`research/LOG.md`).

## Usage

Requires Python 3 with `numpy`, `scipy`, `soundfile` and `torch`. The CPU build of torch works;
an NVIDIA GPU with the CUDA build of torch is much faster (see [GPU](#gpu-acceleration)).
On this machine, `declip.bat in.wav out.wav` uses the GPU environment in `.venv` automatically.

```
python -m betterdeclipper input.wav output.wav                 # automatic analysis, "normal" preset
python -m betterdeclipper input.wav                            # -> "input [auto clip+soft normal].wav"
python -m betterdeclipper input.wav output.wav --preset best   # slowest, most accurate
python -m betterdeclipper input.flac output.wav --clip-level -12   # force the clip level (dBFS)
python -m betterdeclipper in.wav out.wav --format pcm24 --normalize -0.1
```

- Without an output name, the result is written next to the input as
  `<name> [<restoration> <preset>].wav`. The restoration is what `auto` found (`auto clip`,
  `auto smeared+soft`, `auto limiter`, `auto none`, ...; `+soft` when a soft shoulder was restored), or
  the chosen mode (`hard`, `soft`, `limiter`, `legacy`), with the level when one is forced
  (`hard -12dB`, `soft knee -9dB`).
- Output is 32-bit float by default: restored peaks can exceed the clip level (and even 0 dBFS
  when the input was clipped at full scale). For PCM output, use `--normalize` or `--gain`.
- Any sample rate works (window lengths are defined in milliseconds).
- `--clip-level` forces a hard-clip level, and `--knee` forces a soft-clip knee (both skip the analysis).
- **Modes** (`--mode`, default `auto`). `auto` first analyzes how the master was clipped or limited,
  separately per channel and polarity, prints what it found, and restores only what shows signs of
  clipping damage:
  - **clip**: a flat plateau (digital clipping, possibly dithered). Restored samples must lie beyond it.
  - **smeared**: a blurred plateau with overshoots above it: clipped audio that was then lossy-encoded
    (MP3/AAC/Vorbis), resampled, or clipped by an oversampled clipper. Samples above
    `ceiling - 6 x blur` are restored, and overshoots are not forced upwards.
  - **limiter**: a ceiling that is only touched by isolated samples (brickwall / look-ahead limiter).
    Only flat runs of 2+ samples at the ceiling (clipping after the limiter) are restored. A limiter's
    gain riding is not waveform damage, and guessing it back reshapes cleanly limited peaks (audibly,
    e.g. on kicks), so the crest below a limiter ceiling is kept as it is.
  - **soft shoulder**: soft clipping or saturation below a clip or smeared ceiling, or without any
    ceiling (then on stronger evidence). A quick first pass measures how much the restoration lifts
    each level; real waveshaping makes that lift accelerate towards the ceiling, and the knee is placed
    where the acceleration starts. When there is no ceiling and the sign is only weak, the file is left
    as it is and the printout says so (`--mode soft` if it still sounds squashed).

  Every sample above the knee may only grow (`|x| >= |y|`, and `|x| >= ceiling` on the plateau);
  everything below it is kept exactly. `hard` uses only the ceilings (no soft region), `soft` always
  adds one (knee from the first pass, or 0.8 x ceiling, also below limiter ceilings), `legacy` is the
  previous detection.
- `--mode limiter` is **experimental**: `auto` plus a soft region from 0.5 x ceiling below limiter
  ceilings. It raises SDR on synthetic limiter tests (see the AL-1 row above), but on real limited
  masters it reshapes peaks the limiter had left clean, removes intended distortion and can add new
  distortion, so `auto` does not use it.
- `--max-gain DB` is an optional safety cap. Restored samples may exceed the clip level by at most
  DB decibels, and the cap is part of the constraints, so peaks stay smooth. It is off by default for
  clipping (in the example the true peaks are 11.8 dB above the clip level); under limiter ceilings a
  +9 dB cap is applied automatically.

Presets (each averages structurally different models):

| preset | models averaged (below 4 kHz; above 4 kHz only NMF-PnP) |
|--------|-----------------|
| fast   | NMF-PnP (150 it) |
| normal | NMF-PnP (weight 0.65) + stereo A-SPADE (0.35) |
| high   | NMF-PnP + PEW-PnP + stereo A-SPADE |
| best   | like `high` with twice the iterations |

SPADE and PEW improve the low and mid range, but inside long clipped gaps they add jagged
high-frequency errors that can be heard as clicks. Above 4 kHz only the NMF model is used: on 22
hard-clip test files this raised SDR (normal +0.07 dB, high +0.13 dB on average) and removed the clicks.

## GPU acceleration

All processing is PyTorch tensor code (FFTs, elementwise math, small matrix products), so it runs
on an NVIDIA GPU unchanged. `--device auto` (the default) uses the GPU when the installed torch
has CUDA support, otherwise the CPU. Force one with `--device cpu` or `--device cuda`.

One-time setup of a GPU environment (the torch download is ~2.5 GB):
```
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install numpy scipy soundfile
declip.bat input.wav output.wav
```
`cu126` builds support NVIDIA GPUs from the GTX 900 series (Maxwell) onward, with a recent driver.

Measured on a GTX 1660 Ti (6 GB) with an i5-2320 host, before the automatic analysis was added (it adds
a few seconds per song on the GPU). The example numbers are in the tables above.

| input | preset | CPU | GPU | GPU speed vs real time |
|-------|--------|-----|-----|------------------------|
| 174 s, 27 % clipped (worst case) | fast | ~4.7 min * | 13.6 s | 12.8x faster |
| | normal | ~19 min * | 49.7 s | 3.5x faster |
| | high | ~24 min * | 71.9 s | 2.4x faster |
| 15 s, 48 kHz, 29 % clipped | normal | 145 s | 10 s | 1.5x faster |

\* CPU times for the 174 s file are estimated as 8x the measured 21.7 s example time.

GPU memory use is under 2 GB. What makes it fast:
- **Everything stays on the GPU.** Each chunk is uploaded once, and all iterations run there.
- **CUDA graphs.** For the PnP/NMF models, one iteration is recorded once and replayed, so the
  ~60 small kernel launches per iteration don't bottleneck on a slower host CPU. If capture fails,
  the solver falls back to normal execution with identical results. Set `BD_CUDA_GRAPHS=0` to
  disable graphs.
- **SPADE runs over the whole file** in large frame batches sized to the free GPU memory, instead of
  once per 20 s chunk. The working set shrinks in steps of 64 frames, so cuFFT plans are reused.
- **SPADE reuses a transform.** The transform computed for the residual at the end of an A-SPADE
  iteration is the one the next iteration starts from, so it is carried over (2 FFTs per iteration
  instead of 3). Bit-identical output; SPADE ~13 % faster on a GTX 1660 Ti and on the CPU, ~22 % on
  an RTX 4070 SUPER (suggested by a user).
- **Device-specific selection.** SPADE's k-largest selection uses `topk` on GPUs and `kthvalue` on
  CPUs (same result; each is faster on its device).

## How it works

1. **Analysis** (`auto.py`, `detect.py`). The top of the amplitude histogram is fitted with a model of
   "natural density up to a ceiling + clipped mass at the ceiling, blurred by a width sigma". The fit gives
   the ceiling and how blurred it is (dither ~0.03 %, resampling/oversampling ~0.2-0.5 %, lossy codecs
   1-6 %). Runs of samples at the ceiling tell clipping (long flat runs), oversampled clipping (long
   rippled runs) and limiting (isolated touches) apart. A quick first restoration with a low knee
   measures the lift per level; it only accelerates towards the ceiling under real waveshaping, which
   places the soft-shoulder knee (not searched below limiter ceilings).
2. **Consistency**. Samples below the knee are kept exactly. Samples above it may only grow, with their
   sign: `|x| >= min(|y|, ceiling)`, so plateau samples lie beyond the ceiling and overshoots of lossy
   codecs are not forced upward. At limiter ceilings only flat runs of 2+ samples count as clipped.
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
   still consistent. The fusion is two-band (4 kHz crossover): SPADE/PEW only contribute below it.
5. **Speed**. torch float32 FFTs, vectorized frames, slice-add overlap-add, in-place updates,
   and flush-to-zero for denormal floats. Without flush-to-zero, the NMF updates run 10x slower
   on older CPUs. An NVIDIA GPU runs the same code 20x faster (see above).
6. **Chunking**. Processing runs in 20 s chunks with 1.5 s of context and a short crossfade, so
   memory stays bounded. Chunks without clipping are copied through.

The research history, all experiments and their numbers are in `research/LOG.md`.
