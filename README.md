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
| BetterDeclipper `--preset fast` | **24.17 dB** | 22.73 dB | 35 s | 1.5 s |
| BetterDeclipper `--preset normal` | **25.21 dB** | 23.78 dB | 142 s | 6.6 s |
| BetterDeclipper `--preset high` | **25.30 dB** | 23.87 dB | 181 s | 8.9 s |
| BetterDeclipper `--preset best` | **25.37 dB** | 23.94 dB | 313 s | 13.6 s |

Even the `fast` preset beats ProAudioDeclipper by 2.3 dB. GPU and CPU give the same quality (within 0.004 dB).

## Usage

Requires Python 3 with `numpy`, `scipy`, `soundfile` and `torch`. The CPU build of torch works;
an NVIDIA GPU with the CUDA build of torch is much faster (see [GPU](#gpu-acceleration)).
On this machine, `declip.bat in.wav out.wav` uses the GPU environment in `.venv` automatically.

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

| preset | models averaged | example (21.7 s): CPU / GPU |
|--------|-----------------|-----------------------------|
| fast   | NMF-PnP (150 it) | 35 s / 1.5 s |
| normal | NMF-PnP (weight 0.65) + stereo A-SPADE (0.35) | 142 s / 6.6 s |
| high   | NMF-PnP + PEW-PnP + stereo A-SPADE | 181 s / 8.9 s |
| best   | like `high` with twice the iterations | 313 s / 13.6 s |

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

Measured on a GTX 1660 Ti (6 GB) with an i5-2320 host. The example numbers are in the tables above.

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
- **Device-specific selection.** SPADE's k-largest selection uses `topk` on GPUs and `kthvalue` on
  CPUs (same result; each is faster on its device).

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
5. **Speed**. torch float32 FFTs, vectorized frames, slice-add overlap-add, in-place updates,
   and flush-to-zero for denormal floats. Without flush-to-zero, the NMF updates run 10x slower
   on older CPUs. An NVIDIA GPU runs the same code 20x faster (see above).
6. **Chunking**. Processing runs in 20 s chunks with 1.5 s of context and a short crossfade, so
   memory stays bounded. Chunks without clipping are copied through.

The research history, all experiments and their numbers are in `research/LOG.md`.
