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
| BetterDeclipper `--preset fast` | 23.46 dB | 22.03 dB | 0.6 min |
| BetterDeclipper `--preset normal` | **24.55 dB** | 23.12 dB | 2.3 min |
| BetterDeclipper `--preset best` | **25.00 dB** | 23.57 dB | 6.2 min |

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
- `--clip-level` forces a level when auto-detection finds nothing. For example, soft-clipped masters
  have no flat plateau at the top of the waveform.

Presets (the model average gets more diverse as you go down the list):

| preset | models averaged | relative time |
|--------|-----------------|---------------|
| fast   | PnP-PEW 93 ms (200 it) | 0.2x |
| normal | PnP-PEW 93 ms + stereo A-SPADE 93 ms | 1x |
| high   | PnP-PEW 93/46 ms + A-SPADE 93/186 ms | ~2x |
| best   | PnP-PEW 46/93/186 ms + A-SPADE 46/93/186 ms | ~2.7x |

## How it works

1. **Clip detection** (`detect.py`). Clipped samples form a dense plateau in the amplitude histogram.
   The plateau's lower edge becomes the clip level, separately per channel and polarity. This
   tolerates dither and requantization noise, which smears the plateau over a few LSBs.
2. **Consistency**. Unclipped samples are kept exactly. Clipped samples are only allowed to lie
   beyond the clip level, with the sign of the clipped sample.
3. **Restoration models**. Each one finds a consistent signal that is sparse in time-frequency:
   - *PnP-PEW* (`methods/pnp.py`): plug-and-play iterations `x <- PEW(P_consistent(x))` with
     "persistent empirical Wiener" social shrinkage (Siedenburg et al. 2014) in a Parseval STFT,
     FISTA momentum, and a geometrically annealed threshold.
   - *Stereo A-SPADE* (`methods/spade.py`): frame-wise hard-sparsity ADMM (Kitić et al. 2015,
     Záviška et al. 2018), vectorized over all frames, with the k largest coefficients selected
     jointly over both channels.
   - *Stereo coupling*: both models work on PCA-rotated channels (a mid/side-like basis), so
     unclipped samples in one channel inform the other. The minor component is regularized harder.
4. **Fusion** (`engine.py`). The averaged models are structurally different: PEW tends to
   slightly undershoot peaks, while SPADE overshoots (PAD behaves like SPADE). Their errors are only
   weakly correlated (~0.45), so averaging adds 1 to 1.4 dB. The average of consistent signals is
   still consistent.
5. **Chunking**. Processing runs in 20 s chunks with 1.5 s of context and a short crossfade, so
   memory stays bounded. Chunks without clipping are copied through.

The research history, all experiments and their numbers are in `research/LOG.md`.
