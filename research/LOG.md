# BetterDeclipper research log

Goal: restore clipped audio as close as possible to the unclipped source (primary metric: SDR vs
ground truth), matching or beating ProAudioDeclipper (PAD). Offline processing is fine.

## Environment
- Python: `C:\Users\Qupci\AppData\Local\Programs\Python\Python311\python.exe` (numpy 1.26, scipy 1.12,
  torch 2.14 **CPU-only**, soundfile, numba). CPU i5-2320 (4 cores, AVX, no AVX2).
- torch float32 FFT is ~10x faster than numpy here -> all solvers use torch.
- PAD VST3 loads in pedalboard (`research/_vendor`) but acts as a pure passthrough when hosted
  headlessly (even gains are ignored) -> cannot benchmark PAD on new inputs automatically.

## Data
- `ex_sample/`: 21.7 s, 44.1 kHz stereo, clipped at -12.04 dBFS (8192 LSB), **26.8 % samples
  clipped** (runs up to 246 samples), clipped file dithered (+-2 LSB spread around 8192).
  - clipped SDR 10.47 dB; PAD output SDR **21.82 dB** (dSDR +11.35); on 5-10 s excerpt PAD = 23.32 dB.
- `testset/` (built by `make_testset.py`): 4 external 48 kHz tracks x {-12, -6} dBFS clip, 15 s,
  16-bit TPDF dither. -12 dB cases are severe (13-50 % clipped).

## Experiments (5-10 s excerpt of the example unless noted; PAD = 23.32 dB there)
| # | method | settings | SDR |
|---|--------|----------|-----|
| 1 | SS-PEW (FISTA, consistent) | win 1024/2048/4096/8192, hop win/4, neigh 3x7, 400 it | 19.35 / 21.96 / **22.89** / 22.51 |
| 2 | SS-PEW neighborhoods @4096 | 1x1 (EW) / 1x5 / 1x9 / 3x3 / 3x9 / 5x5 / 5x9 | 18.85 / 20.24 / 19.73 / 22.61 / 22.90 / 22.49 / 22.61 |
| 3 | + stereo coupling | none / joint-pool 0.5 / M-S / M-S+joint / PCA / per-bin rot / per-bin adaptive | 22.89 / 23.19 / 24.07 / 23.72 / **24.09** / 23.85 / 23.69 |
| 4 | lambda schedule (PCA) | lam0 {0.3,0.1,0.03} x lam1 {1e-3..3e-5} (rel. to max coef) | 23.79 .. 24.10 (flat) |
| 5 | union of frames | 2048+8192 / 1024+4096 / 2048+4096+8192 / 4096 hop/8 / 4096+16384 | 24.17 / 23.93 / 23.66 / 23.86 / 23.47 |
| 6 | A-SPADE / S-SPADE (no stereo) | 4096 win, red 2, s=r=1, eps 0.1 | 22.31 (294 s) / 19.60 |
| 7 | less shrinkage (alpha, beta) | alpha 2/4 diverge (FISTA + non-convex) | 4-10 (bad) |
| 8 | signal-domain PnP (x <- PEW(P(x))) | 4096, 3x7, PCA | **24.34** |
| 9 | PnP cycle spinning | 2 / 4 shifts | 24.18 / 24.19 (no gain) |
| 10 | PnP max-of-neighborhoods | 3x7+9x1 / +15x1 / +7x3 / 1x9+9x1 / +3x15 | 23.68 / 22.66 / 24.09 / 23.88 / 24.16 |
| 11 | + AR (Janssen) refinement | win 2048, p=256, 3 outer | 24.46 (+0.12, 236 s: not worth it) |

Findings:
- Error analysis: our error profile over frequency is nearly the same shape as PAD's (PAD likely also
  uses a TF-sparsity model); ~50 % of remaining error is < 500 Hz.
- Bias: inside clipped runs we under-estimate the excess over threshold by only 2-5 % (12 % on runs
  >128 samples); PAD over-estimates by 7-13 %. Our MSE is lower than PAD's for all run lengths except
  the longest. => remaining error is waveform *shape*, not amplitude bias.

## Full-file benchmarks (`bench_testset.py`, PEW 4096(93 ms)/hop 1/4, 3x7, PCA, 400 it)
| case | clip % | SDR in | ours | PAD |
|------|--------|--------|------|-----|
| example (full 21.7 s) | 26.8 | 10.47 | **23.44** | 21.82 |
| gris48 -12 / -6 | 28.9 / 3.7 | 8.60 / 20.13 | 20.43 / 34.58 | n/a |
| merlon48 -12 / -6 | 30.0 / 5.3 | 7.69 / 17.47 | 17.94 / 29.34 | n/a |
| lofi48 -12 / -6 | 13.4 / 1.4 | 9.09 / 17.12 | 17.82 / 22.87 | n/a |
| ghostpage48 -12 / -6 | 50.1 / 16.8 | 5.69 / 13.37 | 11.37 / 19.19 | n/a |
Runtime ~7x slower than real time on the i5-2320 (149 s for the 21.7 s example).
