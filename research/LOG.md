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

## Session 2 findings (model fusion, SPADE tuning, NMF, speed)
| # | experiment (5-10 s excerpt; PAD = 23.32) | result |
|---|------------------------------------------|--------|
| 12 | PnP-PEW 4096 + side lambda gain [1, 2.5] (PCA minor comp.) | 24.48 |
| 13 | PnP iterations 100/200/300/400/600/1000 | 23.59/24.12/24.30/24.48/24.56/24.58 |
| 14 | ensemble avg of PnP 2048+4096+8192 (error corr 0.63-0.82) | 24.76 |
| 15 | ours (PnP) + PAD average (error corr only 0.35!) | 25.58 -> model diversity matters |
| 16 | A-SPADE 4096 s=8 (per channel) + PnP average | SPADE 23.14, fused 25.36 |
| 17 | stereo joint A-SPADE (PCA, joint top-k over channels) | 23.62 alone, fused 25.45 |
| 18 | SPADE eps (abs., signal normalized to clip level) 0.1/0.3/1/3/6/12 | 23.62/23.66/23.86/**24.08**/23.33/18.83; eps 3 fused 25.54, 12 s (was 62 s) |
| 19 | SPADE speed: single kthvalue for all active frames (k grows in lockstep) + drop converged frames | 2.2x faster, identical output |
| 20 | 6-model avg (PnP 2048/4096/8192 + SPADE 2048/4096/8192) | 25.85 (oracle LS weights 25.91) |
| 21 | error correlation PAD vs our SPADE 0.57-0.62, vs PnP 0.26-0.36 -> PAD is SPADE-like; ens6+PAD only 26.0 | |
| 22 | **NMF-Wiener PnP** (denoiser gain V/(V+lam^2), V = low-rank KL-NMF of iterate's power spectrogram, templates shared over channels, warm-started, 1 MU step / iteration) | rank 16/32/64/128/256: 23.88/24.67/24.79/**25.46**/25.04 |
| 23 | NMF + PnP-PEW + SPADE average | 26.04 |
| 24 | **denormal floats**: MU updates produce denormals; `torch.set_flush_denormal(True)` makes NMF-PnP 10x faster (165 s -> 16 s) | |
| 25 | overlap-add via slice adds instead of F.fold: 9x faster synthesis | |

Full example (21.7 s) single models via engine (20 s chunks): NMF rank 128/256/512 = 24.58/24.66/24.72 dB
(96/131/200 s), PnP-PEW = 23.80 dB (64 s).

## Presets on the full example (NMF rank = min(128, 0.3 x rows))
| preset | models | SDR | clipped-samples SDR | time |
|--------|--------|-----|---------------------|------|
| fast   | NMF 93 ms, 150 it | 24.17 | 22.73 | 38 s |
| normal | NMF + SPADE | 24.93 | 23.50 | 140 s |
| high   | NMF + PEW + SPADE | 25.30 | 23.87 | 199 s |
| (old best) | NMF93+NMF46+PEW+SPADE93+SPADE186 | 25.30 | 23.87 | 368 s -> no gain on full file |
Uncapped rank (0.3 x rows ~ 564) was slower and not better in ensembles (fast 24.29/78 s, normal 24.89/238 s, high 25.24/299 s).
NMF lambda schedule variants (lam1 1e-3/1e-5, lam0 0.03, chan_gain) -> no gain over defaults.

## Real-world material (blind CD examples) - histogram signatures (research/plots/blind_hist.png)
- greenday: bell-shaped plateau ~23 LSB below max (+-6 LSB) + shoulder: hard clip, then processed.
- scar tissue: sharp plateau spikes, polarity-dependent levels.
- metallica: NO top plateau; broad density bump at 0.7-0.85 x peak = soft clipping / heavy limiting
  (knee ~0.62 x peak). Needs a soft-clip model (constraint |x| >= |y| above a knee).

## Other inputs: 48 kHz test set with the `normal` preset (NMF + SPADE) vs first PEW-only version
| case | SDR in | PEW v1 | normal v3 |
|------|--------|--------|-----------|
| gris48 -12 / -6 | 8.60 / 20.13 | 20.43 / 34.58 | **22.79 / 35.41** |
| merlon48 -12 / -6 | 7.69 / 17.47 | 17.94 / 29.34 | **19.93 / 30.08** |
| lofi48 -12 / -6 | 9.09 / 17.12 | 17.82 / 22.87 | **19.31 / 26.19** |
| ghostpage48 -12 / -6 | 5.69 / 13.37 | 11.37 / 19.19 | **13.19 / 19.53** |
Mean dSDR over the 8 cases: 9.30 -> 10.91 dB. ~80-210 s per 15 s case.

## Soft clipping (synthetic: example GT through tanh soft clipper, knee 0.3, ceiling 0.5, 16-bit)
- histogram pile-up detection does NOT fire for gentle tanh saturation (density keeps decaying);
  it fires for heavy limiting (metallica: knee 0.68-0.78 x peak).
- forced soft mode (constraint |x| >= |y| above knee), fast preset, input 24.07 dB:
  knee 0.30/0.35/0.40/0.43/0.46/0.48 -> 29.51/33.48/**33.98**/30.85/27.09/24.99 dB
  => best declared knee ~0.75-0.8 x peak (above the true knee); used as default for --mode soft.

## Blind CD excerpts (30 s, fast preset; research/outputs/blind/)
- greenday: hard, -8.32 dBFS, 0.46 % flagged; restored peak +4.6 dBFS on a snare onset (short runs
  on a ~4 kHz oscillation; PAD also predicts a big peak there but limits it to 0 dBFS and lowers
  surrounding unclipped samples via its limiter).
- metallica: soft (knee ~-12.8/-11.7 dBFS, peak -9.6), 19.9 % flagged.
- scar tissue: hard, -6.12 dBFS, 1.1 % flagged.
- `--max-gain` cap on the excerpt (normal preset): none 25.85, +12 dB 25.76, +9 dB 24.96 dB.

## Fusion weights
Optimal NMF/SPADE weight (5 s excerpts): example 0.65 (+0.11 vs equal), merlon48-12 0.75 (+0.29 @0.65),
lofi48-12 0.45 (-0.18 @0.65), ghostpage48-6 0.75 (+0.67 @0.65) -> mean +0.22 dB with 0.65/0.35.
3-way (0.45/0.2/0.35) only +0.04 mean -> `high` keeps equal weights.
NMF + local PEW energy hybrid (nmf_smooth 0.3/0.6): 25.37/25.14 vs 25.46 pure NMF -> no gain.
**normal preset with 0.65/0.35: 25.21 dB on the full example** (was 24.93), 158 s.
Robustness: mono asym float, 3 channels, 24-bit container, clean passthrough all OK.
best (2x iterations): 25.37 dB, clipped-samples 23.94 dB, 387 s

## GPU acceleration (GTX 1660 Ti 6 GB, torch 2.14.0+cu126 in project .venv)
- Everything is torch already; added device plumbing (engine/CLI `--device auto|cpu|cuda`).
- Exact CPU speedups found on the way (bit-identical outputs): reuse NMF H@W.T between iterations,
  in-place ops, 2x2 channel mixing as elementwise ops instead of einsum (einsum -> bmm with inner
  dim 2 was the largest GPU kernel and slow on CPU too), SPADE compacted working set.
  CPU example: fast 38 -> 35 s, normal 158 -> 142 s, high 199 -> 181 s, best 387 -> 313 s.
- GPU profiling: small inputs are launch-bound (~60 kernels/iteration on a slow host CPU); 20 s
  chunks are GPU-bound (NMF ~7.5 ms/iteration: ~5.5 ms elementwise passes, 1.3 ms GEMM, 0.55 ms FFT).
- CUDA graphs for the PnP/NMF loop (static buffers, lambda^2 / momentum as device scalars, eager
  fallback): bit-identical to eager GPU; 2x on 5 s inputs, ~0 on 21.7 s (GPU-bound).
- SPADE: whole file in frame batches sized from free memory, working set shrinks in steps of 64
  rows (cuFFT plan reuse), topk on GPU / kthvalue on CPU (identical threshold, faster per device).
  Output via OLA of corrections: float32-level differences only (1.2e-7).
- Results (SDR identical within 0.004 dB):
  | preset | example CPU | example GPU | 174 s file GPU |
  |--------|-------------|-------------|----------------|
  | fast   | 35 s  | 1.5 s  | 13.6 s |
  | normal | 142 s | 6.6 s  | 49.7 s |
  | high   | 181 s | 8.9 s  | 71.9 s |
  | best   | 313 s | 13.6 s | - |
