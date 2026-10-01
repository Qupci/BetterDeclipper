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

## Session 4: mastered material (ex_master, 2026-09-28)
### User examples - signatures (research/master/analyze_inputs.py, plots/master_hist.png, flat_runs.py)
- soft_works (Rome Search, Tarzan): hard plateau (flat runs up to 50-150 samples) PLUS a soft shoulder
  below it (density rises from ~0.9 / ~0.7 x peak). Old auto saw the plateau -> hard -> shoulder untouched.
- bad (Bangarang, Goin' In x2, Purple Lamborghini, All Is Fair): spike at the ceiling but ceiling samples
  are isolated touches (flat-run fraction 0-8 %, no flat runs at any level) -> brickwall limiter last.
- weird (OWSLA 3 aac 264k, Mothership vorbis 192k, signal mp3 128k mono): plateau smeared by the codec,
  overshoots up to +1..+4.9 dBFS; old detection found nothing -> passthrough. signal has passages far above
  the ceiling that are smooth (EQ/filtering after clipping) -> no single ceiling.
- AL-1 (Naturl Audio limiter, hosted with pedalboard; default settings + ceiling -6 dB, detector Slow
  reproduce the user's render at 39 dB): look-ahead limiter, V-shaped gain dips of ~+-3 ms around each
  peak (0.5-0.6 at the peak). 96 % of the error energy within 1 ms of a ceiling touch (ex/petal), but half
  of it on samples below 0.8 x peak -> a level knee cannot isolate it. Dense material (Ruder Buster) also
  has slower gain riding (dynamics). Old auto: petal -6 dB AL-1 18.96 -> 18.98 dB (soft 19.66).
### Master benchmark (research/master/degrade.py, make_bench.py; research/outputs/mbench)
7 unclipped 30 s sources (ex GT, Deltarune: Petal Dance, Ruder Buster, Thrash Machine, Violet Tactics
@44.1k; BIG SHOT, Black Knife @48k) x 14 degradations: hard -6/-9, tanh soft clip (knee 0.6/0.4),
cubic soft clip, soft shoulder + hard ceiling (-6/-9), 4x oversampled hard clip, hard clip + resample
44.1->48k, hard clip + mp3 128k / aac 256k / vorbis 192k, AL-1 -6, 4x-OS clip -4 dB + AL-1 -7.
Fast preset, baseline sweep (mbench_base.json): old auto only wins on plain hard clipping; best fixed
soft knee (x peak) varies 0.5 (cubic) .. 0.9 (oversampled/resampled) -> must be estimated per file.
### Constraint family (auto.py)
flag |y| >= knee; restoration must satisfy |x| >= min(|y|, theta) (same sign). hard: knee = theta;
soft: knee < theta; smeared ceiling: knee = theta - c*sigma. Tolerance below |y| hurts (always best 0).
### Smeared-ceiling fit (detect.fit_ceiling)
histogram model [exp-linear natural density truncated at theta + clipped mass M at theta] * N(0, sigma^2),
Poisson ML. Recovers theta within 0.1-0.3 % (OS clip, resampling), ~1 % (vorbis); mp3/aac plateaus sit
~5 % below the original clip level. sigma/theta: dither 2-4e-4, OS/resampling 1.5-5e-3, codecs 0.008-0.06.
smear:c1 sweep (knee theta - c1 sigma, mbench_smear.json): c1 = 4-8 best everywhere, tolerance 0 best.
e.g. ex: oshard 25.40 -> 40.13 (hard-clip reference 40.87), resamp 39.31, mp3 23.23, aac 25.47, vorbis 31.19.
### Ceiling classification (auto.analyze_ceilings)
sharp ceiling + flat-run fraction >= 0.5 -> clip; sharp + isolated touches -> limiter; sigma/theta in
[1e-3, 0.07] with overshoot > 1.5 sigma -> smeared; else none. Correct on the benchmark (3/224 borderline
channel decisions) and on all user files; unclipped sources -> none.
### Soft-shoulder knee
- histogram quantile matching (compress_est.py): useless, natural-tail extrapolation errors of 20-170 %.
- histogram pile-up vs log-linear trend (shoulder_feat.py): misses mild shoulders (softhard6, tanh6, cubic).
- level-dependent gain cap (envelope, gcap per sample): makes a too-low knee much less harmful
  (hard6 @ knee 0.5: 23.8 -> 32.3 dB) but no single cap setting fits all curves -> not adopted.
- first-pass "raise curve": restore with knee 0.4 theta, median lift |x|/|y| - 1 per level. Uncompressed
  flagged samples are lifted by a level-dependent bias (5-14 % near the top, source dependent) but the
  curve only ACCELERATES towards the ceiling when there is real compression (0.8->0.95 increase: hard6
  -1.7..+0.6, OS clip -0.6..+0.7, softhard +2.8..+5.7, tanh6 +13..+17 points).
### Clicks in normal+ (research/fusion)
Per-band error on the example: normal beats fast below 1.5 kHz (0-150 Hz: -30.66 vs -28.20 dB) but is
worse above 8 kHz; SPADE alone has 887 HF error bursts (> 3x fast) and +4.5 dB error above 16 kHz (it adds
HF that is not in the source): jagged spikes inside long clipped gaps (plots/spade_clicks.png).
Two-band fusion (SPADE only below fc): ex 25.211 -> 25.226 (fc 4k) / 25.241 (8k); fast-HF + normal-LF
25.250 (8k). Frequency-weighted SPADE sparsity: SPADE alone worse (22.61 -> 22.16), fusion +0.04 only.
- knee rule chosen offline from the saved curves (eval_knee_rules.py): knee where the slope of the
  smoothed lift curve has risen by 10 % of its total rise above its minimum ("convexity onset"); no knee
  if the top slope exceeds the minimum by < 0.1/unit (ceiling found) or < 0.45/unit (no ceiling: an
  unclipped source reached 0.31, tanh >= 1.16). Knee >= 0.5. Mean distance to the best fixed knee on
  soft-type cases 0.3-1.0 dB; never fires on hard / OS / resampled clipping (accel <= 0.03).
  First pass on long files: loudest 6 x 10 s only.
- classification refinement: near-ceiling runs >= 4 samples (long) + flatness of in-run steps
  (<= 3 LSB / 1e-4 x level): long+flat = clip, long+rippled = smeared (OS clip), short = limiter.
  Fixed petal OS clip (clip+limiter 23.35 -> smeared 32.31 dB).
### Limiters (AL-1 cases, research/al1)
- gain shape around touches: V-shaped dips over ~+-3 ms, 0.5-0.6 at the peak (plots/al1_gain_shape.png).
- oracle per-event dip depth (triangular, tau 3 ms): petal 17.93 -> 28.48, ex 21.08 -> 26.61,
  bigshot 11.21 -> 19.92, petal OS+AL-1 8.77 -> 19.39 dB -> the model is expressive enough.
- blind: dip projector in the PnP loop (DipProj, one depth per event fitted to the NMF-Wiener output):
  only +0.1..+1.3 dB (NMF prior at 93 ms does not see 6 ms dips); proximity mask (|x| >= |y| within
  +-tau of touches): mean 16.55 (al1_6); plain level knee 0.5 x ceiling: 16.67 (input 15.08, all 7
  cases improved) -> limiter default knee 0.5. Slow limiter (AL-1 Gen 1 continuous, release 123 ms):
  every strategy +-0.02 dB -> the low knee does no harm when there is only gain riding.
### Two-band fusion validation (22 hard-clip cases: 48 kHz test set + mbench hard6/hard9)
normal (SPADE 0.35 below fc, 0 above): fc 3k +0.081, 4k +0.072, 6k +0.054, 8k +0.040 dB mean;
high (HF = NMF only): 4k +0.128 (min +0.011, never worse), 6k +0.088, 8k +0.061 -> fc = 4 kHz.
Example: normal 25.23, high 25.39, best 25.47 dB; HF error bursts vs fast: normal 17 -> 1, high/best 0;
error above 16 kHz normal -2.24 -> -4.28 dB.
### Final new auto (mbench_final.json, fast preset, 7 sources each)
| degradation | input | old auto | new auto | best fixed |
|---|---|---|---|---|
| hard6 / hard9 | 20.10 / 13.12 | 30.11 / 22.77 | 30.11 / 22.77 | 30.39 / 22.95 |
| softhard6 / softhard9 | 19.90 / 13.10 | 24.80 / 20.16 | 27.25 / 21.84 | 27.82 / 22.13 |
| tanh6 / tanh9k4 | 19.00 / 12.22 | 19.53 / 13.83 | 26.42 / 19.28 | 26.45 / 19.56 |
| cubic6 | 17.48 | 17.75 | 21.77 | 22.35 |
| oshard6 / resamp6 | 20.10 / 20.10 | 20.31 / 22.12 | 29.66 / 29.95 | 30.11 / 30.04 |
| mp3 / aac / vorbis | 16.72 / 17.57 / 19.43 | 17.08 / 19.35 / 21.45 | 19.71 / 21.75 / 24.83 | 19.90 / 22.15 / 24.92 |
| al1_6 / osal1 / al1slow | 15.08 / 7.56 / 4.54 | 15.09 / 7.57 / 4.54 | 16.64 / 8.79 / 4.56 | 17.01 / 9.62 / 4.56 |
| all 105 | 15.73 | 18.43 | 21.69 | 22.00 |
Never worse than the old auto on any case; unclipped sources untouched (6/7 bit-exact, 1 at 57 dB).
First pass: 80 NMF iterations give the same knees as 150 (40 flips some) -> 80.
Limiter cap: +9 dB above the observed value / ceiling under limiter ceilings (Purple Lamborghini has
deliberately square-clipped sub-bass under a limiter that was otherwise "restored" to +17.6 dBFS);
no change on the AL-1 benchmark (16.64 / 8.79).
User files (normal preset, research/outputs/master_demo): Rome clip+knee 0.88, Tarzan clip+knee 0.75,
All Is Fair / Bangarang / Goin' In x2 / Purple limiter (knee 0.5, ~20 % flagged), OWSLA smeared+knee 0.50,
Mothership smeared (3.3 %), signal none (no consistent ceiling), Petal AL-1 18.96 -> 20.23 dB,
Petal HARD 23.30 -> 34.19 dB (vs source).

## Session 5 (2026-09-30): faithful auto after the listening review
User review of session 4 (by ear): the smeared-ceiling and soft-knee handling is good (OWSLA 3 "on point",
Rome / Tarzan), but on All Is Fair, Bangarang, Goin' In x2, Purple Lamborghini and Petal AL-1 the auto
restoration damaged more than it restored: intended distortion reduced, undistorted samples "restored",
new audible distortion (kicks). SPACELLEX: odd restoration of the left channel's positive side only.
- Cause (master/report_inputs.py, results/report_inputs_v1.log): all of these were "limiter" -> 0.5 knee
  (19-23 % flagged). Their first-pass lift acceleration is -0.17..+0.06 (no saturation evidence), so the
  tanh/cubic soft-shoulder logic was not involved. SPACELLEX: only L+ passed the limiter hit-rate test
  (37 touches/s vs ~4/s) -> knee on L+ only. SDR gains from the limiter knee (AL-1 +1.5 dB) came from
  reshaping cleanly limited crests, which is audible as added distortion.
- Even freeing only the samples touching a limiter ceiling (knee = ceiling) lifted an isolated true-peak
  touch 0.999 -> 1.29 in SPACELLEX (a click). The touches of a clean limiter carry no clipping evidence.
- New rule: at limiter ceilings only flat runs of >= 2 samples (steps <= max(3 LSB, 1e-4 x ceiling)) are
  freed; no soft-shoulder search below limiter ceilings (also excluded from the first pass); the 0.5 knee
  is the experimental --mode limiter. Flagged now: Bangarang / Goin' In ~0.00-0.01 %, All Is Fair 0.12 %
  (real 2-3 sample clip plateaus after the limiter), SPACELLEX 0.01 %, Petal AL-1 0.0x %.
- CLI prints a per-channel/polarity table (kind, ceiling, blur, restored-from level, flagged %), what each
  kind means and the soft-shoulder decision with its evidence (lift acceleration vs threshold).
- scar tissue (PAD blind example): clip (long 0.7, flat 0.99), lift curve flat up to the plateau
  (acc -0.01) -> hard. --mode soft (0.8 knee) also lifts the limited crest below the clip (same kind of
  guess as the limiter knee) -> not in auto. PAD itself changes 87 % of all samples on its blind
  examples, so it is no reference for the knee.
- Forest (Zapper CD): rounded plateaus (clip followed by smoothing without overshoot, e.g. linear
  interpolation), no exact ceiling; lift acceleration 0.36 < 0.45 (no-ceiling threshold). Unclipped scan
  (master/acc_scan.py, results/acc_scan_v1.log): 40 Deltarune ch5 tracks -0.08..0.27 (Flower King 0.27),
  earlier excerpt 0.31, Forest fast 0.28 -> threshold kept. Tried and rejected as extra evidence (overlap
  with unclipped tracks): histogram shelf excess / cutoff sharpness (master/shelf_feat.py), within-run
  spread near the top (Forest 1.6-2.3 % vs unclipped 4-9 %, but correlated with cutoff sharpness), near-max
  event rate (Forest 0.02-0.9/s, unclipped up to 1.1/s). The printout reports acc 0.25-0.45 without a
  ceiling as a weak sign and suggests --mode soft.
- Zapper tracks credits / ghost_normal pass the no-ceiling threshold (acc 0.52 / 0.71, knee 0.85):
  0.08 / 0.29 % flagged (top samples only).
### Benchmark after session 5 (mbench_s5.json, fast preset, summary_s5.py)
Only limiter-related cases changed (all others within 0.005 dB of session 4):
| degradation | input | old auto | s4 auto | s5 auto | --mode limiter |
|---|---|---|---|---|---|
| al1_6 / osal1 / al1slow | 15.08 / 7.56 / 4.54 | 15.09 / 7.57 / 4.54 | 16.64 / 8.79 / 4.56 | 15.08 / 7.60 / 4.54 | 16.64 / 8.79 / 4.56 |
| oshard6 (ruder has one limiter polarity) | 20.10 | 20.31 | 29.66 | 29.58 | 29.66 |
| all 105 | 15.73 | 18.43 | 21.68 | 21.49 | |
s5 auto vs the pre-session-4 auto: never worse by more than 0.011 dB (knife al1_6 14.023 -> 14.012).
User files with the session-5 auto (normal preset, research/outputs/master_demo_s5, results/demo_s5.log):
All Is Fair 0.12 % flagged (max +3.3 dB), Bangarang / Goin' In x2 0.00 % (max +1.2..1.7 dB), Purple
Lamborghini 0.01 % (a dozen 70-87 sample flat plateaus per channel at 0:52 / 2:23 -> up to +6.2 dB),
Petal AL-1 0.02 % (+0.8 dB max), SPACELLEX 0.01 % (+0.2 dB max). RMS change <= 0.003 dB on all of them.
### A-SPADE transform reuse (user suggestion, research/speed/spade_cache.py)
The A(xn) computed for the residual at the end of an A-SPADE iteration is A(xa) of the next one (xa = xn):
carried over (re-indexed with the working set on compaction), residual computed in zb's storage
(zb.neg_().add_(A(xn)) == A(xn) - zb exactly) -> 2 transforms per iteration instead of 3, same peak memory
(661 MiB on the example). Bit-identical output on GPU and CPU. SPADE time (GTX 1660 Ti, min of 2 runs):
example 21.7 s 3.11 -> 2.71 s, Rome 120 s 9.48 -> 8.25 s, metallica 476 s 122.5 -> 106.1 s (-13 %);
CPU (8 s excerpt) 20.4 -> 17.8 s (-13 %). User measured -22 % on an RTX 4070 SUPER (185 s track).
Example SDR unchanged: normal 25.23 / high 25.39 / best 25.47 dB.
CLI: output is optional -> '<name> [<restoration> <preset>].wav' next to the input (restoration: 'auto
<kinds>[+soft]' / 'auto none', or the mode, with a forced level: 'hard -12dB', 'soft knee -9dB').
### Session 6 (2026-09-30): moderate limiter knee by default, packaging
User listened to --mode soft on the limiter masters (Bangarang etc.) and liked it ("accurate enough") ->
auto below limiter ceilings: knee from the first pass if found, else 0.8 x ceiling (= what --mode soft did
there; auto and soft are bit-identical on all-limiter files). 'none' polarities that reach a limiter
ceiling of the same file (max <= 1.001 x ceiling, >= MIN_EVENTS touches) join it (SPACELLEX: L+ was
limiter at 37 touches/s, the others ~4/s -> now all four, symmetric restoration). Benchmark
(mbench_s6.json, fast): al1_6 15.08 -> 15.52, osal1 7.60 -> 8.17, al1slow 4.54 = 4.54, oshard6 29.58 ->
29.86 (ruder's misclassified limiter polarity 26.26 -> 28.17); hard6/tanh6/mp3/softhard6 bit-identical;
no case below its input; 105-case mean 21.58 (s5 21.49, --mode limiter 0.5 knee 21.68).
Flower Man (Deltarune ch5, guest mastered by Camellia): all four polarities peak at -0.21 dBFS (+-0.01 dB)
= a limiter, but sample peaks touch it only 0.07-0.66 times/s (< 8/s) -> 'none', untouched by auto.
Packaging: pyproject.toml (setuptools, console script `betterdeclipper`, version 0.2.0 from __init__),
lazy `from betterdeclipper import declip` (torch loads on first use). Wheel = the 15 package modules only.
### Reusable analysis files, license (2026-10-01)
auto.py split into analyze() (ceilings + soft-shoulder search = the slow first pass; plain data with a
source fingerprint) and auto_constraints(..., analysis=) which applies any mode to it. CLI
--save-analysis / --load-analysis (JSON, editable), API declip(..., analysis=info['analysis']['data']).
Saved from --mode hard the search still runs (full_analysis), so one file serves auto/hard/soft/limiter.
Checks: reloaded runs bit-identical to fresh ones in all four modes (tarzan, 75 s: 13.7 -> 6.3 s with the
fast preset), via file and in memory; 15 benchmark cases bit-identical to mbench_s5/s6 after the
refactor; another input (album case) is reported as reused, channel mismatch is an error.
License: AGPL-3.0-only (LICENSE = gnu.org agpl-3.0.txt), version 0.3.0.
