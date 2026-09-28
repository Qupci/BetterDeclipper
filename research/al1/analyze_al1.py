"""Characterize the user's AL-1 -6 dB master of Petal Dance against the source (and the HARD clip)."""
import numpy as np, soundfile as sf
from scipy import signal

src, sr = sf.read(r'F:/deltarune/ch5/17 - Petal Dance.flac', dtype='float64', always_2d=True)
al, _ = sf.read('F:/BetterDeclipper/ex_master/AL-1/17 - Petal Dance -6dB AL-1.wav', dtype='float64', always_2d=True)
hd, _ = sf.read('F:/BetterDeclipper/ex_master/AL-1/17 - Petal Dance -6dB HARD.wav', dtype='float64', always_2d=True)
th = 10 ** (-6 / 20)
x, a, h = src[:, 0], al[:, 0], hd[:, 0]
print('clip level', th, 'hard check max|h - clip(x)|', np.abs(h - np.clip(x, -th, th)).max())
# where is AL-1 different from source?
d = a - x
big = np.abs(x) > th
print('frac |x|>th %.4f' % big.mean())
# sample-wise transfer: bins of |x| -> mean |a| and gain
for lo, hi in [(0, .05), (.05, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .45), (.45, .5), (.5, .55), (.55, .6), (.6, .7), (.7, .8), (.8, 1.0)]:
    m = (np.abs(x) >= lo) & (np.abs(x) < hi)
    if m.sum() < 10:
        continue
    r = a[m] / x[m]
    print(f'|x| in [{lo:.2f},{hi:.2f}): n={m.sum():8d}  median gain a/x {np.median(r):.4f}  p5 {np.percentile(r,5):.4f} p95 {np.percentile(r,95):.4f}  max|a| {np.abs(a[m]).max():.4f}')
# gain envelope over time: ratio of |a| to |x| smoothed in 5 ms windows, on samples with |x| > 0.05
w = int(0.005 * sr)
num = np.convolve(np.abs(a * x), np.ones(w), 'same'); den = np.convolve(x * x, np.ones(w), 'same')
g = num / np.maximum(den, 1e-12)
act = den / w > 0.01 ** 2
print('gain env: frac < 0.99: %.4f, < 0.95: %.4f, < 0.9: %.4f (over active)' % ((g[act] < .99).mean(), (g[act] < .95).mean(), (g[act] < .9).mean()))
# spectra of the differences
f, Pa = signal.welch(a - x, sr, nperseg=4096)
_, Ph = signal.welch(h - x, sr, nperseg=4096)
_, Px = signal.welch(x, sr, nperseg=4096)
print('band      err AL-1 (dB rel x)   err HARD (dB rel x)')
for lo, hi in [(0, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 6000), (6000, 12000), (12000, 22050)]:
    m = (f >= lo) & (f < hi)
    print(f'{lo:5d}-{hi:5d} Hz: {10*np.log10(Pa[m].sum()/Px[m].sum()):8.2f}  {10*np.log10(Ph[m].sum()/Px[m].sum()):8.2f}')
# peak statistics
print('AL-1 peak %.5f, frac |a| > 0.999*th: %.5f, > 0.99 th %.5f, > 0.95 th %.5f' % (np.abs(a).max(), (np.abs(a) > .999*th).mean(), (np.abs(a) > .99*th).mean(), (np.abs(a) > .95*th).mean()))
# find a strongly clipped region and print a sample excerpt
i = np.argmax(np.abs(x))
sl = slice(i - 30, i + 30)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
print('around the largest source peak (x, AL-1, HARD):')
print(np.c_[x[sl], a[sl], h[sl]].T)
