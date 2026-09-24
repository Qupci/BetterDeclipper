import numpy as np, soundfile as sf, glob, os
def stats(name, gt, th):
    m = np.abs(gt) > th
    both = np.mean(m[:, 0] & m[:, 1]); one = np.mean(m[:, 0] ^ m[:, 1])
    M = (gt[:, 0] + gt[:, 1]) / np.sqrt(2); S = (gt[:, 0] - gt[:, 1]) / np.sqrt(2)
    r = np.sum(S**2) / np.sum(M**2)
    # side energy at clipped positions
    mm = m[:, 0] | m[:, 1]
    print(f"{name:14s} clipped both {both*100:5.1f}%  only-one {one*100:5.1f}%  S/M energy {10*np.log10(r):6.1f} dB  corrLR {np.corrcoef(gt[:,0], gt[:,1])[0,1]:.3f}")
gt, sr = sf.read("F:/BetterDeclipper/ex_sample/sample_ground_truth.wav")
stats("example", gt, 0.25)
for d in sorted(glob.glob("F:/BetterDeclipper/testset/*/")):
    gt, sr = sf.read(d + "ground_truth.wav")
    for cdb in (-12, -6):
        stats(os.path.basename(d[:-1]) + f"{cdb}", gt, 10 ** (cdb / 20))
