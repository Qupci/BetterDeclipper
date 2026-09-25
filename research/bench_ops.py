import sys, time, torch
dev = sys.argv[1]
N, M = 955, 2 * 8193
c = torch.randn(N, M, dtype=torch.complex64, device=dev); mag = torch.rand(N, M, device=dev)
def sync():
    if dev == "cuda": torch.cuda.synchronize()
def tm(name, f, n=20):
    for _ in range(3): r = f()
    sync(); t = time.time()
    for _ in range(n): r = f()
    sync(); print(f"{dev:4s} {name:34s} {(time.time()-t)/n*1000:8.3f} ms"); return r
a = tm("abs2: real**2 + imag**2", lambda: c.real ** 2 + c.imag ** 2)
b = tm("abs2: view_as_real.square.sum", lambda: torch.view_as_real(c).square().sum(-1))
print("     identical:", torch.equal(a, b))
for k in (500, 2800, 6000):
    t1 = tm(f"kthvalue k={k}", lambda: torch.kthvalue(mag, M - k + 1, dim=-1, keepdim=True)[0])
    t2 = tm(f"topk.min k={k}", lambda: torch.topk(mag, k, dim=-1, sorted=False)[0].min(-1, keepdim=True)[0])
    print("     identical:", torch.equal(t1, t2))
