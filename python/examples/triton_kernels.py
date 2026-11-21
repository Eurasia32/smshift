"""Triton kernels (TMA + wgmma on sm_90) that follow the smshift protocol:
pf_gemm is a persistent GEMM that claims tiles from a global counter and exits
when its SM is no longer allowed."""
import triton
import triton.language as tl


@triton.jit
def _smid():
    return tl.inline_asm_elementwise("mov.u32 $0, %smid;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _claim(allowed_ptr, next_ptr, num_tiles):
    # Atomics on scalars give one value for the whole CTA, so the decision is
    # uniform even if the host rewrites the bitmap concurrently.
    sm = _smid()
    w = tl.atomic_or(allowed_ptr + sm // 32, 0)
    t = num_tiles
    if ((w >> (sm % 32)) & 1) != 0:
        t = tl.atomic_add(next_ptr, 1)
    return t


@triton.jit
def _nanosleep(ns):
    return tl.inline_asm_elementwise("nanosleep.u32 $1;\n mov.u32 $0, 0;", "=r,r", [ns], dtype=tl.int32,
                                     is_pure=False, pack=1)


@triton.jit
def pf_gemm(a_desc, b_desc, c_desc, M, N, K, allowed_ptr, next_ptr, done_ptr, exited_ptr, num_tiles, sleep_ptr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    """Persistent C = A @ B^T (A: MxK, B: NxK). Tiles are claimed from next_ptr;
    tile t computes output tile t % tiles_per_gemm, so num_tiles may span
    several repetitions of the same GEMM. A CTA exits when its SM is not in
    the allowed bitmap or no tile is left. After each tile the CTA sleeps for
    *sleep_ptr ns (read every tile, so the host can change it at runtime; the
    hardware caps one nanosleep at about 1 ms and may sleep less)."""
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    tiles_per_gemm = num_pid_m * num_pid_n
    k_tiles = tl.cdiv(K, BK)
    t = _claim(allowed_ptr, next_ptr, num_tiles)
    while t < num_tiles:
        pid = t % tiles_per_gemm
        group = pid // (GROUP_M * num_pid_n)
        first_m = group * GROUP_M
        size_m = min(num_pid_m - first_m, GROUP_M)
        pid_m = first_m + (pid % size_m)
        pid_n = (pid % (GROUP_M * num_pid_n)) // size_m
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for ki in range(k_tiles):
            a = a_desc.load([pid_m * BM, ki * BK])
            b = b_desc.load([pid_n * BN, ki * BK])
            acc = tl.dot(a, b.T, acc)
        c_desc.store([pid_m * BM, pid_n * BN], acc.to(tl.float16))
        tl.atomic_add(done_ptr, 1)
        ns = tl.atomic_or(sleep_ptr, 0)
        if ns > 0:
            _nanosleep(ns)
        t = _claim(allowed_ptr, next_ptr, num_tiles)
    tl.atomic_add(exited_ptr, 1)
