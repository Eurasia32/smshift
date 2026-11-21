"""Correctness of a real persistent GEMM (Triton, TMA + wgmma) under smshift.
Requires torch and triton (tested with torch 2.9.1, triton 3.5.1).

  test_gemm.py base     full GPU, no migration: result, counters, TFLOPS
  test_gemm.py migrate  start on 16 SMs, spawn onto 64 more, then revoke the
                        original 16 so the original instance completes; 4000
                        launches with other parameters then run on the same
                        stream and reuse its CB0. A second spawn (32 more SMs)
                        after that can only work from the CB0 template; its
                        parameters include host TMA descriptors.
                        Run with SMSHIFT_TEST_NO_CB_COPY=1 as a control.
"""
import ctypes
import os
import sys
import time

os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
import torch  # noqa: E402
import triton  # noqa: E402
from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import smshift  # noqa: E402
from triton_kernels import pf_gemm  # noqa: E402

BM, BN, BK, GROUP_M, WARPS, STAGES = 128, 256, 64, 8, 8, 3
M = N = K = 8192

_cuda = ctypes.CDLL("libcuda.so.1")


def ctas_per_sm(compiled):
    n = ctypes.c_int()
    r = _cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(n), ctypes.c_void_p(compiled.function),
                                                         ctypes.c_int(WARPS * 32), ctypes.c_size_t(compiled.metadata.shared))
    if r != 0:
        raise RuntimeError(f"cuOccupancyMaxActiveBlocksPerMultiprocessor -> {r}")
    return n.value


def u32_tensor(bits):
    return torch.tensor([b & 0xFFFFFFFF for b in bits], dtype=torch.int64).to(torch.uint32).view(torch.int32).cuda()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "base"
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    smshift.init(sibling=True)
    nsm = torch.cuda.get_device_properties(0).multi_processor_count

    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)
    ref = a @ b.T
    da = TensorDescriptor.from_tensor(a, [BM, BK])
    db = TensorDescriptor.from_tensor(b, [BN, BK])
    dc = TensorDescriptor.from_tensor(c, [BM, BN])
    tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)

    allowed = u32_tensor([0xFFFFFFFF] * 8)
    ctr = torch.zeros(3, device="cuda", dtype=torch.int32)  # next, done, exited
    nxt, done, exited = ctr[0:1], ctr[1:2], ctr[2:3]
    nosleep = torch.zeros(1, device="cuda", dtype=torch.int32)

    def launch(grid, total, stream):
        with torch.cuda.stream(stream):
            return pf_gemm[(grid,)](da, db, dc, M, N, K, allowed, nxt, done, exited, total, nosleep, BM=BM, BN=BN, BK=BK,
                                    GROUP_M=GROUP_M, num_warps=WARPS, num_stages=STAGES)

    main_s = torch.cuda.Stream()
    side_s = torch.cuda.Stream()

    # Warm-up (compiles and loads the kernels).
    compiled = launch(1, 0, main_s)
    torch.cuda.synchronize()
    occ = ctas_per_sm(compiled)
    print(f"pf_gemm: {compiled.n_regs} regs, {compiled.metadata.shared} B shared, {occ} CTA/SM")
    ctr.zero_()

    if mode == "base":
        reps = 20
        total = tiles * reps
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        smshift.arm(smshift.valid_tpcs())
        launch(nsm * occ, total, main_s)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tflops = 2.0 * M * N * K * reps / dt / 1e12
        h = ctr.tolist()
        err = (c.float() - ref.float()).abs().max().item()
        print(f"{dt * 1e3:.1f} ms, {tflops:.0f} TFLOPS; next {h[0]} (expect {total + nsm * occ}), done {h[1]}/{total}, "
              f"exited {h[2]}/{nsm * occ}; max |C - ref| {err:.3f}")
        ok = h[1] == total and h[2] == nsm * occ and h[0] == total + nsm * occ and err < 1.0
    else:
        T = smshift.valid_tpcs()
        s0 = T[:8]
        s1 = T[8:40]
        b0, b1 = smshift.probe_sms(s0), smshift.probe_sms(s1)
        b01 = [x | y for x, y in zip(b0, b1)]
        reps = 90
        total = tiles * reps
        grid0 = smshift.popcount(b0) * occ
        add = smshift.popcount(b1) * occ
        allowed.copy_(u32_tensor(b0))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        smshift.arm(s0)
        launch(grid0, total, main_s)
        k = smshift.kernel_handle()
        time.sleep(0.2)
        # The original grid has only grid0 CTAs; the sibling gets its own width.
        smshift.spawn(side_s.cuda_stream, k, s1, resume=0, grid_width=add, allowed_bits=b01,
                     d_allowed=allowed.data_ptr())
        spawned = add
        side_s.synchronize()
        # Queued behind the original instance; they run once it completes and
        # reuse CUDA's CB0 memory while the sibling is still computing.
        # Same kernel, other parameters: dummy descriptors and counters, no
        # tiles. This overwrites the whole parameter area of a reused CB0
        # (a kernel with a short parameter list would only touch padding).
        xa = torch.zeros(BM, BK, device="cuda", dtype=torch.float16)
        xb = torch.zeros(BN, BK, device="cuda", dtype=torch.float16)
        xc = torch.zeros(BM, BN, device="cuda", dtype=torch.float16)
        xda, xdb, xdc = (TensorDescriptor.from_tensor(xa, [BM, BK]), TensorDescriptor.from_tensor(xb, [BN, BK]),
                         TensorDescriptor.from_tensor(xc, [BM, BN]))
        xctr = torch.zeros(3, device="cuda", dtype=torch.int32)
        xallowed = u32_tensor([0xFFFFFFFF] * 8)
        with torch.cuda.stream(main_s):
            for i in range(4000):
                smshift.arm_all()
                pf_gemm[(1,)](xda, xdb, xdc, BM, BN, BK, xallowed, xctr[0:1], xctr[1:2], xctr[2:3], 0, nosleep, BM=BM, BN=BN,
                              BK=BK, GROUP_M=GROUP_M, num_warps=WARPS, num_stages=STAGES)
        time.sleep(0.2)
        smshift.set_allowed(side_s.cuda_stream, allowed.data_ptr(), b1)  # revoke the original's SMs
        # Spawn again after the original instance has completed and its CB0
        # has been reused: these CTAs can only get correct parameters (TMA
        # descriptors, pointers) from the template.
        time.sleep(0.2)
        s2 = T[40:56]
        b2 = smshift.probe_sms(s2)
        add2 = smshift.popcount(b2) * occ
        smshift.spawn(side_s.cuda_stream, k, s2, resume=0, grid_width=add2,
                     allowed_bits=[x | y for x, y in zip(b1, b2)], d_allowed=allowed.data_ptr())
        spawned += add2
        smshift.fence(main_s.cuda_stream, done.data_ptr(), total)
        deadline = time.perf_counter() + 30
        while not main_s.query() and time.perf_counter() < deadline:
            time.sleep(0.01)
        if not main_s.query():
            print("not complete after 30 s (sibling stalled or read wrong parameters)")
            sys.stdout.flush()
            os._exit(1)
        dt = time.perf_counter() - t0
        torch.cuda.synchronize()
        h = ctr.tolist()
        err = (c.float() - ref.float()).abs().max().item()
        print(f"dummy counters (must be 0 tiles done): {xctr.tolist()[1]}")
        print(f"start {smshift.popcount(b0)} SMs, +{smshift.popcount(b1)} SMs at 0.2 s, original revoked at 0.4 s, "
              f"+{smshift.popcount(b2)} SMs at 0.6 s; "
              f"{dt:.2f} s; done {h[1]}/{total}, exited {h[2]}/{grid0 + spawned}; max |C - ref| {err:.3f}")
        ok = h[1] == total and h[2] == grid0 + spawned and err < 1.0 and xctr.tolist()[1] == 0
    print("result:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
