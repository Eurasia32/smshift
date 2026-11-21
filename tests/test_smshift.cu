// test_smshift.cu - Tests for the smshift library (blockIdx-claim kernels).
// Build: make test  (see Makefile)
//
//   test_smshift fence   grow 8 -> 32 SMs, then grow/shrink/grow; the stream must
//                       not complete before all work is done.
//   test_smshift churn   grow 8 -> 32 SMs, then 4000 launches with other
//                       parameters on the same stream while the sibling still
//                       runs. Every CTA checks its parameters. Run once more
//                       with SMSHIFT_TEST_NO_CB_COPY=1 as a sensitivity control
//                       (expected to report mismatches).
// MIG_ITERS sets per-block work (default 700000).
#include "smshift.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#define CHECK(x)                                                                                    \
  do {                                                                                              \
    cudaError_t e_ = (x);                                                                           \
    if (e_ != cudaSuccess) {                                                                        \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));     \
      exit(1);                                                                                      \
    }                                                                                               \
  } while (0)

static const unsigned kMagic = 0x5a5a1234u;

// - CTAs on SMs outside allowed[] exit without work;
// - claim[] makes each blockIdx do its work once;
// - every CTA checks that it sees its own launch parameters.
__global__ void coop_work(unsigned magic, unsigned* sink, unsigned long iters, unsigned* claim, const unsigned* allowed,
                          unsigned* done, unsigned* maxb, unsigned* bad) {
  __shared__ int s_real;
  const unsigned b = blockIdx.x;
  if (threadIdx.x == 0) {
    if (magic != kMagic) atomicAdd(bad, 1u);
    int real = magic == kMagic && smshift_allowed(allowed) && !atomicExch(&claim[b], 1u);
    if (real) atomicMax(maxb, b);
    s_real = real;
  }
  __syncthreads();
  if (!s_real) return;
  unsigned acc = threadIdx.x;
  for (unsigned long i = 0; i < iters; i++) acc = acc * 1664525u + 1013904223u;
  __syncthreads();
  if (threadIdx.x == 0) {
    sink[b] = acc;
    __threadfence();
    atomicAdd(done, 1u);
  }
}

__global__ void churn(unsigned v, unsigned* out) { out[0] = v; }


static int popcount8(const unsigned* b) {
  int c = 0;
  for (int i = 0; i < 8; i++) c += __builtin_popcount(b[i]);
  return c;
}

int main(int argc, char** argv) {
  setenv("CUDA_MODULE_LOADING", "EAGER", 1);
  std::string mode = argc > 1 ? argv[1] : "fence";
  CHECK(cudaSetDevice(0));
  if (smshift_init(/*sibling=*/true)) {
    fprintf(stderr, "smshift_init failed\n");
    return 1;
  }

  const unsigned blocks = 12000;
  const int threads = 1024;
  const unsigned long iters = getenv("MIG_ITERS") ? strtoul(getenv("MIG_ITERS"), nullptr, 0) : 700000;

  unsigned *sink, *claim, *allowed, *done, *maxb, *bad, *scratch;
  CHECK(cudaMalloc(&sink, blocks * 4));
  CHECK(cudaMalloc(&claim, blocks * 4));
  CHECK(cudaMalloc(&allowed, 32));
  CHECK(cudaMalloc(&done, 4));
  CHECK(cudaMalloc(&maxb, 4));
  CHECK(cudaMalloc(&bad, 4));
  CHECK(cudaMalloc(&scratch, 4));
  CHECK(cudaMemset(claim, 0, blocks * 4));
  CHECK(cudaMemset(allowed, 0xff, 32));
  CHECK(cudaMemset(done, 0, 4));
  CHECK(cudaMemset(maxb, 0, 4));
  CHECK(cudaMemset(bad, 0, 4));
  CHECK(cudaDeviceSynchronize());

  // First 4 and next 16 TPCs that exist (the mask bit layout has gaps).
  int bits[128];
  const int nb = smshift_valid_tpcs(bits, 128);
  if (nb < 20) {
    fprintf(stderr, "need at least 20 TPCs, found %d\n", nb);
    return 1;
  }
  std::vector<int> small(bits, bits + 4), big(bits + 4, bits + 20);
  unsigned sms_small[8], sms_big[8];
  smshift_probe_sms(small.data(), small.size(), sms_small);
  smshift_probe_sms(big.data(), big.size(), sms_big);

  cudaStream_t main_s, side_s;
  CHECK(cudaStreamCreateWithFlags(&main_s, cudaStreamNonBlocking));
  CHECK(cudaStreamCreateWithFlags(&side_s, cudaStreamNonBlocking));

  unsigned magic = kMagic;
  void* args[] = {&magic, &sink, (void*)&iters, &claim, &allowed, &done, &maxb, &bad};
  smshift_kernel k;
  auto t0 = std::chrono::steady_clock::now();
  auto now = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  if (smshift_launch(main_s, (const void*)coop_work, blocks, threads, 0, args, small.data(), small.size(), &k)) {
    fprintf(stderr, "launch failed\n");
    return 1;
  }

  // Steps: (time ms, TPC set). "fence" also shrinks back and grows again.
  struct Step {
    int ms;
    const std::vector<int>* tpcs;
    const unsigned* sms;
  };
  std::vector<Step> plan = {{700, &big, sms_big}};
  if (mode == "fence") plan = {{500, &big, sms_big}, {800, &small, sms_small}, {1100, &big, sms_big}};
  printf("mode %s: start on %d SMs, %zu step(s)\n", mode.c_str(), popcount8(sms_small), plan.size());

  for (auto& st : plan) {
    std::this_thread::sleep_until(t0 + std::chrono::milliseconds(st.ms));
    smshift_spawn_args a = {};
    a.tpcs = st.tpcs->data();
    a.n = st.tpcs->size();
    a.d_maxb = maxb;
    a.margin = 256;
    a.allowed_bits = st.sms;
    a.d_allowed = allowed;
    a.silent = true;
    if (smshift_spawn(side_s, &k, &a)) {
      fprintf(stderr, "spawn failed\n");
      return 1;
    }
    CHECK(cudaStreamSynchronize(side_s));
    printf("t=%.2fs spawned on %d SMs\n", now(), popcount8(st.sms));
  }

  if (mode == "churn") {
    // Runs once the original instance completes; reuses CUDA's CB0 memory.
    for (unsigned i = 0; i < 4000; i++) {
      smshift_arm_all();
      churn<<<1, 1, 0, main_s>>>(0xdead0000u | i, scratch);
    }
    CHECK(cudaGetLastError());
  }
  smshift_fence(main_s, done, blocks);
  // Poll instead of blocking, so a broken run reports what it saw.
  cudaStream_t io;
  CHECK(cudaStreamCreateWithFlags(&io, cudaStreamNonBlocking));
  while (cudaStreamQuery(main_s) == cudaErrorNotReady && now() < 15.0)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  if (cudaStreamQuery(main_s) != cudaSuccess) {
    unsigned d = 0, b = 0;
    CHECK(cudaMemcpyAsync(&d, done, 4, cudaMemcpyDeviceToHost, io));
    CHECK(cudaMemcpyAsync(&b, bad, 4, cudaMemcpyDeviceToHost, io));
    CHECK(cudaStreamSynchronize(io));
    printf("t=%.2fs not complete: done %u/%u, CTAs with wrong parameters %u\nresult: FAIL\n", now(), d, blocks, b);
    fflush(stdout);
    _exit(1);
  }
  double t_sync = now();

  unsigned h_done = 0, h_bad = 0;
  std::vector<unsigned> h_claim(blocks);
  CHECK(cudaMemcpy(&h_done, done, 4, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&h_bad, bad, 4, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_claim.data(), claim, blocks * 4, cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());
  unsigned unclaimed = 0;
  for (unsigned c : h_claim) unclaimed += c != 1;

  printf("t=%.2fs stream synchronized: done %u/%u, unclaimed %u, CTAs with wrong parameters %u\n", t_sync, h_done,
         blocks, unclaimed, h_bad);
  bool ok = h_done == blocks && unclaimed == 0 && h_bad == 0;
  printf("result: %s\n", ok ? "PASS" : "FAIL");
  smshift_cleanup();
  return ok ? 0 : 1;
}
