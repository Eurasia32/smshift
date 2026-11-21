// pd_demo.cu - Single-GPU prefill/decode co-location with runtime SM re-allocation.
// Build: make examples  (see Makefile)
//
// Synthetic workload (spin loops), for scheduling behaviour only:
//   prefill: one persistent kernel; each CTA loops pulling ~200 us items from a
//            global counter (tile-scheduler structure of persistent GEMM and
//            attention kernels). A CTA exits when its SM is not allowed.
//   decode:  one kernel every PERIOD_MS; LOW or HIGH blocks per step, switching
//            every PHASE_MS. HIGH does not fit the period on 16 SMs.
//
//   pd_demo <mode>
//     shared        no partition; decode on a high-priority stream
//     static-small  decode on 16 SMs, prefill on the other 116
//     static-large  decode on 64 SMs, prefill on the other 68
//     chunk         prefill split into kernels of CHUNK_ITEMS items; the
//                   partition follows the decode phase at kernel boundaries
//     dynamic       one prefill kernel; partition follows the decode phase at
//                   runtime (shrink: allowed bitmap; grow: smshift_spawn)
// Env: HORIZON_MS PERIOD_MS PHASE_MS LOW HIGH P_ITERS D_ITERS CHUNK_ITEMS
#include "smshift.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CHECK(x)                                                                                \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(1);                                                                                  \
    }                                                                                           \
  } while (0)

static const int kThreads = 1024;

__global__ void prefill_k(const unsigned* allowed, unsigned* next, unsigned total, unsigned* items_done,
                          unsigned* exited, unsigned long iters, unsigned* sink) {
  __shared__ unsigned s_item;
  unsigned acc = threadIdx.x;
  while (true) {
    if (threadIdx.x == 0) s_item = smshift_allowed(allowed) ? atomicAdd(next, 1u) : 0xffffffffu;
    __syncthreads();
    const unsigned item = s_item;
    __syncthreads();
    if (item >= total) break;
    for (unsigned long i = 0; i < iters; i++) acc = acc * 1664525u + 1013904223u;
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(items_done, 1u);
  }
  if (threadIdx.x == 0) {
    sink[blockIdx.x % 4096] = acc;
    __threadfence();
    atomicAdd(exited, 1u);
  }
}

__global__ void decode_k(unsigned long iters, unsigned* sink) {
  unsigned acc = threadIdx.x;
  for (unsigned long i = 0; i < iters; i++) acc = acc * 1664525u + 1013904223u;
  if (threadIdx.x == 0) sink[blockIdx.x % 4096] = acc;
}

static long env(const char* k, long d) { return getenv(k) ? strtol(getenv(k), nullptr, 0) : d; }

static std::vector<int> minus(const std::vector<int>& a, const std::vector<int>& b) {
  std::vector<int> r;
  for (int x : a)
    if (std::find(b.begin(), b.end(), x) == b.end()) r.push_back(x);
  return r;
}

static int popcount8(const unsigned* b) {
  int c = 0;
  for (int i = 0; i < 8; i++) c += __builtin_popcount(b[i]);
  return c;
}

struct TpcSet {
  std::vector<int> tpcs;
  unsigned sms[8];
  int n_sms;
};
static TpcSet make_set(std::vector<int> tpcs) {
  TpcSet s;
  s.tpcs = std::move(tpcs);
  smshift_probe_sms(s.tpcs.data(), s.tpcs.size(), s.sms);
  s.n_sms = popcount8(s.sms);
  return s;
}

static double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, (size_t)(p * (v.size() - 1) + 0.5))];
}

int main(int argc, char** argv) {
  setenv("CUDA_MODULE_LOADING", "EAGER", 1);
  const std::string mode = argc > 1 ? argv[1] : "dynamic";
  const long horizon_ms = env("HORIZON_MS", 4000), period_us = env("PERIOD_MS", 5) * 1000, phase_ms = env("PHASE_MS", 500);
  const unsigned low = env("LOW", 32), high = env("HIGH", 640);
  const unsigned long p_iters = env("P_ITERS", 35000), d_iters = env("D_ITERS", 52500);
  const unsigned chunk_items = env("CHUNK_ITEMS", 50000);
  if (mode != "shared" && mode != "static-small" && mode != "static-large" && mode != "chunk" && mode != "dynamic") {
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
  }
  CHECK(cudaSetDevice(0));
  if (smshift_init(/*sibling=*/true)) {
    fprintf(stderr, "smshift_init failed\n");
    return 1;
  }

  // TPC_DISABLE_MASK bits that have SMs (the layout has gaps).
  int bits[128];
  const int nb = smshift_valid_tpcs(bits, 128);
  std::vector<int> valid(bits, bits + (nb > 0 ? nb : 0));
  int dev_sms = 0;
  CHECK(cudaDeviceGetAttribute(&dev_sms, cudaDevAttrMultiProcessorCount, 0));
  const TpcSet all = make_set(valid);
  if (nb < 32 || all.n_sms != dev_sms) {
    fprintf(stderr, "TPC map covers %d of %d SMs with %d bits; need >= 32 TPCs\n", all.n_sms, dev_sms, nb);
    return 1;
  }
  const std::vector<int> dec_small_t(valid.begin(), valid.begin() + 8), dec_large_t(valid.begin(), valid.begin() + 32);
  const TpcSet dec_small = make_set(dec_small_t), dec_large = make_set(dec_large_t);
  const TpcSet pf_large = make_set(minus(valid, dec_small_t)), pf_small = make_set(minus(valid, dec_large_t));
  const TpcSet delta = make_set(minus(dec_large_t, dec_small_t));

  int occ_p = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_p, prefill_k, kThreads, 0));

  int prio_lo = 0, prio_hi = 0;
  CHECK(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
  cudaStream_t pf_s, dec_s, side_s, io_s;
  CHECK(cudaStreamCreateWithPriority(&pf_s, cudaStreamNonBlocking, prio_lo));
  CHECK(cudaStreamCreateWithPriority(&dec_s, cudaStreamNonBlocking, prio_hi));
  CHECK(cudaStreamCreateWithFlags(&side_s, cudaStreamNonBlocking));
  CHECK(cudaStreamCreateWithFlags(&io_s, cudaStreamNonBlocking));

  const int kMaxChunks = 4096;
  unsigned *allowed, *next, *items_done, *exited, *psink, *dsink;
  CHECK(cudaMalloc(&allowed, 32));
  CHECK(cudaMalloc(&next, kMaxChunks * 4));
  CHECK(cudaMalloc(&items_done, 4));
  CHECK(cudaMalloc(&exited, 4));
  CHECK(cudaMalloc(&psink, 4096 * 4));
  CHECK(cudaMalloc(&dsink, 4096 * 4));
  CHECK(cudaMemset(allowed, 0xff, 32));
  CHECK(cudaMemset(next, 0, kMaxChunks * 4));
  CHECK(cudaMemset(items_done, 0, 4));
  CHECK(cudaMemset(exited, 0, 4));
  CHECK(cudaDeviceSynchronize());

  auto read_u32 = [&](unsigned* p) {
    unsigned v = 0;
    CHECK(cudaMemcpyAsync(&v, p, 4, cudaMemcpyDeviceToHost, io_s));
    CHECK(cudaStreamSynchronize(io_s));
    return v;
  };

  const unsigned kInf = 0xffffffffu;
  unsigned launched_ctas = 0;
  int chunks = 0, spawns = 0;
  smshift_kernel pk = {0, -1};
  const TpcSet* pf_running = nullptr;  // chunk mode: set of the running chunk
  const TpcSet* dec_set = nullptr;     // nullptr = unmasked
  unsigned pf_grid = 0;

  auto launch_prefill = [&](const TpcSet* set, unsigned* nxt, unsigned total) {
    pf_grid = occ_p * (set ? set->n_sms : dev_sms);
    void* args[] = {&allowed, &nxt, &total, &items_done, &exited, (void*)&p_iters, &psink};
    if (set) {
      if (smshift_launch(pf_s, (const void*)prefill_k, pf_grid, kThreads, 0, args, set->tpcs.data(), set->tpcs.size(),
                        &pk)) {
        fprintf(stderr, "prefill launch failed\n");
        exit(1);
      }
    } else {
      CHECK(cudaLaunchKernel((const void*)prefill_k, pf_grid, kThreads, args, 0, pf_s));
    }
    launched_ctas += pf_grid;
  };
  auto launch_chunk = [&](const TpcSet* set) {
    if (chunks == kMaxChunks) {
      fprintf(stderr, "too many chunks\n");
      exit(1);
    }
    launch_prefill(set, next + chunks++, chunk_items);
    pf_running = set;
    dec_set = set == &pf_large ? &dec_small : &dec_large;
  };

  auto t0 = std::chrono::steady_clock::now();
  auto now_us = [&] { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(); };

  // Initial state: phase 0 (low decode load).
  if (mode == "shared") {
    launch_prefill(nullptr, next, kInf);
  } else if (mode == "static-small" || mode == "dynamic") {
    CHECK(cudaMemcpy(allowed, pf_large.sms, 32, cudaMemcpyHostToDevice));
    launch_prefill(&pf_large, next, kInf);
    dec_set = &dec_small;
  } else if (mode == "static-large") {
    CHECK(cudaMemcpy(allowed, pf_small.sms, 32, cudaMemcpyHostToDevice));
    launch_prefill(&pf_small, next, kInf);
    dec_set = &dec_large;
  } else {  // chunk
    launch_chunk(&pf_large);
  }
  const unsigned grid0 = pf_grid;

  bool stopped = false;
  double t_stop = 0;
  unsigned items_at_stop = 0;
  auto stop_prefill = [&] {
    const unsigned zero[8] = {0};
    CHECK(cudaMemcpy(allowed, zero, 32, cudaMemcpyHostToDevice));
    stopped = true;
    t_stop = now_us();
    items_at_stop = read_u32(items_done);
  };

  struct Rec {
    int phase;
    double t_sched, lat;
  };
  std::vector<Rec> recs;
  std::vector<double> phase_items[2];
  int cur_phase = 0;
  double t_phase = 0;
  unsigned items_phase = 0;
  unsigned blocks_args = 0;
  void* dargs[] = {(void*)&d_iters, &dsink};

  for (long k = 0;; k++) {
    const double t_k = (double)k * period_us;
    if (t_k >= horizon_ms * 1000.0) break;
    // Wait for the step time. Chunk mode keeps polling so the next prefill
    // kernel follows the previous one without a host-induced gap.
    while (now_us() < t_k) {
      if (mode == "chunk" && cudaStreamQuery(pf_s) == cudaSuccess) {
        const int ph = (int)((long)(now_us() / 1000) / phase_ms) % 2;
        launch_chunk(ph ? &pf_small : &pf_large);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(mode == "chunk" ? 10 : 50));
    }
    const int phase = (int)((long)(t_k / 1000) / phase_ms) % 2;
    if (phase != cur_phase && !stopped) {
      unsigned it = read_u32(items_done);
      phase_items[cur_phase].push_back((it - items_phase) / ((now_us() - t_phase) / 1e6));
      items_phase = it;
      t_phase = now_us();
      if (mode == "dynamic") {
        if (phase == 1) {  // shrink prefill, decode takes the freed SMs
          dec_set = &dec_large;
          if (smshift_set_allowed(side_s, allowed, pf_small.sms)) exit(1);
        } else {           // decode back to 16 SMs, prefill grows into them
          dec_set = &dec_small;
          const unsigned add = occ_p * delta.n_sms;
          smshift_spawn_args a = {};
          a.tpcs = delta.tpcs.data();
          a.n = delta.tpcs.size();
          a.resume = grid0 - add;  // dispatch exactly `add` CTAs
          a.allowed_bits = pf_large.sms;
          a.d_allowed = allowed;
          a.silent = true;
          if (smshift_spawn(side_s, &pk, &a)) {
            fprintf(stderr, "spawn failed\n");
            exit(1);
          }
          launched_ctas += add;
          spawns++;
        }
      }
      cur_phase = phase;
    }
    if (mode == "chunk" && cudaStreamQuery(pf_s) == cudaSuccess) launch_chunk(phase ? &pf_small : &pf_large);

    blocks_args = phase ? high : low;
    if (dec_set) {
      if (smshift_launch(dec_s, (const void*)decode_k, blocks_args, kThreads, 0, dargs, dec_set->tpcs.data(),
                        dec_set->tpcs.size(), nullptr)) {
        fprintf(stderr, "decode launch failed\n");
        exit(1);
      }
    } else {
      CHECK(cudaLaunchKernel((const void*)decode_k, blocks_args, kThreads, dargs, 0, dec_s));
    }
    cudaError_t q;
    while ((q = cudaStreamQuery(dec_s)) == cudaErrorNotReady) {
      if (!stopped && now_us() >= horizon_ms * 1000.0) stop_prefill();
      if (mode == "chunk" && !stopped && cudaStreamQuery(pf_s) == cudaSuccess) launch_chunk(phase ? &pf_small : &pf_large);
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    CHECK(q);
    recs.push_back({phase, t_k, (now_us() - t_k) / 1000.0});
  }
  if (!stopped) stop_prefill();
  if (items_at_stop >= items_phase && t_stop > t_phase)
    phase_items[cur_phase].push_back((items_at_stop - items_phase) / ((t_stop - t_phase) / 1e6));
  // Wait for every prefill CTA of every instance to exit.
  if (mode != "chunk") smshift_fence(pf_s, exited, launched_ctas);
  CHECK(cudaStreamSynchronize(pf_s));
  CHECK(cudaDeviceSynchronize());
  const unsigned exited_h = read_u32(exited);

  // ---- Report ----
  const double period_ms = period_us / 1000.0;
  printf("mode %-12s  prefill %u items in %.2fs = %.0f items/s", mode.c_str(), items_at_stop, t_stop / 1e6,
         items_at_stop / (t_stop / 1e6));
  double low_rate = 0, high_rate = 0;
  for (double r : phase_items[0]) low_rate += r / phase_items[0].size();
  for (double r : phase_items[1]) high_rate += r / phase_items[1].size();
  if (mode == "shared")
    printf(" (per-phase rates not sampled: the decode loop is blocked)\n");
  else
    printf(" (low-phase %.0f/s, high-phase %.0f/s)\n", low_rate, high_rate);
  for (int ph = 0; ph < 2; ph++) {
    std::vector<double> l, first;
    int miss = 0;
    for (auto& r : recs) {
      if (r.phase != ph) continue;
      l.push_back(r.lat);
      miss += r.lat > period_ms;
      double into = fmod(r.t_sched / 1000.0, (double)phase_ms);
      if (into < 100) first.push_back(r.lat);
    }
    printf("  decode %s (%u blocks): %zu steps, p50 %.2f ms, p99 %.2f ms, max %.2f ms, >%.0f ms %.1f%%, "
           "max in first 100 ms of phase %.2f ms\n",
           ph ? "high" : "low ", ph ? high : low, l.size(), pct(l, 0.5), pct(l, 0.99), pct(l, 1.0), period_ms,
           l.empty() ? 0 : 100.0 * miss / l.size(), pct(first, 1.0));
  }
  printf("  prefill CTAs launched %u, exited %u%s; chunks %d; spawns %d; SMs: decode %d/%d, prefill %d/%d (%s)\n",
         launched_ctas, exited_h, mode == "chunk" ? "" : (exited_h == launched_ctas ? " (OK)" : " (MISMATCH)"), chunks,
         spawns, dec_small.n_sms, dec_large.n_sms, pf_large.n_sms, pf_small.n_sms,
         mode == "shared" ? "unpartitioned" : "small/large");
  smshift_cleanup();
  return 0;
}
