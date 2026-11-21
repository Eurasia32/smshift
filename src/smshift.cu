// smshift.cu - Implementation of smshift (runtime SM re-allocation, Hopper).
// Field positions: open-gpu-doc classes/compute/clcbc0qmd.h (QMD V04_00).
#include "smshift.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>

extern "C" CUresult cuGetExportTable(const void** table, const CUuuid* id);

// ---- QMD V04_00 field positions (32-bit word index or bit index) ----
static const int kWordMask = 76;      // TPC_DISABLE_MASK(0..3)
static const int kWordGridX = 32;     // GRID_WIDTH
static const int kWordResumeX = 44;   // GRID_WIDTH_RESUME
static const int kWordResumeYZ = 45;  // GRID_HEIGHT_RESUME 15:0, GRID_DEPTH_RESUME 31:16
static const int kWordCbValid = 13;   // CONSTANT_BUFFER_VALID(i) = word 13 bit 4i
static const int kWordCb = 48;        // CONSTANT_BUFFER_ADDR_LOWER_SHIFTED6(i) = word 48+2i;
                                      // word 49+2i: ADDR_UPPER_SHIFTED6 18:0, SIZE_SHIFTED4 31:19
static const int kNumCb = 8;
static const int kQmdWords = 96;      // 384 bytes
static const int kByteQmdVersion = 72;  // QMD_MAJOR_VERSION 7:4, QMD_MINOR_VERSION 3:0
static const uint8_t kQmdV04_00 = 0x40;
static const int kBitMaskValid = 31;  // TPC_DISABLE_MASK_VALID
// libcuda uses dependent slot 0 for stream ordering; slot 1 is free.
static const int kBitDepEnable1 = 373, kBitDepAction1 = 374, kBitDepPtr1 = 480;
static const int kActionSchedule = 1;
static const int kQmdDomain = 0xb, kPreUpload = 1, kPostUpload = 2;

// ---- Low-VA buffer layout ----
static const size_t kBufBytes = 8u << 20;
static const size_t kQmdSlotBytes = 512;  // QMD pointers are addr >> 8
static const unsigned kSibSlots = 2048;   // [0, 1 MB): sibling QMDs (ring)
static const size_t kTmplQmdBase = 1u << 20;
static const unsigned kTmplSlots = 64;    // [1 MB, 1 MB + 32 KB): template QMDs (ring)
static const size_t kTmplCbBase = 2u << 20;
static const unsigned kTmplCbBytes = 36u << 10;  // >= 0x210 header + 32764 B of params

static const CUuuid kCallbackFuncsId = {{0x2c, (char)0x8e, 0x0a, (char)0xd8, 0x07, 0x10, (char)0xab, 0x4e,
                                         (char)0x90, (char)0xdd, 0x54, 0x71, (char)0x9f, (char)0xe5, (char)0xf7, 0x4b}};

// ---- State ----
struct Edit {
  bool active = false;
  unsigned mask[4] = {0, 0, 0, 0};
  bool dep = false;
  uint32_t dep_ptr = 0;
};
// The QMD callbacks run inside cudaLaunchKernel on the launching thread.
static thread_local Edit t_edit;
static thread_local uint64_t t_last_va = 0;

static bool s_sibling = false;
static volatile bool s_bad_qmd_version = false;  // set by the callback, never cleared
static bool s_cb_copy = true;  // SMSHIFT_TEST_NO_CB_COPY=1 disables it (tests only)
static uint64_t s_qbuf = 0;
static unsigned s_next_sib = 0, s_next_tmpl = 0;
static unsigned* s_scratch = nullptr;
static unsigned* s_sink = nullptr;
static cudaStream_t s_probe_stream = nullptr;

static void set_bits(uint32_t* t, int lo, int width, uint32_t v) {
  for (int i = 0; i < width; i++) {
    int b = lo + i;
    t[b / 32] = (t[b / 32] & ~(1u << (b % 32))) | (((v >> i) & 1u) << (b % 32));
  }
}

static void tpc_disable_mask(const int* tpcs, int n, unsigned m[4]) {
  for (int i = 0; i < 4; i++) m[i] = ~0u;
  for (int i = 0; i < n; i++) m[tpcs[i] / 32] &= ~(1u << (tpcs[i] % 32));
}

static void qmd_callback(void*, int, int cbid, const void* in_params) {
  const uint64_t* w = (const uint64_t*)in_params;
  if (cbid == kPostUpload) {
    t_last_va = w[5];
    return;
  }
  if (cbid != kPreUpload || !t_edit.active) return;
  uint32_t* t = (uint32_t*)w[4];
  if (((const uint8_t*)t)[kByteQmdVersion] != kQmdV04_00) {
    // Unknown layout: leave the QMD untouched (the launch runs unrestricted).
    s_bad_qmd_version = true;
    t_edit = Edit();
    return;
  }
  for (int i = 0; i < 4; i++) t[kWordMask + i] = t_edit.mask[i];
  set_bits(t, kBitMaskValid, 1, 1);
  if (t_edit.dep) {
    set_bits(t, kBitDepPtr1, 32, t_edit.dep_ptr);
    set_bits(t, kBitDepAction1, 3, kActionSchedule);
    set_bits(t, kBitDepEnable1, 1, 1);
  }
  t_edit = Edit();
}

// ---- Device side ----

__device__ static unsigned dev_smid() {
  unsigned r;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}
__device__ static unsigned long long dev_time() {
  unsigned long long r;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r));
  return r;
}
__device__ static unsigned long long cb_addr(const unsigned* q, int i) {
  return ((unsigned long long)(q[kWordCb + 2 * i + 1] & 0x7FFFFu) << 38) | ((unsigned long long)q[kWordCb + 2 * i] << 6);
}
__device__ static unsigned cb_size(const unsigned* q, int i) { return (q[kWordCb + 2 * i + 1] >> 19) << 4; }
__device__ static void set_cb_addr(unsigned* q, int i, unsigned long long a) {
  q[kWordCb + 2 * i] = (unsigned)(a >> 6);
  q[kWordCb + 2 * i + 1] = (q[kWordCb + 2 * i + 1] & ~0x7FFFFu) | (unsigned)((a >> 38) & 0x7FFFFu);
}

__global__ static void k_mark_sms(unsigned* bits, unsigned long iters, unsigned* sink) {
  if (threadIdx.x == 0) atomicOr(&bits[dev_smid() / 32], 1u << (dev_smid() % 32));
  unsigned acc = threadIdx.x;
  for (unsigned long i = 0; i < iters; i++) acc = acc * 1664525u + 1013904223u;
  if (threadIdx.x == 0) sink[blockIdx.x] = acc;
}

// One thread. Order: allowed bitmap, progress, then the QMD that gets scheduled.
//   src:  QMD to copy from (the original, or the launch's template)
//   tmpl: if non-null, first snapshot src (+ CB0) into tmpl and copy from it
//   sib:  if non-null, the QMD to modify and schedule; else src is modified in place
__global__ static void k_spawn(unsigned* src, unsigned* tmpl, unsigned* tmpl_cb, int cb_copy, unsigned* sib,
                               unsigned* allowed, uint4 al_lo, uint4 al_hi, int set_allowed, uint4 mask,
                               unsigned resume, const unsigned* maxb, unsigned margin, const unsigned* counter,
                               int silent, unsigned grid_width, unsigned* at) {
  if (set_allowed) {
    unsigned na[8] = {al_lo.x, al_lo.y, al_lo.z, al_lo.w, al_hi.x, al_hi.y, al_hi.z, al_hi.w};
    for (int i = 0; i < 8; i++) allowed[i] = na[i];
    __threadfence_system();
  }
  unsigned c = counter ? *(volatile const unsigned*)counter : 0;
  unsigned p = resume;
  if (maxb) {
    unsigned next = *(volatile const unsigned*)maxb + 1;
    p = next > margin ? next - margin : 0;
  }
  if (tmpl) {
    for (int i = 0; i < kQmdWords; i++) tmpl[i] = src[i];
    if (cb_copy) {
      // CB0 holds the launch's parameters; CUDA reuses it once the original
      // instance completes. CB5 was observed to alias CB0 on every launch, so
      // every valid bank pointing at CB0 is redirected.
      const unsigned long long a0 = cb_addr(tmpl, 0);
      const unsigned size = cb_size(tmpl, 0);
      if (size > kTmplCbBytes) __trap();
      const unsigned* from = (const unsigned*)a0;
      for (unsigned i = 0; i < size / 4; i++) tmpl_cb[i] = from[i];
      for (int b = 0; b < kNumCb; b++)
        if (((tmpl[kWordCbValid] >> (4 * b)) & 1u) && cb_addr(tmpl, b) == a0)
          set_cb_addr(tmpl, b, (unsigned long long)tmpl_cb);
    }
    src = tmpl;
  }
  unsigned* q = src;
  if (sib) {
    for (int i = 0; i < kQmdWords; i++) sib[i] = src[i];
    sib[5] = sib[6] = sib[7] = sib[9] = 0;                 // HW_ONLY_* state
    sib[8] = (unsigned)((unsigned long long)sib >> 8);     // as observed on uploaded QMDs
    if (grid_width) sib[kWordGridX] = grid_width;
    q = sib;
  }
  q[kWordMask + 0] = mask.x, q[kWordMask + 1] = mask.y, q[kWordMask + 2] = mask.z, q[kWordMask + 3] = mask.w;
  q[kWordResumeX] = p;
  q[kWordResumeYZ] = 0;
  if (silent) {
    q[10] &= ~((1u << 0) | (1u << 16));  // RELEASE_ENABLE(0), RELEASE_ENABLE(1)
    q[11] &= ~((1u << 0) | (1u << 16));  // RELEASE_ENABLE(2), DEPENDENT_QMD0_ENABLE
  }
  __threadfence_system();
  if (at) {
    at[0] = c;
    at[1] = p;
    *(unsigned long long*)(at + 2) = dev_time();
  }
}

__global__ static void k_nop(unsigned* p) { p[0] += 0; }

__global__ static void k_fence(const unsigned* v, unsigned target) {
  while (*(volatile const unsigned*)v < target) __nanosleep(5000);
}

// ---- Low-VA buffer ----
// Device-memory reservation hints are not honored under UVA in practice; host
// memory mapped at a fixed low address works (device VA == host VA) and the
// scheduler reads QMDs from it.
static int alloc_low_va_buffer() {
  const unsigned long long hints[] = {0x210000000ull, 0x400000000ull, 0x3000000000ull};
  for (unsigned long long hint : hints) {
    CUdeviceptr va = 0;
    if (cuMemAddressReserve(&va, kBufBytes, kBufBytes, (CUdeviceptr)hint, 0) != CUDA_SUCCESS) continue;
    if (va >> 40) {
      cuMemAddressFree(va, kBufBytes);
      continue;
    }
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    CUmemGenericAllocationHandle h;
    CUmemAccessDesc acc = {};
    acc.location = prop.location;
    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemCreate(&h, kBufBytes, &prop, 0) || cuMemMap(va, kBufBytes, 0, h, 0) ||
        cuMemSetAccess(va, kBufBytes, &acc, 1) || cuMemsetD8(va, 0, kBufBytes)) {
      cuMemAddressFree(va, kBufBytes);
      continue;
    }
    s_qbuf = va;
    return 0;
  }
  for (unsigned long long hint : hints) {
    void* p = mmap((void*)hint, kBufBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
                   -1, 0);
    if (p == MAP_FAILED) continue;
    memset(p, 0, kBufBytes);
    CUdeviceptr dp = 0;
    if (cuMemHostRegister(p, kBufBytes, CU_MEMHOSTREGISTER_DEVICEMAP) || cuMemHostGetDevicePointer(&dp, p, 0) ||
        dp >> 40) {
      munmap(p, kBufBytes);
      continue;
    }
    s_qbuf = dp;
    return 0;
  }
  fprintf(stderr, "smshift: no GPU VA below 1 TB available\n");
  return -1;
}

// ---- Public API ----

int smshift_init(bool sibling) {
  int dev = 0, major = 0;
  if (cudaGetDevice(&dev) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
    return SMSHIFT_ERR_CUDA;
  if (major != 9) return SMSHIFT_ERR_UNSUPPORTED;
  if (cudaMalloc(&s_scratch, 4096) != cudaSuccess) return SMSHIFT_ERR_CUDA;
  if (cudaMalloc(&s_sink, 16384 * 4) != cudaSuccess) return SMSHIFT_ERR_CUDA;
  if (cudaStreamCreateWithFlags(&s_probe_stream, cudaStreamNonBlocking) != cudaSuccess) return SMSHIFT_ERR_CUDA;
  k_nop<<<1, 1>>>(s_scratch);
  if (cudaDeviceSynchronize() != cudaSuccess) return SMSHIFT_ERR_CUDA;

  uintptr_t* tbl = nullptr;
  if (cuGetExportTable((const void**)&tbl, &kCallbackFuncsId) != CUDA_SUCCESS || !tbl) return SMSHIFT_ERR_UNSUPPORTED;
  auto subscribe = (int (*)(uint32_t*, void (*)(void*, int, int, const void*), void*))tbl[3];
  auto enable = (int (*)(uint32_t, uint32_t, int, int))tbl[6];
  uint32_t h = 0;
  if (subscribe(&h, qmd_callback, nullptr) || enable(1, h, kQmdDomain, kPreUpload) ||
      enable(1, h, kQmdDomain, kPostUpload))
    return SMSHIFT_ERR_UNSUPPORTED;

  s_cb_copy = getenv("SMSHIFT_TEST_NO_CB_COPY") == nullptr;
  s_sibling = sibling;
  if (sibling && alloc_low_va_buffer()) return SMSHIFT_ERR_NO_LOW_VA;
  return SMSHIFT_OK;
}

void smshift_arm(const int* tpcs, int n) {
  t_edit = Edit();
  t_edit.active = true;
  tpc_disable_mask(tpcs, n, t_edit.mask);
}

void smshift_arm_all(void) {
  t_edit = Edit();
  t_edit.active = true;  // valid bit with nothing disabled
}

uint64_t smshift_last_va(void) { return t_last_va; }

int smshift_launch(cudaStream_t stream, const void* func, dim3 grid, dim3 block, size_t shmem, void** args,
                  const int* tpcs, int n, smshift_kernel* out) {
  if (n <= 0 || !tpcs) return SMSHIFT_ERR_ARG;
  smshift_arm(tpcs, n);
  t_last_va = 0;
  cudaError_t e = cudaLaunchKernel(func, grid, block, args, shmem, stream);
  if (out) {
    out->qmd_va = t_last_va;
    out->tmpl = -1;
  }
  if (s_bad_qmd_version) return SMSHIFT_ERR_QMD_VERSION;
  return e == cudaSuccess && t_last_va ? SMSHIFT_OK : SMSHIFT_ERR_CUDA;
}

int smshift_spawn(cudaStream_t side, smshift_kernel* k, const smshift_spawn_args* a) {
  if (!k || !k->qmd_va || !a || !a->tpcs || a->n <= 0 || (a->allowed_bits && !a->d_allowed) ||
      (a->grid_width && !s_sibling))
    return SMSHIFT_ERR_ARG;
  if (s_bad_qmd_version) return SMSHIFT_ERR_QMD_VERSION;
  unsigned m[4];
  tpc_disable_mask(a->tpcs, a->n, m);
  const unsigned zero[8] = {0};
  const unsigned* al = a->allowed_bits ? a->allowed_bits : zero;

  unsigned* src = (unsigned*)k->qmd_va;
  unsigned *tmpl = nullptr, *tmpl_cb = nullptr, *sib = nullptr;
  if (s_sibling) {
    if (k->tmpl < 0) {
      k->tmpl = (int)(s_next_tmpl++ % kTmplSlots);
      tmpl = (unsigned*)(s_qbuf + kTmplQmdBase + kQmdSlotBytes * k->tmpl);
      tmpl_cb = (unsigned*)(s_qbuf + kTmplCbBase + (size_t)kTmplCbBytes * k->tmpl);
    } else {
      src = (unsigned*)(s_qbuf + kTmplQmdBase + kQmdSlotBytes * k->tmpl);
    }
    sib = (unsigned*)(s_qbuf + kQmdSlotBytes * (s_next_sib++ % kSibSlots));
  }

  smshift_arm_all();
  k_spawn<<<1, 1, 0, side>>>(src, tmpl, tmpl_cb, s_cb_copy ? 1 : 0, sib, a->d_allowed,
                             make_uint4(al[0], al[1], al[2], al[3]), make_uint4(al[4], al[5], al[6], al[7]),
                             a->allowed_bits ? 1 : 0, make_uint4(m[0], m[1], m[2], m[3]), a->resume, a->d_maxb,
                             a->margin, a->d_counter, a->silent ? 1 : 0, a->grid_width, a->d_at);
  // The helper's completion makes the scheduler fetch and schedule the QMD.
  smshift_arm_all();
  t_edit.dep = true;
  t_edit.dep_ptr = (uint32_t)((sib ? (uint64_t)sib : (uint64_t)src) >> 8);
  k_nop<<<1, 1, 0, side>>>(s_scratch + 200);
  return cudaGetLastError() == cudaSuccess ? SMSHIFT_OK : SMSHIFT_ERR_CUDA;
}

int smshift_set_allowed(cudaStream_t side, unsigned* d_allowed, const unsigned bits[8]) {
  return cudaMemcpyAsync(d_allowed, bits, 32, cudaMemcpyHostToDevice, side) == cudaSuccess ? SMSHIFT_OK
                                                                                            : SMSHIFT_ERR_CUDA;
}

int smshift_fence(cudaStream_t stream, const unsigned* d_value, unsigned target) {
  smshift_arm_all();
  k_fence<<<1, 1, 0, stream>>>(d_value, target);
  return cudaGetLastError() == cudaSuccess ? SMSHIFT_OK : SMSHIFT_ERR_CUDA;
}

void smshift_probe_sms(const int* tpcs, int n, unsigned out[8]) {
  cudaMemsetAsync(s_scratch, 0, 32, s_probe_stream);
  smshift_arm(tpcs, n);
  k_mark_sms<<<2048, 128, 0, s_probe_stream>>>(s_scratch, 2000, s_sink);
  cudaMemcpyAsync(out, s_scratch, 32, cudaMemcpyDeviceToHost, s_probe_stream);
  cudaStreamSynchronize(s_probe_stream);
}

int smshift_tpc_sms(int bit, unsigned out[8]) {
  if (bit < 0 || bit >= 128) return SMSHIFT_ERR_ARG;
  int all[128], rest[127], nr = 0;
  for (int i = 0; i < 128; i++) {
    all[i] = i;
    if (i != bit) rest[nr++] = i;
  }
  unsigned with_bit[8], without_bit[8];
  smshift_probe_sms(all, 128, with_bit);
  smshift_probe_sms(rest, nr, without_bit);
  for (int i = 0; i < 8; i++) out[i] = with_bit[i] & ~without_bit[i];
  return cudaGetLastError() == cudaSuccess ? SMSHIFT_OK : SMSHIFT_ERR_CUDA;
}

int smshift_valid_tpcs(int* bits, int max) {
  int n = 0;
  for (int b = 0; b < 128 && n < max; b++) {
    unsigned sms[8];
    if (smshift_tpc_sms(b, sms) != SMSHIFT_OK) return SMSHIFT_ERR_CUDA;
    bool any = false;
    for (int i = 0; i < 8; i++) any |= sms[i] != 0;
    if (any) bits[n++] = b;
  }
  return n;
}

void smshift_cleanup(void) {
  if (s_scratch) cudaFree(s_scratch), s_scratch = nullptr;
  if (s_sink) cudaFree(s_sink), s_sink = nullptr;
  if (s_probe_stream) cudaStreamDestroy(s_probe_stream), s_probe_stream = nullptr;
  // The low-VA buffer lives for the process: siblings may still be cached by
  // the scheduler, and unmapping it under them is not safe.
  s_sibling = false;
}
