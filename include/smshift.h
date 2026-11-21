// smshift.h - Runtime SM re-allocation via QMD rewriting (Hopper H100, QMD V04_00).
//
// What the library does:
//   - launch a kernel restricted to a TPC set (TPC_DISABLE_MASK in its QMD);
//   - while it runs, spawn another hardware instance of the same launch on a
//     different TPC set, starting at a chosen blockIdx (GRID_WIDTH_RESUME), via
//     DEPENDENT_QMD_ACTION=SCHEDULE from a helper kernel;
//   - a fence kernel that keeps the launch's stream from completing before all
//     instances' work is done.
//
// What the user's kernel must do (the hardware does not stop old instances):
//   - CTAs on SMs outside the allowed-SM bitmap exit without doing work
//     (smshift_allowed());
//   - work must be claimed so that two instances never do the same work twice:
//     either a per-blockIdx claim flag, or a persistent loop pulling items from
//     a global counter;
//   - count finished work, and put smshift_fence() after the launch.
// See test_smshift.cu (blockIdx kernels) and pd_demo.cu (persistent kernels).
//
// Sibling mode (recommended): each spawn schedules a private copy of the QMD,
// so no QMD has two active instances (Xid 13 SKEDCHECK22_INVALIDATE_ACTIVE_QMD).
// On the first spawn of a launch, its QMD and CB0 (kernel parameters) are
// snapshotted into a template; all siblings use the template, so they stay
// valid after the original instance completes and CUDA reuses its CB0.
//
// Threading: launch interception state is thread-local; all spawns for one
// launch must be issued on the same side stream.
#pragma once
#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {  // C ABI so the library can be loaded from Python (ctypes)
#endif

// Return codes of the int-returning functions.
enum {
  SMSHIFT_OK = 0,
  SMSHIFT_ERR_CUDA = -1,         // a CUDA call failed
  SMSHIFT_ERR_UNSUPPORTED = -2,  // not compute capability 9.x, or the driver hook is unavailable
  SMSHIFT_ERR_QMD_VERSION = -3,  // a launch used a QMD layout other than V04_00 (left unmodified)
  SMSHIFT_ERR_ARG = -4,          // invalid argument
  SMSHIFT_ERR_NO_LOW_VA = -5,    // sibling mode: no GPU VA below 1 TB could be mapped
};

// ---- Lifecycle ----

// Install the QMD debug callback and allocate internal buffers. In sibling
// mode also map an 8 MB buffer below 1 TB GPU VA (DEPENDENT_QMD*_POINTER is
// addr >> 8 in 32 bits). Call once with an active CUDA context.
int smshift_init(bool sibling);
void smshift_cleanup(void);

// ---- Launch interception ----

// Arm the next kernel launch from this thread to run only on the given TPCs.
// smshift_arm_all() sets a valid mask with every TPC enabled; launches armed
// like this are not serialized behind other masked grids.
void smshift_arm(const int* tpcs, int n);
void smshift_arm_all(void);

// GPU VA of the QMD uploaded by this thread's most recent launch.
uint64_t smshift_last_va(void);

// A launch that can be spawned again.
typedef struct smshift_kernel {
  uint64_t qmd_va;
  int tmpl;  // template slot, -1 until the first spawn
} smshift_kernel;

// arm + cudaLaunchKernel. `out` (nullable) receives the launch handle.
int smshift_launch(cudaStream_t stream, const void* func, dim3 grid, dim3 block, size_t shmem, void** args,
                  const int* tpcs, int n, smshift_kernel* out);

// ---- Spawning another instance ----

typedef struct smshift_spawn_args {
  const int* tpcs;              // hardware TPC set of the new instance
  int n;
  unsigned resume;              // first blockIdx of the new instance, used if d_maxb is NULL
  const unsigned* d_maxb;       // if set: resume = *d_maxb + 1 - margin (read on the GPU)
  unsigned margin;
  const unsigned* allowed_bits; // host, 8 words: written to d_allowed before the spawn; NULL = keep
  unsigned* d_allowed;
  const unsigned* d_counter;    // optional: snapshot into d_at[0]
  unsigned* d_at;               // optional, 4 words: counter, resume, globaltimer (64-bit)
  bool silent;                  // clear release semaphores and DEPENDENT_QMD0 on the new instance
  unsigned grid_width;          // 0 = keep; else GRID_WIDTH of the new instance (sibling mode only).
                                // The kernel must not depend on gridDim.x, which still reports
                                // the original launch's value.
} smshift_spawn_args;

// Asynchronous on `side`. The first spawn of a launch must happen while the
// original instance is still running (it snapshots QMD and CB0). 0 on success.
int smshift_spawn(cudaStream_t side, smshift_kernel* k, const smshift_spawn_args* a);

// Write the allowed-SM bitmap (8 words) without spawning (shrink).
int smshift_set_allowed(cudaStream_t side, unsigned* d_allowed, const unsigned bits[8]);

// Launch a 1-thread kernel on `stream` that finishes when *d_value >= target.
int smshift_fence(cudaStream_t stream, const unsigned* d_value, unsigned target);

// ---- Utility ----

// SMs reachable from a TPC set, as an 8-word bitmap. Synchronizes internally.
void smshift_probe_sms(const int* tpcs, int n, unsigned out[8]);

// SMs behind one TPC_DISABLE_MASK bit (0..127); all zero if the bit has no
// TPC. The bit layout has gaps (on H100 SXM, 66 TPCs sit in bits 0..83).
int smshift_tpc_sms(int bit, unsigned out[8]);

// Fill `bits` with the mask bits that have SMs, in increasing order. Returns
// the count (>= 0) or an error code. Runs 256 short probe kernels.
int smshift_valid_tpcs(int* bits, int max);

#ifdef __cplusplus
}
#endif

// ---- Device helpers ----

#ifdef __CUDACC__
__device__ inline unsigned smshift_smid() {
  unsigned r;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}
__device__ inline unsigned long long smshift_globaltimer() {
  unsigned long long r;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(r));
  return r;
}
// True if the calling CTA's SM is in the allowed bitmap (8 words).
__device__ inline bool smshift_allowed(const unsigned* allowed) {
  unsigned sm = smshift_smid();
  return (*(volatile const unsigned*)&allowed[sm / 32] >> (sm % 32)) & 1u;
}
#endif
