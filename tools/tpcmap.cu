// tpcmap - print which SMs sit behind each TPC_DISABLE_MASK bit.
// Build: make tools. Run on an idle GPU; it launches a few hundred short kernels.
#include <cuda_runtime.h>

#include <cstdio>

#include "smshift.h"

int main() {
  if (cudaSetDevice(0) != cudaSuccess) return 1;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const int rc = smshift_init(false);
  if (rc != SMSHIFT_OK) {
    fprintf(stderr, "smshift_init failed (%d)\n", rc);
    return 1;
  }
  printf("%s, %d SMs\n", prop.name, prop.multiProcessorCount);
  unsigned all[8] = {0};
  int n_tpc = 0;
  for (int bit = 0; bit < 128; bit++) {
    unsigned sms[8];
    if (smshift_tpc_sms(bit, sms) != SMSHIFT_OK) {
      fprintf(stderr, "probe of bit %d failed\n", bit);
      return 1;
    }
    int n = 0;
    for (int i = 0; i < 256; i++) n += (sms[i / 32] >> (i % 32)) & 1;
    if (!n) continue;
    n_tpc++;
    printf("bit %3d -> SM", bit);
    for (int i = 0; i < 256; i++)
      if ((sms[i / 32] >> (i % 32)) & 1) printf(" %d", i);
    printf("\n");
    for (int i = 0; i < 8; i++) all[i] |= sms[i];
  }
  int covered = 0;
  for (int i = 0; i < 256; i++) covered += (all[i / 32] >> (i % 32)) & 1;
  printf("%d TPC bits with SMs, %d of %d SMs covered%s\n", n_tpc, covered, prop.multiProcessorCount,
         covered == prop.multiProcessorCount ? "" : " (INCOMPLETE)");
  smshift_cleanup();
  return covered == prop.multiProcessorCount ? 0 : 1;
}
