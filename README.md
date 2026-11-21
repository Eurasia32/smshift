# smshift

Re-allocate the SMs of a **running** CUDA kernel on NVIDIA Hopper, without relaunching it.

[libsmctrl](#credits) restricts a kernel to a set of TPCs when it is launched. smshift goes further: after launch, it can move a kernel onto more, fewer, or different TPCs while the kernel keeps running. It edits the kernel's QMD (the hardware launch descriptor) and asks the hardware scheduler to dispatch the remaining work again.

> **Status: research prototype.** smshift relies on undocumented driver and hardware behaviour, observed on one machine. A wrong QMD can raise a GPU exception (Xid 13). So far such an exception has only ended the offending process's context. Do not run smshift next to work you cannot afford to lose.

## What it does

- **Launch on a TPC set**: `smshift_launch` / `smshift_arm` sets `TPC_DISABLE_MASK` in the launch's QMD.
- **Grow / move at runtime**: `smshift_spawn` makes the hardware dispatch another instance of the same launch, on another TPC set. The new instance starts at a chosen `blockIdx` (`GRID_WIDTH_RESUME`) or with its own grid width. It runs from a private copy of the QMD and of the kernel parameters (CB0), so it stays valid after the original instance completes.
- **Shrink at runtime**: the kernel checks an allowed-SM bitmap in device memory, so `smshift_set_allowed` alone makes CTAs on revoked SMs exit.
- **Correct completion**: `smshift_fence` keeps the stream from completing before the work of all instances is done.
- **Topology**: `smshift_valid_tpcs` / `tools/tpcmap` find which mask bits have SMs. The layout has gaps: on H100 SXM, 66 TPCs sit in bits 0..83.

Measured on H100 (details in [docs/design.md](docs/design.md)):
- Growing a running grid from 8 to 32 SMs takes 10-25 µs until the first CTA starts on a new SM.
- In every test, each block did its work exactly once, through repeated grow / shrink / move steps.
- This holds for a Triton persistent GEMM with TMA descriptors (TMA + wgmma) as well as for plain CUDA kernels.

## Requirements

- Compute capability 9.x (Hopper). Only QMD version V04_00 is supported; launches with another QMD layout are left untouched, and calls return `SMSHIFT_ERR_QMD_VERSION`.
- Tested with: H100 80GB HBM3 (SXM module), driver 590.48.01, CUDA 12.6, Linux x86_64. Other drivers may change the internal interfaces used here.
- No root access, no kernel module.
- Python binding (optional): any Python 3 with `ctypes`. The examples need torch and triton (tested with torch 2.9.1 and triton 3.5.1).

## Build and check

```sh
make                 # build/libsmshift.{so,a}, tests, tools, examples
make check           # needs an idle GPU: TPC map + migration tests
```

`CUDA=/path/to/cuda` and `ARCH=sm_90` can be overridden.

Run every program with `CUDA_MODULE_LOADING=EAGER`. With lazy loading, the first launch of a kernel waits for running kernels to finish, which breaks concurrent partitions. The tests and examples set this themselves.

## Using it

### Kernel side

The hardware does not stop an old instance, so the kernel itself has to cooperate:

1. A CTA on an SM outside the allowed bitmap exits without doing work: `if (!smshift_allowed(allowed)) return;`.
2. Work is claimed, so two instances never do the same work. Use either a per-`blockIdx` claim flag, or a persistent loop that pulls work items from a global counter (the tile-scheduler structure of persistent GEMM kernels).
3. Finished work is counted, and the launch is followed by `smshift_fence(stream, d_done, total)`.

With a spawned `grid_width`, `blockIdx.x` can exceed the original grid while `gridDim.x` still reports the original value. Kernels that pull work from a counter do not depend on either.

### Host side (C)

```c
smshift_init(/*sibling=*/true);
int bits[128];
int n = smshift_valid_tpcs(bits, 128);           // usable TPC mask bits

smshift_kernel k;
smshift_launch(stream, (const void*)my_kernel, grid, block, 0, args, bits, 4, &k);  // first 4 TPCs

// later, while it runs: grow onto TPCs 4..19
unsigned sms[8]; smshift_probe_sms(bits, 20, sms);
smshift_spawn_args a = {0};
a.tpcs = bits + 4; a.n = 16;
a.d_maxb = d_maxb; a.margin = 256;              // resume after the highest claimed block
a.allowed_bits = sms; a.d_allowed = d_allowed;  // allow all 20 TPCs
a.silent = true;
smshift_spawn(side_stream, &k, &a);

smshift_fence(stream, d_done, total_blocks);
```

See `tests/test_smshift.cu` for a blockIdx-claim kernel and `examples/pd_demo.cu` for a persistent kernel.

### Python

`python/smshift.py` binds the same API with `ctypes`. It looks for the library in `$SMSHIFT_LIB`, then in `../build/`, then on the system library path. `python/examples/test_triton_gemm.py` runs a persistent Triton GEMM and migrates it while it runs; `triton_kernels.py` shows the kernel-side protocol in Triton.

Interception state is thread-local: `arm()` and the launch it applies to must run on the same thread.

## Examples

- `examples/pd_demo.cu` co-locates a persistent "prefill" kernel and periodic "decode" kernels on one GPU. It compares static partitions with runtime re-allocation (synthetic spin kernels).
- `python/examples/test_triton_gemm.py` migrates a persistent Triton GEMM. Its `migrate` mode with `SMSHIFT_TEST_NO_CB_COPY=1` is the control showing why kernel parameters are copied.

## Limitations

- Shrinking takes effect at the kernel's next check, e.g. after the current work item.
- The hardware does not isolate partitions completely. Co-located compute-bound kernels on disjoint SMs still share L2, HBM bandwidth and the power budget.
- Ring buffers in the library are reused: 2048 sibling QMDs and 64 parameter templates. A launch that lives through more spawns than that is not supported.
- Only one GPU (device 0 context) has been tested.

## How it works

See [docs/design.md](docs/design.md): the QMD fields used, the launch callbacks, sibling QMDs and parameter templates, and the hardware checks that fail when something is wrong.

## Credits

- The launch interception uses the libcuda debug-callback entry point that **libsmctrl** (Joshua Bakita, UNC) found and documented: J. Bakita and J. H. Anderson, "Hardware Compute Partitioning on NVIDIA GPUs", RTAS 2023. smshift does not contain libsmctrl code.
- QMD field positions come from NVIDIA's [open-gpu-doc](https://github.com/NVIDIA/open-gpu-doc) (`classes/compute/clcbc0qmd.h`).

## License

MIT, see [LICENSE](LICENSE).
