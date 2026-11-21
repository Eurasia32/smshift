# smshift design notes

Everything here was observed on H100 80GB HBM3, driver 590.48.01, CUDA 12.6. Field names and bit positions come from `clcbc0qmd.h` (QMD V04_00) in NVIDIA's open-gpu-doc. The behaviour described is what we measured, not documented semantics.

## 1. Intercepting launches

`cuGetExportTable` with UUID `2c8e0ad8-0710-ab4e-90dd-54719fe5f74b` returns a table of libcuda debug-callback functions (as used by libsmctrl): entry 3 subscribes a callback, entry 6 enables it for `(domain, cbid)`. Domain `0xb` carries QMD events:

| cbid | when | parameters (64-bit words) |
|---|---|---|
| 1 | before upload | `[4]` = host staging copy of the QMD (editable) |
| 2 | after upload | `[5]` = GPU VA of the uploaded QMD |

cbids 0 and 3 can be enabled but never fired for ordinary launches.

The callbacks run inside `cudaLaunchKernel` on the launching thread, so the per-launch edit is thread-local. The QMD version byte is at offset 72 (`0x40` = V04_00); any other value leaves the QMD untouched.

## 2. QMD fields used

| field | bits | use |
|---|---|---|
| `TPC_DISABLE_MASK_VALID` | 31 | must be set for the mask to apply |
| `TPC_DISABLE_MASK(0..3)` | 2432-2559 (words 76-79) | 1 = TPC disabled |
| `GRID_WIDTH` | 1024-1055 (word 32) | sibling grid width |
| `GRID_WIDTH_RESUME` / `_HEIGHT_` / `_DEPTH_` | 1408-1471 (words 44-45) | first block of a new instance |
| `DEPENDENT_QMD_ENABLE/ACTION(1)`, `DEPENDENT_QMD1_POINTER` | 373, 374-376, 480-511 | helper kernel triggers `SCHEDULE` on the target |
| `RELEASE_ENABLE(0..2)`, `DEPENDENT_QMD_ENABLE(0)` | 320, 336, 352, 368 | cleared on siblings ("silent") |
| `CONSTANT_BUFFER_VALID(i)` | word 13 bit 4i | which constant banks are valid |
| `CONSTANT_BUFFER_ADDR_LOWER/UPPER_SHIFTED6(i)`, `SIZE_SHIFTED4(i)` | word 48+2i; word 49+2i bits 0-18 / 19-31 | parameter (CB0) relocation |

libcuda itself uses `DEPENDENT_QMD0` (action `DECREMENT_DEPENDENCE`) for stream ordering, so smshift uses slot 1.

`TPC_DISABLE_MASK` bits do not map to SMs contiguously, and some bits have no TPC. `smshift_tpc_sms(bit)` finds the SMs behind a bit by launching a probe kernel with and without it.

## 3. Re-dispatching a running launch

Rewriting the mask in the running QMD's memory has no effect: the scheduler keeps a cached copy. What works:

1. A helper kernel writes the target QMD (new mask, `GRID_WIDTH_RESUME`).
2. A second, 1-thread helper kernel is launched with `DEPENDENT_QMD1_POINTER` = target and action `QMD_SCHEDULE`. When it completes, the scheduler fetches the target QMD again and dispatches it as an additional instance, starting at the resume block.

The original instance keeps dispatching its own remaining blocks. Hence the kernel-side protocol (allowed bitmap + claims) and `smshift_fence`.

Measured: the first CTA on a new SM starts 10-25 µs after the spawn kernel runs. On revoked SMs, the original instance's remaining CTAs drained within ~3 ms (each exits at once).

## 4. Sibling QMDs and parameter templates

Scheduling the original QMD a second time puts two active instances on one QMD. When one completes, the scheduler auto-invalidates the QMD while the other is still active → **Xid 13 `SKEDCHECK22_INVALIDATE_ACTIVE_QMD`**. smshift therefore schedules a copy (sibling) of the QMD:

- `DEPENDENT_QMD*_POINTER` holds `addr >> 8` in 32 bits, so the copy must sit below 1 TB of GPU VA. The public allocation APIs do not return such addresses for device memory. smshift maps host memory at a fixed low address with `mmap` and registers it (`cuMemHostRegister`); under UVA its device VA equals its host VA. The scheduler reads QMDs from system memory without problems.
- CB0 holds the launch's parameters. CUDA reuses it soon after the original instance completes: with Triton launches, within ~10 launches on the same stream. On the first spawn of a launch, smshift copies QMD + CB0 into a template and redirects every valid constant bank that points at CB0 (CB5 aliases CB0 on every launch observed). All siblings are made from the template. Controls (`SMSHIFT_TEST_NO_CB_COPY=1`): without the copy, 9096 of 12000 CTAs of a blockIdx kernel, and all CTAs of a second Triton GEMM sibling, read another launch's parameters.
- Siblings are "silent": release semaphores and `DEPENDENT_QMD0` are cleared, so only the original instance signals the stream.
- `HW_ONLY_*` words (5-9) are cleared; word 8 (`HW_ONLY_SKED_NEXT_QMD_POINTER`) is set to the copy's own `addr >> 8`, as observed on uploaded QMDs.

## 5. Resume point

`GRID_WIDTH_RESUME` must stay within the grid, or the scheduler raises **Xid 13 `SKEDCHECK24_CTA_RESUME`**. Compute it from the highest block that started real work (`d_maxb`), not from a count of started CTAs, which includes CTAs that only skipped.

## 6. Completion

The original instance's completion releases the stream. With the protocol above, its remaining CTAs exit early after a shrink, so it can complete before siblings finish. `smshift_fence` launches a 1-thread kernel on the same stream that waits until the work counter reaches the target. It runs with a valid all-TPC mask, so it is not serialized behind masked grids.

## 7. Scheduling behaviour that matters in practice

- Two grids with disjoint valid masks run concurrently, even while the first still has blocks to dispatch. Grids with the same mask, or without a valid mask, are dispatched in FIFO order.
- With lazy module loading, the first launch of a kernel waits for running kernels. Use `CUDA_MODULE_LOADING=EAGER`.
- An SM holding resident CTAs cannot change its L1/shared-memory split. A kernel with little shared memory that is resident first blocks a large-shared-memory kernel on the same SMs; setting the small kernel's preferred carveout to 100 avoids this.

## 8. Hardware checks seen (Xid 13 "Graphics Exception")

| check | cause |
|---|---|
| `SKEDCHECK22_INVALIDATE_ACTIVE_QMD` | a QMD completed (auto-invalidate) while another instance of it was active |
| `SKEDCHECK24_CTA_RESUME` | `GRID_WIDTH_RESUME` beyond the grid |

In every case only the process that caused it lost its context. Check `journalctl -k | grep Xid` after experiments.
