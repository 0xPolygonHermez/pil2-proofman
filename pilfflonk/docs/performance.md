# pilfflonk performance

Time and memory of the setup, the prover and the JS verifier on the CPU, and the GPU path with its
speed-up. Every proof in these measurements verifies.

## CPU

### Method

- **Machine.** 2 × AMD EPYC 7773X (256 threads, 2 NUMA nodes, 1.5 GB of L3), 1 TB of RAM, Ubuntu 22.04
  (Linux 5.15), GCC 11.4 with `-O3` and AVX2, libomp 14. Shared: each run records the machine's load,
  which ranged from 6 to 225.
- **Build.** `cargo build --release --features proofman-starks-lib-c/cpu-only`; OpenMP on all 256
  threads unless said otherwise.
- **Programs.** The fixtures with `N` as a parameter, in `pilfflonk/bench/`: `fibonacci` (stage 1: two
  columns and an im pol; 2 fixed; `qDeg = 1`) and `all_sum` (`all` on the sum bus: 14 committed
  columns of stage 1 and 6 of stage 2, 10 fixed, `qDeg = 2`), and `all_prod` at two sizes. Packed
  (`--extra-muls 2`) and `--no-packing`.
- **SRS.** A test ptau with a fixed public `τ` (good for measuring only), of 75,497,536 powers
  (`9·2^23 + 64`), 4.8 GB.
- **Phases.** The prover's C++ timers (`PILFFLONK_*`, logged at `-vv`): the key (`LOAD_SRS`,
  `LOAD_AIRS` with the INTT of the fixed columns, `FIXED_COMMITMENTS`), the instance, each stage
  (`STAGE_<s>`: `HINT_COLUMNS_<s>`, `IM_POLS_<s>`, and per `f` `INTT_<f>` and `COMMIT_<f>`, its MSM), `Q`
  (`Q_EXTEND`, `Q_DOMAIN`, `Q_EVALUATE`, `Q_INTERPOLATE`, `Q_COMMIT`), `EVALUATIONS` and the opening
  (`OPEN`: `SHPLONK_W`, `SHPLONK_COMMIT_W`, `SHPLONK_WP`, `SHPLONK_COMMIT_WP`). They cost nothing
  measurable. The memory of a phase is the peak `VmRSS`, sampled every 0.2 s.
- **Repetitions.** Three runs of each setup and proof (two at some points); the tables give medians.

### CPU results

| Program | N | Layout | Setup (s) | Setup (GB) | Prove (s) | Prove (GB) | Verify (s) | Proof (bytes) |
|---|---|---|---|---|---|---|---|---|
| `fibonacci` | 2^10 | packed | 0.34 | 0.0 | 1.38 | 0.0 | 0.30 | 704 |
| `fibonacci` | 2^16 | packed | 1.10 | 2.0 | 3.42 | 2.0 | 0.30 | 704 |
| `fibonacci` | 2^20 | packed | 2.30 | 2.6 | 9.62 | 2.7 | 0.31 | 704 |
| `fibonacci` | 2^22 | packed | 6.24 | 4.3 | 18.77 | 5.0 | 0.30 | 704 |
| `fibonacci` | 2^24 | packed | 20.18 | 11.2 | 59.12 | 14.0 | 0.30 | 704 |
| `fibonacci` | 2^24 | `--no-packing` | 16.70 | 7.8 | 55.45 | 13.0 | 0.30 | 672 |
| `all_sum` | 2^16 | packed | 1.91 | 2.2 | 9.54 | 2.3 | 0.31 | 1,760 |
| `all_sum` | 2^20 | packed | 8.57 | 5.0 | 26.13 | 5.8 | 0.32 | 1,760 |
| `all_sum` | 2^20 | `--no-packing` | 12.17 | 3.1 | 42.94 | 5.1 | 0.32 | 2,624 |
| `all_sum` | 2^22 | packed | 27.19 | 13.8 | 76.00 | 18.1 | 0.31 | 1,760 |
| `all_sum` | 2^23 | packed | 53.88 | 25.6 | 137.83 | 36.3 | 0.32 | 1,760 |
| `all_prod` | 2^20 | packed | 14.69 | 4.3 | 33.90 | 5.6 | 0.32 | 1,664 |

- **The proof and the verifier do not depend on `N`.** The proof's size depends only on the number of
  `f` and of evaluations. The JS verifier takes 0.30–0.33 s at every size, nearly all of it starting
  Node and ffjavascript.
- **Packing** hardly changes the `fibonacci`, which only packs its two fixed columns, and makes
  `all_sum` 17–50 % faster (9 `f` instead of 31: fewer MSMs), for an SRS 4.5 times larger and twice the
  setup's memory.
- **Growth.** Below `2^18` time barely grows: it is mostly MSMs, each about 100 ms even with 2,000
  points, as ffiasm's MSM opens some 300 OpenMP regions with 256 threads. From `2^20` to `2^24` it grows
  less than `N log N`, as the MSMs are linear in `N`. Memory grows linearly from `2^20`: about 900 bytes
  per row for the `fibonacci` and 4.6 KB per row for `all_sum`.

**Where the time goes** at the largest sizes (packed, 256 threads):

- MSMs, 36–38 % (up to 50 % unpacked): about 6 million points per second. ffiasm keeps 2^16 buckets of
  96 bytes per thread, 1.6 GB with 256 threads, nearly all the memory of a proof up to `2^18`.
- `W` and `W'`, 16–18 %: a sequential `divByMonic` per `f` and offset, the packing and the sums.
- Loading the key, 16–18 %: the SRS, the `.const`, the INTT of the fixed columns and, mostly,
  recomputing the fixed commitments on every proof (3.2 s at `fibonacci` `2^24`).
- `Q`, 20 %: the LDE of its columns, zeroing buffers, the domain and the MSM; the bytecode itself takes
  0.3–0.5 s.
- The witness, 5–9 %: reading 1–2 GB and checking each value three times.

**Threads.** With `OMP_NUM_THREADS=64`, small and medium proofs are 2 to 7 times faster and use 4 times
less memory, as the MSM pays fewer regions and buckets (`fibonacci` `2^10`: 1.38 s → 0.18 s); at `2^22`
and `2^24` both take the same.

**The limit is the compiler.** The prover reaches `N = 2^24`, and nothing suggests it would not with
larger programs (`all_sum` at `2^24` would need about 5 minutes and 75 GB). pil2com does not: `all`
keeps four full-width fixed columns in the pilout, and at `2^24` the pilout is 2.48 GB, above what Node
writes in one call (`2^31 − 1`) and above protobuf's 2 GiB, after 2 h 14 min of compiling; at `2^23` it
takes 48 minutes and 40 GB. Any program with a std connection has that limit.

### Q in parts

Evaluating `Q` on the whole coset ([protocol.md#q-in-parts](protocol.md#q-in-parts)) extended every
column `Q` reads to `N'` points: 16 GB of extended columns for `all_sum` at `2^22`, four times all the
committed polynomials. One coset of `H` at a time:

| Program | N | Layout | Peak of `Q` (GB) | Peak of the proof (GB) | `Q` (s) | Prove (s) |
|---|---|---|---|---|---|---|
| `fibonacci` | 2^24 | packed | 16.0 → 13.8 | 16.0 → 14.0 | 14.58 → 11.74 | 63.07 → 59.12 |
| `all_sum` | 2^20 | packed | 7.1 → 4.3 | 7.2 → 5.8 | 8.17 → 4.31 | 31.40 → 26.13 |
| `all_sum` | 2^22 | `--no-packing` | 26.8 → 15.4 | 26.8 → 15.4 | 28.83 → 15.18 | 108.06 → 91.42 |
| `all_sum` | 2^22 | packed | 28.5 → 17.1 | 28.5 → 18.1 | 27.23 → 15.59 | 88.77 → 76.00 |

The memory of `Q` drops 40 % for `all_sum` and 15 % for the `fibonacci`, and `Q` is no longer the
proof's peak.

### Not implemented

None changes the proof or the protocol: an MSM that scales (shared buckets, a window by size and
threads, or the GPU's); fewer threads per MSM by its number of points (1.3–7× from `2^10` to `2^20`);
not recomputing the fixed commitments on every proof (5–7 % at `2^22`–`2^24`); a parallel division for
`W` and `W'`; reserving `Q`'s buffers without zeroing them; a single canonicity check of the witness.

## GPU

### What runs on the GPU

`proofman-cli pilfflonk prove -g/--gpu` runs the MSMs and the NTTs of the key and of the proof on the
GPU; everything else stays where it is. No MSM, NTT or kernel of its own: `PilFflonk::Gpu`
(`pil2-stark/src/pilfflonk/pilfflonk_gpu.{hpp,cpp}`) only calls the GPU entry points pil2-stark has,
as the PLONK GPU prover calls them (`rapidsnark/plonk_prover_gpu.c.cuh`):

- `msm_bn128_gpu_dev_ptr` (`bn128/src/msm/msm_bn128.cu`, sppark's Pippenger) with `montgomery = true`,
  on a copy of the SRS's powers `[τ^i]₁` kept on the device;
- `ntt_bn128_gpu_dev_ptr` and `intt_bn128_gpu_dev_ptr` (`bn128/src/ntt/ntt_bn128.cu`, sppark's NTT), in
  natural order in and out;
- `gpu_plonk_cuda_malloc`, `_free`, `gpu_plonk_memcpy_h2d`, `_d2h`, `gpu_plonk_cuda_device_sync` and
  `gpu_plonk_set_device` (`rapidsnark/plonk_prover.cu`) for the device's memory, and sppark's
  `cuda_available` to know whether there is a GPU.

Two points of the prover go to the GPU, and only two:

- `Srs::commit`, every MSM: the commitments of each stage, of `Q`, of `W` and `W'`, and the fixed
  commitments checked when the key is loaded;
- the transforms of `Lde`: the INTT of `Lde::intt` (each stage's columns and, at load, the fixed ones),
  the FFT of each part in `extendCosetPart`, and the INTT of `interpolateCoset`.

The elementwise work around the transforms (the coset shift and the folding of each part, the scaling
by `g^−j`, trimming a polynomial to its degree), the blinding and the packing stay on the CPU, so the
data goes to the device and back on every transform. The PLONK GPU prover has no coset helper on the
device, and one would be a new kernel. The interpreter, the SHPLONK divisions, the evaluations at `ξ`,
the hints, the im pols and reading the files stay on the CPU too.

### Why the proof is the same

sppark's NTT roots for BN254 are ffiasm's, `ω_{2^k} = 5^((r−1)/2^k)` for `k ≤ 28`, in the same
Montgomery form (its `group_gen` is 5, ffiasm's `nqr`), and its inverse scales by `1/2^k`, as ffiasm's
`ifft`. Both sides compute in exact field arithmetic and leave every element in its canonical
Montgomery form, below `r`. The MSM returns a Jacobian point `(X, Y, Z)`, ffiasm's extended
`(X, Y, Z², Z³)`, and a commitment leaves the prover in affine coordinates, which are unique. So every
value is the same element at the same place, and the proof is the same bit for bit.

### Selection, memory and errors

- **Selection.** `-g/--gpu`, the CPU by default, in a build that found `nvcc`
  (`provers/starks-lib-c/build.rs`; never with `--features proofman-starks-lib-c/cpu-only`). It does
  not go through the STARK backend's global switch (`set_gpu_mode`), whose `get_num_gpus` exits on a
  machine without a driver: the device is given explicitly, `ProvingKey::load_on(dir, Device::Gpu)`
  (Rust), `pilfflonk_ctx_new_on(dir, PILFFLONK_DEVICE_GPU)` (C), `ProvingKey::load(dir, Device::Gpu)`
  (C++). `proofman_pilfflonk::gpu_available()` (`pilfflonk_gpu_available`) says whether there is a GPU.
  Without one (a CPU library, or no GPU), `--gpu` is refused before the SRS is read. A GPU library
  without `--gpu` is the CPU, and the CPU library has no GPU code at all (`__USE_CUDA__`).
- **Memory.** The SRS is copied to the device once, when the key loads (timer `PILFFLONK_GPU_SRS`), for
  every MSM of the key and its proofs, as the PLONK prover's `d_ptau`. One device buffer, grown to the
  largest request, holds the scalars of an MSM or the data of an NTT; every call holds a lock on it, so a
  key shared by several threads stays safe. Device 0. The device holds `64·nG1` bytes of SRS, 32 bytes
  per buffer element (up to `max(nG1, N')`), and what sppark's MSM reserves per call.
- **Errors.** A CUDA failure inside the reused helpers (no device memory, a lost device) aborts the
  process, as in the PLONK GPU prover (`CHECKCUDAERR`): the one exception to the C API's rule. sppark's
  MSM reports its own failure as the point at infinity; `Gpu::msm` turns that, for scalars not all zero,
  into an error rather than a proof that does not verify.

### GPU results

RTX 5090 (`sm_120`), CUDA 13.0, 32 CPU threads (`OMP_NUM_THREADS=32`), packed keys, the median of three
proofs on each device with the same `--insecure-blinding-seed`. Every GPU proof is the CPU's byte for
byte, and the JS verifier accepts them. Seconds, CPU → GPU:

| Program | N | Proof | Fixed commitments | Stage MSMs | `Q` MSM | `W`, `W'` MSMs | Opening |
|---|---|---|---|---|---|---|---|
| `fibonacci` | 2^16 | 0.74 → 0.52 (1.42×) | 0.19 → 0.21 | 0.205 → 0.012 | 0.068 → 0.003 | 0.18 → 0.011 | 0.21 → 0.03 |
| `fibonacci` | 2^18 | 1.95 → 1.34 (1.46×) | 0.26 → 0.81 | 0.581 → 0.035 | 0.177 → 0.011 | 0.58 → 0.022 | 0.67 → 0.11 |
| `fibonacci` | 2^20 | 5.67 → 4.87 (1.16×) | 0.67 → 3.49 | 1.515 → 0.062 | 0.484 → 0.019 | 1.65 → 0.036 | 2.04 → 0.43 |
| `fibonacci` | 2^22 | 18.02 → 17.17 (1.05×) | 2.02 → 12.95 | 4.155 → 0.173 | 1.374 → 0.059 | 4.81 → 0.094 | 6.47 → 1.59 |
| `all_sum` | 2^16 | 2.57 → 1.22 (2.11×) | 0.38 → 0.42 | 0.940 → 0.053 | 0.132 → 0.006 | 0.58 → 0.024 | 0.68 → 0.13 |
| `all_sum` | 2^18 | 7.30 → 3.89 (1.88×) | 0.96 → 1.91 | 2.377 → 0.111 | 0.266 → 0.016 | 1.77 → 0.039 | 2.26 → 0.51 |
| `all_sum` | 2^20 | 23.47 → 12.97 (1.81×) | 2.68 → 6.80 | 6.800 → 0.268 | 0.856 → 0.033 | 5.36 → 0.103 | 7.08 → 1.90 |
| `all_prod` | 2^18 | 7.26 → 3.82 (1.90×) | 1.08 → 1.91 | 2.264 → 0.089 | 0.396 → 0.013 | 1.65 → 0.036 | 2.10 → 0.50 |

- **The MSMs** of the stages, of `Q` and of `W` and `W'` are 15–52× faster, and the transforms of the
  stages and of `Q` 2–7×.
- **The whole proof** is 1.05–2.1× faster: what stays on the CPU (the SHPLONK divisions, the
  interpreter, the transfers around each transform, the witness) and the check below dominate.
- **The fixed-commitment check** when the key loads is up to 6.4× *slower* on the GPU: 12.95 s against
  2.02 s at `fibonacci` `2^22`, where it costs as much as the MSMs save. This is open and under
  investigation.

### Open

- The fixed-commitment check on the GPU, above.
- A coset on the device: the transforms of `Lde` make a host-to-device round trip per column, as the
  elementwise work is on the CPU. sppark has `NTT::Type::coset` with `group_gen = 5`, but no entry point
  of `ntt_bn128.cu` exposes it: that needs a new entry point or a kernel.
- Pinned, asynchronous copies, as the PLONK GPU prover overlaps its copies with the computation; this
  path copies synchronously from pageable memory.
- The interpreter stays on the CPU, as it does not dominate.

## Reproducing

**CPU** (`pilfflonk/bench/bench.sh`; its header lists the variables `BENCH_DIR`, `BENCH_PTAU`,
`BENCH_PACKING`, `BENCH_REPEATS`, `BENCH_KEEP`, `BENCH_NODE_HEAP_MB` and `OMP_NUM_THREADS`):

```sh
cargo build --release --features proofman-starks-lib-c/cpu-only \
    --bin proofman-cli --bin proofman-setup --example pilfflonk_bench_inputs
export BENCH_DIR=/tmp/pilfflonk-bench PIL2C_EXEC=<pil2-compiler>/src/pil.js
pilfflonk/bench/bench.sh ptau 75497536          # 9·2^23 + 64 powers: 4.8 GB, about 20 min
BENCH_PACKING="packed nopacking" BENCH_REPEATS=3 pilfflonk/bench/bench.sh run fibonacci 10 12 14 16 18 20 22 24
BENCH_PACKING="packed nopacking" BENCH_REPEATS=3 pilfflonk/bench/bench.sh run all_sum 10 12 14 16 18 20
pilfflonk/bench/bench.sh summary
```

`run` compiles once (the pilouts stay in `$BENCH_DIR/pilouts`), sets up, writes the witness with the
example `pilfflonk_bench_inputs` (`pilfflonk/bench/inputs.rs`), proves and verifies, and writes
`$BENCH_DIR/{compile,setup,prove}.tsv`; `summary` (`summary.mjs`) makes the tables. Each size deletes
its keys, witness and proofs unless `BENCH_KEEP=1`. Check the disk first: the `fibonacci` at `2^24`
needs about 5 GB besides the ptau, `all_sum` at `2^23` about 13 GB.

**GPU** (`pilfflonk/bench/gpu_check.sh`):

1. Make the key and the witness with the CPU tools: after `bench.sh ptau <powers>`,
   `BENCH_KEEP=1 BENCH_REPEATS=1 pilfflonk/bench/bench.sh run fibonacci <bits>` leaves the key in
   `$BENCH_DIR/<program>/<bits>/build_packed/provingKey` and the witness in
   `$BENCH_DIR/<program>/<bits>/witness`.
2. On the GPU machine, from the repository root:

   ```sh
   export PATH=/usr/local/cuda/bin:$PATH
   CUDA_ARCHS=120 cargo build --release -p proofman-cli
   CUDA_ARCHS=120 PILFFLONK_GPU=1 make -C pil2-stark pilfflonk_gpu_test
   OMP_NUM_THREADS=<threads> pilfflonk/bench/gpu_check.sh <provingKey> <witness>
   ```

   `cargo build` goes first: `build.rs` initialises the submodules `external/{sppark,blst}` and builds
   blst, which the test links too. `CUDA_ARCHS=120` builds for the RTX 5090 only; without it the
   Makefile detects the architecture with `nvidia-smi`, or builds for all majors. Use the same
   `CUDA_ARCHS` for both commands: a change of gencode rebuilds every GPU object.
   `pilfflonk_gpu_test` compares the GPU with the CPU byte for byte (the NTTs, an `Lde`, `Srs::commit`
   around sppark's window sizes, whole proofs); without a GPU it skips them and says so, and with
   `PILFFLONK_GPU=1` a skip fails. `gpu_check.sh` proves on each device with the same blinding seed
   (`GPU_CHECK_SEED`, `GPU_CHECK_REPEATS` times), compares every `proof.json` and `publics.json` with the
   first CPU proof's, prints each phase's median, and exits with 0 only if every proof is the same.
3. Back on a machine with Node, verify the GPU proofs:
   `proofman-cli pilfflonk verify <vkey> <out>/gpu.1/publics.json <out>/gpu.1/proof.json`.
