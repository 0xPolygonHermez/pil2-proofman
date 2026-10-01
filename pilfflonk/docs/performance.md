# pilfflonk performance

Time and memory of the setup, the prover and the JS verifier on the CPU, the GPU path with its
speed-up, and the final SNARK wrap with pilfflonk against FFLONK and PLONK. Every proof in these
measurements verifies.

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
  measurable. The Rust side logs its own the same way: `KEY_FILES` (the globalInfo, the vkey and the
  pilfflonkinfo), `WITNESS_READ` (a witness directory's trace and its range check, read while the key
  loads: [the start of a proof](#the-start-of-a-proof)) and `WRITE_PROOF`; with `--gpu`, `GPU_INIT`
  is what is left of CUDA's initialisation. The memory of a phase is the peak `VmRSS`, sampled every
  0.2 s.
- **Repetitions.** Three runs of each setup and proof (two at some points); the tables give medians.

### CPU results

Measured before [the parallel SHPLONK division](#the-shplonk-division) and
[the overlapped start of a proof](#the-start-of-a-proof), which make `all_sum` and `all_prod` at
`2^18`, and every program from `2^20` on, 12–24 % faster on 32 threads: the times of the proofs are
higher than today's.

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
- `W` and `W'`, 16–18 %: a sequential `divByMonic` per `f` and offset, the packing and the sums. Now
  parallel: `W` 6–13 times faster from `2^18` on, `W'` about 6 times
  ([The SHPLONK division](#the-shplonk-division)).
- Loading the key, 16–18 %: the SRS, the `.const` (both read by one thread then, the `.const` a byte
  at a time: [The start of a proof](#the-start-of-a-proof)), the INTT of the fixed columns and, mostly,
  recomputing the fixed commitments on every proof (3.2 s at `fibonacci` `2^24`).
- `Q`, 20 %: the LDE of its columns, zeroing buffers, the domain and the MSM; the bytecode itself takes
  0.3–0.5 s.
- The witness, 5–9 %: reading 1–2 GB and checking each value three times. Now read while the key
  loads, and checked in Rust without a `BigUint` per value
  ([The start of a proof](#the-start-of-a-proof)).

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

### The SHPLONK division

`W` divides each `f_i − r_i` by `X^k − ξ·ω^s` once per offset `s`, and `W'` divides `L` by `X − y`
([protocol.md#shplonk-opening](protocol.md#shplonk-opening)). rapidsnark's `divByMonic` runs the
recurrence `q_j = a_{j+m} + β·q_{j+m}` of the quotient by `X^m − β` on `m` threads, one per residue of
`j` mod `m`: on one thread for `W'` and for every `f` of `k = 1`. `Polynomial::divByMonicInPlace`, a
method added to rapidsnark next to it, which `PilFflonk::divideExactly` calls, runs it as a blocked
scan, over blocks of about `2^12` coefficients of the quotient, a whole number `R` of rows of `m`:

1. every block but the lowest, in parallel, from zero carries: the lowest `m` coefficients it gives,
   and the lowest `m` coefficients of the dividend it is about to overwrite;
2. the true carries, from the top block down, `m` multiplications per block: from zero carries to the
   true ones, a block's lowest coefficients move by `β^R` times the carries above it;
3. every block again, in parallel, from its true carries, in place.

It is twice the serial work, on every thread, in place (`divByMonic` clears a second polynomial), and
it gives the serial quotient: the quotient is unique and the arithmetic exact. It returns whether the
remainder is zero, which `divideExactly` checks as before. `W` and `W'` also pack each `f_i` into one
buffer, where `W` divides it as a polynomial over it that does not clear it
(`Polynomial::fromReservedBuffer`), instead of allocating, clearing and copying a polynomial per
`f_i`; and `pack()` calls `CPolynomial::getCoefficients`, added next to `getPolynomial`, which
interleaves `f` in one parallel pass over its coefficients, without clearing a power-of-two buffer and
scanning it for the degree, which every commitment gains from too. pilfflonk has no division or
packing of its own: it calls these rapidsnark methods, and rapidsnark's existing ones are unchanged
for their other callers. On the CPU, 32 threads (`OMP_NUM_THREADS=32 taskset -c 0-31` on the
256-thread machine), packed keys, the median of three proofs, before → after, in seconds; every proof
the same, byte for byte:

| Program | N | `W` | `W'` | Opening |
|---|---|---|---|---|
| `fibonacci` | 2^18 | 0.093 → 0.014 | 0.030 → 0.005 | 0.55 → 0.43 |
| `fibonacci` | 2^20 | 0.478 → 0.038 | 0.158 → 0.027 | 1.68 → 1.11 |
| `fibonacci` | 2^22 | 1.680 → 0.150 | 0.615 → 0.097 | 5.41 → 3.37 |
| `all_sum` | 2^18 | 0.349 → 0.053 | 0.192 → 0.033 | 1.66 → 1.20 |
| `all_sum` | 2^20 | 1.627 → 0.164 | 0.773 → 0.118 | 5.78 → 3.68 |
| `all_prod` | 2^18 | 0.408 → 0.050 | 0.179 → 0.032 | 1.59 → 1.09 |

What is left of the opening is the MSMs of `[W]₁` and `[W']₁`.

### The start of a proof

Before stage 1 the prover loads the key and reads the witness. Both were sequential, and the files were
read by one thread:

- **The SRS and the `.const`** are read by every thread, a chunk of 8 MiB each (one `pread` of the
  whole section before, and the `.const` a byte at a time through `std::istreambuf_iterator`).
- **The witness directory** is read while the C++ core loads the key
  (`PilfflonkWitnessArgs::load_with_key` in the CLI, for `prove` and `check`): the Rust side reads the
  key's own files first (`ProvingKeyFiles`), which give the witness's shape, and a thread opens the
  directory and reads the trace (`WITNESS_READ`) meanwhile. The errors are those of before, in the order of before: the key's,
  then the witness's. A witness library still runs after the key.
- **The range check** of the trace in Rust compares two 128-bit halves with `r`, where it allocated a
  `BigUint` per value: 0.7 s for the 16.8 million values of `all_sum` at `2^20`, a few tens of ms now.
- **CUDA** (`--gpu`): a thread calls `pilfflonk_gpu_available` before any file is read, so that
  CUDA's initialisation, sppark's device list and its streams, runs while the key's files and the
  witness are read; the C++ core's first call (`GPU_INIT`) waits only for what is left. `--gpu` is
  still refused before the SRS is read.

The start is from the first line of the log to stage 1; the witness, from "Reading the witness" to
stage 1, the instance included (and, before, the read and its check); `WITNESS_READ`, the read, in
parallel with the key. As above:

| Program | N | Start | `LOAD_SRS` | `LOAD_AIRS` | Witness | `WITNESS_READ` | Proof |
|---|---|---|---|---|---|---|---|
| `fibonacci` | 2^20 | 1.51 → 1.01 | 0.101 → 0.059 | 0.361 → 0.087 | 0.167 → 0.049 | 0.049 | 5.28 → 4.04 |
| `fibonacci` | 2^22 | 3.57 → 2.26 | 0.348 → 0.097 | 1.061 → 0.373 | 0.645 → 0.192 | 0.201 | 15.18 → 11.95 |
| `all_sum` | 2^18 | 2.08 → 1.21 | 0.114 → 0.051 | 0.474 → 0.113 | 0.357 → 0.101 | 0.094 | 6.55 → 5.17 |
| `all_sum` | 2^20 | 5.41 → 2.80 | 0.377 → 0.108 | 1.512 → 0.449 | 1.370 → 0.346 | 0.363 | 19.88 → 15.21 |
| `all_prod` | 2^18 | 1.77 → 1.50 | 0.089 → 0.045 | 0.423 → 0.109 | 0.357 → 0.097 | 0.092 | 5.99 → 5.27 |

What is left of the start on the CPU is mostly `FIXED_COMMITMENTS`, the fixed columns' MSMs, and the
instance, which copies the trace into the columns of the stages. Below `2^18` nothing changes: the
proofs take 1–2.5 s, mostly MSMs.

### Not implemented

None changes the proof or the protocol: an MSM that scales (shared buckets, a window by size and
threads, or the GPU's); fewer threads per MSM by its number of points (1.3–7× from `2^10` to `2^20`);
not recomputing the fixed commitments on every proof (5–7 % at `2^22`–`2^24`); reserving `Q`'s buffers
without zeroing them; a single canonicity check of the witness; the instance's columns cleared by one
thread before the parallel copy of the trace (`std::vector::assign`, in `PILFFLONK_INSTANCE`,
0.19–0.34 s at `fibonacci` `2^22` and `all_sum` `2^20`); one division for the `f_i` of `W` that share
their `Z_{T_i}` (the three `f` of stage 1 of `fibonacci`: about 0.05 s at `2^22`, but a refused
division would then name no single `f_i`); not freeing the instance and the key at the end of the
command (0.25–0.33 s, though the process's exit frees them too).

## GPU

### What runs on the GPU

`proofman-cli pilfflonk prove -g/--gpu` keeps the key and the proof on the device (`GpuKey`,
`pil2-stark/src/pilfflonk/pilfflonk_key_gpu.hpp`): the SRS's powers `[τ^i]₁`, the fixed columns'
coefficients and the bytecode while the key lives, and every polynomial of a proof in one arena of
device memory. On the device:

- at load, the fixed columns' INTT and the fixed commitments, and the sum of the MSM shift for every
  length of an MSM of the key's proofs;
- each stage's INTTs, blinding and commitments (`InstanceGpu::commitStage`);
- all of `Q` (`InstanceGpu::computeQ` and `commitQ`): the LDE of each column it reads on each part of
  the coset (`LdeGpu`), Zi of the part, its bytecode (`ExpressionsGpu`), its interpolation, the check of
  its bound, its pieces with their blinding, and their commitments; the pieces stay there for the
  opening;
- SHPLONK's evaluations, `W`, `W'` and their commitments (`OpeningGpu`).

On the host: the transcript, the blinding's draws (sent up in the CPU's order), the std's hints and
the im pols (on copies of the stage columns), and the opening's scalars. The MSMs and the NTTs are the
GPU entry points pil2-stark has, called as the PLONK GPU prover calls them
(`rapidsnark/plonk_prover_gpu.c.cuh`):

- `msm_bn128_gpu_dev_ptr` (`bn128/src/msm/msm_bn128.cu`, sppark's Pippenger) with `montgomery = true`,
  on the device's copy of the SRS's powers;
- `ntt_bn128_gpu_dev_ptr` and `intt_bn128_gpu_dev_ptr` (`bn128/src/ntt/ntt_bn128.cu`, sppark's NTT), in
  natural order in and out;
- the PLONK prover's helpers (`rapidsnark/plonk_prover.cu`): device memory, copies, its tables of
  powers (`gpu_plonk_precompute_omega_tables_async`) and its exact division by `X − β`
  (`gpu_plonk_compute_div_zerofier`), and sppark's `cuda_available` to know whether there is a GPU.

The elementwise work around them is pilfflonk's own kernels (`pilfflonk_kernels.cu`,
`pilfflonk_lde.cu`, `pilfflonk_expressions.cu`, `pilfflonk_shplonk.cu`), each tested against the CPU's
code byte for byte. A proof copies the witness to the device, and back to the host the stage columns
and the committed polynomials (which the host's hints read, and of which `Instance::polynomial` hands
out copies), and a few elements per step otherwise; `-vv` logs the bytes of each phase
(`PILFFLONK_COPIES_<phase>`).

### Why the proof is the same

sppark's NTT roots for BN254 are ffiasm's, `ω_{2^k} = 5^((r−1)/2^k)` for `k ≤ 28`, in the same
Montgomery form (its `group_gen` is 5, ffiasm's `nqr`), and its inverse scales by `1/2^k`, as ffiasm's
`ifft`. Both sides compute in exact field arithmetic and leave every element in its canonical
Montgomery form, below `r`. The MSM returns a Jacobian point `(X, Y, Z)`, ffiasm's extended
`(X, Y, Z², Z³)`, and a commitment leaves the prover in affine coordinates, which are unique. So every
value is the same element at the same place, and the proof is the same bit for bit.

### The shifted MSM

sppark's Pippenger is fast when the scalars look random and slow when many of them repeat: in each
window, every point whose scalar has the same digit goes to one bucket, and few threads sum it. Fixed
polynomials repeat: every coefficient of `L1` is `1/N`. On the RTX 5090, with the same 8,388,608
points:

| Scalars | GPU, unshifted | GPU, shifted | CPU (ffiasm, 32 threads) |
|---|---|---|---|
| random | 0.047 s | 0.063 s | 2.39 s |
| all equal | 25.79 s | 0.070 s | 1.17 s |
| every other one equal | 14.02 s | 0.060 s | 1.85 s |

So the MSM of the device path (`GpuKey::commit`) shifts its scalars. For a fixed `h`,
`ρ_i = h^(i+1)`, and it computes

```
Σ s_i·[τ^i]₁ = Σ (s_i + ρ_i)·[τ^i]₁ − Σ ρ_i·[τ^i]₁
```

`s + ρ` looks random whatever `s` is, and `Σ ρ_i·[τ^i]₁` depends only on the length `n`, so it is
computed once for each `n` and kept. The difference is the same point, so the commitment, in affine
coordinates, is the CPU's bit for bit. Every MSM has the static length of its `f`'s degree bound in the
layout (zero scalars add nothing), so the sums are computed when the key loads, one MSM per length;
the `n` additions `s_i + ρ_i` are in the kernel that packs the scalars. `ρ` needs to be neither secret
nor random, as it changes how the point is computed, not the point.

### Selection, memory and errors

- **Selection.** `-g/--gpu`, the CPU by default, in a build that found `nvcc`
  (`provers/starks-lib-c/build.rs`; never with `--features proofman-starks-lib-c/cpu-only`). It does
  not go through the STARK backend's global switch (`set_gpu_mode`), whose `get_num_gpus` exits on a
  machine without a driver: the device is given explicitly, `ProvingKey::load_on(dir, Device::Gpu)`
  (Rust), `pilfflonk_ctx_new_on(dir, PILFFLONK_DEVICE_GPU)` (C), `ProvingKey::load(dir, Device::Gpu)`
  (C++). `proofman_pilfflonk::gpu_available()` (`pilfflonk_gpu_available`) says whether there is a GPU.
  Without one (a CPU library, or no GPU), `--gpu` is refused before the SRS is read. A GPU library
  without `--gpu` is the CPU, and the CPU library has no GPU code at all (`__USE_CUDA__`).
- **Memory.** The SRS is copied to the device once, when the key loads (timer `PILFFLONK_GPU_SRS`), as
  the PLONK prover's `d_ptau`, and so are each AIR's fixed coefficients, bytecode and tables. A proof
  keeps its data in one arena, whose phases reuse each other's bytes (`ArenaLayout`); one proof at a
  time uses it, and another thread's waits. The key checks, as it loads each AIR and before it copies
  anything of it, that the device holds it and a whole proof of it, with what sppark's MSM reserves per
  call; if not, the load fails, naming the AIR and the bytes it needs and the device has: a proof never
  runs out of device memory midway, and there is no fallback to the host. The arena holds `Q` in its
  default parts, of `2^nBits` points; a larger `q_part_bits` whose parts it does not hold is refused
  when it is set (`pilfflonk_instance_set_q_part_bits`), naming the parts and the bytes they need and
  the arena has, and never changed to another size. Device 0.
- **Errors.** A CUDA failure inside the reused helpers (no device memory, a lost device) aborts the
  process, as in the PLONK GPU prover (`CHECKCUDAERR`): the one exception to the C API's rule. sppark's
  MSM reports its own failure as the point at infinity; `GpuKey::commit` turns that, for shifted scalars
  not all zero or for the shift itself, into an error rather than a proof that does not verify. A
  witness the constraints refuse is refused with the CPU's error, word for word.

### GPU results

The first GPU path, which ran only the MSMs and the NTTs on the GPU from host memory (before the device
path above). RTX 5090 (`sm_120`), CUDA 13.0, 32 CPU threads (`OMP_NUM_THREADS=32`), packed keys, the
median of three proofs on each device with the same `--insecure-blinding-seed`. Every GPU proof is
the CPU's byte for byte, and the JS verifier accepts them. Seconds, CPU → GPU:

| Program | N | Proof | Fixed commitments | Stage MSMs | `Q` MSM | `W`, `W'` MSMs | Opening |
|---|---|---|---|---|---|---|---|
| `fibonacci` | 2^16 | 0.71 → 0.35 (2.03×) | 0.17 → 0.014 | 0.212 → 0.018 | 0.069 → 0.008 | 0.18 → 0.024 | 0.20 → 0.04 |
| `fibonacci` | 2^18 | 1.80 → 0.50 (3.60×) | 0.26 → 0.026 | 0.576 → 0.045 | 0.175 → 0.022 | 0.55 → 0.045 | 0.57 → 0.07 |
| `fibonacci` | 2^20 | 5.26 → 1.22 (4.31×) | 0.65 → 0.049 | 1.529 → 0.070 | 0.491 → 0.029 | 1.66 → 0.094 | 1.77 → 0.25 |
| `fibonacci` | 2^22 | 16.01 → 2.85 (5.62×) | 2.01 → 0.142 | 4.088 → 0.191 | 1.373 → 0.085 | 4.77 → 0.252 | 5.12 → 0.60 |
| `all_sum` | 2^16 | 2.41 → 0.93 (2.59×) | 0.35 → 0.053 | 0.936 → 0.156 | 0.135 → 0.012 | 0.58 → 0.053 | 0.63 → 0.09 |
| `all_sum` | 2^18 | 6.78 → 1.75 (3.87×) | 0.93 → 0.087 | 2.393 → 0.188 | 0.275 → 0.027 | 1.81 → 0.105 | 1.94 → 0.27 |
| `all_sum` | 2^20 | 20.95 → 4.34 (4.83×) | 2.64 → 0.203 | 6.709 → 0.400 | 0.837 → 0.049 | 5.34 → 0.279 | 5.75 → 0.70 |
| `all_prod` | 2^18 | 6.79 → 1.79 (3.79×) | 1.10 → 0.105 | 2.260 → 0.179 | 0.399 → 0.013 | 1.69 → 0.092 | 1.81 → 0.26 |

- **The whole proof** is 2.0–5.6× faster (`fibonacci` `2^22`: 16.01 s against 2.85 s). On the GPU,
  the largest phases left are `Q`'s LDE around its transforms (0.27 s at `fibonacci` `2^22`, 0.93 s at
  `all_sum` `2^20`: the folding and the transfers), what remains of CUDA's initialisation once the
  warm-up has overlapped what it could (0.28 s), the interpreter (0.14–0.37 s) and `W` and `W'`
  (0.32–0.39 s together).
- **The MSMs** of the stages, of `Q` and of `W` and `W'` are 6–31× faster. Without the shift they were
  15–52× faster, but the fixed-commitment check was up to 6.4× slower than the CPU (12.95 s against
  2.02 s at `fibonacci` `2^22`), and the whole proof only 1.05–2.1× faster; with it, that check takes
  0.14–0.20 s.
- **The transforms** of the stages and of `Q` are 2–7× faster.
- Before [the parallel SHPLONK division](#the-shplonk-division) and [the overlapped start of a
  proof](#the-start-of-a-proof), the GPU proof took 4.82 s at `fibonacci` `2^22` (3.7×) and 6.90 s at
  `all_sum` `2^20` (3.4×).

### Open

- The std's hints and the im pols on the device, and with them the copies of the stage columns and
  of the committed polynomials back to the host.
- CUDA's initialisation in parallel with the SRS's read: `--gpu` would then be refused after the SRS
  is read, not before.
- The SRS's copy to the device (`GPU_SRS`) in parallel with the reading of the `.const`.

## The wrap

The final SNARK of `prove-snark` with pilfflonk, on `examples/fibonacci-square`, against the FFLONK
and PLONK wraps of the same program. **M** marks a measurement, **E** an estimate.

### Method

- **The chain.** `setup -r --hash Poseidon2` and `prove -a` give the vadcop_final proof.
  `setup-snark --final-snark pilfflonk` makes the recursivef (BN128 Merkle trees in custom mode), the
  final circuit with the custom templates `PoseidonT(5)` and `Num2Bytes`, its AIR in plonk2pil's
  BN254 family (PoseidonBN254, layout L1, with range checks) and the pilfflonk key of that AIR.
  `prove-snark` proves the recursivef, computes the wrap's witness in its own process and proves it
  with pilfflonk; `verify-snark` runs the JS verifier.
- **Machine and build** as in [CPU](#method): release, `--features proofman-starks-lib-c/cpu-only`.
  64 threads (`OMP_NUM_THREADS=RAYON_NUM_THREADS=64`, `taskset -c 0-63`), and 32 (`taskset -c 0-31`)
  for the comparison with FFLONK and PLONK, measured at 32. Load average 5–42.
- **SRS.** The Hermez `powersOfTau28_hez_final_24.ptau` (power 24: 2^25 − 1 powers `[τ^i]₁`, 19 GB;
  its blake2b is the one snarkjs's README gives), read in place.
- **Measures.** The wall time and peak RSS of each command (`/usr/bin/time -v`; GB are its kB / 10^6,
  as for FFLONK and PLONK), the RSS of its process tree once a second, and `prove-snark`'s timers
  (`-v`; with `-vv` also the prover's `PILFFLONK_*`). Three proofs at each thread count: the tables
  give medians.

### The end-to-end run

On 64 threads, **M**:

| Step | Time | Peak RSS (GB) |
|---|---|---|
| `compile-pil` | 0.99 s | 0.27 |
| `setup -r --hash Poseidon2` | 11 min 18 s | 3.40 (5.78 with the circom and pil2com it runs) |
| `gen-custom-commits-fixed` | 1.6 s | 0.74 |
| `prove -a` (the vadcop_final proof) | 25.2 s | 5.08 |
| `verify-stark` | 0.13 s | 0.03 |
| `setup-snark --final-snark pilfflonk` | 3 min 49 s | 3.48 (4.06 with its children) |
| `prove-snark` | 22.2 s | 6.50 |
| `verify-snark` | 0.33 s | 0.08 |

- **The key.** The final circuit has 296,115 r1cs constraints and 633,854 PLONK constraints (631,857
  once copies are merged); its AIR has 2^19 rows, 9 `f` (`k` = [13, 13, 1, 2, 3, 2, 3, 4, 1]), `qDeg`
  5 and no im pol, and needs 6,815,756 powers of the SRS (`13·N + 12`). The pilfflonk key is 0.89 GB
  (the SRS, 436 MB, and the `.const`, 453 MB), the final circuit's witness files (`final.so`,
  `final.dat`, `final.exec`) 73 MB, and the recursivef's `.consttree` 878 MB, written by the first
  `prove-snark` (28.1 s instead of 22.2 s).
- **`setup-snark`** spends the last 180 s of its 229 s waiting for the final circuit's witness
  calculator (`final.so`, g++ on circom's C++), which compiles in the background from the moment
  circom writes it. Before that: the recursivef's circom 9.1 s, its AIR and constant tree 12.2 s, the
  final circuit's circom 15.4 s, plonk2pil 1.8 s, pil2com over BN254 3.9 s and setup-pilfflonk 6.1 s,
  where the peak is.
- **The key is reproducible.** A fresh `setup -r` and `setup-snark` give the vkey, the
  `pilfflonk.verifier.sol` and the project contract of an earlier run byte for byte; its `final.exec`
  differs in the header only, which now records the r1cs's `n_vars`, and has the same body.
- **`verify-snark`** accepts the proof, and refuses it with a byte of an evaluation or of a commitment
  changed, with its public changed, against the vkey of the same AIR set up with another SRS (CI's
  test ptau), and against the vkey of the 2^22 key below ("a proof of this shape has 2048 bytes, not
  2208").

`prove-snark`'s phases, **M**:

| Phase | 64 threads (s) | 32 threads (s) | Peak RSS (GB) |
|---|---|---|---|
| `INITIALIZING_FINAL_SNARK_PROVER`: the pilfflonk key, its fixed commitments checked | 2.69 | 3.28 | 5.3 |
| `GENERATE_RECURSIVEF` | 9.76 | 17.41 | 4.8 |
| `CALCULATE_FINAL_WITNESS`: circom 0.53 s, the exec 0.29 s | 0.84 | 0.83 | 5.8 |
| `CALCULATE_FINAL_PROOF`: the pilfflonk proof | 7.04 | 9.64 | 6.5 |
| `GENERATING_WRAPPER_SNARK_PROOF`: the three above | 17.65 | 27.91 | 6.5 |
| The command | 22.2 | 32.9 | 6.50 / 6.29 |

On 32 threads the pilfflonk proof is stage 1 1.60 s, stage 2 0.62 s, `Q` 4.39 s (the LDE of its
columns 2.50 s, its MSM 0.59 s), the evaluations 0.04 s and the opening 2.72 s (the MSMs of `[W]₁`
and `[W']₁`, 1.24 s each); loading the key, 2.61 s of the 3.28 s are its fixed commitments.

### Against FFLONK and PLONK

FFLONK and PLONK are rapidsnark's provers on the stock recursivef and final circuit (measured before
pilfflonk's wrap existed, on the same machine and program, 32 threads; FFLONK with a fixed-`τ` test
ptau of `9·2^24` powers, as the Hermez one is too small for it). The 2^22 pilfflonk key is the one
before the range-check gates, whose final circuit checks the Goldilocks ranges bit by bit
(`Num2Bits`), on 64 threads.

| | pilfflonk, 2^19 | pilfflonk, 2^22, no range checks | FFLONK | PLONK |
|---|---|---|---|---|
| Final circuit (r1cs) | 296,115 | 5,379,109 | 7,605,644 | 7,605,644 |
| PLONK constraints | 633,854 | 10,572,480 | 13,774,131 | 13,774,131 |
| Key | 0.89 GB | — | 38.7 GB zkey | 25.8 GB zkey |
| `setup-snark` | 229 s, 3.48 GB (64 thr); 235 s, 3.5 GB (32 thr) | 268 s, 15.9 GB | 360 s, 29.7 GB | 367 s, 16.0 GB |
| Final proof, CPU 32 threads | **9.64 s** | 62.6 s (64 thr) | 127.8 s | 45.0 s |
| Wrapper, CPU 32 threads | **27.9 s** | 80.1 s (64 thr) | 146.3 s | 63.9 s |
| `prove-snark`, CPU 32 threads | 32.9 s | 109 s (64 thr) | 218 s | 73–82 s |
| Peak RSS | **6.29 GB** | 27.4 GB (64 thr) | 134.4 GB | 54.5 GB |
| Final proof, RTX 5090 | **1.86 s** | — | no GPU prover | 0.86 s |
| Wrapper, RTX 5090 | **2.73 s** | — | — | 1.65 s |
| `prove-snark`, RTX 5090 | 3.98 s | — | — | 3.65 s |
| Host RSS, RTX 5090 | 4.74 GB | — | — | 13.2 GB |
| GPU memory | 2.6 GiB | — | — | 29.8 GiB |
| Proof | 2,208 B, 69 words | 2,048 B, 64 words | 768 B (E: snarkjs's 24 words) | 768 B |
| `verifyProof` gas | 332,661 | 310,932 | 182,681 | 263,175 |
| Verifier's runtime | 21,569 B | 19,306 B | 14,078 B | 5,850 B |

All **M** but FFLONK's proof size. On 32 threads the pilfflonk proof is 4.7 times faster than
PLONK's and 13 times faster than FFLONK's, and the wrapper 2.3 and 5.2 times; its peak memory is
8.7 and 21 times smaller. Its `verifyProof` costs 1.82 times FFLONK's gas (the limit set for this
port was twice) and 1.26 times PLONK's. The range checks took the AIR from 2^22 rows to 2^19 and
the proof from 62.6 s to 7.0 s on 64 threads, for 160 more bytes and 21,729 more gas. On the RTX
5090 ([On the GPU](#on-the-gpu)), with a GPU prover not yet finished, the pilfflonk proof takes 2.2
times PLONK's, the wrapper 1.65 times and the whole command 0.33 s more; it needs a third of PLONK's
host memory and an eleventh of its GPU memory.

### On the GPU

The same `prove-snark`, with `--gpu`, on worker-13: RTX 5090 (`sm_120`), CUDA 13.0, 32 CPU threads
(`OMP_NUM_THREADS=RAYON_NUM_THREADS=32`), on this run's vadcop_final proof and `provingKeySnark/`.
Built from `e5dd3060b` (the CPU measurements above are of `8343ef2c4`, whose CPU prover gives the
same proofs), with the GPU prover half-way through its port: on the device are the key (the SRS,
the fixed columns' INTT and commitments), each stage's INTT and commitment, `Q`'s LDE,
interpolation and commitment, and the SHPLONK opening (the evaluations, `W`, `W'` and their
commitments); on the CPU, still, are the evaluation of `Q`'s bytecode and the zerofiers of its
domain (their device versions exist, not yet wired in) and the stage-2 hint columns. Three runs
with an `nvidia-smi` sampler running, which keeps the driver's state up between processes (a warm
GPU), and one without (cold). Every proof verifies with `verify-snark`, with the CPU proof's publics
and length. **M**:

| Phase | Warm (s) | Cold (s) |
|---|---|---|
| `LOADING_RECURSIVE_F_SETUP` | 0.26 | 0.26 |
| `INITIALIZING_FINAL_SNARK_PROVER` | 0.69 | 0.76 |
| `GENERATE_RECURSIVEF` | 0.35 | 0.32 |
| `CALCULATE_FINAL_WITNESS` | 0.39 | 0.39 |
| `CALCULATE_FINAL_PROOF` | 1.86 | 1.89 |
| `GENERATING_WRAPPER_SNARK_PROOF` | 2.73 | 2.72 |
| The command | 3.98 | 5.27 |

The medians of the warm runs; the first of them also writes the recursivef's `.consttree`
(`LOADING_RECURSIVE_F_SETUP` 11.48 s, the command 15.16 s). The host's peak RSS is 4.74 GB, and the
GPU's memory in use peaks at 2,681 MiB.

- **The pilfflonk proof**, 1.86 s (9.64 s on 32 threads of the CPU machine): the instance 0.07 s,
  stage 1 0.07 s, stage 2 0.20 s (0.17 s of it the hint columns, on the CPU), `Q` 1.36 s and the
  opening 0.11 s. `Q` is 0.91 s of bytecode and 0.07 s of zerofiers on the CPU over its 8 parts,
  0.28 s of LDE, 0.02 s of interpolation and 0.06 s of MSM. More than half the proof, 1.15 s, is
  the work still on the CPU.
- **The key**, 0.69 s: CUDA's initialisation 0.11 s, the SRS 0.04 s and its copy 0.04 s, and the AIR
  0.49 s, with the fixed INTT 0.08 s, the shift sums 0.17 s and the fixed commitments 0.09 s.
- **Against PLONK's GPU wrap** (rapidsnark's GPU prover, measured on the same 5090): its final proof
  takes 0.86 s, its wrapper 1.65 s and its command 3.50–3.65 s, with 13.2 GB of host RSS and
  29.8 GiB of GPU memory. With `Q`'s bytecode on the device (0.083 s against 5.10 s on the CPU for
  the L1 layout's `Q` at 2^21), its zerofiers and the hints there too, the pilfflonk proof would take
  about 0.8 s (**E**: 1.86 s less the 1.15 s on the CPU, plus about 0.05 s on the device).

### On chain

The project contract `provingKeySnark/final/BuildVerifier.sol` (`BuildVerifier is
PilfflonkVerifier`), which setup-snark writes beside the key's `pilfflonk.verifier.sol`, with solc
0.8.37 (optimizer, 200 runs) and Foundry v1.8.3 (offline, evm `osaka`), on the run's proof, **M**.
The gas is that of the call (`gasleft()` around it), as for FFLONK and PLONK, and that of a
transaction on anvil (with the intrinsic 21,000 and the calldata):

| Call | Outcome | Gas | Transaction gas |
|---|---|---|---|
| `verifyProof(proof, [public])` | `true` | 332,661 | 388,876 |
| the same, a commitment's `x` plus 1 | `false` | 3,845 | |
| the same, the last evaluation plus 1 | `false` | 90,468 | |
| the same, the public plus 1 | `false` | 23,711 | |
| `verifySnarkProof(programVK, rootCVadcopFinal, publicValues, proof)` | returns | 353,287 | 403,330 |
| the same, the last evaluation plus 1 | reverts `InvalidProof()` | 108,728 | |
| the same, a public value changed | reverts | 41,998 | |
| the same, another `programVK` | reverts | 42,026 | |
| the same, `publicValues` = `snark_proof.bin`'s `public_bytes` | reverts | 42,053 | |

- **Size.** `BuildVerifier`'s runtime is 22,905 B (initcode 22,933 B), `PilfflonkVerifier`'s alone
  21,569 B: under EIP-170's 24,576 B, with 1,671 B to spare, and EIP-3860's 49,152 B. One contract
  is enough; the wrap does not need the original's two (`PilFflonkVerifier` and `ShPlonkVerifier`).
- **The arguments of `verifySnarkProof`.** `programVK` is the vadcop_final proof's `rom_root` (the
  publics_info's `verificationKey`), each value big-endian, and `rootCVadcopFinal` the contract's
  `getRootCVadcopFinal()`. `publicValues` are the other publics as the final circuit hashes them
  (`getSha256Inputs`): each value's bytes little-endian, `0x1900000000000000…` for `module` = 25.
  `snark_proof.bin`'s `public_bytes` are big-endian (`get_public_bytes_solidity`:
  `0x0000000000000019…`), so `hashPublicValues` of them is not the proof's public and the contract
  refuses them. The PLONK and FFLONK wraps share both the circuit's template and
  `get_public_bytes_solidity` (not measured here).
- **`eth_estimateGas` is not a measure of `verifyProof`.** The verifier returns `false` and does not
  revert, so the least gas that does not revert is one at which a precompile runs out of gas and
  `verifyProof` returns `false`: 209,223 on anvil, which stops at its second `ecMul`. The 178,137
  estimated earlier for the 2^22 key on anvil is of that kind: Foundry gives 310,932 for its proof.
  `verifySnarkProof` reverts on a `false` verdict, so its estimate, 410,211, is that of an accepted
  call.

### Reproducing the wrap

From the repository root, with the release binaries
(`cargo build --release -p pil2-stark-setup -p proofman-cli -p fibonacci-square --features
proofman-starks-lib-c/cpu-only`), circom 2.2.3 on the `PATH` (`setup/circom`) and `PIL2C_EXEC`
naming a pil2com that honours `prime` ([README.md#compile-pil](README.md#compile-pil)):

```sh
B=<build dir>
target/release/proofman-setup compile-pil --pil examples/fibonacci-square/pil/build.pil \
    -I pil2-components/lib/std/pil -o $B/build.pilout -u $B/build/fixed --fixed-to-file
target/release/proofman-setup setup -a $B/build.pilout -b $B/build -r -u $B/build/fixed --hash Poseidon2
target/release/proofman-cli gen-custom-commits-fixed --witness-lib target/release/libfibonacci_square.so \
    --proving-key $B/build/provingKey/ --custom-commits rom=$B/build/rom.bin
target/release/proofman-cli prove --witness-lib target/release/libfibonacci_square.so \
    --proving-key $B/build/provingKey/ --public-inputs examples/fibonacci-square/src/inputs.json \
    --output-dir $B/proofs --custom-commits rom=$B/build/rom.bin -a
target/release/proofman-setup setup-snark -b $B/build --final-snark pilfflonk --powers-of-tau <ptau> \
    --publics-info examples/fibonacci-square/src/publics_info.json
target/release/proofman-cli prove-snark -p $B/proofs/vadcop_final_proof.bin -k $B/build/provingKeySnark \
    -o $B/wrap
target/release/proofman-cli verify-snark -p $B/wrap/snark_proof.bin \
    -k $B/build/provingKeySnark/final/provingKey/final/pilfflonk/pilfflonk.vkey.json
```

The ptau needs at least 6,815,756 powers `[τ^i]₁`: Hermez's `powersOfTau28_hez_final_22` (2^23 − 1)
or a larger one, or, for a test,
`target/release/examples/pilfflonk_bench_inputs ptau 6815756 <out.ptau>` (a public `τ`; 3 min 23 s
on 64 threads). The CI job `test-pilfflonk-wrap` runs these steps with that test ptau, and then
`cli/tests/snark_pilfflonk.rs` on the key and the proof; it waits for the compiler's branch to be
published (its comment says how to turn it on). On the Solidity side, `proofman-cli pilfflonk
calldata -k <vkey> -p <proof.bin> --publics <publics.json> --format hex` gives `verifyProof`'s
calldata from the proof's bytes, and its words are `verifySnarkProof`'s `proofBytes`.

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
