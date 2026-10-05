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
remainder is zero, which `divideExactly` checks as before. `W` also packs each `f_i` into one buffer,
where it divides it as a polynomial over it that does not clear it (`Polynomial::fromReservedBuffer`),
instead of allocating, clearing and copying a polynomial per `f_i`; `W'` packs none, and adds each
component `p_j` of `f_i` into `L` where the packing puts it (coefficient `c` at `c·k + j`), as the
device does; and `pack()` calls `CPolynomial::getCoefficients`, added next to `getPolynomial`, which
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

**M** marks a measurement, **E** an estimate.

### What runs on the GPU

`proofman-cli pilfflonk prove -g/--gpu` keeps the key and the proof on the device (`GpuKey`,
`pil2-stark/src/pilfflonk/pilfflonk_key_gpu.hpp`): the SRS's powers `[τ^i]₁`, the fixed columns'
coefficients and the bytecode while the key lives, and every polynomial of a proof in one arena of
device memory. On the device:

- at load, the fixed columns' INTT and the fixed commitments, and the sum of the MSM shift for every
  length of an MSM of the key's proofs ([loading a key](#loading-a-key-on-the-gpu));
- each stage (`InstanceGpu::commitStage`): its columns, the std's hints and the im pols, with the
  bytecode on the device (`computeStageColumns`), and its INTTs, blinding and commitments;
- all of `Q` (`InstanceGpu::computeQ` and `commitQ`): the LDE of each column it reads on each part of
  the coset (`LdeGpu`), Zi of the part, its bytecode (`ExpressionsGpu`), its interpolation, the check of
  its bound, its pieces with their blinding, and their commitments;
- SHPLONK's evaluations, `W`, `W'` and their commitments (`OpeningGpu`), on the committed polynomials
  and `Q`'s pieces where the device keeps them.

On the host: the transcript, the blinding's draws (sent up in the CPU's order) and the opening's
scalars.

### Rules of the device path

The code cites these rules by name:

- **No fallback.** A key whose AIRs, or a whole proof of one of them, the device cannot hold is
  refused when it loads, naming the AIR, the bytes it needs and the bytes the device has. A proof
  never runs out of device memory midway, and nothing falls back to the host.
- **One part size.** A proof's arena holds `Q` in the part size it was sized for. A larger part size
  that the arena does not hold is refused when it is set, and is never changed to another.
- **Recomputed, not kept.** What the host rarely needs is not kept for it: the fixed columns'
  evaluations on `H` are computed from their coefficients when a check asks for them.
- **Copies on demand.** What the host asks for of the device's data is copied when it asks, and
  once. The MSMs and the NTTs are the GPU entry points pil2-stark has, called as the PLONK GPU prover
calls them (`rapidsnark/plonk_prover_gpu.c.cuh`):

- `msm_bn128_gpu_dev_ptr` (`bn128/src/msm/msm_bn128.cu`, sppark's Pippenger) with `montgomery = true`,
  on the device's copy of the SRS's powers;
- `ntt_bn128_gpu_dev_ptr` and `intt_bn128_gpu_dev_ptr` (`bn128/src/ntt/ntt_bn128.cu`, sppark's NTT), in
  natural order in and out;
- the PLONK prover's helpers (`rapidsnark/plonk_prover.cu`): device memory, copies, its tables of
  powers (`gpu_plonk_precompute_omega_tables_async`), its running product
  (`gpu_plonk_prefix_scan_multiply`) and its exact division by `X − β`
  (`gpu_plonk_compute_div_zerofier`), and sppark's `cuda_available` to know whether there is a GPU.

The elementwise work around them is pilfflonk's own kernels (`pilfflonk_kernels.cu`,
`pilfflonk_lde.cu`, `pilfflonk_expressions.cu`, `pilfflonk_hints.cu`, `pilfflonk_shplonk.cu`), each
tested against the CPU's code byte for byte.

**What crosses.** `-vv` logs the bytes of each phase (`PILFFLONK_COPIES_<phase>`). A proof copies its
witness to the device, and a few kilobytes otherwise:

- up: the blinding factors of each stage and of `Q`'s boundaries, the opening's interpolants and the
  descriptors of its evaluations; and, not counted, the operand tables of each evaluation of the
  bytecode, a few KB, by a plain copy;
- down: the count of each committed polynomial's coefficients (its degree), the row of each hint's
  first zero denominator, `Q`'s counts, the evaluations, an element or two of each division, and the
  commitments, which sppark's MSM returns.

The key copies the SRS's powers and its fixed columns up once, and back only the counts of the fixed
columns' coefficients; the host keeps nothing of the fixed columns. What the host asks for of the
device's data is copied when it asks, once ([copies on demand](#rules-of-the-device-path)): a column (`Instance::column`), the witness
columns (`check`), a committed polynomial (`Instance::polynomial`), `Q`'s pieces
(`Instance::qPiece`), the fixed polynomials (`AirKey::fixedPolynomial`), and the fixed columns on `H`
(`AirKey::fixedEvaluations`, which `check` reads: their coefficients, evaluated on the host by
`Lde::ntt`); the columns only before `commitQ`, which reuses their memory, and are refused after.

### Why the proof is the same

sppark's NTT roots for BN128 are ffiasm's, `ω_{2^k} = 5^((r−1)/2^k)` for `k ≤ 28`, in the same
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
layout (zero scalars add nothing), so the sums are computed when the key loads (`addShiftSums`): in
increasing order, each from the sum of the next shorter length and an MSM of the powers between the
two, so that all of them take the MSM of the longest length's points. The `n` additions `s_i + ρ_i`
are in the kernel that packs the scalars. `ρ` needs to be neither secret nor random, as it changes how
the point is computed, not the point.

### Loading a key on the GPU

`ProvingKey::load` on the GPU overlaps the host's reading of the files with the device's work, as the
PLONK GPU prover's `round0` overlaps the copy of its `PTau` and the NTTs of its constant polynomials
with the rest:

1. CUDA's initialisation, started by a thread before any file is read ([the start of a
   proof](#the-start-of-a-proof)); `GPU_INIT` is what is left of it.
2. Each AIR's `.const` is read on a thread of its own: the first from here on, and each next one from
   when the AIR before it takes its own (`AirKey::ConstantsSource`).
3. The SRS is read and checked (`LOAD_SRS`), and its powers go to the device through the pinned,
   double-buffered staging (`GPU_SRS`), after a check that the device holds them.
4. For each AIR (`LOAD_AIRS`), first what its pilfflonkinfo and `.bin` give alone, while its `.const`
   may still be read: its degrees, layout and hints; its whole budget checked against the device's
   free memory ([no fallback](#rules-of-the-device-path)) before anything of it goes up; its bytecode and its interpreter on the
   device; and the shift's sums (`GPU_SHIFT_SUMS`). Then its fixed columns, decoded from the
   `.const`, go up, are interpolated (`FIXED_INTT`; only their counts come back) and committed
   (`FIXED_COMMITMENTS`).

On the RTX 5090, 32 threads, a warm GPU, **M**. "Before" is the previous loading, which read the
files one after the other and copied the fixed columns' coefficients back to the host (three runs
in the same session); "after" is the first GPU proof of `gpu_check`, and for the wrap its warm runs:

| Phase (s) | L1 `2^21`, before | after | wrap `2^19`, before | after |
|---|---|---|---|---|
| `LOAD_SRS` | 0.14–0.16 | 0.205 | 0.038–0.039 | 0.077–0.089 |
| `GPU_SRS` | 0.11 | 0.126 | 0.038–0.039 | 0.042 |
| `GPU_SHIFT_SUMS` | 0.36–0.39 | 0.124 | 0.188–0.210 | 0.084–0.111 |
| `FIXED_INTT` | 0.29–0.30 | 0.128 | 0.077–0.079 | 0.037 |
| `FIXED_COMMITMENTS` | 0.155 | 0.154 | 0.077–0.092 | 0.062–0.063 |
| `LOAD_AIRS` | 1.11–1.14 | 0.590 | 0.443–0.474 | 0.279–0.318 |
| The key's copies down | 1.68 GB | 200 B | 453 MB | 216 B |

- **The shift's sums** take one MSM of the longest length instead of one per length: 0.36–0.39 s
  against 0.12 s at L1 `2^21`, whose lengths reach `12·N`.
- **`FIXED_INTT`** no longer copies the coefficients back to the host (1.68 GB at L1 `2^21`).
- **`LOAD_AIRS`** less those three is the reading of the files and the derivations: 0.30 s before,
  0.18 s after, as the `.const` is read during the SRS's load and the shift's sums. `LOAD_SRS` takes
  longer, 0.15 against 0.205 s, as both files are read at once.
- In all, the key's phases at L1 `2^21` take 0.92 s, against 1.36–1.41 s.

### Selection, memory and errors

- **Selection.** `-g/--gpu`, the CPU by default, in a build that found `nvcc`
  (`provers/starks-lib-c/build.rs`; never with `--features proofman-starks-lib-c/cpu-only`). It does
  not go through the STARK backend's global switch (`set_gpu_mode`), whose `get_num_gpus` exits on a
  machine without a driver: the device is given explicitly, `ProvingKey::load_on(dir, Device::Gpu)`
  (Rust), `pilfflonk_ctx_new_on(dir, PILFFLONK_DEVICE_GPU)` (C), `ProvingKey::load(dir, Device::Gpu)`
  (C++). `proofman_pilfflonk::gpu_available()` (`pilfflonk_gpu_available`) says whether there is a GPU.
  Without one (a CPU library, or no GPU), `--gpu` is refused before the SRS is read. A GPU library
  without `--gpu` is the CPU, and the CPU library has no GPU code at all (`__USE_CUDA__`).
- **Memory.** The SRS's powers and each AIR's fixed coefficients, bytecode (its args and numbers,
  which the interpreter reads) and tables are on the device while the key lives. A proof keeps its
  data in one arena, whose phases reuse each other's bytes (`ArenaLayout`); one proof at a time uses
  it, from its instance's creation to its end, and another thread's instance waits. On the thread
  that holds it (the thread of the latest call of that instance or of its opening: an instance may
  move between threads), a second instance is refused, as it would wait for itself
  (`GpuKey::Lease`; the C API's concurrent instances, `pilfflonk_api.hpp`). The key checks, as
  it loads each AIR and before anything of it goes to the device, that the device holds it and a
  whole proof of it, with what sppark's MSM reserves per call; if not, the load fails, naming the AIR
  and the bytes it needs and the device has: a proof never runs out of device memory midway, and
  there is no fallback to the host ([no fallback](#rules-of-the-device-path)). The arena holds `Q` in its default parts, of
  `2^nBits` points; a larger `q_part_bits` whose parts it does not hold is refused when it is set
  (`pilfflonk_instance_set_q_part_bits`), naming the parts and the bytes they need and the arena has,
  and never changed to another size ([one part size](#rules-of-the-device-path)). Device 0, whatever the calling thread's current
  device: every call of the GPU path, and the release of its device memory, makes it current while it
  runs and leaves the thread's as it was (`DeviceScope`), as the STARK's GPU entry points set theirs.
- **Host memory.** A key on the GPU keeps of each AIR neither its fixed columns on `H` (1.68 GB at L1
  `2^21`) nor the LDE's table of the `N'` roots of unity (512 MB at `N' = 2^24`), which its proofs do
  not read: `Lde` builds the table for its first transform on the host, and the only one a key on the
  GPU runs is the evaluation of its fixed columns when `check` asks for them
  ([recomputed, not kept](#rules-of-the-device-path)). **M** (two GPU proofs each): the peak RSS of a
  proof of L1 `2^21` is 5.66 GB, against 6.18 GB when the key kept both, and of `fibonacci` `2^22`
  1.53 GB against 1.79 GB, the table of each (512 and 256 MiB); the fixed
  columns on `H`, decoded from the `.const` while the AIR loads, are freed once they are on the
  device, so they leave the key's memory but not the load's peak. The `.const`'s bytes themselves
  are released as soon as they are decoded, before the columns are interpolated and committed.
- **The arena given.** The arena can be a buffer of the caller's ([the wrap's device
  buffer](#the-wraps-device-buffer)): `GpuKeyOptions::arena` (C++), `pilfflonk_ctx_new_on_device_buffer`
  (C), `ProvingKey::load_on_device_buffer` (Rust). The key never writes it while it loads (its
  scratch is its own then), synchronises the device when a proof takes it and when it gives it back,
  and never frees it; a buffer smaller than a proof's arena is refused, saying how many bytes the
  arena needs and the buffer has. `ProvingKey::requiredDeviceBytes` (`pilfflonk_gpu_device_bytes`,
  `proofman_pilfflonk::gpu_device_bytes`) gives, from the key's files and without loading it (and
  without the LDE's table), the arena's bytes and the most the key needs beside it, as the key
  reserves them: its device memory, the scratch of an AIR's loading, which it allocates beside a given
  arena while the AIR loads, and what a proof allocates besides. A key on a given arena loads with
  exactly `beside` bytes free (or a `memoryLimit` of it), and is refused with one byte less; one on
  its own arena, whose loading's scratch is in it, needs at most `arena + beside`.
  `ProvingKey::freeDeviceBytes` (`pilfflonk_gpu_free_bytes`, `proofman_pilfflonk::gpu_free_bytes`) is
  the free memory a load checks that against.
- **Errors.** A CUDA failure inside the reused helpers (no device memory, a lost device) aborts the
  process, as in the PLONK GPU prover (`CHECKCUDAERR`): the one exception to the C API's rule. sppark's
  MSM reports its own failure as the point at infinity; `GpuKey::commit` turns that, for shifted scalars
  not all zero or for the shift itself, into an error rather than a proof that does not verify. A
  witness the constraints refuse is refused with the CPU's error, word for word.

### The wrap's device buffer

In `prove-snark` with pilfflonk on the GPU, the key's proofs keep their arena in the wrap's device
memory, as rapidsnark's PLONK prover is carved out of it by `pre_allocate_final_snark_prover_c`
(`pilfflonk_wrap::wrap_arena`, `proofman/src/pilfflonk_wrap.rs`):

- with `d_buffers`, proofman's unified buffer (`SnarkWrapper::new_with_preallocated_buffers`): all of
  it, which the key refuses, as a GPU without the memory ([no fallback](#rules-of-the-device-path)), if it holds less than a
  proof's arena; the unified buffer must be on device 0, where pilfflonk proves. The wrapper then
  re-uploads the STARK's fixed columns that the proof overwrote, as after PLONK's
  (`reload_fixed_pols_gpu`): the flag is set as each wrap's proof ends, whatever its outcome, as a
  proof that fails may have written the buffer too (for PLONK and FFLONK as for pilfflonk);
- without (`prove-snark`): the recursivef's prover buffer, which the wrapper now holds for its life, as
  for PLONK, grown to the arena if it is smaller (`reserve_recursivef_aux_trace`), or refused if the
  device cannot hold it.

The recursivef is done with either buffer when pilfflonk proves. The key's own device memory is
beside it, from the wrapper's start with `preload`: once the arena is in place, the wrap checks that
the device has what the key needs beside it free (`requiredDeviceBytes`' `beside`, against
`gpu_free_bytes`), before the key loads anything. At the wrap's `2^19`, **M**: the arena is
1,057,572,864 bytes of the recursivef's prover buffer of 2,665,131,360, and the key needs
1,556,689,640 bytes beside it at most (`ProvingKey::requiredDeviceBytes`, its loading's scratch
included; 1,338,585,832 without it); [On the GPU](#on-the-gpu) has the
run. The unified buffer's path is measured only by the C++ tests of an arena given to a key, which
prove the CPU's proofs on it and refuse one a byte short.

### GPU results

The device path on an RTX 5090 (`sm_120`), CUDA 13.0, 32 CPU threads
(`OMP_NUM_THREADS=32`), packed keys, `gpu_check.sh` with three proofs on each device and the same
`--insecure-blinding-seed`, and an `nvidia-smi` sampler running, which keeps the driver's state up
between processes (a warm GPU), as persistence mode would; without it, cold.

**Checks, M.** Every GPU proof is the CPU's byte for byte: `fibonacci` `2^16`–`2^22`, `all_sum`
`2^16`–`2^20`, `all_prod` `2^18`, the wrap's layout L1 at `2^21` (three proofs on each device), and 20
fixture keys (one on each): the synthetic constraint domains, `Q` split, im pols, unpacked, and two
stages.
`pilfflonk_gpu_test` with `PILFFLONK_GPU=1` passes: those 20 keys proved with `Q` in every part size
the arena holds, 40 changed witnesses refused with the CPU's errors, 23 bytecodes through the
interpreter on the device, the stage columns of the 3 AIRs of two stages, and every kernel, copy and
memory test.

**As each part moved to the device, M.** The whole command (`wall`, from the first line of the log
to the last), warm, each row with everything of the rows above it:

| Point | L1 `2^21` | `fibonacci` `2^22` | `all_sum` `2^20` |
|---|---|---|---|
| The MSMs and NTTs on the GPU, from host memory | 16.94 | 2.79–2.88 | 4.26–4.36 |
| The key and the stage commitments on the device | 16.2–16.4 | 2.52–2.68 | 3.94–3.97 |
| and `Q`'s LDE and the opening | 9.74–9.76 | 1.78–1.82 | 2.51–2.57 |
| and the stage-2 hints | 9.37–9.47 | | 2.26–2.36 |
| and `Q`'s bytecode, all of `Q` | 2.83–2.86 | 1.22–1.25 | 1.54–1.70 |
| and no copies back, the overlapped load | **2.11–2.20** | **0.89–0.91** | **1.14–1.19** |

`all_prod` `2^18`: 0.89–1.01 s with all of `Q` on the device, 0.75–0.92 s now. Against that step's
CLI in the same session (L1 2.76–2.79 s, `fibonacci` `2^22` 1.06 s), the last row takes 0.65 s and
0.16 s less: the key's load ([above](#loading-a-key-on-the-gpu)) and the copies back.

**Every key, M.** The whole command on each device, the median and the range of the three runs of
`gpu_check` (the same build, 32 threads, a warm GPU):

| Program | N | CPU (s) | GPU (s) | CPU/GPU |
|---|---|---|---|---|
| `fibonacci` | 2^16 | 0.76 (0.74–0.80) | 0.31 (0.31–0.34) | 2.5 |
| `fibonacci` | 2^18 | 1.79 (1.78–1.81) | 0.45 (0.45–0.53) | 4.0 |
| `fibonacci` | 2^20 | 5.24 (5.22–5.40) | 0.63 (0.60–0.65) | 8.3 |
| `fibonacci` | 2^22 | 15.84 (15.66–15.90) | 0.90 (0.89–0.91) | 17.6 |
| `all_sum` | 2^16 | 2.51 (2.47–2.54) | 0.60 (0.52–0.65) | 4.2 |
| `all_sum` | 2^18 | 6.76 (6.72–6.78) | 0.77 (0.71–0.84) | 8.8 |
| `all_sum` | 2^20 | 20.79 (20.79–20.81) | 1.14 (1.14–1.19) | 18.2 |
| `all_prod` | 2^18 | 6.62 (6.56–6.69) | 0.80 (0.75–0.92) | 8.3 |
| L1 | 2^21 | 61.50 (61.41–61.70) | 2.12 (2.11–2.20) | 29.0 |

The command includes the key's load and CUDA's initialisation, which dominate the GPU's time at the
smaller sizes.

**Copies at L1 `2^21`, M** (`PILFFLONK_COPIES_*`, bytes up / down):

| Phase | With copies back | Now |
|---|---|---|
| `KEY` | 3.29 GB / 1.68 GB | 3,288,361,577 / 200 |
| `INSTANCE` | 0.60 GB / 0.60 GB | 603,979,776 / 0 |
| `STAGE_1` | 736 / 0.60 GB | 736 / 72 |
| `STAGE_2` | 352 / 0.34 GB | 352 / 40 |
| `Q` | 0 / 16 | 0 / 16 |
| `EVALUATIONS` | 1,504 / 1,504 | 1,504 / 1,504 |
| `OPEN` | 1,504 / 3,056 | 1,504 / 3,056 |

A proof copies its witness, 604 MB, and 4.1 KB up, and 4.7 KB down, where it copied 1.54 GB back
before. The key copies its SRS's powers (1.61 GB) and fixed columns (1.68 GB) up, and 25 counts back.

**Warm and cold, M.** Cold, `fibonacci` `2^22` takes 2.06–2.10 s against 0.89–0.91 s: each process
then adds 1.2 s of CUDA's initialisation, which a warm driver keeps.

**Device memory, M.** At the wrap's `2^19`: the arena, 1.06 GB, and 1.56 GB beside it at most
(`requiredDeviceBytes`, its loading's scratch included). Over every key of the run, the GPU's memory
in use, sampled once a second, peaks at 8,257 MiB, L1 `2^21`'s: the SRS's powers (1.61 GB), the
fixed coefficients (1.68 GB), the arena, whose largest phase is `Q`'s (the 39 columns `Q` reads on a
part of `N` points, 2.6 GB, after 0.94 GB of committed polynomials), sppark's MSM of `12·N` points,
and CUDA's context.

### Open

- The interpreter's operand tables go up with a plain copy for each evaluation, a few KB, outside
  `PILFFLONK_COPIES_*`.
- CUDA's initialisation in parallel with the SRS's read: `--gpu` would then be refused after the SRS
  is read, not before.
- The SRS's host copy is kept (the CPU's checks, and `fixedCommitments` with another SRS): 1.61 GB of
  host memory at L1 `2^21`, the one large datum a key on the GPU keeps on the host, now that it keeps
  neither the fixed columns on `H` (1.68 GB) nor the LDE's table of roots (512 MB)
  ([host memory](#selection-memory-and-errors)).
- Not done: a smaller packing of the fixed columns (`W` and `W'` from `12·N` to about `5·N`
  coefficients), and the wrap's `.exec` additions on the device, as PLONK's.

## The wrap

The final SNARK of `prove-snark` with pilfflonk, on `examples/fibonacci-square`, against the FFLONK
and PLONK wraps of the same program. **M** marks a measurement, **E** an estimate.

### Method

- **The chain.** `setup -r --hash Poseidon2` and `prove -a` give the vadcop_final proof.
  `setup-snark --final-snark pilfflonk` makes the recursivef (BN128 Merkle trees in custom mode), the
  final circuit with the custom templates `PoseidonT(5)` and `Num2Bytes`, its AIR in plonk2pil's
  BN128 family (PoseidonBN128, layout L1, with range checks) and the pilfflonk key of that AIR.
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
  final circuit's circom 15.4 s, plonk2pil 1.8 s, pil2com over BN128 3.9 s and setup-pilfflonk 6.1 s,
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
| Peak RSS | **6.12 GB** | 27.4 GB (64 thr) | 134.4 GB | 54.5 GB |
| Final proof, RTX 5090 | **0.30 s** | — | no GPU prover | 0.86 s |
| Wrapper, RTX 5090 | **1.00 s** | — | — | 1.65 s |
| `prove-snark`, RTX 5090 | **2.17 s** | — | — | 3.50–3.65 s |
| Host RSS, RTX 5090 | **2.95 GB** | — | — | 13.2 GB |
| GPU memory | **5.0 GiB** | — | — | 29.8 GiB |
| Proof | 2,208 B, 69 words | 2,048 B, 64 words | 768 B (E: snarkjs's 24 words) | 768 B |
| `verifyProof` gas | 332,661 | 310,932 | 182,681 | 263,175 |
| Verifier's runtime | 21,569 B | 19,306 B | 14,078 B | 5,850 B |

All **M** but FFLONK's proof size. pilfflonk's peak RSS on 32 threads is that of the current prover,
measured on the RTX 5090 machine's CPU (two runs, 6.12 GB both, against 6.30 GB there for the prover
before its last cleanup, as the CPU machine's 6.29 GB); its times are the CPU machine's. On 32
threads the pilfflonk proof is 4.7 times faster than PLONK's and 13 times faster than FFLONK's, and
the wrapper 2.3 and 5.2 times; its peak memory is 8.9 and 22 times smaller. Its `verifyProof` costs
1.82 times FFLONK's gas (the limit set for this port was twice) and 1.26 times PLONK's. The range
checks took the AIR from 2^22 rows to 2^19 and the proof from 62.6 s to 7.0 s on 64 threads, for 160
more bytes and 21,729 more gas. On the RTX 5090 ([On the GPU](#on-the-gpu)), the pilfflonk proof is
2.9 times faster than PLONK's, the wrapper 1.7 times and the whole command 1.6–1.7 times; it needs a
quarter of PLONK's host memory and a sixth of its GPU memory.

### On the GPU

The same `prove-snark`, with `--gpu`, on an RTX 5090 (`sm_120`), CUDA 13.0, 32 CPU threads
(`OMP_NUM_THREADS=RAYON_NUM_THREADS=32`), on this run's vadcop_final proof and `provingKeySnark/`,
with the GPU prover of [GPU](#gpu): the key, every stage with its hints and im pols, `Q` whole and
the SHPLONK opening on the device, and pilfflonk's arena in the recursivef's prover buffer ([the
wrap's device buffer](#the-wraps-device-buffer)). The CPU figures above were measured before the
device path held whole proofs; the CPU prover's proofs have not changed since. Three runs with an
`nvidia-smi` sampler running, which keeps the driver's state up between processes (a warm GPU), and
one without (cold). Every proof verifies with `verify-snark`. **M**, the medians of the warm runs and
the cold run:

| Phase | Warm (s) | Cold (s) |
|---|---|---|
| `LOADING_RECURSIVE_F_SETUP` | 0.26 | 0.26 |
| `INITIALIZING_FINAL_SNARK_PROVER` | 0.50 (0.43–0.50) | 0.51 |
| `GENERATE_RECURSIVEF` | 0.35 | 0.35 |
| `CALCULATE_FINAL_WITNESS` | 0.34 | 0.34 |
| `CALCULATE_FINAL_PROOF` | **0.30** (0.28–0.32) | 0.28 |
| `GENERATING_WRAPPER_SNARK_PROOF` | **1.00** (0.98–1.03) | 0.98 |
| The command | **2.17** (2.16–2.22) | 3.47 |

The recursivef's `.consttree` was there from earlier runs. Against the CLI before the last cleanup
of the prover and the wrap (the same session, alternating, three warm runs each): the final proof
0.32, the final circuit's witness 0.39 and the wrapper 1.07–1.08 s; its proofs are the same byte for
byte with a fixed blinding seed. The host's peak RSS is 2.95 GB: a key on the GPU keeps neither its
fixed columns on `H` nor the LDE's table on the host ([host memory](#selection-memory-and-errors)).
The GPU's memory in use, sampled once a second, peaks at 5,157 MiB.

- **The pilfflonk proof**, 0.30 s (9.64 s on 32 threads of the CPU machine): the instance
  0.013–0.016 s, stage 1 0.05–0.07 s, stage 2 0.018–0.031 s, `Q` 0.088–0.090 s, the evaluations
  0.001 s and the opening 0.10–0.12 s. A proof copies its witness up, 168 MB, and 4.5 KB up and 5.2
  KB down otherwise; nothing of the witness or of the committed polynomials comes back to the host.
- **The key**, 0.50 s: the SRS 0.07 s and its copy 0.04 s, and the AIR 0.27–0.33 s, with the
  shift's sums 0.08–0.10 s, the fixed INTT 0.04 s and the fixed commitments 0.06–0.08 s
  ([loading a key](#loading-a-key-on-the-gpu)). CUDA's initialisation comes earlier, with the
  recursivef's device buffers, which the wrapper allocates at its start.
- **The wrapper**, 1.00 s: the recursivef, the final circuit's witness and the pilfflonk proof. The
  wrapper holds the recursivef's device buffers for its life, as for PLONK, instead of allocating
  and freeing them (with 512 MB of pinned host memory) for each proof.
- **GPU memory.** The wrapper holds the recursivef's device buffers (its prover buffer, 2.67 GB, and
  its constant tree) throughout, as for PLONK, with pilfflonk's arena (1.06 GB) inside the prover
  buffer, and the key's memory beside them, 1.56 GB at most (`requiredDeviceBytes`).
- **Against PLONK's GPU wrap** (rapidsnark's GPU prover, measured on the same 5090): its final proof
  takes 0.86 s, its wrapper 1.65 s and its command 3.50–3.65 s, with 13.2 GB of host RSS and
  29.8 GiB of GPU memory. pilfflonk's proof is 2.9 times faster, its wrapper 1.7 times and its
  command 1.6–1.7 times, with a quarter of the host memory and a sixth of the GPU memory.

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
naming a pil2com that has `--field` ([README.md#compile-pil](README.md#compile-pil)):

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
`cli/tests/snark_pilfflonk.rs` on the key and the proof; it runs when the repository variable
`PILFFLONK_BN128_CI` is `"true"` (its comment says where to set it). On the Solidity side, `proofman-cli pilfflonk
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
   `pilfflonk_gpu_test` compares the GPU with the CPU byte for byte (the kernels, the LDE, the
   interpreter, `GpuKey`'s commitments around sppark's window sizes, whole proofs, and what a proof
   and a key copy); without a GPU it skips them and says so, and with
   `PILFFLONK_GPU=1` a skip fails. `gpu_check.sh` proves on each device with the same blinding seed
   (`GPU_CHECK_SEED`, `GPU_CHECK_REPEATS` times), compares every `proof.json` and `publics.json` with the
   first CPU proof's, prints each phase's median, and exits with 0 only if every proof is the same.
3. Back on a machine with Node, verify the GPU proofs:
   `proofman-cli pilfflonk verify <vkey> <out>/gpu.1/publics.json <out>/gpu.1/proof.json`.
