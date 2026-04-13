# CPU/GPU Interface Harmonization for Goldilocks NTT & Poseidon2

## 1. Context

The GPU side (`NTTGoldilocksGPU`, `Poseidon2GoldilocksGPU<W>`) was recently cleaned up. The CPU side (`NTT_Goldilocks`, `Poseidon2Goldilocks<W>`) predates the cleanup and has:
- Drifted naming and parameter shapes vs. GPU.
- A bug in the `merkletree_batch` wrapper (dead code — delete, not fix).
- A half-implemented AVX512 path that is never dispatched correctly.
- A larger public surface than the GPU — many entry points are tests/benches-only.

**Goal**: bring CPU and GPU to a single coherent API; preserve bit-exact correctness and current performance at every step; defer all AVX512 implementation work to the very last phase (no AVX512 host currently available).

---

## 2. Design decisions (locked)

1. **AVX selection stays compile-time** (`#ifdef __AVX2__` / `#ifdef __AVX512__`). No runtime function-pointer dispatch.
2. **`Poseidon2Mode` enum** — one public method per operation, mode as a parameter:
   ```cpp
   enum class Poseidon2Mode : uint8_t {
       Auto = 0,        // resolves to best variant compiled in via resolveAuto()
       Scalar,
       Avx,
       AvxBatch,
       Avx512,          // Phase 7 only — aborts loudly if not compiled in
       Avx512Batch,     // Phase 7 only
   };
   ```
   - `Auto` centralizes the `#ifdef` cascade via `resolveAuto()` (one place, not per call site).
   - Explicit modes abort loudly if the requested SIMD level was not compiled in — misuse is a build-config bug, not a silent fallback.
3. **`resolveAuto()` helper** — single inline that encapsulates the `#ifdef` cascade:
   ```cpp
   enum class Poseidon2OpKind : uint8_t { SingleSponge, Aggregating };

   inline Poseidon2Mode resolveAuto(Poseidon2Mode m, Poseidon2OpKind op) {
       if (m != Poseidon2Mode::Auto) return m;
   #ifdef __AVX512__
       return (op == Poseidon2OpKind::Aggregating) ? Poseidon2Mode::Avx512Batch
                                                   : Poseidon2Mode::Avx512;
   #elif defined(__AVX2__)
       return (op == Poseidon2OpKind::Aggregating) ? Poseidon2Mode::AvxBatch
                                                   : Poseidon2Mode::Avx;
   #else
       return Poseidon2Mode::Scalar;
   #endif
   }
   ```
4. **Per-operation valid modes** — single-sponge ops reject Batch modes at runtime (abort with clear message):

   | Operation | Valid modes |
   |---|---|
   | `hashFullResult`, `hash` | `Auto`, `Scalar`, `Avx`, `Avx512` |
   | `linearHash` | `Auto`, `Scalar`, `Avx`, `AvxBatch`, `Avx512`, `Avx512Batch` |
   | `merkletree` | `Auto`, `Scalar`, `AvxBatch`, `Avx512Batch` |

5. **`Layout` enum for GPU** — no default, every call site must be explicit:
   ```cpp
   enum class Layout : uint8_t { RowMajor, Tiles };
   ```
6. **No `NTTTuning` struct in this refactor.** CPU NTT keeps current parameters as-is; only rename `extendPol` → `LDE`.
7. **Hard renames, no `[[deprecated]]` shims** — every caller updated in the same commit.
8. **AVX512 deferred to Phase 7** — no AVX512 host available; Phases 0–6 compile-check only.

---

## 3. Audit: what the prover uses vs. what is dead

### 3.1 NTT_Goldilocks — used by the prover

| Method | Call sites |
|---|---|
| `NTT(dst, src, size)` | [fri.hpp:91](pil2-stark/src/starkpil/fri/fri.hpp#L91), [fri.hpp:248](pil2-stark/src/starkpil/fri/fri.hpp#L248) |
| `NTT(dst, src, size, ncols, buffer)` | [starks.hpp:223](pil2-stark/src/starkpil/starks.hpp#L223), [starks.hpp:225](pil2-stark/src/starkpil/starks.hpp#L225) |
| `INTT(dst, src, size)` | [starks.hpp:197,199,278](pil2-stark/src/starkpil/starks.hpp#L197), [stark_verify.hpp:681](pil2-stark/src/starkpil/stark_verify.hpp#L681), [fri.hpp:93,250](pil2-stark/src/starkpil/fri/fri.hpp#L93) |
| `extendPol(out, in, NExt, N, ncols [, buffer])` | [const_pols.hpp:38,58](pil2-stark/src/starkpil/const_pols.hpp#L38), [build_const_tree.cpp:44](pil2-stark/src/bctree/build_const_tree.cpp#L44), [starks_api.cpp:634,682](pil2-stark/src/api/starks_api.cpp#L634), [starks.hpp:125-158](pil2-stark/src/starkpil/starks.hpp#L125) |

**Dead in production**: `nphase`, `nblock`, `inverse=true`, `extend=true` parameters; `int extension_` ctor arg with values other than 1.

### 3.2 Poseidon2Goldilocks<W> — used by the prover

| Method | Call sites |
|---|---|
| `hash_full_result_seq(out, in)` | [transcriptGL.cpp:22,25,28](pil2-stark/src/starkpil/transcript/transcriptGL.cpp#L22), [stark_verify.hpp:199](pil2-stark/src/starkpil/stark_verify.hpp#L199) |
| `hash_seq(state, in)` | [merkleTreeGL.cpp:236,250,264](pil2-stark/src/starkpil/merkleTree/merkleTreeGL.cpp#L236) |
| `linear_hash_seq(out, in, size)` | [merkleTreeGL.cpp:182,185,188](pil2-stark/src/starkpil/merkleTree/merkleTreeGL.cpp#L182) |
| `merkletree_batch_avx512` / `_batch_avx` / `_seq` | [merkleTreeGL.cpp:280-306](pil2-stark/src/starkpil/merkleTree/merkleTreeGL.cpp#L280) — manual `#ifdef` cascade |
| `Poseidon2GoldilocksGrinding::grinding(...)` | [gen_proof.hpp:266](pil2-stark/src/starkpil/gen_proof.hpp#L266) |

**Dead-in-production** (never called by the prover):
- `merkletree(...)` / `merkletree_batch(...)` wrappers at [poseidon2_goldilocks.hpp:99-127](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp#L99) — the prover has its own `#ifdef` cascade; **Severity-A bug is in dead code; fix = delete**.
- `partial_merkle_tree` — public but no prover caller.
- Single-sponge AVX: `hash_full_result_avx`, `hash_avx`, `linear_hash_avx`, `merkletree_avx` — tests/benches only.
- 4-lane AVX batch: `hash_full_result_batch_avx`, `hash_batch_avx`, `linear_hash_batch_avx` — tests/benches; `merkletree_batch_avx` pulls them internally.
- AVX512 batch: `hash_full_result_batch_avx512`, `hash_batch_avx512`, `linear_hash_batch_avx512` — internal to `merkletree_batch_avx512`.

---

## 4. Identified incoherences (prioritized)

### Severity A — real bugs / dead-but-broken
1. **`merkletree_batch` wrapper** ([poseidon2_goldilocks.hpp:117-127](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp#L117)) — `batch_size` passed as `arity`; calls non-batch `_avx`. Dead; **delete**.
2. **AVX512 wrapper dispatch never reached** — `#if defined(__AVX2__) || defined(__AVX512__)` always picks AVX2 path. Subsumed by wrapper deletion.
3. **AVX512 single-element primitives missing** — `pow7_avx512`, `add_avx512`, `hash_avx512`, etc. all commented out. **Phase 7.**

### Severity B — naming / API drift
4. `extendPol` vs GPU's `LDE`.
5. `_seq`/`_avx`/`_avx512` suffixes leak into public API.
6. GPU exposes `linearHash` + `linearHashTiled` as two functions; a single `Layout` param is cleaner.

### Severity C — hygiene
7. Mixed `u_int64_t` / `uint64_t` in CPU sources.
8. `partial_merkle_tree` public but unused.

---

## 5. Final target API

### 5.1 NTT — CPU vs GPU

| Operation | CPU (after Phase 4) | GPU (unchanged) |
|---|---|---|
| Forward | `void NTT(Element *dst, Element *src, uint64_t size, uint64_t ncols=1, Element *buffer=NULL, uint64_t nphase=3, uint64_t nblock=1, bool inverse=false, bool extend=false);` | `void NTT(gl64_t *dst, uint64_t nBits, uint64_t nCols, cudaStream_t s);` |
| Inverse | `void INTT(Element *dst, Element *src, uint64_t size, uint64_t ncols=1, Element *buffer=NULL, uint64_t nphase=3, uint64_t nblock=1, bool extend=false);` | `void INTT(gl64_t *dst, uint64_t nBits, uint64_t nCols, cudaStream_t s);` |
| LDE | `void LDE(Element *output, Element *input, uint64_t N_Extended, uint64_t N, uint64_t ncols, Element *buffer=NULL, uint64_t nphase=3, uint64_t nblock=1);` | `void LDE(gl64_t *d_dst, uint64_t dst_off, gl64_t *d_src, uint64_t src_off, uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, TimerGPU &t, cudaStream_t s);` |

### 5.2 Poseidon2 — CPU vs GPU

| Operation | CPU (after Phase 3+7) | GPU (after Phase 5) |
|---|---|---|
| Single-sponge full output | `static void hashFullResult(Element *out, const Element *in, Poseidon2Mode mode);` | (device-only) |
| Single-sponge compressed | `static void hash(Element (&state)[CAPACITY], const Element (&in)[W], Poseidon2Mode mode);` | `static void hash(uint64_t *out, const uint64_t *in, cudaStream_t s=0);` |
| Linear hash | `static void linearHash(Element *out, Element *in, uint64_t size, Poseidon2Mode mode);` | `static void linearHash(uint64_t *d_out, uint64_t *d_in, uint64_t nCols, uint64_t nRows, Layout layout, cudaStream_t s);` |
| Merkle tree | `static void merkletree(Element *tree, Element *in, uint64_t nCols, uint64_t nRows, uint64_t arity, int nThreads, uint64_t dim, Poseidon2Mode mode);` | `static void merkletree(uint32_t arity, uint64_t *d_tree, uint64_t *d_in, uint64_t nCols, uint64_t nRows, Layout layout, cudaStream_t s);` |
| Grinding | `static void grinding(uint64_t &out_idx, const uint64_t *in, uint32_t n_bits);` | `static void grinding(uint64_t *d_nonce, uint64_t *d_nonceBlock, const uint64_t *d_in, uint32_t n_bits, cudaStream_t s);` |

**Removed from public CPU**: every `_seq`/`_avx`/`_avx512`/`_batch_*` symbol; `partial_merkle_tree`; legacy wrappers.
**Removed from public GPU**: `linearHashTiled`, `merkletreeTiled`, `buildMerkleTreeTilesGPU`.

### 5.3 Caller examples (target state)

```cpp
// Hot prover path — best implementation compiled in
Poseidon2Goldilocks<16>::merkletree(tree, src, nCols, nRows, arity,
                                    /*nThreads=*/0, /*dim=*/1,
                                    Poseidon2Mode::Auto);

// Transcript single hash (was hash_full_result_seq)
Poseidon2Goldilocks<8>::hashFullResult(out, in, Poseidon2Mode::Scalar);

// GPU prover commit (was buildMerkleTreeTilesGPU)
buildMerkleTreeGPU(arity, pNodes, src, nCols, NExtended, Layout::Tiles, stream);

// GPU FRI (was buildMerkleTreeGPU, row-major layout)
buildMerkleTreeGPU(arity, treeFRI->nodes, treeFRI->source, treeFRI->width,
                   treeFRI->height, Layout::RowMajor, stream);
```

---

## 6. Verification gates (run at end of EVERY step)

### (a) CPU tests
```bash
cd /home/rick/pil2-proofman/pil2-stark
make -j testscpu && ./testscpu      # scalar + AVX2 variants both exercised
# Phase 7 only, on AVX512 host:
# make -j testscpu_avx512 && ./testscpu_avx512
```

### (b) Benchmarks (≤ 2 % regression vs baseline)
```bash
cd /home/rick/pil2-proofman/pil2-stark
make -j benchscpu && ./benchscpu | tee /tmp/bench.txt
./scripts/bench_compare.sh src/goldilocks/benchs/baseline/$(hostname).txt /tmp/bench.txt
```

### (c) Full end-to-end proof
```bash
cd /home/rick/pil2-proofman
cargo build --bin proofman-cli
cargo run --bin proofman-cli prove \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --public-inputs examples/fibonacci-square/src/inputs.json \
     --custom-commits rom=examples/fibonacci-square/build/rom.bin \
     --verify-proofs --aggregation --compressed \
     --output-dir examples/fibonacci-square/build/proofs
```

All three gates must pass after every step. Gate (c) is load-bearing — Merkle-root divergence makes the verifier reject.

---

## 7. Granular implementation steps

### Phase 0 — Test & bench scaffolding (no behavior change)

**Files**: [pil2-stark/src/goldilocks/Makefile](pil2-stark/src/goldilocks/Makefile), [tests.cpp](pil2-stark/src/goldilocks/tests/tests.cpp), new [pil2-stark/scripts/bench_compare.sh](pil2-stark/scripts/bench_compare.sh), new `src/goldilocks/benchs/baseline/`.

- [x] **0.1** `Makefile`: add `testscpu_avx512` target — `-mavx512f -D__AVX512__`; compiles on any host, runs only on AVX512 hardware (Phase 7). Single `testscpu` binary covers scalar + AVX2 via explicit cross-checks.
  - *Commit: `test: add testscpu_avx512 build target and fix pre-existing AVX512 template errors`*
- [ ] **0.2** `scripts/bench_compare.sh baseline.txt fresh.txt` — exits non-zero if any line regresses > 2 %
  - *Commit: `test: add bench_compare.sh regression detector`*
- [ ] **0.3** `tests.cpp`: add `extendPol` correctness test — multiple `(N, N_Ext, ncols)` pairs; reference: INTT → multiply by shift powers → NTT
  - *Commit: `test: add extendPol end-to-end correctness test`*
- [ ] **0.4** `tests.cpp`: add merkletree cross-check — `merkletree_seq ≡ merkletree_batch_avx` (and `≡ _batch_avx512` under `#ifdef __AVX512__`) for widths {4,8,12,16}, rows ∈ {2¹⁰, 2¹⁵}, cols ∈ {1, 8, 64, 100}
  - *Commit: `test: add merkletree seq≡avx_batch cross-check`*
- [ ] **0.5** `tests.cpp`: add explicit test exercising `merkletree(...)` and `merkletree_batch(...)` wrappers — characterizes the Severity-A bug so Phase 1 deletion is documented
  - *Commit: `test: document merkletree_batch wrapper Severity-A bug`*
- [ ] **0.6** Record bench baseline: `make -j benchscpu && ./benchscpu > src/goldilocks/benchs/baseline/$(hostname).txt`
  - *Commit: `bench: record $(hostname) baseline`*

**Verify**: gates (a), (b), (c) green.

---

### Phase 1 — Delete dead `merkletree`/`merkletree_batch` wrappers

**Files**: [poseidon2_goldilocks.hpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp).

- [ ] **1.1** Grep confirm: no caller in `pil2-stark/src/` or `pil2-proofman/` references the *wrapper* `Poseidon2Goldilocks<W>::merkletree(` (non-variant signature). If found, investigate before continuing.
  - *Commit: (none — audit step)*
- [ ] **1.2** Delete the `merkletree(...)` wrapper from [poseidon2_goldilocks.hpp:104-115](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp#L104)
  - *Commit: `refactor: delete dead merkletree wrapper (Severity-A bug)`*
- [ ] **1.3** Delete the `merkletree_batch(...)` wrapper from [poseidon2_goldilocks.hpp:117-127](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp#L117)
  - *Commit: `refactor: delete dead merkletree_batch wrapper`*
- [ ] **1.4** Remove Phase-0 wrapper-characterization test (0.6) — replace with comment confirming wrappers are gone
  - *Commit: `test: remove wrapper-bug test (wrappers deleted)`*

**Verify**: gates (a), (b), (c) green.

---

### Phase 2 — Introduce `Poseidon2Mode` + `resolveAuto()` + new public API alongside old

**Files**: [poseidon2_goldilocks.hpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp), [poseidon2_goldilocks.cpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.cpp) (or `.hpp` if header-only).

- [ ] **2.1** Add `Poseidon2OpKind` enum (`SingleSponge`, `Aggregating`) near top of `poseidon2_goldilocks.hpp`
  - *Commit: `api: add Poseidon2OpKind enum`*
- [ ] **2.2** Add `Poseidon2Mode` enum (`Auto`, `Scalar`, `Avx`, `AvxBatch`, `Avx512`, `Avx512Batch`) near top of `poseidon2_goldilocks.hpp`
  - *Commit: `api: add Poseidon2Mode enum`*
- [ ] **2.3** Add `resolveAuto(Poseidon2Mode m, Poseidon2OpKind op)` inline helper that centralizes the `#ifdef` cascade (see §2 design decisions)
  - *Commit: `api: add resolveAuto() centralized SIMD dispatcher`*
- [ ] **2.4** Add `static void hashFullResult(Element *out, const Element *in, Poseidon2Mode mode)` — dispatches to existing `_seq`/`_avx`; `Avx512*` aborts with clear message until Phase 7
  - *Commit: `api: add hashFullResult(mode) alongside hash_full_result_seq`*
- [ ] **2.5** Add `static void hash(Element (&state)[CAPACITY], const Element (&in)[W], Poseidon2Mode mode)`
  - *Commit: `api: add hash(mode) alongside hash_seq`*
- [ ] **2.6** Add `static void linearHash(Element *out, Element *in, uint64_t size, Poseidon2Mode mode)`
  - *Commit: `api: add linearHash(mode) alongside linear_hash_seq`*
- [ ] **2.7** Add `static void merkletree(Element *tree, Element *in, uint64_t nCols, uint64_t nRows, uint64_t arity, int nThreads, uint64_t dim, Poseidon2Mode mode)` — `Auto` uses `resolveAuto(m, Aggregating)`; `Avx512*` aborts until Phase 7
  - *Commit: `api: add merkletree(mode) replacing per-site #ifdef cascade`*
- [ ] **2.8** `tests.cpp`: add equivalence tests — `hashFullResult(..., Scalar) ≡ hash_full_result_seq(...)`, `merkletree(..., Auto) ≡ merkletree(..., Scalar)` on same input
  - *Commit: `test: add new-API≡old-API equivalence tests`*

**Verify**: gates (a), (b), (c) green. Old API still public — no callers migrated yet.

---

### Phase 3 — Migrate all callers to new API; make old methods private

**Files**: [transcriptGL.cpp](pil2-stark/src/starkpil/transcript/transcriptGL.cpp), [stark_verify.hpp](pil2-stark/src/starkpil/stark_verify.hpp), [merkleTreeGL.cpp](pil2-stark/src/starkpil/merkleTree/merkleTreeGL.cpp), [tests.cpp](pil2-stark/src/goldilocks/tests/tests.cpp), [bench.cpp](pil2-stark/src/goldilocks/benchs/bench.cpp), [poseidon2_goldilocks.hpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp).

- [ ] **3.1** `transcriptGL.cpp:22,25,28` — `hash_full_result_seq(out, in)` → `hashFullResult(out, in, Poseidon2Mode::Scalar)`
  - *Commit: `migrate: transcriptGL hash_full_result_seq → hashFullResult`*
- [ ] **3.2** `stark_verify.hpp:199` — same migration
  - *Commit: `migrate: stark_verify hash_full_result_seq → hashFullResult`*
- [ ] **3.3** `merkleTreeGL.cpp:182,185,188` — `linear_hash_seq(...)` → `linearHash(..., Poseidon2Mode::Scalar)`
  - *Commit: `migrate: merkleTreeGL linear_hash_seq → linearHash`*
- [ ] **3.4** `merkleTreeGL.cpp:236,250,264` — `hash_seq(...)` → `hash(..., Poseidon2Mode::Scalar)`
  - *Commit: `migrate: merkleTreeGL hash_seq → hash`*
- [ ] **3.5** `merkleTreeGL.cpp:280-306` — replace the manual per-arity `#ifdef` cascade (12 lines × 3 arities) with:
  ```cpp
  // arity 2:
  Poseidon2Goldilocks<8>::merkletree(nodes, source, width, height, arity, 0, 1,
                                     Poseidon2Mode::Auto);
  // arity 3: Poseidon2Goldilocks<12>; arity 4: Poseidon2Goldilocks<16>
  ```
  - *Commit: `migrate: merkleTreeGL::merkelize() replace #ifdef cascade with merkletree(Auto)`*
- [ ] **3.6** `tests.cpp`: update all test calls to Mode-parameter API; iterate over `{Scalar, Avx}` in cross-checks
  - *Commit: `test: migrate tests to Mode-parameter API`*
- [ ] **3.7** `bench.cpp`: update all bench calls to Mode-parameter API
  - *Commit: `bench: migrate benches to Mode-parameter API`*
- [ ] **3.8** Move all `_seq`/`_avx`/`_avx512`/`_batch_*` public methods to `private:` in `poseidon2_goldilocks.hpp`
  - *Commit: `api: make _seq/_avx/_batch_* private`*
- [ ] **3.9** Move `partial_merkle_tree` to `private:` (or delete if no internal caller surfaced)
  - *Commit: `api: make partial_merkle_tree private`*

**Verify**: gates (a), (b), (c) green. Any missed call site will fail to compile or produce a wrong Merkle root caught by (c).

---

### Phase 4 — NTT rename: `extendPol` → `LDE`

**Files**: [ntt_goldilocks.hpp](pil2-stark/src/goldilocks/src/ntt_goldilocks.hpp), [ntt_goldilocks.cpp](pil2-stark/src/goldilocks/src/ntt_goldilocks.cpp), all callers.

- [ ] **4.1** Rename in `ntt_goldilocks.hpp` declaration + `ntt_goldilocks.cpp` definition; keep all parameters identical
  - *Commit: `api: rename extendPol → LDE in ntt_goldilocks`*
- [ ] **4.2** Update [const_pols.hpp:38](pil2-stark/src/starkpil/const_pols.hpp#L38) and [const_pols.hpp:58](pil2-stark/src/starkpil/const_pols.hpp#L58)
  - *Commit: `migrate: extendPol → LDE in const_pols.hpp`*
- [ ] **4.3** Update [build_const_tree.cpp:44](pil2-stark/src/bctree/build_const_tree.cpp#L44)
  - *Commit: `migrate: extendPol → LDE in build_const_tree.cpp`*
- [ ] **4.4** Update [starks_api.cpp:634](pil2-stark/src/api/starks_api.cpp#L634) and [starks_api.cpp:682](pil2-stark/src/api/starks_api.cpp#L682)
  - *Commit: `migrate: extendPol → LDE in starks_api.cpp`*
- [ ] **4.5** Update [starks.hpp:125-158](pil2-stark/src/starkpil/starks.hpp#L125) (all occurrences)
  - *Commit: `migrate: extendPol → LDE in starks.hpp`*
- [ ] **4.6** Grep sweep: confirm zero remaining `extendPol` in `pil2-stark/src/` (excluding comments)
  - *Commit: (none — sweep; fix any stragglers found)*

**Verify**: gates (a), (b), (c) green.

---

### Phase 5 — GPU `Layout` parameter unification

**Files**: [poseidon2_goldilocks.cuh](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.cuh), [poseidon2_goldilocks.cu](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.cu), and all GPU callers.

- [ ] **5.1** Add `enum class Layout : uint8_t { RowMajor, Tiles };` near top of `poseidon2_goldilocks.cuh`
  - *Commit: `api: add Layout enum to poseidon2_goldilocks.cuh`*
- [ ] **5.2** Collapse `linearHash` + `linearHashTiled` → `linearHash(..., Layout layout)` in header; body dispatches to existing kernels
  - *Commit: `api: collapse linearHash+linearHashTiled → linearHash(Layout)`*
- [ ] **5.3** Collapse `merkletree` + `merkletreeTiled` → `merkletree(..., Layout layout)` in header
  - *Commit: `api: collapse merkletree+merkletreeTiled → merkletree(Layout)`*
- [ ] **5.4** Collapse `buildMerkleTreeGPU` + `buildMerkleTreeTilesGPU` → `buildMerkleTreeGPU(..., Layout layout, ...)`
  - *Commit: `api: collapse buildMerkleTree*GPU → buildMerkleTreeGPU(Layout)`*
- [ ] **5.5** Update `poseidon2_goldilocks.cu` implementations to match new signatures
  - *Commit: `impl: update poseidon2_goldilocks.cu for Layout parameter`*
- [ ] **5.6** Update `starks_gpu.cu` — all `buildMerkleTreeTilesGPU(...)` → `buildMerkleTreeGPU(..., Layout::Tiles, ...)`; FRI site ([starks_gpu.cu:823](pil2-stark/src/starkpil/starks_gpu.cu#L823)) → `..., Layout::RowMajor, ...`
  - *Commit: `migrate: starks_gpu.cu → buildMerkleTreeGPU(Layout)`*
- [ ] **5.7** Update `starks_api.cu` callers
  - *Commit: `migrate: starks_api.cu → buildMerkleTreeGPU(Layout)`*
- [ ] **5.8** Update `gen_commit.cuh` callers
  - *Commit: `migrate: gen_commit.cuh → buildMerkleTreeGPU(Layout)`*
- [ ] **5.9** Update GPU `tests.cu`
  - *Commit: `migrate: tests.cu → new Layout-parameter API`*
- [ ] **5.10** Update GPU `bench.cu`
  - *Commit: `migrate: bench.cu → new Layout-parameter API`*
- [ ] **5.11** Delete old `linearHashTiled`, `merkletreeTiled`, `buildMerkleTreeTilesGPU` symbols
  - *Commit: `api: delete superseded *Tiled and buildMerkleTreeTilesGPU symbols`*

**Verify**: gates (a), (b), (c) green. A `Layout` swap silently corrupts Merkle roots — gate (c) `--verify-proofs` is the canary.

---

### Phase 6 — Hygiene cleanup

**Files**: CPU goldilocks sources.

- [ ] **6.1** `ntt_goldilocks.hpp` + `ntt_goldilocks.cpp`: `u_int64_t` → `uint64_t`, `u_int32_t` → `uint32_t` throughout
  - *Commit: `cleanup: u_int*_t → uint*_t in ntt_goldilocks`*
- [ ] **6.2** `poseidon2_goldilocks.hpp` and AVX headers: same type substitution for any remaining occurrences
  - *Commit: `cleanup: u_int*_t → uint*_t in poseidon2_goldilocks`*
- [ ] **6.3** Delete `partial_merkle_tree` body + declaration if no internal caller surfaced after Phase 3
  - *Commit: `cleanup: delete partial_merkle_tree (no callers)`*
- [ ] **6.4** Final grep sweep — zero `_seq`/`_avx`/`_avx512` outside `private:`; zero `merkletree_batch`; zero `extendPol`
  - *Commit: (none — sweep; fix any found)*

**Verify**: gates (a), (b), (c) green.

---

### Phase 7 — AVX512 implementation (requires AVX512 host)

**Files**: [poseidon2_goldilocks_avx512.hpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks_avx512.hpp), [poseidon2_goldilocks.hpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp).

- [ ] **7.1** Implement `pow7_avx512` — `__m256i` → `__m512i`, 4 lanes → 8
  - *Commit: `avx512: implement pow7_avx512`*
- [ ] **7.2** Implement `add_avx512`
  - *Commit: `avx512: implement add_avx512`*
- [ ] **7.3** Implement `add_avx512_small`
  - *Commit: `avx512: implement add_avx512_small`*
- [ ] **7.4** Implement `matmul_external_avx512`
  - *Commit: `avx512: implement matmul_external_avx512`*
- [ ] **7.5** Implement `hash_avx512` (single-element, uses 7.1–7.4)
  - *Commit: `avx512: implement hash_avx512`*
- [ ] **7.6** Implement `hash_full_result_avx512`
  - *Commit: `avx512: implement hash_full_result_avx512`*
- [ ] **7.7** Implement `linear_hash_avx512`
  - *Commit: `avx512: implement linear_hash_avx512`*
- [ ] **7.8** Implement `merkletree_avx512`
  - *Commit: `avx512: implement merkletree_avx512`*
- [ ] **7.9** Replace `Avx512`/`Avx512Batch` abort branches in Phase-2 dispatchers with real calls to new AVX512 functions; update `resolveAuto()` if needed
  - *Commit: `avx512: wire Avx512/Avx512Batch branches in mode dispatcher`*
- [ ] **7.10** Run `testscpu_avx512` on AVX512 host; add cross-check: `Scalar ≡ Avx ≡ Avx512` for all (width, size) pairs
  - *Commit: `test: validate AVX512 correctness on AVX512 host`*
- [ ] **7.11** Bench AVX512 vs AVX2; record `benchs/baseline/$(hostname)_avx512.txt`
  - *Commit: `bench: record AVX512 baseline on AVX512 host`*
- [ ] **7.12** Delete `testscpu_avx512` target from Makefile — AVX512 is now covered by `testscpu` auto-detection; compile-check target no longer needed
  - *Commit: `cleanup: remove testscpu_avx512 target (AVX512 validated)`*

**Verify (AVX512 host)**: all 3 gates + `testscpu_avx512`. On non-AVX512 hosts: behavior identical to Phase 6.

---

### Phase 8 — Comment audit (all phases complete)

Sweep all files touched during Phases 0–7 and remove development scaffolding comments, leaving only comments that are meaningful to a future reader of the code.

**Files**: all modified `.hpp`, `.cpp`, `.cu`, `.cuh`, `tests.cpp`, `bench.cpp`, `Makefile`.

- [ ] **8.1** Remove all `// TODO Phase N:` comments — these are development notes, not code documentation
  - *Commit: `cleanup: remove TODO Phase N development comments`*
- [ ] **8.2** Remove `// Phase 7 only`, `// build-only`, and similar scaffolding notes that described temporary constraints now resolved
  - *Commit: `cleanup: remove scaffolding comments (constraints resolved)`*
- [ ] **8.3** Uncomment or delete `// TODO Phase 7` blocks in `tests.cpp` — either the test is now live (uncomment + adapt) or it was superseded (delete)
  - *Commit: `cleanup: resolve all TODO Phase 7 test stubs`*
- [ ] **8.4** Final grep sweep: zero occurrences of `TODO Phase`, `Phase 7 only`, `build-only`, `Severity-A`, `Severity-B`, `Severity-C` in non-comment prose; zero dead `#if 0` blocks introduced during this refactor
  - *Commit: (none — sweep; fix any found)*

**Verify**: gates (a), (b), (c) green.

---

## 8. Progress

| Phase | Description | Steps | Done |
|---|---|---|---|
| 0 | Test & bench scaffolding | 6 | 1 |
| 1 | Delete dead wrappers | 4 | 0 |
| 2 | Introduce Poseidon2Mode + new API | 8 | 0 |
| 3 | Migrate callers; make old API private | 9 | 0 |
| 4 | NTT rename extendPol → LDE | 6 | 0 |
| 5 | GPU Layout parameter | 11 | 0 |
| 6 | Hygiene cleanup | 4 | 0 |
| 7 | AVX512 implementation (AVX512 host) | 12 | 0 |
| 8 | Comment audit | 4 | 0 |
| **Total** | | **64** | **1** |

---

## 9. Critical files

| Phase | Files modified |
|---|---|
| 0 | [pil2-stark/Makefile](pil2-stark/Makefile), [tests.cpp](pil2-stark/src/goldilocks/tests/tests.cpp), new [scripts/bench_compare.sh](pil2-stark/scripts/bench_compare.sh) |
| 1 | [poseidon2_goldilocks.hpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp) |
| 2 | [poseidon2_goldilocks.hpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp) |
| 3 | [poseidon2_goldilocks.hpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp), [transcriptGL.cpp](pil2-stark/src/starkpil/transcript/transcriptGL.cpp), [stark_verify.hpp](pil2-stark/src/starkpil/stark_verify.hpp), [merkleTreeGL.cpp](pil2-stark/src/starkpil/merkleTree/merkleTreeGL.cpp), [tests.cpp](pil2-stark/src/goldilocks/tests/tests.cpp), [bench.cpp](pil2-stark/src/goldilocks/benchs/bench.cpp) |
| 4 | [ntt_goldilocks.hpp](pil2-stark/src/goldilocks/src/ntt_goldilocks.hpp), [ntt_goldilocks.cpp](pil2-stark/src/goldilocks/src/ntt_goldilocks.cpp), [const_pols.hpp](pil2-stark/src/starkpil/const_pols.hpp), [build_const_tree.cpp](pil2-stark/src/bctree/build_const_tree.cpp), [starks_api.cpp](pil2-stark/src/api/starks_api.cpp), [starks.hpp](pil2-stark/src/starkpil/starks.hpp) |
| 5 | [poseidon2_goldilocks.cuh](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.cuh), [poseidon2_goldilocks.cu](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.cu), [starks_gpu.cu](pil2-stark/src/starkpil/starks_gpu.cu), [starks_api.cu](pil2-stark/src/api/starks_api.cu), [gen_commit.cuh](pil2-stark/src/starkpil/gen_commit.cuh), [tests.cu](pil2-stark/src/goldilocks/tests/tests.cu), [bench.cu](pil2-stark/src/goldilocks/benchs/bench.cu) |
| 6 | CPU goldilocks sources |
| 7 | [poseidon2_goldilocks_avx512.hpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks_avx512.hpp), [poseidon2_goldilocks.hpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp) |

Reference (gold standard, unchanged except Phase 5):
- [ntt_goldilocks.cuh](pil2-stark/src/goldilocks/src/ntt_goldilocks.cuh)
- [poseidon2_goldilocks.cuh](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.cuh)
