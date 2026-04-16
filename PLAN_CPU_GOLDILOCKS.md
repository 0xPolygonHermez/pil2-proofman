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
       Auto = 0,        // resolves to the best variant compiled in for that operation
       Scalar,
       Avx,
       AvxBatch,        // backs merkletree_batch_avx, linear_hash_batch_avx, etc.
       Avx512,          // single-sponge AVX512 — only available after Phase 7
       Avx512Batch,     // backs merkletree_batch_avx512; already implemented today
   };
   ```
   - `Auto` centralizes the `#ifdef` cascade inside the new public methods (one place per operation, not per call site).
   - Explicit modes abort loudly if the requested SIMD level was not compiled in — misuse is a build-config bug, not a silent fallback.
3. **Per-operation `Auto` resolution** — each public method resolves `Auto` according to the backends that make sense for that operation:

   | Operation | `Auto` resolves to |
   |---|---|
   | `hashFullResult`, `hash` | `Avx512` / `Avx` / `Scalar` |
   | `linearHash`, `merkletree` | `Avx512Batch` / `AvxBatch` / `Scalar` |

   If implementation convenience suggests a helper, it stays private to the implementation rather than becoming part of the public design.
4. **Per-operation valid modes** — rule: *if the backing primitive exists, expose the mode*. Single-sponge ops (`hashFullResult`, `hash`) have no meaningful batched semantics, so `*Batch` modes abort at runtime. Everything else accepts all six modes.

   | Operation | Valid modes | `Auto` resolves to |
   |---|---|---|
   | `hashFullResult`, `hash`, `linearHash` | `Auto`, `Scalar`, `Avx`, `Avx512` | `Avx512` / `Avx` / `Scalar` |
   | `merkletree` | all six modes | `Avx512Batch` / `AvxBatch` / `Scalar` |

   `linearHash` is single-sponge only: the `linear_hash_batch_avx` / `linear_hash_batch_avx512` primitives hash 4 / 8 contiguous rows per call (different contract, not "same op with vectorization") and are internal building blocks of `merkletree`, not callable under the single-row public API. They remain private and reachable only via `merkletree(..., AvxBatch/Avx512Batch)`.

   For `merkletree`, the single-sponge `Avx` / `Avx512` modes produce correct trees (just slower than batched) and are exposed for benchmarking and A/B testing. Production callers use `Auto`.

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

Signatures below are final as of Phase 3. Phase 7 only wires up the single-sponge AVX512 backends for modes that already exist in the enum.

| Operation | CPU (final signature; Phase 3) | GPU (after Phase 5) |
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
     --output-dir examples/fibonacci-square/build/proofs 2>&1 | tee /tmp/prove.log

# Compare proof time against reference (zisk1, min of 3 baseline runs = 1052 ms)
t=$(grep -oP 'GENERATE_VADCOP_FINAL_COMPRESSED_PROOF \(\K\d+' /tmp/prove.log | tail -1)
echo "prove time: ${t} ms   reference: 1052 ms   delta: $(( (t - 1052) * 100 / 1052 ))%"
```

**Proof-time reference**: `1052 ms` on `zisk1` (minimum of 3 runs before Phase 1).
**Regression threshold**: unset for now — record the delta after each step; once we have 5–10 samples through Phases 1–2 we'll see the natural jitter band and pick a number. Until then, investigate anything >10 %.

All three gates must pass after every step. Gate (c) is load-bearing — Merkle-root divergence makes the verifier reject.

---

## 7. Granular implementation steps

### Phase 0 — Test & bench scaffolding (no behavior change)

**Files**: [pil2-stark/src/goldilocks/Makefile](pil2-stark/src/goldilocks/Makefile), [tests.cpp](pil2-stark/src/goldilocks/tests/tests.cpp), new [pil2-stark/scripts/bench_compare.sh](pil2-stark/scripts/bench_compare.sh), new `src/goldilocks/benchs/baseline/`.

- [x] **0.1** `Makefile`: add `testscpu_avx512` target — `-mavx512f -D__AVX512__`; compiles on any host, runs only on AVX512 hardware (Phase 7). Single `testscpu` binary covers scalar + AVX2 via explicit cross-checks.
  - *Commit: `test: add testscpu_avx512 build target and fix pre-existing AVX512 template errors`*
- [x] **0.2** `scripts/bench_compare.sh baseline.txt fresh.txt` — exits non-zero if any line regresses > 2 %
  - *Commit: `test: add bench_compare.sh regression detector`*
- [x] **0.3** `tests.cpp`: add `extendPol` correctness test — multiple `(N, N_Ext, ncols)` pairs; reference: Horner evaluation of p at each coset point
  - *Commit: `test: add extendPol end-to-end correctness test`*
- [x] **0.4** `tests.cpp`: add merkletree cross-check — `merkletree_seq ≡ merkletree_batch_avx` (and `≡ _batch_avx512` under `#ifdef __AVX512__`) for widths {4,8,12,16}, rows ∈ {2¹⁰, 2¹⁵}, cols ∈ {1, 8, 64, 100}
  - *Commit: `test: add merkletree seq≡avx_batch cross-check`*
- [x] **0.5** `tests.cpp`: add explicit test exercising `merkletree(...)` and `merkletree_batch(...)` wrappers — characterizes the Severity-A bug so Phase 1 deletion is documented
  - *Commit: `test: document merkletree_batch wrapper Severity-A bug`*
- [x] **0.6** `bench.cpp`: fill benchmark coverage gaps using current `_seq`/`_avx` API (before Phase 2 introduces the Mode parameter):
  - `hash_full_result_seq` + `hash_full_result_avx` for widths {4, 8, 12}
  - `linear_hash_seq` + `linear_hash_avx` for widths {8, 12} — W=4 has RATE=0 so linear_hash doesn't terminate; skipped
  - `merkletree_seq` + `merkletree_batch_avx` for (W=12, arity=3) and (W=16, arity=4)
  - Standalone `INTT` benchmark (currently implicit inside extendPol only)
  - *Commit: `bench: add missing width/arity/INTT benchmark coverage`*
- [x] **0.7** Record bench baseline: `make -j benchscpu && ./benchscpu > src/goldilocks/benchs/baseline/$(hostname).txt`
  - *Commit: `bench: record $(hostname) baseline`*

**Verify**: gates (a), (b), (c) green.

---

### Phase 1 — Delete dead `merkletree`/`merkletree_batch` wrappers

The name `merkletree` is reclaimed here so Phase 2 can rebuild it correctly with the mode parameter. The underlying primitives (`merkletree_seq`, `merkletree_avx`, `merkletree_batch_avx`, `merkletree_batch_avx512`) are untouched — every one is exposed again in Phase 2 via a `Poseidon2Mode` (see §2 valid-modes table), so no capability is lost.

**Files**: [poseidon2_goldilocks.hpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp), [tests.cpp](pil2-stark/src/goldilocks/tests/tests.cpp).

**Execution order**: 1.1 → 1.4 → 1.2 → 1.3 (the characterization test calls both wrappers, so deleting a wrapper before the test breaks mid-series compilation — bisect-hostile).

- [x] **1.1** Grep confirm: no caller in `pil2-stark/src/` or `pil2-proofman/` references the *wrapper* `Poseidon2Goldilocks<W>::merkletree(` (non-variant signature). If found, investigate before continuing.
  - *Commit: (none — audit step)*
- [x] **1.4** Remove Phase-0 wrapper-characterization test (0.5) — replace with comment confirming wrappers are gone
  - *Commit: `test: remove wrapper-bug test (wrappers deleted)`*
- [x] **1.2** Delete the `merkletree(...)` wrapper from [poseidon2_goldilocks.hpp:104-115](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp#L104)
  - *Commit: `refactor: delete dead merkletree wrapper (Severity-A bug)`*
- [x] **1.3** Delete the `merkletree_batch(...)` wrapper from [poseidon2_goldilocks.hpp:117-127](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp#L117)
  - *Commit: `refactor: delete dead merkletree_batch wrapper`*

**Verify**: gates (a), (b), (c) green.

---

### Phase 2 — Introduce `Poseidon2Mode` + new public API alongside old

**Files**: [poseidon2_goldilocks.hpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp), [poseidon2_goldilocks.cpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.cpp) (or `.hpp` if header-only).

- [x] **2.1** Add `Poseidon2Mode` enum (`Auto`, `Scalar`, `Avx`, `AvxBatch`, `Avx512`, `Avx512Batch`) near top of `poseidon2_goldilocks.hpp`
  - *Commit: `api: add Poseidon2Mode enum`*
- [x] **2.2** Add `static void hashFullResult(Element *out, const Element *in, Poseidon2Mode mode)` — dispatches to existing `_seq`/`_avx`; `Auto` resolves locally to `Avx512` / `Avx` / `Scalar`; `Avx512*` aborts with clear message until Phase 7
  - *Commit: `api: add hashFullResult(mode) alongside hash_full_result_seq`*
- [x] **2.3** Add `static void hash(Element (&state)[CAPACITY], const Element (&in)[W], Poseidon2Mode mode)` — `Auto` resolves locally to `Avx512` / `Avx` / `Scalar`
  - *Commit: `api: add hash(mode) alongside hash_seq`*
- [x] **2.4** Add `static void linearHash(Element *out, Element *in, uint64_t size, Poseidon2Mode mode)` — single-sponge only; valid modes `Auto`/`Scalar`/`Avx`/`Avx512`; `Auto` resolves locally to `Avx512` / `Avx` / `Scalar`; `*Batch` modes abort (those primitives have a 4/8-row contract, reachable only via `merkletree`)
  - *Commit: `api: add linearHash(mode) alongside linear_hash_seq`*
- [x] **2.5** Add `static void merkletree(Element *tree, Element *in, uint64_t nCols, uint64_t nRows, uint64_t arity, int nThreads, uint64_t dim, Poseidon2Mode mode)` — accepts all six modes; `Auto` resolves locally to `Avx512Batch` / `AvxBatch` / `Scalar`; `Scalar` → `merkletree_seq`, `Avx` → `merkletree_avx`, `AvxBatch` → `merkletree_batch_avx`, `Avx512Batch` → `merkletree_batch_avx512`; `Avx512` aborts until Phase 7 (step 7.8)
  - *Commit: `api: add merkletree(mode) replacing per-site #ifdef cascade`*
- [x] **2.6** `tests.cpp`: add equivalence tests — `hashFullResult(..., Scalar) ≡ hash_full_result_seq(...)`; for `merkletree`, iterate over every compiled-in mode (`Scalar`, `Avx`, `AvxBatch`, plus `Avx512Batch` under `#ifdef __AVX512__`) and assert identical roots; assert `merkletree(..., Auto)` matches the backend that the per-operation `Auto` resolution selected
  - *Commit: `test: add new-API≡old-API equivalence tests`*

**Verify**: gates (a), (b), (c) green. Old API still public — no callers migrated yet.

---

### Phase 3 — Migrate all callers to new API; make old methods private

**Files**: [transcriptGL.cpp](pil2-stark/src/starkpil/transcript/transcriptGL.cpp), [stark_verify.hpp](pil2-stark/src/starkpil/stark_verify.hpp), [merkleTreeGL.cpp](pil2-stark/src/starkpil/merkleTree/merkleTreeGL.cpp), [tests.cpp](pil2-stark/src/goldilocks/tests/tests.cpp), [bench.cpp](pil2-stark/src/goldilocks/benchs/bench.cpp), [poseidon2_goldilocks.hpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp).

- [x] **3.1** `transcriptGL.cpp:22,25,28` — `hash_full_result_seq(out, in)` → `hashFullResult(out, in, Poseidon2Mode::Scalar)`
  - *Commit: `migrate: transcriptGL hash_full_result_seq → hashFullResult`*
- [x] **3.2** `stark_verify.hpp:199` — same migration
  - *Commit: `migrate: stark_verify hash_full_result_seq → hashFullResult`*
- [x] **3.3** `merkleTreeGL.cpp:182,185,188` — `linear_hash_seq(...)` → `linearHash(..., Poseidon2Mode::Scalar)`
  - *Commit: `migrate: merkleTreeGL linear_hash_seq → linearHash`*
- [x] **3.4** `merkleTreeGL.cpp:236,250,264` — `hash_seq(...)` → `hash(..., Poseidon2Mode::Scalar)`
  - *Commit: `migrate: merkleTreeGL hash_seq → hash`*
- [x] **3.5** `merkleTreeGL.cpp:280-306` — replace the manual per-arity `#ifdef` cascade with `merkletree(..., Poseidon2Mode::Auto)`
  - *Commit: `migrate: merkleTreeGL::merkelize() replace #ifdef cascade with merkletree(Auto)`*
- [x] **3.6** `tests.cpp`: migrate tests to Mode-parameter API; tests now exercise only the public Mode interface (private primitives are validated implicitly via mode-equivalence checks). Direct-primitive tests (`poseidon2_avx_batch`, `merkletree_seq_avxbatch_cross_check`) deleted — covered by `mode_merkletree_equivalence`.
  - *Commit: `test: migrate tests to Mode-parameter API`*
- [x] **3.7** `bench.cpp`: migrate to Mode API; benches that targeted private primitives without a public Mode equivalent (`POSEIDON2_BENCH_FULL_AVX_BATCH`, `MERKLETREE_BATCH_BENCH`) deleted.
  - *Commit: `bench: migrate benches to Mode-parameter API`*
- [x] **3.8** Move all `_seq`/`_avx`/`_batch_*` to `private:` in `poseidon2_goldilocks.hpp`. Public surface is now Mode API + `grinding` + `partial_merkle_tree`.
  - *Commit: `api: make _seq/_avx/_batch_* private`*
- [x] **3.9** `partial_merkle_tree` — kept public after audit. It has live callers in [merkleTreeGL.hpp:78,81,84](pil2-stark/src/starkpil/merkleTree/merkleTreeGL.hpp#L78) (Merkle proof verification path) — backs no Mode but is a distinct standalone op, not a primitive of `merkletree`. Plan §3.2 originally listed it as unused; that was incorrect. No commit needed.
  - *Commit: (none — audit; status unchanged)*

**Verify**: gates (a), (b), (c) green. Any missed call site will fail to compile or produce a wrong Merkle root caught by (c).

---

### Phase 4 — NTT rename: `extendPol` → `LDE`

**Files**: [ntt_goldilocks.hpp](pil2-stark/src/goldilocks/src/ntt_goldilocks.hpp), [ntt_goldilocks.cpp](pil2-stark/src/goldilocks/src/ntt_goldilocks.cpp), all callers.

- [x] **4.1** Rename in `ntt_goldilocks.hpp` declaration + `ntt_goldilocks.cpp` definition; keep all parameters identical. Error message string updated too.
  - *Commit: `api: rename extendPol → LDE in ntt_goldilocks`*
- [x] **4.2** Updated [const_pols.hpp](pil2-stark/src/starkpil/const_pols.hpp) (2 call sites)
  - *Commit: `migrate: extendPol → LDE in const_pols.hpp`*
- [x] **4.3** Updated [build_const_tree.cpp](pil2-stark/src/bctree/build_const_tree.cpp) (1 call site)
  - *Commit: `migrate: extendPol → LDE in build_const_tree.cpp`*
- [x] **4.4** Updated [starks_api.cpp](pil2-stark/src/api/starks_api.cpp) (2 call sites)
  - *Commit: `migrate: extendPol → LDE in starks_api.cpp`*
- [x] **4.5** Updated [starks.hpp](pil2-stark/src/starkpil/starks.hpp) (4 call sites)
  - *Commit: `migrate: extendPol → LDE in starks.hpp`*
- [x] **4.6** Grep sweep: zero remaining `extendPol` under `pil2-stark/`. Also migrated `tests.cpp` (test renamed `extendPol_correctness` → `LDE_correctness`), `tests.cu` (3 call sites), `bench.cpp` (1 call site + `EXTENDEDPOL_BENCH` renamed to `LDE_API_BENCH` — the pre-existing `LDE_BENCH`/`LDE_BLOCK_BENCH` names refer to a different manual-composition bench and were kept unchanged).
  - *Commit: (none — sweep)*

**Verify**: gates (a), (b), (c) green.

---

### Phase 5 — GPU `Layout` parameter unification

**Files**: [poseidon2_goldilocks.cuh](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.cuh), [poseidon2_goldilocks.cu](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.cu), and all GPU callers.

- [x] **5.1** Add `enum class Layout : uint8_t { RowMajor, Tiles };` near top of `poseidon2_goldilocks.cuh`
  - *Commit: `api: add Layout enum to poseidon2_goldilocks.cuh`*
- [x] **5.2** Collapse `linearHash` + `linearHashTiled` → `linearHash(..., Layout layout)` in header; body dispatches to existing kernels
  - *Commit: `api: collapse linearHash+linearHashTiled → linearHash(Layout)`*
- [x] **5.3** Collapse `merkletree` + `merkletreeTiled` → `merkletree(..., Layout layout)` in header
  - *Commit: `api: collapse merkletree+merkletreeTiled → merkletree(Layout)`*
- [x] **5.4** Collapse `buildMerkleTreeGPU` + `buildMerkleTreeTilesGPU` → `buildMerkleTreeGPU(..., Layout layout, ...)`
  - *Commit: `api: collapse buildMerkleTree*GPU → buildMerkleTreeGPU(Layout)`*
- [x] **5.5** Update `poseidon2_goldilocks.cu` implementations to match new signatures
  - *Commit: `impl: update poseidon2_goldilocks.cu for Layout parameter`*
- [x] **5.6** Update `starks_gpu.cu` — all `buildMerkleTreeTilesGPU(...)` → `buildMerkleTreeGPU(..., Layout::Tiles, ...)`; FRI site → `..., Layout::RowMajor, ...`
  - *Commit: `migrate: starks_gpu.cu → buildMerkleTreeGPU(Layout)`*
- [x] **5.7** Update `starks_api.cu` callers (3 sites)
  - *Commit: `migrate: starks_api.cu → buildMerkleTreeGPU(Layout)`*
- [x] **5.8** Update `gen_commit.cuh` callers (1 site)
  - *Commit: `migrate: gen_commit.cuh → buildMerkleTreeGPU(Layout)`*
- [x] **5.9** Update GPU `tests.cu`
  - *Commit: `migrate: tests.cu → new Layout-parameter API`*
- [x] **5.10** Update GPU `bench.cu`
  - *Commit: `migrate: bench.cu → new Layout-parameter API`*
- [x] **5.11** Delete old `linearHashTiled`, `merkletreeTiled`, `buildMerkleTreeTilesGPU` symbols. (The internal `linearHashTiledKernel` stays — it's the device kernel launched by the `Layout::Tiles` branch.)
  - *Commit: `api: delete superseded *Tiled and buildMerkleTreeTilesGPU symbols`*

**Verify**: gates (a), (b), (c) green. A `Layout` swap silently corrupts Merkle roots — gate (c) `--verify-proofs` is the canary.

---

### Phase 6 — Hygiene cleanup

**Files**: CPU goldilocks sources.

- [x] **6.1** `ntt_goldilocks.hpp` + `ntt_goldilocks.cpp`: `u_int64_t` → `uint64_t`, `u_int32_t` → `uint32_t` throughout (80 occurrences). Also cleaned 4 occurrences in `bench.cpp` (from Phase 0.6 additions).
  - *Commit: `cleanup: u_int*_t → uint*_t in ntt_goldilocks`*
- [x] **6.2** `poseidon2_goldilocks.hpp` and AVX headers: already clean (zero occurrences). Scope extended to `goldilocks_base_field.cpp` (4 occurrences) to leave the entire goldilocks tree consistent.
  - *Commit: `cleanup: u_int*_t → uint*_t in poseidon2_goldilocks`*
- [x] **6.3** Deleted `merkletree_batch_seq` (confirmed dead after Phase 3 — backs no mode, no caller). `partial_merkle_tree` kept — live callers in [merkleTreeGL.hpp:78,81,84](pil2-stark/src/starkpil/merkleTree/merkleTreeGL.hpp#L78) (see Post-refactor follow-up for integration options).
  - *Commit: `cleanup: delete dead merkletree_batch_seq primitive`*
- [x] **6.4** Final grep sweep — zero `_seq`/`_avx`/`_avx512` outside `private:`; zero `extendPol`; `merkletree_batch` only appears in the still-live `_avx`/`_avx512` variants (which back `AvxBatch`/`Avx512Batch` modes — retained per plan §2).
  - *Commit: (none — sweep)*

**Verify**: gates (a), (b), (c) green.

---

### Phase 7 — AVX512 validation on AVX512 host

**Decision (supersedes the original Phase 7 plan)**: single-sponge AVX512 variants are **not implemented**. At the state sizes used by Poseidon2 (4–16 elements) an 8-lane `__m512i` offers no meaningful gain over the existing 4-lane `__m256i` single-sponge path, and the real AVX512 win case (hashing 8 parallel sponges) is already covered by `Avx512Batch`. Rather than invest engineering on a variant that would at best match AVX2 performance, Phase 7 is scoped to: validate the existing `Avx512Batch` path on AVX512 hardware, fix the pre-existing bugs that validation surfaces, and update the `Auto` resolution so callers don't inadvertently reach the unimplemented `Avx512` branch.

The `Poseidon2Mode::Avx512` enum value is retained as a reserved-but-unimplemented value: explicit requests abort loudly (per plan §2.2 — "misuse is a build-config bug, not a silent fallback"), and `Auto` never resolves there.

**Files**: [poseidon2_goldilocks.hpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp), [poseidon2_goldilocks.cpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.cpp), [poseidon2_goldilocks_avx512.hpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks_avx512.hpp), [Makefile](pil2-stark/src/goldilocks/Makefile).

- [x] **7.1** Update `Poseidon2Mode::Avx512` enum comment to document that single-sponge AVX512 is intentionally unimplemented (reason + pointer to `Avx512Batch`).
  - *Commit: `avx512: document single-sponge Avx512 as intentionally unimplemented`*
- [x] **7.2** Change `Auto` resolution in `hashFullResult` / `hash` / `linearHash` to pick `Avx` (not `Avx512`) on AVX512 hosts. `merkletree`'s `Auto` already resolves to `Avx512Batch` — unchanged.
  - *Commit: `avx512: Auto resolution skips unimplemented single-sponge Avx512`*
- [x] **7.3** Delete commented-out stale AVX512 single-sponge declarations in [poseidon2_goldilocks.hpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.hpp) and commented-out stale implementation blocks in [poseidon2_goldilocks.cpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks.cpp) / [poseidon2_goldilocks_avx512.hpp](pil2-stark/src/goldilocks/src/poseidon2_goldilocks_avx512.hpp). These referenced a retired Poseidon layout (SPONGE_WIDTH=24) and would never compile.
  - *Commit: `avx512: delete stale commented-out single-sponge code`*
- [x] **7.4** Fix pre-existing bug in `hash_full_result_batch_avx512`: constants were hardcoded to `C12`/`D12` regardless of `SPONGE_WIDTH`. For W=16 this read past the end of the 118-element `C12` / 12-element `D12` arrays and stack-smashed (first surfaced now that an AVX512 host exists). Port the same `SPONGE_WIDTH`-dispatched selector pattern used by `hash_full_result_batch_avx`.
  - *Commit: `avx512: fix hardcoded C12/D12 in hash_full_result_batch_avx512`*
- [x] **7.5** Fix pre-existing bug in `matmul_external_batch_avx512`: three `matmul_m4_batch_avx512` calls and an 8-element sum pattern were hardcoded for W=12. For W=8 this read past end of the register array (stack smash); for W=16 it missed `x[12..15]`. Port the `SPONGE_WIDTH`-parameterized loop used by `matmul_external_batch_avx`.
  - *Commit: `avx512: fix W=12-hardcoded matmul_external_batch_avx512`*
- [x] **7.6** Delete `testscpu_avx512` Makefile target and associated `BUILD_DIR_AVX512` machinery. Since Phase 0's introduction, AVX512 auto-detects when the host supports it, so a separate compile-check target serves no purpose. Help text and clean rule updated in lockstep.
  - *Commit: `cleanup: remove testscpu_avx512 target (AVX512 now auto-detected)`*

**Verify**: gate (a) `testscpu` — 35/35 pass on AVX512 host (including `mode_merkletree_equivalence` which now exercises `Avx512Batch` for W={8,12,16}, the path that surfaced the two pre-existing bugs). Gate (b) `benchscpu` — blocked by pre-existing libbenchmark 1.5.0 incompatibility (`->Name()` / `kSecond` unavailable); unrelated to Phase 7. Gate (c) end-to-end proof — run to verify `Avx512Batch` production path.

**Not implemented**: the original 7.1–7.11 (real `pow7_avx512`, `matmul_external_avx512`, `hash_full_result_avx512`, `linear_hash_avx512`, `merkletree_avx512`). See decision note above.

---

### Phase 8 — Comment audit (all phases complete)

Sweep all files touched during Phases 0–7 and remove development scaffolding comments, leaving only comments that are meaningful to a future reader of the code.

**Files**: all modified `.hpp`, `.cpp`, `.cu`, `.cuh`, `tests.cpp`, `bench.cpp`, `Makefile`.

- [x] **8.1** Removed `// TODO Phase 7` comment and the empty `poseidon_avx512` test shell (lines 1610-1665). The test ran zero assertions — superseded by `mode_merkletree_equivalence` which exercises `Avx512Batch` on AVX512 hosts.
  - *Commit: `cleanup: remove TODO Phase N development comments`*
- [x] **8.2** Removed all `Phase N`, `Step N.N`, `Severity-A` references from code comments in tests.cpp, bench.cpp, poseidon2_goldilocks.hpp, poseidon2_goldilocks.cuh. Comments now describe what things ARE, not which refactor step created them.
  - *Commit: `cleanup: remove scaffolding comments (constraints resolved)`*
- [x] **8.3** Deleted the entire pre-existing `#if 0` dead-test block (280 lines) in tests.cpp. It contained legacy tests referencing the nonexistent `PoseidonGoldilocks::` class, empty `merkletree_seq` shells, and `merkletree_avx` tests calling private primitives.
  - *Commit: `cleanup: delete #if 0 dead-test block`*
- [x] **8.4** Final grep sweep: zero `TODO Phase`, `Phase N`, `Severity-*` in code files; zero `#if 0` blocks introduced during this refactor. Pre-existing `#if 0 //__USE_CUDA__` block (CUDA tests, not our refactor) left as-is.
  - *Commit: (none — sweep)*

**Verify**: gates (a), (b), (c) green.

---

### Post-refactor follow-up — `partial_merkle_tree` integration

After Phase 8 lands, revisit `Poseidon2Goldilocks<W>::partial_merkle_tree`. It currently sits in the public API alongside the Mode-dispatched methods but doesn't follow that shape (single backend, `snake_case` name, no mode). Options to weigh:

1. **Fold into the Mode API** — rename to `partialMerkleTree`, add a `Poseidon2Mode` parameter even though only `Scalar` is implemented today. Consistent shape; room to grow if an AVX variant ever lands.
2. **Rename only** — `partialMerkleTree` (camelCase) without Mode, documenting it as a single-impl op parallel to `grinding`. No behavioral change.
3. **Leave as-is** — `partial_merkle_tree` stays snake_case as a recognized exception for historical reasons.

Live callers: [merkleTreeGL.hpp:78,81,84](pil2-stark/src/starkpil/merkleTree/merkleTreeGL.hpp#L78) (Merkle-proof verification path). Any choice requires updating all three call sites in lockstep. Decision should factor in whether an AVX partial-merkle implementation is on the roadmap.

---

## 8. Progress

| Phase | Description | Steps | Done |
|---|---|---|---|
| 0 | Test & bench scaffolding | 7 | 7 |
| 1 | Delete dead wrappers | 4 | 4 |
| 2 | Introduce Poseidon2Mode + new API | 6 | 6 |
| 3 | Migrate callers; make old API private | 9 | 9 |
| 4 | NTT rename extendPol → LDE | 6 | 6 |
| 5 | GPU Layout parameter | 11 | 11 |
| 6 | Hygiene cleanup | 4 | 4 |
| 7 | AVX512 validation on AVX512 host (scope reduced — see §7) | 6 | 6 |
| 8 | Comment audit | 4 | 4 |
| **Total** | | **57** | **57** |

---

## 9. Critical files

| Phase | Files modified |
|---|---|
| 0 | [pil2-stark/src/goldilocks/Makefile](pil2-stark/src/goldilocks/Makefile), [tests.cpp](pil2-stark/src/goldilocks/tests/tests.cpp), new [scripts/bench_compare.sh](pil2-stark/scripts/bench_compare.sh) |
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
