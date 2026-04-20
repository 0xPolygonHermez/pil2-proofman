---
name: Poseidon v1 port — pending AVX512 validation
description: Pending validation step for the Poseidon v1 port on feature/add_poseidon1 — the AVX512 code paths have never been executed on real hardware.
type: project
originSessionId: 34e13233-947b-480a-b334-aad942efa84d
---
Poseidon v1 port on branch `feature/add_poseidon1`: CPU port complete and all 17 scalar + AVX2 tests pass against 0.14.0 golden vectors on the user's current machine. The AVX512 batch code paths (`permute_batch_avx512`, `compress_batch_avx512`, `linear_hash_batch_avx512`, `merkletree_batch_avx512`) compile but **have never been executed**: this machine's CPU lacks AVX512 (`grep -m1 avx512f /proc/cpuinfo` returns nothing), so the Makefile does not enable `-mavx512f -D__AVX512__` and those tests / benches are `#ifdef`'d out.

**Why:** User said they will test on an AVX512-capable server later and asked to be reminded.

**How to apply:** When the user is next on an AVX512 host (or when a CI runner with AVX512 is available), remind them to:
  1. Rebuild in the other environment: `cd pil2-stark/src/goldilocks && make testscpu`. Expected: `-D__AVX512__` is auto-detected via `/proc/cpuinfo`.
  2. Run `./testscpu --gtest_filter='PoseidonV1*Avx512*'` — must be green (the `merkletree_binary_fib128x64_avx512batch` test is the one gated by `#ifdef __AVX512__`).
  3. Run `./benchscpu --benchmark_filter='.*AVX512.*'` — record timings.
  4. If any AVX512 test fails, the most likely culprit is `poseidon_goldilocks_avx512.hpp` (port of 0.14.0 primitives templated to `<W>`) or the batch reduction stride in `poseidon_goldilocks.cpp` (already fixed to `arity * CAPACITY`, not `SPONGE_WIDTH`).

## GPU optimization pass (feature/add_poseidon1)

Three optimizations landed in [poseidon_goldilocks.cuh](pil2-stark/src/goldilocks/src/poseidon_goldilocks.cuh) / [.cu](pil2-stark/src/goldilocks/src/poseidon_goldilocks.cu):
1. **Shared-memory state** for `linearHashKernel_pos1` / `linearHashTiledKernel_pos1` / `merkleNodeKernel_pos1` — removes the 96-byte stack spill present on every thread; drops regs/thread 63→48 on linearHash. Launched with dynamic smem = `TPB_POS1 * SPONGE_WIDTH * sizeof(uint64_t)`.
2. **Fused partial round** (0.14.0 style): `dot + prod+add` interleaved in a single inner loop, halving scratchpad traffic in partial rounds.
3. **`state[0]` register-cached across all 22 partial rounds** — saves 43 smem ops per permutation. Only state[0] (1 register); caching the full W-element state blew reg count to 255 and tanked occupancy (-40%), reverted.

Result: ~20% speedup across `linearHash` and `merkletree` at W=12 arity=2. v1/v2 ratio tightened from ~2.85× to ~2.20–2.30×.

Tried and rejected: `__launch_bounds__(TPB, N)` (compiler gave itself more regs, slower); TPB sweep 64–256 (no change — occupancy is smem-limited at 58% per ncu, hitting the hardware ceiling); full unrolling of `mvp_smem_` outer loop (blew register count, slower).

Further gains would require algorithm-level rewrites (warp-shuffle MDS multiply across 12-thread sponge groups) or accepting Poseidon2 where interop allows.

## Poseidon2 GPU optimization pass (for same prover critical path)

Ran the same optimization playbook on the Poseidon2 GPU (`poseidon2_goldilocks.cu` / `.cuh`) — **no meaningful wins available**. The Poseidon2 GPU code was already highly optimized:
- Zero stack spill (in contrast to Poseidon v1's 96-byte spill — the big lever on v1).
- ncu: 57.9% achieved / 58.33% theoretical occupancy (smem-limited at 7 blocks/SM).
- 29% memory throughput → compute-bound, not memory-bound.
- Warp cycles/issued: 11.64 → latency-bound within a warp.

Attempted & reverted (each measured as a no-op or regression):
1. **state[0] in register + fused sum+mul in partial rounds**: reverted. The inlined `sumAllSmem` + `partialRoundMulSmem` sequence had adjacent smem accesses that nvcc already fuses during register allocation.
2. **Fused sbox+mds in full rounds**: reverted. Same reason — adjacent inlined calls already held state in registers across the two passes.
3. **TPB sweep (64..256)** + **launch_bounds(TPB, 2..8)**: all within 1%, occupancy cap at smem limit.

Conclusion: Poseidon2 GPU permute throughput is at the arch ceiling for sm_120. The only remaining levers are algorithm-level (warp-shuffle MDS, persistent kernels spanning merkle levels) which are major rewrites. Searched ICICLE / Plonky3 / SP1 / Polygon / OpenVM / HorizenLabs — no open-source implementation beats the current one using conventional techniques. Lita Foundation's NVRTC impl claims 5.3× over ICICLE via runtime compilation (non-portable, untested).

**Operative take**: v1 was the hot target (fixed spill, fused partial rounds → −20%). v2 is already cold-optimized.
