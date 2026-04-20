---
name: goldilocks_gpu_improvement
description: Status and context for the goldilocks GPU cleanup and performance improvement project
type: project
---

Working on: /home/rick/pil2-proofman/pil2-stark/src/goldilocks/

**Phase 0 COMPLETE (2026-03-18)** — baseline tests + benchmarks added and passing.

**Why:** Systematic cleanup of NTT/Poseidon2/merkle GPU code that grew organically — mixed concerns, dead code, poor naming, no GPU NTT tests.

**How to apply:** Check current phase before starting work each session; verify each step with `./testsgpu` + prover run.

## Prover verification command (corrected):
```bash
cd /home/rick/pil2-proofman/pil2-stark && make -j starks_lib_gpu
cd /home/rick/pil2-proofman && touch provers/starks-lib-c/build.rs && cargo build --features gpu --bin proofman-cli
cargo run --features gpu --bin proofman-cli prove \
    --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
    --proving-key examples/fibonacci-square/build/provingKey/ \
    --public-inputs examples/fibonacci-square/src/inputs.json \
    --output-dir examples/fibonacci-square/build/proofs \
    --custom-commits rom=examples/fibonacci-square/build/rom_gpu.bin -y -f -vv
```

## Baseline timings (captured 2026-03-18):
- INTT_GPU 2^20: 0.155 ms, 2^22: 0.580 ms, 2^24: 3.80 ms
- LINEAR_HASH12 (24 cols, 2^23 rows): 18.9 ms
- MERKLETREE12 (24 cols, 2^23 rows, arity=3): 22.1 ms

## free_twiddle_factors_and_r pitfall:
Do NOT call `NTT_Goldilocks_GPU::free_twiddle_factors_and_r()` in benchmarks or tests that run in sequence. The function resets `maxLogDomainSize` to 0, so subsequent `NTT_Goldilocks_GPU(24, ...)` constructor calls skip re-init (condition `24 > 0` would pass but there's a static check), causing a segfault on the freed twiddle factor pointers. The twiddle factors are static GPU memory — let them persist for the process lifetime.

## Phase 1 — Revise GPU interfaces (in progress)
- Plan details in `GPU_IMPROVEMENT_PLAN.md` (repo file, single source of truth)
- **Step 1.1 COMPLETE (2026-03-20)**: Removed 2 dead transpose kernels, dropped unused TimerGPU param from prepare_blocks_trace. All tests pass, benchmarks unchanged, prover verified.
- **Proof timing baseline**: Inner proofs 929ms, final proof 234ms, total ~1.2s (release build, RTX 5090)
- Steps 1.2–1.6 remaining: extract toBlockTiled free fn, split LDE_MerkleTree_GPU/computeQ_MerkleTree_inplace, create MerkleTreeGoldilocksGPU class, clean dead Poseidon2 code
