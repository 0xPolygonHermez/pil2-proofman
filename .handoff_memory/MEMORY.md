# pil2-proofman Project Memory

## Key Architecture
- **pil2-stark** submodule contains C++/CUDA code for STARK and PLONK provers
- Three key PLONK GPU files: `plonk_prover_gpu.cuh` (header), `plonk_prover_gpu.c.cuh` (logic), `plonk_prover.cu` (CUDA kernels)
- Build: `cd pil2-stark && make -j starks_lib_gpu` then `touch provers/starks-lib-c/build.rs && cargo build --release --features gpu --bin proofman-cli`
- Test: `cargo run --release --features gpu --bin proofman-cli -- prove-snark -k examples/fibonacci-square/build/provingKeySnark -p examples/fibonacci-square/build/proofs/vadcop_final_proof.bin -vv -o tmp`
- Verify: `cargo run --release --features gpu --bin proofman-cli -- verify-snark -p tmp/snark_proof.bin -k examples/fibonacci-square/build/provingKeySnark/final/final.verkey.json`

## Critical Pitfalls
- **Polynomial constructor zeros buffer**: `new Polynomial<Engine>(E, buffer, length, blindLength)` calls `initialize()` which zeros the entire buffer via `ThreadUtils::parset`. Always create the Polynomial BEFORE writing data into its `coef` buffer, or the data will be overwritten.
- **GPU OOM during recursive F proof**: d_eval buffers must be allocated AFTER recursive STARK proof completes (lazy alloc in prove(), not in setZkey), because recursive proof needs GPU memory too.
- **GPU OOM during MSM**: Temp MSM allocs need ~1.5 GB headroom. Don't allocate persistent buffers right before MSM calls.
- **sizeof(FrElement)=32, sizeof(G1PointAffine)=64**, domainSize=2^24, sDomain=512MB, fullBytes=2GB
- **CUDA stream sync with NTT**: `ntt_bn128_gpu_dev_ptr`/`intt_bn128_gpu_dev_ptr` use their OWN non-blocking stream (sppark). Default-stream kernels (zero_pad, gate kernels) must call `cudaDeviceSynchronize()` before NTT/INTT, or the NTT reads stale data. This caused "T Polynomial is not well calculated" errors.
- **blindCoefficients dual modification**: `blindCoefficients(bf, len)` in polynomial.c.hpp modifies BOTH `coef[N+i] += bf[i]` AND `coef[i] -= bf[i]`. GPU dPolCoef buffers (from IFFT) are unblinded — when reading poly coefficients on GPU for round5, must account for both high and low blinding corrections.
- **Fused gather kernel nConstraints bounds**: `kernelGatherZRatios` runs over `domainSize` threads but maps only have `nConstraints` entries. Must pass `nConstraints` and set a=b=c=0 for i >= nConstraints (zero-pad region).
- **GPU Fr::toMontgomery**: Use `Fr::toMontgomery(val)` (static method, takes Element&), NOT `val.to()` which doesn't exist on GPU `BN128GPUScalarField::Element`.
- **GPU poly eval blinding correction**: GPU has unblinded IFFT coefficients. After GPU eval, add CPU correction: `sum_j bf[j]*(x^(N+j) - x^j)`. For A/B/C (2 blind factors) and Z (3 blind factors). xiw^N = xi^N since omega^N=1.
- **d_scanWork overflow into adjacent buffers**: `d_scanWork` lives at `d_aux[4N..]` and needs `N/1024 + N/(1024*1024) + 1` elements for the 3-level prefix scan. Must include this in d_aux allocation size. Moving omega precompute to round0 (before round2 scan) exposed this — scan overwrote omega tables.
- **PLONK_GPU_TIMING flag**: `#define PLONK_GPU_TIMING` in plonk_prover_gpu.cuh controls all sub-round timing prints. Comment out to disable. Total `Execution time:` print and all `cudaDeviceSynchronize()` calls remain unconditional.

## GPU Optimization Status (feature/plonk_gpu_2)
Phases 1-13 complete:
- Phase 1: Deferred blinding + CPU correction for MSM commitments
- Phase 2b: Incremental computeT (4 CUDA kernels) + persistent PTau — 6.05s
- Phase 3: Unified GPU buffer (27 GiB single cudaMalloc) — ~5.4s
- Phase 4: GPU parallel prefix scans — ~4.16s
- Phase 5: GPU-resident MSM + divZh+split kernels — ~2.7s
- Phase 6: Fused CPU computeR+Wxi loop — ~2.42s
- Phase 7: Async D2H overlap with MSMs — ~2.08s
- Phase 8: GPU computeR+Wxi kernel — ~1.92s
- Phase 9: GPU witness gather + fused z_ratios — ~1.90s
- Phase 10: Async H2D zkey + GPU poly eval — ~1.73s
- Phase 11: Precomputed omega powers for gate kernels — ~1.66s
- Phase 12: Remove dead A/B/C D2H + CPU Polynomial + blindCoefficients — ~1.43s (−14%)
- Phase 13: Remove dead T1/T2/T3 D2H + CPU Polynomial — ~1.30s (−9%)
- **Phase 14: Early zkey coef transfer — split into 2 batches during computeT — ~1.25s (−0.05s / −4%)**
  - Batch 1 (QC,S1,S2,S3 coefs → slot 8): launched in computeT after gate_C frees QC eval slot, uses pinnedS
  - Batch 2 (QM,QL,QR,QO coefs → slot 6): launched after computeT, uses pinnedQ
  - cuda_device_sync() after gate_C ensures slot 8 safe; QMPerm's internal sync ensures slot 6 safe
  - Zkey join wait: 0.050s → ~0s. Round 4: 0.07s → 0.03s
  - Destinations moved from d_piBuffer/d_lagBuffer to d_staticEvalsBuffer slots 6 and 8

- **Phase 15: Remove dead parset + GPU-native computeWxiw + remove Z CPU Poly chain — ~1.10s (−12%)**
  - Removed 1.5 GB ThreadUtils::parset on buffers["A"] (buffers["B"/"C"] dead, nPublic elements set explicitly)
  - Removed CPU `polynomials["Z"]` creation in computeZ, D2H+parcpy+fixDegree in round2, blindCoefficients in computeT
  - GPU-native computeWxiw: D2D from dPolCoefZ + small D2H/H2D patches (96B each) for blinding corrections
  - computeWxiw: 0.041s → 0.006s, MSM Z: 0.075s → 0.049s, Round 2: 0.13s → 0.09s, parset: 0.05s → 0s
  - Bug fix: `polynomials["Wxiw"]->getLength()` was dead after Z removal — replaced with `domainSize + 2`

- **Phase 16: Cleanup + precompute all omega tables in round0 — ~1.12s**
  - Removed dead GPU functions: `computeZRatiosKernel`, `gpu_plonk_compute_z_ratios`, `gpu_plonk_compute_z_ratios_no_d2h`, `gpu_plonk_memcpy_d2h_async`
  - Added `PLONK_GPU_TIMING` compile flag — wraps all sub-round timing in `#ifdef`
  - Precomputed omega (N-root) tables in round0 for `kernelGatherZRatios` — replaces per-thread `Fr::pow(omega, i)`
  - Fixed d_scanWork overflow: expanded d_aux by `scanWorkElems` to prevent scan from overwriting omega tables
  - Removed unnecessary braces around bfZ in round2

### Current timing breakdown (~1.12s total, ~0.86s rounds sum):
- Round 1 (GPU gather+3×IFFT + 3×MSM devptr): 0.22s
- Round 2 (fused gather+z_ratios + scan + IFFT + MSM devptr): 0.09s
- Round 3 (precompute omega + computeT + 3×MSM devptr): 0.40s
- Round 4 (GPU poly eval + blinding correction): 0.03s
- Round 5 (GPU R+Wxi + divByZerofier + 2×MSM devptr): 0.11s
- Setup overhead: calculateAdditions 0.25s (dominant remaining bottleneck)
- Remaining bottlenecks: MSMs (~50% of round time), calculateAdditions (0.25s)

## CPU Memory Cleanup (verified)
- Removed `inverses`/`products` arrays (256 MB) — GPU prefix scan replaced batchInverse()
- Removed `buffers["tmp"]` (32 MB) — polynomialFromMontgomery() unused in GPU path
- Removed dead functions: `batchInverse()`, `polynomialFromMontgomery()`, `multiExponentiationGPU()`, `computeWirePolynomial()`
- Total saved: ~288 MB CPU memory. Proof verified correct.

## Related Project Files
- [project_goldilocks_gpu.md](project_goldilocks_gpu.md): Goldilocks GPU cleanup project — current phase, baseline timings, pitfalls
- [project_poseidon_v1_port.md](project_poseidon_v1_port.md): Poseidon v1 port (branch feature/add_poseidon1) — AVX512 paths pending validation on AVX512-capable hardware
