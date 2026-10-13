// Device port of STIR::prove (stir.hpp); step labels follow it.
#include "stir_gpu.cuh"
#include "stir_math_gpu.cuh"
#include "starks_gpu.cuh"
#include "transcriptGL.cuh"
#include "goldilocks_trace_layout.cuh"
#include "starks_api_internal.cuh"   // buildMerkleTreeGPU, runGrindingGPU

void stirProve_gpu(const StirParams &p, const StirGpuIo &io, TranscriptGL_GPU *tr, TranscriptGL_GPU *helper,
                   const std::function<void()> &openStageTrees, TimerGPU &timer, cudaStream_t stream)
{
    using namespace stirgpu;
    const uint64_t M = p.M(), s = p.numOodSamples;
    const StirLayout &l = io.layout;
    const Goldilocks::Element shift = Goldilocks::shift();
    gl64_t *st = io.base + l.state, *pr = io.base + l.proof, *B = io.base + l.ntt;
    gl64_t *rFold = st + l.rFold, *rFoldNext = st + l.rFoldNext, *rComb = st + l.rComb, *rho = st + l.rho;
    gl64_t *chal = st + l.challenge, *rOut = st + l.rOut, *gHat = st + l.gHat;
    uint64_t *nG = (uint64_t *)(st + l.nG), *flag = (uint64_t *)(pr + l.oodFlag), *cosetIdx = (uint64_t *)(io.base + l.queries);

    auto treeHeight = [&](uint64_t i) { return (1ull << p.logDomainSizes[i]) >> p.logFoldingFactors[i]; };
    auto treeWidth = [&](uint64_t i) { return (1ull << p.logFoldingFactors[i]) * FIELD_EXTENSION; };
    auto commit = [&](uint64_t i, const gl64_t *src, bool colMajor) {
        gl64_t *leaves = io.base + l.treeSrc[i], *nodes = io.base + l.treeNodes[i];
        cosetLeaves(leaves, src, p.logDomainSizes[i], p.logFoldingFactors[i], colMajor, stream);
        buildMerkleTreeGPU(p.merkleTreeArity, (uint64_t *)nodes, (uint64_t *)leaves, treeWidth(i), treeHeight(i), Layout::RowMajor, stream);
        tr->put((Goldilocks::Element *)&nodes[l.treeNumNodes[i] - HASH_SIZE], HASH_SIZE, stream);
    };
    // Shift queries of iteration i (1..M): grind, permute, open T_{i-1}; i = 1 also opens the stage trees.
    auto shiftQueries = [&](uint64_t i) {
        tr->getField((uint64_t *)chal, stream);
        CHECKCUDAERR(cudaMemcpyAsync(io.inputHashNonce, chal, FIELD_EXTENSION * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
        runGrindingGPU(io.nonce, io.nonceBlocks, io.inputHashNonce, p.grindingBitsQueries[i - 1], stream);
        CHECKCUDAERR(cudaMemcpyAsync(pr + l.nonces + (i - 1), io.nonce, sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
        helper->reset(stream);
        helper->put2((Goldilocks::Element *)chal, FIELD_EXTENSION, (Goldilocks::Element *)io.nonce, 1, stream);
        helper->getPermutations(io.rawQueriesL0, p.numQueries[i - 1], p.logDomainSizes[i - 1], (Goldilocks::Element *)io.permScratch, stream);
        if (i == 1) openStageTrees();
        modQueries(cosetIdx, io.rawQueriesL0, p.numQueries[i - 1], treeHeight(i - 1), stream);
        proveLeafQueries_gpu(io.queriesBuff + (i - 1) * io.queryBlockStride, io.base + l.treeSrc[i - 1], io.base + l.treeNodes[i - 1],
                             treeWidth(i - 1), treeHeight(i - 1), cosetIdx, p.numQueries[i - 1], p.merkleTreeArity, p.lastLevelVerification, stream);
    };
    // Fold f_i in place (first |L_i|/k_i points of io.f) and leave ĝ_{i+1} (d_{i+1} coefficients) in gHat.
    auto foldAndInterpolate = [&](uint64_t i) {
        foldStir(io.f, rFold, p.logDomainSizes[i], p.logFoldingFactors[i], timer, stream);
        const uint64_t logLk = p.logDomainSizes[i] - p.logFoldingFactors[i];
        evalsToCoeffs(gHat, 1ull << p.logDegrees[i + 1], B, io.f, logLk, Goldilocks::exp(shift, 1ull << p.logFoldingFactors[i]), stream);
    };

    CHECKCUDAERR(cudaMemsetAsync(flag, 0, sizeof(uint64_t), stream));
    commit(0, io.f, /*colMajor=*/false);                                      // T_0 commits f_0
    tr->getField((uint64_t *)rFold, stream);                                  // r^fold_0

    for (uint64_t i = 1; i < M; i++) {
        // 2(a) g_i := Fold(f_{i-1}), committed on L_i (B keeps it column-major for the quotient).
        foldAndInterpolate(i - 1);
        coeffsToEvalsColMajor(B, gHat, 1ull << p.logDegrees[i], p.logDomainSizes[i], shift, stream);
        commit(i, B, /*colMajor=*/true);
        // 2(b) r_out; an in-domain draw raises the flag instead of re-squeezing (no host read).
        for (uint64_t j = 0; j < s; j++) tr->getField((uint64_t *)(rOut + j * FIELD_EXTENSION), stream);
        oodFlag(flag, rOut, s, shift, p.logDomainSizes[i], stream);
        // 2(c) β_{i,j} := ĝ_i(r_out^{i,j})
        gl64_t *beta = pr + l.betas + (i - 1) * s * FIELD_EXTENSION;
        for (uint64_t j = 0; j < s; j++) horner(beta + j * FIELD_EXTENSION, gHat, 1ull << p.logDegrees[i], rOut + j * FIELD_EXTENSION, stream);
        tr->put((Goldilocks::Element *)beta, s * FIELD_EXTENSION, stream);
        // 2(d) r^fold_i, r_comb^i, r_shift
        tr->getField((uint64_t *)rFoldNext, stream);
        tr->getField((uint64_t *)rComb, stream);
        shiftQueries(i);
        // 2(e) G_i and Ans_i; io.f still holds Fold(f_{i-1}) on L_{i-1}^{k_{i-1}}.
        const uint64_t logLk = p.logDomainSizes[i - 1] - p.logFoldingFactors[i - 1];
        buildG(st + l.gPoints, st + l.gValues, nG, rOut, beta, s, cosetIdx, p.numQueries[i - 1],
               io.f, Goldilocks::exp(shift, 1ull << p.logFoldingFactors[i - 1]), logLk, stream);
        // 2(f) Âns_i and Sh_i, zero-padded to s + t_{i-1}, absorbed before ρ_i.
        const uint64_t nPad = s + p.numQueries[i - 1];
        interpolate(pr + l.ans[i - 1], nPad, st + l.gPoints, st + l.gValues, st + l.gTmp, st + l.zCoeffs, st + l.interpT, nG, stream);
        shake(pr + l.shake[i - 1], nPad, pr + l.ans[i - 1], st + l.gPoints, st + l.powerSums, nG, stream);
        tr->put((Goldilocks::Element *)(pr + l.ans[i - 1]), nPad * FIELD_EXTENSION, stream);
        tr->put((Goldilocks::Element *)(pr + l.shake[i - 1]), nPad * FIELD_EXTENSION, stream);
        tr->getField((uint64_t *)rho, stream);
        // f_i := DegCor(Quotient(g_i)) on L_i into io.f; io.f (|L_0|·3, dead since buildG) also holds 6·|L_i| of NTT scratch.
        assert(p.logDomainSizes[i] < p.logDomainSizes[0]);
        quotientDegCor(io.f, B, p.logDomainSizes[i], shift, pr + l.ans[i - 1], st + l.zCoeffs, nPad, nG, rComb, stream);
        CHECKCUDAERR(cudaMemcpyAsync(rFold, rFoldNext, FIELD_EXTENSION * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
    }

    // Final step: p := ĝ_M, sent in the clear (or its hash absorbed).
    foldAndInterpolate(M - 1);
    const uint64_t dM = 1ull << p.logDegrees[M];
    CHECKCUDAERR(cudaMemcpyAsync(pr + l.finalPol, gHat, dM * FIELD_EXTENSION * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
    if (!p.hashCommits) {
        tr->put((Goldilocks::Element *)(pr + l.finalPol), dM * FIELD_EXTENSION, stream);
    } else {
        helper->reset(stream);
        helper->put((Goldilocks::Element *)(pr + l.finalPol), dM * FIELD_EXTENSION, stream);
        helper->getState((Goldilocks::Element *)chal, stream);
        tr->put((Goldilocks::Element *)chal, HASH_SIZE, stream);
    }
    shiftQueries(M);
}
