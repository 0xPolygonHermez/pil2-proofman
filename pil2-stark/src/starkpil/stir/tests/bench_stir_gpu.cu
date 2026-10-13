// Isolated low-degree-test benchmarks under blake3: GPU STIR vs GPU FRI vs CPU STIR.
// All DISABLED_: ./build-gpu/stirBenchGpu --gtest_also_run_disabled_tests [--gtest_filter=...]
#include <gtest/gtest.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>
#include "stir_test_helpers.hpp"
#include "stir_math_gpu.cuh"
#include "starks_gpu.cuh"
#include "transcriptGL.cuh"
#include "merkleTreeGL.hpp"
#include "poseidon2_goldilocks.cuh"
#include "blake3_goldilocks.hpp"
#include "goldilocks_trace_layout.cuh"
#include "starks_api_internal.cuh"
#include "stir_gpu.cuh"

using namespace stirtest;

namespace
{
constexpr int WARMUP = 3, TIMED = 10, CPU_TIMED = 3;

struct DevBuf {   // bench-only RAII device buffer
    gl64_t *p = nullptr;
    explicit DevBuf(uint64_t n) { CHECKCUDAERR(cudaMalloc(&p, n * sizeof(gl64_t))); CHECKCUDAERR(cudaMemset(p, 0, n * sizeof(gl64_t))); }
    ~DevBuf() { cudaFree(p); }
    DevBuf(const DevBuf &) = delete;
    DevBuf &operator=(const DevBuf &) = delete;
};

// What generate_stark_struct + the security solver give blake3 at blowup 1 (query counts are the
// same for batch sizes 32..128); nBits 19 is the blowup-2 shape of the hashes Blake3 / recursion airs.
struct Sched {
    StirParams stir;
    std::vector<uint64_t> friLogDomains;
    uint64_t friQueries, friGrinding;
};

StirParams blake3Stir(std::vector<uint64_t> k, std::vector<uint64_t> d, std::vector<uint64_t> L, std::vector<uint64_t> t)
{
    StirParams p;
    p.logFoldingFactors = k; p.logDegrees = d; p.logDomainSizes = L; p.numOodSamples = 1;
    p.numQueries = t; p.grindingBitsQueries.assign(k.size(), 24);
    p.merkleTreeArity = 2; p.lastLevelVerification = 4; p.merkleTreeCustom = true; p.transcriptArity = 2; p.hashCommits = true;
    return p;
}

Sched sched(uint64_t nBits)
{
    switch (nBits) {
    case 17: return {blake3Stir({4, 4, 2}, {17, 13, 9, 7}, {18, 17, 16, 15}, {211, 53, 31}), {18, 14, 10, 6}, 211, 24};
    case 19: return {blake3Stir({4, 4, 4}, {19, 15, 11, 7}, {21, 20, 19, 18}, {106, 43, 27}), {21, 17, 13, 10, 7}, 106, 24};
    case 20: return {blake3Stir({4, 4, 4, 1}, {20, 16, 12, 8, 7}, {21, 20, 19, 18, 17}, {211, 53, 31, 22}), {21, 17, 13, 9, 6}, 211, 24};
    case 22: return {blake3Stir({4, 4, 4, 3}, {22, 18, 14, 10, 7}, {23, 22, 21, 20, 19}, {211, 53, 31, 22}), {23, 18, 14, 10, 6}, 211, 24};
    }
    throw std::runtime_error("no schedule for this nBits");
}

double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

void report(const char *what, uint64_t nBits, const std::vector<double> &ms)
{
    printf("BENCH %s nBits=%lu median=%.3f ms min=%.3f max=%.3f (n=%zu)\n", what, nBits, median(ms),
           *std::min_element(ms.begin(), ms.end()), *std::max_element(ms.begin(), ms.end()), ms.size());
    fflush(stdout);
}

bool blake3PowOk(const uint64_t *c, uint64_t nonce, uint32_t bits)
{
    if (bits == 0) return true;
    Goldilocks::Element x[8] = {}, out[8];
    x[0].fe = c[0]; x[1].fe = c[1]; x[2].fe = c[2]; x[3] = Goldilocks::fromU64(nonce);
    Blake3Goldilocks::permute(out, x);
    return Goldilocks::toU64(out[0]) < (1ULL << (64 - bits));
}

void blake3Grind(uint64_t &nonce, const uint64_t *c, uint32_t bits)
{
    if (bits == 0) { nonce = 0; return; }   // the CPU grinder's 1 << 64 is UB at 0 bits
    Blake3Goldilocks::grinding(nonce, c, bits);
}

MerkleTreeGL tree(const StirParams &p, uint64_t height, uint64_t width)
{
    return MerkleTreeGL(p.merkleTreeArity, p.lastLevelVerification, p.merkleTreeCustom, height, width, false, false);
}

// ---- GPU STIR, laid out as test_stir_gpu.cu's proveOnGpu ------------------------------------------
struct StirGpu {
    StirParams p;
    std::unique_ptr<DevBuf> base, pristine, seed;
    StirGpuIo io;
    std::unique_ptr<TranscriptGL_GPU> t, h;
    TimerGPU timer;
    cudaStream_t s;

    StirGpu(const StirParams &p_, const std::vector<FE> &f0, cudaStream_t s_) : p(p_), s(s_)
    {
        const uint64_t nQ = p.numQueries[0];
        uint64_t maxWidth = 0, maxProof = 0;
        for (uint64_t i = 0; i < p.M(); i++) {
            MerkleTreeGL mt = tree(p, (1ull << p.logDomainSizes[i]) >> p.logFoldingFactors[i], (1ull << p.logFoldingFactors[i]) * 3);
            maxWidth = std::max(maxWidth, mt.getMerkleTreeWidth()); maxProof = std::max(maxProof, mt.getMerkleProofSize());
        }
        const uint64_t stride = nQ * (maxWidth + maxProof), fSize = (1ull << p.logDomainSizes[0]) * 3;
        uint64_t at = 0;
        const uint64_t oF = at; at += fSize;
        const uint64_t oQ = at; at += p.M() * stride;
        const uint64_t oRaw = at; at += nQ;
        const uint64_t oPerm = at; at += (nQ * p.logDomainSizes[0] + 62) / 63 + 1;
        const uint64_t oNonce = at; at += 1;
        const uint64_t oBlocks = at; at += GRIND_NONCE_BLOCKS_MAX;
        const uint64_t oIn = at; at += HASH_SIZE;
        StirLayout l = makeStirLayout(p, stirAlign(at), [&](uint64_t hgt) { return tree(p, hgt, 3).numNodes; });
        base = std::make_unique<DevBuf>(l.total);
        pristine = std::make_unique<DevBuf>(fSize);
        gl64_t *b = base->p;
        CHECKCUDAERR(cudaMemcpy(pristine->p, f0.data(), fSize * 8, cudaMemcpyHostToDevice));
        io = StirGpuIo{b, l, b + oF, (uint64_t *)(b + oRaw), (uint64_t *)(b + oPerm), (uint64_t *)(b + oNonce),
                       (uint64_t *)(b + oBlocks), (uint64_t *)(b + oIn), b + oQ, stride};
        t = std::make_unique<TranscriptGL_GPU>(p.transcriptArity, p.merkleTreeCustom, s);
        h = std::make_unique<TranscriptGL_GPU>(p.transcriptArity, p.merkleTreeCustom, s);
        seed = std::make_unique<DevBuf>(4);
        CHECKCUDAERR(cudaMemcpy(seed->p, SEED, 4 * 8, cudaMemcpyHostToDevice));
        timer.init(s);
    }
    // Untimed: a fresh transcript and f_0 (the driver clobbers io.f).
    void prepare()
    {
        CHECKCUDAERR(cudaMemcpyAsync(io.f, pristine->p, (1ull << p.logDomainSizes[0]) * 3 * 8, cudaMemcpyDeviceToDevice, s));
        t->reset(s);
        t->put((Goldilocks::Element *)seed->p, 4, s);
    }
    void run() { stirProve_gpu(p, io, t.get(), h.get(), [] {}, timer, s); }
};

// ---- GPU FRI low-degree test, mirroring gen_proof.cuh's FRI branch (stage-tree openings excluded,
// as the STIR harness's openStageTrees is a no-op) -----------------------------------------------
struct FriGpu {
    StirParams g;   // commitment geometry only
    std::vector<uint64_t> L;
    uint64_t nQ, bits;
    std::unique_ptr<DevBuf> base, pristine, seed;
    std::vector<uint64_t> oSrc, oNodes, oBuff, nNodes;
    uint64_t oF, oChal, oHash, oRaw, oIdx, oPerm, oNonce, oBlocks, oIn;
    std::unique_ptr<TranscriptGL_GPU> t, h;
    TimerGPU timer;
    cudaStream_t s;

    FriGpu(const Sched &sc, const std::vector<FE> &f0, cudaStream_t s_) : g(sc.stir), L(sc.friLogDomains), nQ(sc.friQueries), bits(sc.friGrinding), s(s_)
    {
        const uint64_t nT = L.size() - 1, fSize = (1ull << L[0]) * 3;
        uint64_t maxWidth = 0, maxProof = 0;
        for (uint64_t i = 0; i < nT; i++) {
            MerkleTreeGL mt = tree(g, 1ull << L[i + 1], (1ull << (L[i] - L[i + 1])) * 3);
            nNodes.push_back(mt.numNodes);
            maxWidth = std::max(maxWidth, mt.getMerkleTreeWidth()); maxProof = std::max(maxProof, mt.getMerkleProofSize());
        }
        const uint64_t stride = nQ * (maxWidth + maxProof);
        uint64_t at = 0;
        oF = at; at += fSize;
        for (uint64_t i = 0; i < nT; i++) { oSrc.push_back(at); at += (1ull << L[i]) * 3; oNodes.push_back(at); at += nNodes[i]; oBuff.push_back(at); at += stride; }
        oChal = at; at += 4;
        oHash = at; at += HASH_SIZE;
        oRaw = at; at += nQ;
        oIdx = at; at += nQ;
        oPerm = at; at += (nQ * L[0] + 62) / 63 + 1;
        oNonce = at; at += 1;
        oBlocks = at; at += GRIND_NONCE_BLOCKS_MAX;
        oIn = at; at += HASH_SIZE;
        base = std::make_unique<DevBuf>(at);
        pristine = std::make_unique<DevBuf>(fSize);
        CHECKCUDAERR(cudaMemcpy(pristine->p, f0.data(), fSize * 8, cudaMemcpyHostToDevice));
        t = std::make_unique<TranscriptGL_GPU>(g.transcriptArity, g.merkleTreeCustom, s);
        h = std::make_unique<TranscriptGL_GPU>(g.transcriptArity, g.merkleTreeCustom, s);
        seed = std::make_unique<DevBuf>(4);
        CHECKCUDAERR(cudaMemcpy(seed->p, SEED, 4 * 8, cudaMemcpyHostToDevice));
        timer.init(s);
    }
    gl64_t *at(uint64_t o) { return base->p + o; }
    void prepare()
    {
        CHECKCUDAERR(cudaMemcpyAsync(at(oF), pristine->p, (1ull << L[0]) * 3 * 8, cudaMemcpyDeviceToDevice, s));
        t->reset(s);
        t->put((Goldilocks::Element *)seed->p, 4, s);
        t->getField((uint64_t *)at(oChal), s);   // in the prover the step-0 challenge comes from the DEEP stage
    }
    void run()
    {
        const uint64_t nSteps = L.size();
        Goldilocks::Element *chal = (Goldilocks::Element *)at(oChal);
        for (uint64_t step = 0; step < nSteps; step++) {
            if (step > 0) fold_inplace(step, oF, chal, L[0], L[step - 1], L[step], base->p, timer, s);
            if (step < nSteps - 1) {
                // merkelizeFRI_inplace: coset-leaf transpose (cosetLeaves == transposeFRI), tree, root
                stirgpu::cosetLeaves(at(oSrc[step]), at(oF), L[step], L[step] - L[step + 1], false, s);
                buildMerkleTreeGPU(g.merkleTreeArity, (uint64_t *)at(oNodes[step]), (uint64_t *)at(oSrc[step]),
                                   (1ull << (L[step] - L[step + 1])) * 3, 1ull << L[step + 1], Layout::RowMajor, s);
                t->put((Goldilocks::Element *)at(oNodes[step] + nNodes[step] - HASH_SIZE), HASH_SIZE, s);
            } else {
                // hashCommits: absorb hash(final evaluations), as calculateHash does
                h->reset(s);
                h->put((Goldilocks::Element *)at(oF), (1ull << L[step]) * 3, s);
                h->getState((Goldilocks::Element *)at(oHash), s);
                t->put((Goldilocks::Element *)at(oHash), HASH_SIZE, s);
            }
            t->getField((uint64_t *)chal, s);
        }
        CHECKCUDAERR(cudaMemcpyAsync(at(oIn), chal, 3 * 8, cudaMemcpyDeviceToDevice, s));
        runGrindingGPU((uint64_t *)at(oNonce), (uint64_t *)at(oBlocks), (uint64_t *)at(oIn), bits, s);
        h->reset(s);
        h->put2(chal, 3, (Goldilocks::Element *)at(oNonce), 1, s);
        h->getPermutations((uint64_t *)at(oRaw), nQ, L[0], (Goldilocks::Element *)at(oPerm), s);
        for (uint64_t step = 0; step + 1 < nSteps; step++) {
            stirgpu::modQueries((uint64_t *)at(oIdx), (uint64_t *)at(oRaw), nQ, 1ull << L[step + 1], s);
            proveLeafQueries_gpu(at(oBuff[step]), at(oSrc[step]), at(oNodes[step]), (1ull << (L[step] - L[step + 1])) * 3,
                                 1ull << L[step + 1], (uint64_t *)at(oIdx), nQ, g.merkleTreeArity, g.lastLevelVerification, s);
        }
    }
};

template <class Runner>
std::vector<double> timeGpu(Runner &r)
{
    cudaEvent_t a, b;
    CHECKCUDAERR(cudaEventCreate(&a)); CHECKCUDAERR(cudaEventCreate(&b));
    for (int i = 0; i < WARMUP; i++) { r.prepare(); r.run(); }
    CHECKCUDAERR(cudaStreamSynchronize(r.s));
    std::vector<double> ms;
    for (int i = 0; i < TIMED; i++) {
        r.prepare();
        CHECKCUDAERR(cudaStreamSynchronize(r.s));
        cudaProfilerStart();   // nsys --capture-range=cudaProfilerApi records only the timed windows
        CHECKCUDAERR(cudaEventRecord(a, r.s));
        r.run();
        CHECKCUDAERR(cudaEventRecord(b, r.s));
        CHECKCUDAERR(cudaEventSynchronize(b));
        cudaProfilerStop();
        float x; CHECKCUDAERR(cudaEventElapsedTime(&x, a, b)); ms.push_back(x);
    }
    CHECKCUDAERR(cudaGetLastError());
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms;
}

// The prover replays this region as a CUDA graph; time that (launch gaps differ from eager runs).
template <class Runner>
cudaGraphExec_t captureRun(Runner &r)
{
    for (int i = 0; i < WARMUP; i++) { r.prepare(); r.run(); }   // lazy init must not happen under capture
    r.prepare();
    CHECKCUDAERR(cudaStreamSynchronize(r.s));
    cudaGraph_t g; cudaGraphExec_t ge;
    CHECKCUDAERR(cudaStreamBeginCapture(r.s, cudaStreamCaptureModeThreadLocal));
    r.run();
    CHECKCUDAERR(cudaStreamEndCapture(r.s, &g));
    CHECKCUDAERR(cudaGraphInstantiate(&ge, g, 0));
    CHECKCUDAERR(cudaGraphDestroy(g));
    return ge;
}

template <class Runner>
std::vector<double> timeGpuGraph(Runner &r)
{
    cudaGraphExec_t ge = captureRun(r);
    cudaEvent_t a, b;
    CHECKCUDAERR(cudaEventCreate(&a)); CHECKCUDAERR(cudaEventCreate(&b));
    for (int i = 0; i < WARMUP; i++) { r.prepare(); CHECKCUDAERR(cudaGraphLaunch(ge, r.s)); }
    std::vector<double> ms;
    for (int i = 0; i < TIMED; i++) {
        r.prepare();
        CHECKCUDAERR(cudaStreamSynchronize(r.s));
        CHECKCUDAERR(cudaEventRecord(a, r.s));
        CHECKCUDAERR(cudaGraphLaunch(ge, r.s));
        CHECKCUDAERR(cudaEventRecord(b, r.s));
        CHECKCUDAERR(cudaEventSynchronize(b));
        float x; CHECKCUDAERR(cudaEventElapsedTime(&x, a, b)); ms.push_back(x);
    }
    cudaEventDestroy(a); cudaEventDestroy(b); cudaGraphExecDestroy(ge);
    return ms;
}

// Throughput: N independent runs, one graph per stream, launched together; returns ms per run.
template <class Runner>
std::vector<double> timeGpuConcurrent(std::vector<std::unique_ptr<Runner>> &rs)
{
    std::vector<cudaGraphExec_t> ge;
    for (auto &r : rs) ge.push_back(captureRun(*r));
    cudaEvent_t a, b;
    CHECKCUDAERR(cudaEventCreate(&a)); CHECKCUDAERR(cudaEventCreate(&b));
    std::vector<double> ms;
    for (int it = 0; it < WARMUP + TIMED; it++) {
        for (auto &r : rs) r->prepare();
        CHECKCUDAERR(cudaDeviceSynchronize());
        CHECKCUDAERR(cudaEventRecord(a, rs[0]->s));
        for (uint64_t k = 1; k < rs.size(); k++) CHECKCUDAERR(cudaStreamWaitEvent(rs[k]->s, a, 0));
        for (uint64_t k = 0; k < rs.size(); k++) CHECKCUDAERR(cudaGraphLaunch(ge[k], rs[k]->s));
        CHECKCUDAERR(cudaDeviceSynchronize());   // b after all streams: recorded on the legacy-synchronised end
        CHECKCUDAERR(cudaEventRecord(b, rs[0]->s));
        CHECKCUDAERR(cudaEventSynchronize(b));
        float x; CHECKCUDAERR(cudaEventElapsedTime(&x, a, b));
        if (it >= WARMUP) ms.push_back(x / rs.size());
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    for (auto e : ge) cudaGraphExecDestroy(e);
    return ms;
}

void cpuStir(Proof &proof, const StirParams &p, const std::vector<FE> &f0)
{
    proof = Stir::makeProof(p);
    TranscriptGL tc = freshTranscript(p);
    ProofTree<FE> stageQ(HASH_SIZE, p.numQueries[0], p.merkleTreeArity, p.lastLevelVerification);
    Stir::prove(proof, p, f0.data(), tc, stageQ, nullptr, 0, blake3Grind);
}

// ---- reading the GPU proof back (copy of test_stir_gpu.cu's readGpuProof / expectSameProof) --------
Proof readGpuProof(const StirParams &p, const StirGpuIo &io)
{
    Proof proof = Stir::makeProof(p);
    const StirLayout &l = io.layout;
    auto down = [&](const gl64_t *src, uint64_t n) {
        std::vector<FE> hv(n); CHECKCUDAERR(cudaMemcpy(hv.data(), src, n * 8, cudaMemcpyDeviceToHost)); return hv;
    };
    const gl64_t *pr = io.base + l.proof;
    const uint64_t M = p.M(), s = p.numOodSamples;
    for (uint64_t i = 0; i + 1 < M; i++) proof.betas[i] = down(pr + l.betas + i * s * 3, s * 3);
    proof.finalPol = down(pr + l.finalPol, proof.finalPol.size());
    { auto n = down(pr + l.nonces, M); for (uint64_t i = 0; i < M; i++) proof.nonces[i] = n[i].fe; }
    for (uint64_t i = 1; i < M; i++) { proof.ansCoeffs[i - 1] = down(pr + l.ans[i - 1], proof.ansCoeffs[i - 1].size());
                                       proof.shakeCoeffs[i - 1] = down(pr + l.shake[i - 1], proof.shakeCoeffs[i - 1].size()); }
    for (uint64_t i = 0; i < M; i++) {
        uint64_t k = 1ull << p.logFoldingFactors[i], height = (1ull << p.logDomainSizes[i]) >> p.logFoldingFactors[i];
        MerkleTreeGL mt = tree(p, height, k * 3);
        auto nodes = down(io.base + l.treeNodes[i], mt.numNodes);
        for (int c = 0; c < HASH_SIZE; c++) proof.trees[i].root[c] = nodes[mt.numNodes - HASH_SIZE + c];
        if (p.lastLevelVerification > 0) { mt.setNodes(nodes.data()); mt.getLevel(&proof.trees[i].last_levels[0]); }
        uint64_t buffSize = k * 3 + mt.getMerkleProofSize();
        auto q = down(io.queriesBuff + i * io.queryBlockStride, p.numQueries[i] * buffSize);
        for (uint64_t j = 0; j < p.numQueries[i]; j++)
            proof.trees[i].polQueries[j] = {MerkleProof<FE>(mt.getMerkleTreeWidth(), mt.getMerkleProofLength(), mt.getNumSiblings(), &q[j * buffSize])};
    }
    return proof;
}

uint64_t u(const FE &x) { return Goldilocks::toU64(x); }

void same(const std::vector<FE> &a, const std::vector<FE> &b, const char *what)
{
    ASSERT_EQ(a.size(), b.size()) << what;
    for (uint64_t i = 0; i < b.size(); i++) ASSERT_EQ(u(a[i]), u(b[i])) << what << " at " << i;
}

void expectSameProof(const Proof &g, const Proof &c)
{
    ASSERT_EQ(g.trees.size(), c.trees.size());
    for (uint64_t i = 0; i < c.trees.size(); i++) {
        for (uint64_t k = 0; k < HASH_SIZE; k++) ASSERT_EQ(u(g.trees[i].root[k]), u(c.trees[i].root[k])) << "root " << i;
        same(g.trees[i].last_levels, c.trees[i].last_levels, "last level");
        for (uint64_t q = 0; q < c.trees[i].polQueries.size(); q++) {
            const auto &a = g.trees[i].polQueries[q][0], &b = c.trees[i].polQueries[q][0];
            for (uint64_t e = 0; e < b.v.size(); e++) ASSERT_EQ(u(a.v[e][0]), u(b.v[e][0])) << "tree " << i << " q " << q;
            for (uint64_t l = 0; l < b.mp.size(); l++) for (uint64_t e = 0; e < b.mp[l].size(); e++)
                ASSERT_EQ(u(a.mp[l][e]), u(b.mp[l][e])) << "sibling " << i << "/" << q;
        }
    }
    for (uint64_t i = 0; i < c.betas.size(); i++) same(g.betas[i], c.betas[i], "beta");
    for (uint64_t i = 0; i < c.ansCoeffs.size(); i++) { same(g.ansCoeffs[i], c.ansCoeffs[i], "ans"); same(g.shakeCoeffs[i], c.shakeCoeffs[i], "shake"); }
    same(g.finalPol, c.finalPol, "finalPol");
    for (uint64_t i = 0; i < c.nonces.size(); i++) ASSERT_EQ(g.nonces[i], c.nonces[i]) << "nonce " << i;
}

cudaStream_t benchStream()
{
    static cudaStream_t s = [] { cudaStream_t x; CHECKCUDAERR(cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking)); return x; }();
    return s;
}

std::vector<FE> f0For(uint64_t nBits, const StirParams &p) { std::vector<FE> f0; honestF0(f0, p, 1000 + nBits); return f0; }

void benchGpuStir(uint64_t nBits)
{
    Sched sc = sched(nBits);
    StirGpu r(sc.stir, f0For(nBits, sc.stir), benchStream());
    report("GPU_STIR", nBits, timeGpu(r));
}

void benchGpuFri(uint64_t nBits)
{
    Sched sc = sched(nBits);
    FriGpu r(sc, f0For(nBits, sc.stir), benchStream());
    report("GPU_FRI_LDT", nBits, timeGpu(r));
}

cudaStream_t newStream() { cudaStream_t x; CHECKCUDAERR(cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking)); return x; }

template <class Runner>
void benchGraphAndConcurrent(const char *what, uint64_t nBits)
{
    Sched sc = sched(nBits);
    std::vector<FE> f0 = f0For(nBits, sc.stir);
    {
        Runner r(sc, f0, benchStream());
        report((std::string(what) + "_GRAPH").c_str(), nBits, timeGpuGraph(r));
    }
    for (int n : {2, 4, 8}) {
        std::vector<std::unique_ptr<Runner>> rs;
        for (int k = 0; k < n; k++) rs.push_back(std::make_unique<Runner>(sc, f0, newStream()));
        report((std::string(what) + "_CONC" + std::to_string(n) + "_PER_PROOF").c_str(), nBits, timeGpuConcurrent(rs));
    }
}
struct StirGpuS : StirGpu { StirGpuS(const Sched &sc, const std::vector<FE> &f0, cudaStream_t s) : StirGpu(sc.stir, f0, s) {} };

void benchCpuStir(uint64_t nBits)
{
    Sched sc = sched(nBits);
    std::vector<FE> f0 = f0For(nBits, sc.stir);
    Proof proof;
    cpuStir(proof, sc.stir, f0);   // one warm-up
    std::vector<double> ms;
    for (int i = 0; i < CPU_TIMED; i++) {
        auto a = std::chrono::steady_clock::now();
        cpuStir(proof, sc.stir, f0);
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
    }
    report("CPU_STIR", nBits, ms);
}
} // namespace

class StirBench : public ::testing::Test {
protected:
    static void SetUpTestSuite()
    {
        define_hash_family(HashFamily::Blake3);
        NTTGoldilocksGPU::initConstants(23);
        int dev = 0; CHECKCUDAERR(cudaGetDevice(&dev));
        uint32_t gpuId = (uint32_t)dev;
        TranscriptGL_GPU::init_const(&gpuId, 1, 2);
        Poseidon2GoldilocksGPUGrinding::initConstants(&gpuId, 1);   // also loads the Merkle widths
        TimerGPU::prewarmEvents(256);
    }
};

TEST_F(StirBench, DISABLED_BenchGpuStir17) { benchGpuStir(17); }
TEST_F(StirBench, DISABLED_BenchGpuStir19b2) { benchGpuStir(19); }
TEST_F(StirBench, DISABLED_BenchGpuStir20) { benchGpuStir(20); }
TEST_F(StirBench, DISABLED_BenchGpuStir22) { benchGpuStir(22); }
TEST_F(StirBench, DISABLED_BenchGpuFri17) { benchGpuFri(17); }
TEST_F(StirBench, DISABLED_BenchGpuFri19b2) { benchGpuFri(19); }
TEST_F(StirBench, DISABLED_BenchGpuFri20) { benchGpuFri(20); }
TEST_F(StirBench, DISABLED_BenchGpuFri22) { benchGpuFri(22); }
TEST_F(StirBench, DISABLED_BenchGraphStir19b2) { benchGraphAndConcurrent<StirGpuS>("GPU_STIR", 19); }
TEST_F(StirBench, DISABLED_BenchGraphStir20) { benchGraphAndConcurrent<StirGpuS>("GPU_STIR", 20); }
TEST_F(StirBench, DISABLED_BenchGraphFri19b2) { benchGraphAndConcurrent<FriGpu>("GPU_FRI_LDT", 19); }
TEST_F(StirBench, DISABLED_BenchGraphFri20) { benchGraphAndConcurrent<FriGpu>("GPU_FRI_LDT", 20); }
TEST_F(StirBench, DISABLED_BenchCpuStir17) { benchCpuStir(17); }
TEST_F(StirBench, DISABLED_BenchCpuStir19b2) { benchCpuStir(19); }
TEST_F(StirBench, DISABLED_BenchCpuStir20) { benchCpuStir(20); }
TEST_F(StirBench, DISABLED_BenchCpuStir22) { benchCpuStir(22); }

// Sanity at the smallest size: blake3 GPU STIR is byte-identical to the CPU prover, and verifies.
TEST_F(StirBench, DISABLED_BenchSanityBlake3MatchesCpu17)
{
    Sched sc = sched(17);
    std::vector<FE> f0 = f0For(17, sc.stir);
    Proof cpu;
    cpuStir(cpu, sc.stir, f0);
    StirGpu r(sc.stir, f0, benchStream());
    r.prepare(); r.run();
    CHECKCUDAERR(cudaStreamSynchronize(r.s));
    Proof gpu = readGpuProof(sc.stir, r.io);
    expectSameProof(gpu, cpu);
    TranscriptGL tv = freshTranscript(sc.stir);
    std::string why;
    auto checkF0 = [&](uint64_t, uint64_t idx, const stir::E3 &v) { return stir::equal(v, (const stir::E3 &)f0[idx * 3]); };
    EXPECT_TRUE(Stir::verify(gpu, sc.stir, tv, checkF0, blake3PowOk, &why)) << why;
}

// The FRI grind on the GPU returns the CPU grinder's minimal nonce for each benchmarked shape.
TEST_F(StirBench, DISABLED_BenchSanityFriNonceMatchesCpu)
{
    for (uint64_t nBits : {17, 19, 20, 22}) {
        Sched sc = sched(nBits);
        FriGpu r(sc, f0For(nBits, sc.stir), benchStream());
        r.prepare(); r.run();
        CHECKCUDAERR(cudaStreamSynchronize(r.s));
        uint64_t chal[3], gpu = 0, cpu = 0;
        CHECKCUDAERR(cudaMemcpy(chal, r.at(r.oIn), sizeof(chal), cudaMemcpyDeviceToHost));
        CHECKCUDAERR(cudaMemcpy(&gpu, r.at(r.oNonce), sizeof(gpu), cudaMemcpyDeviceToHost));
        blake3Grind(cpu, chal, r.bits);
        EXPECT_EQ(gpu, cpu) << "nBits " << nBits;
        printf("FRI nBits=%llu grind %llu bits nonce %llu\n", (unsigned long long)nBits, (unsigned long long)r.bits, (unsigned long long)gpu);
    }
}

// Blake3 query sampling: GPU getPermutations equals the CPU transcript's and leaves the same state.
TEST_F(StirBench, DISABLED_BenchSanityBlake3PermutationsMatchCpu)
{
    // nBits 20/22/17/13 do not divide 63, so queries straddle field boundaries.
    for (auto [n, nBits] : std::vector<std::pair<uint64_t, uint64_t>>{{211, 20}, {53, 22}, {7, 17}, {300, 13}, {1, 21}}) {
        TranscriptGL cpu(2, true);
        cpu.put((Goldilocks::Element *)SEED, 4);
        std::vector<uint64_t> want(n);
        cpu.getPermutations(want.data(), n, nBits);
        uint64_t wantNext[3]; cpu.getField(wantNext);

        TranscriptGL_GPU gpu(2, true, 0);
        DevBuf seed(4), res(n), scratch((n * nBits + 62) / 63 + 1), next(3);
        CHECKCUDAERR(cudaMemcpy(seed.p, SEED, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice));
        gpu.put((Goldilocks::Element *)seed.p, 4, 0);
        gpu.getPermutations((uint64_t *)res.p, n, nBits, (Goldilocks::Element *)scratch.p, 0);
        gpu.getField((uint64_t *)next.p, 0);
        CHECKCUDAERR(cudaDeviceSynchronize());
        std::vector<uint64_t> got(n), gotNext(3);
        CHECKCUDAERR(cudaMemcpy(got.data(), res.p, n * 8, cudaMemcpyDeviceToHost));
        CHECKCUDAERR(cudaMemcpy(gotNext.data(), next.p, 3 * 8, cudaMemcpyDeviceToHost));
        EXPECT_EQ(got, want) << "n " << n << " nBits " << nBits;
        for (int c = 0; c < 3; c++) EXPECT_EQ(gotNext[c] % GOLDILOCKS_PRIME, wantNext[c] % GOLDILOCKS_PRIME) << "next field, n " << n;
    }
}
