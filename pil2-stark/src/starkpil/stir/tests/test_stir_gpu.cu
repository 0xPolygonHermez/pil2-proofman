// GPU STIR: every kernel against its stir_math.hpp twin, then the driver against STIR::prove.
// Field arithmetic is exact, so all comparisons are bit-for-bit.
#include <gtest/gtest.h>
#include <random>
#include <set>
#include <vector>
#include "stir_test_helpers.hpp"
#include "stir_math_gpu.cuh"
#include "starks_gpu.cuh"
#include "transcriptGL.cuh"
#include "merkleTreeGL.hpp"
#include "poseidon2_goldilocks.cuh"
#include "poseidon_goldilocks.cuh"
#include "poseidon_goldilocks.hpp"
#include "blake3_goldilocks.cuh"
#include "blake3_core.hpp"
#include "stir_gpu.cuh"

using namespace stirtest;

namespace
{
struct DevBuf {   // test-only RAII device buffer
    gl64_t *p = nullptr; uint64_t n = 0;
    explicit DevBuf(uint64_t n_) : n(n_) { CHECKCUDAERR(cudaMalloc(&p, n * sizeof(gl64_t))); CHECKCUDAERR(cudaMemset(p, 0, n * sizeof(gl64_t))); }
    ~DevBuf() { cudaFree(p); }
    void up(const std::vector<FE> &h) { CHECKCUDAERR(cudaMemcpy(p, h.data(), h.size() * sizeof(FE), cudaMemcpyHostToDevice)); }
    std::vector<FE> down(uint64_t cnt) const { std::vector<FE> h(cnt); CHECKCUDAERR(cudaMemcpy(h.data(), p, cnt * sizeof(FE), cudaMemcpyDeviceToHost)); return h; }
};
void expectEq(const std::vector<FE> &a, const std::vector<FE> &b, uint64_t n)
{
    for (uint64_t i = 0; i < n; i++) ASSERT_EQ(Goldilocks::toU64(a[i]), Goldilocks::toU64(b[i])) << "at " << i;
}
std::vector<FE> randomE3(uint64_t n, uint64_t seed)
{
    std::mt19937_64 g(seed);
    std::vector<FE> v(n * FIELD_EXTENSION);
    for (auto &x : v) x = Goldilocks::fromU64(g() % GOLDILOCKS_PRIME);
    return v;
}
} // namespace

class StirGpuTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        define_hash_family(HashFamily::Poseidon2);
        NTTGoldilocksGPU::initConstants(20);
    }
};

TEST_F(StirGpuTest, coeffs_evals_roundtrip_matches_cpu)
{
    const uint64_t logN = 10, N = 1 << logN, d = 300;
    stir::Domain L{Goldilocks::shift(), logN};
    std::vector<FE> coeffs = randomE3(d, 1), evalsCpu(N * FIELD_EXTENSION);
    stir::coefficientsToEvaluations(evalsCpu.data(), coeffs.data(), d, L);

    DevBuf dC(N * FIELD_EXTENSION), dB(N * FIELD_EXTENSION), dE(N * FIELD_EXTENSION);
    dC.up(coeffs);
    stirgpu::coeffsToEvalsColMajor(dB.p, dC.p, d, logN, L.shift, 0);
    // back to interleaved for the comparison
    std::vector<FE> col = dB.down(N * FIELD_EXTENSION), evalsGpu(N * FIELD_EXTENSION);
    for (uint64_t r = 0; r < N; r++) for (uint64_t c = 0; c < 3; c++) evalsGpu[r * 3 + c] = col[c * N + r];
    expectEq(evalsGpu, evalsCpu, N * FIELD_EXTENSION);

    dE.up(evalsCpu);
    stirgpu::evalsToCoeffs(dC.p, d, dB.p, dE.p, logN, L.shift, 0);
    expectEq(dC.down(d * FIELD_EXTENSION), coeffs, d * FIELD_EXTENSION);
}

TEST_F(StirGpuTest, coset_leaves_match_cpu_commit_layout)
{
    const uint64_t logN = 9, logK = 3, N = 1 << logN, nLeaves = N >> logK, k = 1 << logK;
    std::vector<FE> f = randomE3(N, 2);
    DevBuf dF(N * 3), dT(N * 3);
    dF.up(f);
    stirgpu::cosetLeaves(dT.p, dF.p, logN, logK, /*srcColMajor=*/false, 0);
    std::vector<FE> t = dT.down(N * 3);
    for (uint64_t m = 0; m < nLeaves; m++) for (uint64_t j = 0; j < k; j++) for (uint64_t c = 0; c < 3; c++)
        ASSERT_EQ(Goldilocks::toU64(t[(m * k + j) * 3 + c]), Goldilocks::toU64(f[(m + j * nLeaves) * 3 + c]));
}

class StirGpuFold : public StirGpuTest, public ::testing::WithParamInterface<uint64_t> {};
TEST_P(StirGpuFold, fold_matches_cpu)
{
    const uint64_t logL = 12, logK = GetParam(), N = 1 << logL;
    stir::Domain L{Goldilocks::shift(), logL};
    std::vector<FE> f = randomE3(N, 3), r = randomE3(1, 4), cpu((N >> logK) * 3);
    stir::fold(cpu.data(), f.data(), L, logK, (const E3 &)r[0]);

    DevBuf dF(N * 3), dR(3);
    dF.up(f); dR.up(r);
    TimerGPU timer;   // use the same construction gen_proof.cuh uses if TimerGPU needs arguments
    stirgpu::foldStir(dF.p, dR.p, logL, logK, timer, 0);
    expectEq(dF.down((N >> logK) * 3), cpu, (N >> logK) * 3);
}
INSTANTIATE_TEST_SUITE_P(Factors, StirGpuFold, ::testing::Values(1, 3, 4, 5));

TEST_F(StirGpuTest, horner_matches_cpu)
{
    for (uint64_t n : {1ull, 7ull, 1000ull, 4096ull}) {
        std::vector<FE> c = randomE3(n, 10 + n), x = randomE3(1, 11), cpu(3);
        stir::evalPoly((E3 &)cpu[0], c.data(), n, (const E3 &)x[0]);
        DevBuf dC(n * 3), dX(3), dO(3);
        dC.up(c); dX.up(x);
        stirgpu::horner(dO.p, dC.p, n, dX.p, 0);
        expectEq(dO.down(3), cpu, 3);
    }
}

// The full G pipeline against QuotientContext, with forced duplicate shift indices.
TEST_F(StirGpuTest, g_set_interpolation_and_shake_match_cpu)
{
    const uint64_t s = 1, logLk = 6, Lk = 1 << logLk;
    const Goldilocks::Element shiftK = Goldilocks::exp(Goldilocks::shift(), 8);
    std::vector<uint64_t> idx = {3, 17, 3, 40, 17, 63, 0};   // two repeats -> |G| = 1 + 5
    const uint64_t t = idx.size(), nPad = s + t;
    std::vector<FE> fold = randomE3(Lk, 20), rOut = randomE3(s, 21), beta = randomE3(s, 22);

    stir::QuotientContext ctx;
    E3 zero; Goldilocks3::zero(zero);
    ctx.reset(zero);
    ctx.add((const E3 &)rOut[0], (const E3 &)beta[0]);
    stir::Domain LK{shiftK, logLk};
    for (uint64_t i : idx) { E3 pt; stir::embed(pt, LK.point(i)); ctx.add(pt, (const E3 &)fold[i * 3]); }
    ctx.build();
    std::vector<FE> sh; ctx.shake(sh);
    ASSERT_EQ(ctx.size(), 6u);

    DevBuf dPts(nPad * 3), dVals(nPad * 3), dTmp(nPad * 3), dZ((nPad + 1) * 3), dT(nPad * 3), dAns(nPad * 3), dSh(nPad * 3), dPow(nPad * 3),
           dRo(3), dBe(3), dFold(Lk * 3), dIdx(t), dNG(1);
    dRo.up(rOut); dBe.up(beta); dFold.up(fold);
    CHECKCUDAERR(cudaMemcpy(dIdx.p, idx.data(), t * sizeof(uint64_t), cudaMemcpyHostToDevice));
    stirgpu::buildG(dPts.p, dVals.p, (uint64_t *)dNG.p, dRo.p, dBe.p, s, (uint64_t *)dIdx.p, t, dFold.p, shiftK, logLk, 0);
    stirgpu::interpolate(dAns.p, nPad, dPts.p, dVals.p, dTmp.p, dZ.p, dT.p, (uint64_t *)dNG.p, 0);
    stirgpu::shake(dSh.p, nPad, dAns.p, dPts.p, dPow.p, (uint64_t *)dNG.p, 0);

    std::vector<FE> ansCpu(nPad * 3, Goldilocks::zero()), shCpu(nPad * 3, Goldilocks::zero());
    std::copy(ctx.ansCoeffs.begin(), ctx.ansCoeffs.end(), ansCpu.begin());
    std::copy(sh.begin(), sh.end(), shCpu.begin());
    expectEq(dPts.down(6 * 3), ctx.points, 6 * 3);
    expectEq(dVals.down(6 * 3), ctx.values, 6 * 3);
    expectEq(dAns.down(nPad * 3), ansCpu, nPad * 3);
    expectEq(dSh.down(nPad * 3), shCpu, nPad * 3);
}

TEST_F(StirGpuTest, g_set_of_one_point_has_empty_shake)
{
    // |G| = 1: Ans-hat is the constant y, Sh is empty (all padding).
    std::vector<FE> rOut = randomE3(1, 30), beta = randomE3(1, 31), fold = randomE3(4, 32);
    DevBuf dPts(3), dVals(3), dTmp(3), dZ(6), dT(3), dAns(3), dSh(3), dPow(3), dRo(3), dBe(3), dFold(12), dNG(1), dIdx(1);
    dRo.up(rOut); dBe.up(beta); dFold.up(fold);
    stirgpu::buildG(dPts.p, dVals.p, (uint64_t *)dNG.p, dRo.p, dBe.p, 1, (uint64_t *)dIdx.p, 0, dFold.p, Goldilocks::shift(), 2, 0);
    stirgpu::interpolate(dAns.p, 1, dPts.p, dVals.p, dTmp.p, dZ.p, dT.p, (uint64_t *)dNG.p, 0);
    stirgpu::shake(dSh.p, 1, dAns.p, dPts.p, dPow.p, (uint64_t *)dNG.p, 0);
    expectEq(dAns.down(3), beta, 3);
    expectEq(dSh.down(3), std::vector<FE>(3, Goldilocks::zero()), 3);
}

// s + t > blockDim with many repeats, an r_out equal to a later shift point and a repeated r_out:
// the survivors must come out in QuotientContext::add's first-occurrence order.
TEST_F(StirGpuTest, g_set_many_candidates_first_occurrence)
{
    const uint64_t s = 3, t = 600, logLk = 9, Lk = 1 << logLk;
    const Goldilocks::Element shiftK = Goldilocks::exp(Goldilocks::shift(), 8);
    stir::Domain LK{shiftK, logLk};
    std::mt19937_64 g(60);
    std::vector<uint64_t> idx(t);
    for (auto &x : idx) x = g() % Lk;
    std::vector<FE> fold = randomE3(Lk, 61), rOut = randomE3(s, 62), beta = randomE3(s, 63);
    { E3 pt; stir::embed(pt, LK.point(idx[5])); std::copy(pt, pt + 3, &rOut[3]); }   // r_out[1] collides with shift point 5
    std::copy(&rOut[0], &rOut[3], &rOut[6]);                                          // r_out[2] == r_out[0]

    stir::QuotientContext ctx;
    E3 zero; Goldilocks3::zero(zero);
    ctx.reset(zero);
    for (uint64_t j = 0; j < s; j++) ctx.add((const E3 &)rOut[j * 3], (const E3 &)beta[j * 3]);
    for (uint64_t i : idx) { E3 pt; stir::embed(pt, LK.point(i)); ctx.add(pt, (const E3 &)fold[i * 3]); }
    const uint64_t n = ctx.size();
    ASSERT_LT(n, s + t);
    ASSERT_GT(n, 256u);

    DevBuf dPts((s + t) * 3), dVals((s + t) * 3), dRo(s * 3), dBe(s * 3), dFold(Lk * 3), dIdx(t), dNG(1);
    dRo.up(rOut); dBe.up(beta); dFold.up(fold);
    CHECKCUDAERR(cudaMemcpy(dIdx.p, idx.data(), t * sizeof(uint64_t), cudaMemcpyHostToDevice));
    stirgpu::buildG(dPts.p, dVals.p, (uint64_t *)dNG.p, dRo.p, dBe.p, s, (uint64_t *)dIdx.p, t, dFold.p, shiftK, logLk, 0);
    ASSERT_EQ(Goldilocks::toU64(dNG.down(1)[0]), n);
    expectEq(dPts.down(n * 3), ctx.points, n * 3);
    expectEq(dVals.down(n * 3), ctx.values, n * 3);
}

// n = 300 > blockDim; the padding past n - 1 must be written as zeros.
TEST_F(StirGpuTest, shake_of_many_points_matches_cpu)
{
    const uint64_t n = 300, nPad = n + 7;
    std::vector<FE> pts = randomE3(n, 70), vals = randomE3(n, 71);
    stir::QuotientContext ctx;
    E3 zero; Goldilocks3::zero(zero);
    ctx.reset(zero);
    for (uint64_t i = 0; i < n; i++) ASSERT_TRUE(ctx.add((const E3 &)pts[i * 3], (const E3 &)vals[i * 3]));
    ctx.build();
    std::vector<FE> sh; ctx.shake(sh);
    sh.resize(nPad * 3, Goldilocks::zero());
    std::vector<FE> ans = ctx.ansCoeffs; ans.resize(nPad * 3, Goldilocks::zero());

    DevBuf dPts(nPad * 3), dAns(nPad * 3), dSh(nPad * 3), dPow(nPad * 3), dNG(1);
    dPts.up(pts); dAns.up(ans);
    CHECKCUDAERR(cudaMemcpy(dNG.p, &n, sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemset(dSh.p, 0xff, nPad * 3 * sizeof(gl64_t)));
    stirgpu::shake(dSh.p, nPad, dAns.p, dPts.p, dPow.p, (uint64_t *)dNG.p, 0);
    expectEq(dSh.down(nPad * 3), sh, nPad * 3);
}

// More points than one block has threads, padded past n; n = 0 must give all zeros.
TEST_F(StirGpuTest, interpolation_of_many_points_matches_cpu)
{
    for (uint64_t n : {0ull, 300ull}) {
        const uint64_t nPad = n + 5;
        std::vector<FE> pts = randomE3(n, 50), vals = randomE3(n, 51), cpu;
        stir::interpolate(cpu, pts.data(), vals.data(), n);
        cpu.resize(nPad * 3, Goldilocks::zero());
        DevBuf dPts(nPad * 3), dVals(nPad * 3), dTmp(nPad * 3), dZ((nPad + 1) * 3), dT(nPad * 3), dAns(nPad * 3), dNG(1);
        dPts.up(pts); dVals.up(vals);
        CHECKCUDAERR(cudaMemcpy(dNG.p, &n, sizeof(uint64_t), cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemset(dAns.p, 0xff, nPad * 3 * sizeof(gl64_t)));   // padding must be written, not inherited
        stirgpu::interpolate(dAns.p, nPad, dPts.p, dVals.p, dTmp.p, dZ.p, dT.p, (uint64_t *)dNG.p, 0);
        expectEq(dAns.down(nPad * 3), cpu, nPad * 3);
    }
}

// Builds a QuotientContext like Prover::degreeCorrect and compares quotientAndDegreeCorrect.
static void checkQuotient(const std::vector<FE> &rComb)
{
    const uint64_t logL = 10, N = 1 << logL;
    stir::Domain L{Goldilocks::shift(), logL};
    std::vector<FE> g = randomE3(N, 40), pts = randomE3(5, 41), vals = randomE3(5, 42);
    stir::QuotientContext ctx; ctx.reset((const E3 &)rComb[0]);
    for (int i = 0; i < 5; i++) ctx.add((const E3 &)pts[i * 3], (const E3 &)vals[i * 3]);
    ctx.build();
    std::vector<FE> cpu(N * 3);
    stir::quotientAndDegreeCorrect(cpu.data(), g.data(), L, ctx);

    // Z(X) = Π(X - a), zero-padded to nPad+1 like zCoeffs; Âns zero-padded to nPad like the proof region.
    const uint64_t n = 5, nPad = 7;
    std::vector<FE> z((nPad + 1) * 3, Goldilocks::zero()), ans(nPad * 3, Goldilocks::zero());
    z[0] = Goldilocks::one();
    for (uint64_t j = 0; j < n; j++)
        for (uint64_t m = j + 2; m-- > 0;) {   // descending, so z[m-1] is still the old value
            E3 hi; Goldilocks3::mul(hi, (const E3 &)z[m * 3], (const E3 &)ctx.points[j * 3]);
            E3 lo = {Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero()};
            if (m > 0) std::copy(&z[(m - 1) * 3], &z[m * 3], lo);
            Goldilocks3::sub((E3 &)z[m * 3], lo, hi);
        }
    std::copy(ctx.ansCoeffs.begin(), ctx.ansCoeffs.end(), ans.begin());

    std::vector<FE> gCol(N * 3);
    for (uint64_t r = 0; r < N; r++) for (int c = 0; c < 3; c++) gCol[c * N + r] = g[r * 3 + c];
    DevBuf dG(N * 3), dF(2 * N * 3), dA(nPad * 3), dZ((nPad + 1) * 3), dR(3), dN(1);
    dG.up(gCol); dA.up(ans); dZ.up(z); dR.up(rComb);
    CHECKCUDAERR(cudaMemcpy(dN.p, &n, 8, cudaMemcpyHostToDevice));
    stirgpu::quotientDegCor(dF.p, dG.p, logL, L.shift, dA.p, dZ.p, nPad, (uint64_t *)dN.p, dR.p, 0);
    expectEq(dF.down(N * 3), cpu, N * 3);
}

TEST_F(StirGpuTest, quotient_matches_cpu) { checkQuotient(randomE3(1, 43)); }

// r_comb·x = 1 at x = L.point(7): the geometric sum takes its (|G|+1) branch there.
TEST_F(StirGpuTest, quotient_geometric_one)
{
    stir::Domain L{Goldilocks::shift(), 10};
    std::vector<FE> rComb(3, Goldilocks::zero());
    rComb[0] = Goldilocks::inv(L.point(7));
    checkQuotient(rComb);
}

TEST_F(StirGpuTest, ood_flag)
{
    stir::Domain L{Goldilocks::shift(), 8};
    std::vector<FE> in(3, Goldilocks::zero()), out = randomE3(1, 44);
    in[0] = L.point(5);
    DevBuf dIn(3), dOut(3), dFlag(1);
    dIn.up(in); dOut.up(out);
    stirgpu::oodFlag((uint64_t *)dFlag.p, dOut.p, 1, L.shift, 8, 0);
    EXPECT_EQ(Goldilocks::toU64(dFlag.down(1)[0]), 0u);
    stirgpu::oodFlag((uint64_t *)dFlag.p, dIn.p, 1, L.shift, 8, 0);
    EXPECT_EQ(Goldilocks::toU64(dFlag.down(1)[0]), 1u);
}

// ---- The driver against STIR::prove ---------------------------------------------------------------
namespace
{
// Reads the device proof back into a StirProof, from the same buffers writeProof will read.
Proof readGpuProof(const StirParams &p, const StirGpuIo &io)
{
    Proof proof = Stir::makeProof(p);
    const StirLayout &l = io.layout;
    auto down = [&](const gl64_t *src, uint64_t n) {
        std::vector<FE> h(n); CHECKCUDAERR(cudaMemcpy(h.data(), src, n * 8, cudaMemcpyDeviceToHost)); return h;
    };
    const gl64_t *pr = io.base + l.proof;
    const uint64_t M = p.M(), s = p.numOodSamples;
    for (uint64_t i = 0; i + 1 < M; i++) { auto b = down(pr + l.betas + i * s * 3, s * 3); proof.betas[i] = b; }
    proof.finalPol = down(pr + l.finalPol, proof.finalPol.size());
    { auto n = down(pr + l.nonces, M); for (uint64_t i = 0; i < M; i++) proof.nonces[i] = n[i].fe; }
    for (uint64_t i = 1; i < M; i++) { proof.ansCoeffs[i - 1] = down(pr + l.ans[i - 1], proof.ansCoeffs[i - 1].size());
                                       proof.shakeCoeffs[i - 1] = down(pr + l.shake[i - 1], proof.shakeCoeffs[i - 1].size()); }
    for (uint64_t i = 0; i < M; i++) {
        uint64_t k = 1ull << p.logFoldingFactors[i], height = (1ull << p.logDomainSizes[i]) >> p.logFoldingFactors[i];
        MerkleTreeGL mt(p.merkleTreeArity, p.lastLevelVerification, p.merkleTreeCustom, height, k * 3, false, false);
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

struct GpuRun {   // test-only owner of the device buffers the STARK would carve from aux_trace
    std::unique_ptr<DevBuf> base;
    StirGpuIo io;
    Proof proof;
};

GpuRun proveOnGpu(const StirParams &p, const std::vector<FE> &f0)
{
    GpuRun r;
    const uint64_t nQ = p.numQueries[0];
    auto numNodes = [&](uint64_t h) { return MerkleTreeGL(p.merkleTreeArity, p.lastLevelVerification, p.merkleTreeCustom, h, 3, false, false).numNodes; };
    uint64_t maxWidth = 0, maxProof = 0;
    for (uint64_t i = 0; i < p.M(); i++) {
        uint64_t h = (1ull << p.logDomainSizes[i]) >> p.logFoldingFactors[i];
        MerkleTreeGL mt(p.merkleTreeArity, p.lastLevelVerification, p.merkleTreeCustom, h, (1ull << p.logFoldingFactors[i]) * 3, false, false);
        maxWidth = std::max(maxWidth, mt.getMerkleTreeWidth()); maxProof = std::max(maxProof, mt.getMerkleProofSize());
    }
    const uint64_t stride = nQ * (maxWidth + maxProof);
    const uint64_t fSize = (1ull << p.logDomainSizes[0]) * 3;
    // [f | queries | raw | perm | nonce | nonceBlocks | inputHashNonce | STIR layout]
    uint64_t at = 0;
    const uint64_t oF = at; at += fSize;
    const uint64_t oQ = at; at += p.M() * stride;
    const uint64_t oRaw = at; at += nQ;
    const uint64_t oPerm = at; at += (nQ * p.logDomainSizes[0] + 62) / 63 + 1;
    const uint64_t oNonce = at; at += 1;
    const uint64_t oBlocks = at; at += GRIND_NONCE_BLOCKS_MAX;
    const uint64_t oIn = at; at += HASH_SIZE;
    StirLayout l = makeStirLayout(p, stirAlign(at), numNodes);
    r.base = std::make_unique<DevBuf>(l.total);
    gl64_t *b = r.base->p;
    CHECKCUDAERR(cudaMemcpy(b + oF, f0.data(), fSize * 8, cudaMemcpyHostToDevice));
    r.io = StirGpuIo{b, l, b + oF, (uint64_t *)(b + oRaw), (uint64_t *)(b + oPerm), (uint64_t *)(b + oNonce),
                     (uint64_t *)(b + oBlocks), (uint64_t *)(b + oIn), b + oQ, stride};

    int dev = 0; CHECKCUDAERR(cudaGetDevice(&dev));
    uint32_t gpuId = (uint32_t)dev;
    TranscriptGL_GPU::init_const(&gpuId, 1, p.transcriptArity);
    Poseidon2GoldilocksGPUGrinding::initConstants(&gpuId, 1);   // also loads the Merkle widths
    TranscriptGL_GPU t(p.transcriptArity, p.merkleTreeCustom, 0), h(p.transcriptArity, p.merkleTreeCustom, 0);
    DevBuf seed(4); seed.up(std::vector<FE>(SEED, SEED + 4));
    t.put((Goldilocks::Element *)seed.p, 4, 0);
    TimerGPU timer;
    stirProve_gpu(p, r.io, &t, &h, [] {}, timer, 0);
    CHECKCUDAERR(cudaStreamSynchronize(0));
    r.proof = readGpuProof(p, r.io);
    return r;
}

void expectSameProof(const Proof &g, const Proof &c)
{
    ASSERT_EQ(g.trees.size(), c.trees.size());
    for (uint64_t i = 0; i < c.trees.size(); i++) {
        for (uint64_t k = 0; k < HASH_SIZE; k++) ASSERT_EQ(Goldilocks::toU64(g.trees[i].root[k]), Goldilocks::toU64(c.trees[i].root[k])) << "root " << i;
        for (uint64_t k = 0; k < c.trees[i].last_levels.size(); k++)
            ASSERT_EQ(Goldilocks::toU64(g.trees[i].last_levels[k]), Goldilocks::toU64(c.trees[i].last_levels[k])) << "last level " << i;
        for (uint64_t q = 0; q < c.trees[i].polQueries.size(); q++) {
            const auto &a = g.trees[i].polQueries[q][0], &b = c.trees[i].polQueries[q][0];
            for (uint64_t e = 0; e < b.v.size(); e++) ASSERT_EQ(Goldilocks::toU64(a.v[e][0]), Goldilocks::toU64(b.v[e][0])) << "tree " << i << " q " << q;
            for (uint64_t l = 0; l < b.mp.size(); l++) for (uint64_t e = 0; e < b.mp[l].size(); e++)
                ASSERT_EQ(Goldilocks::toU64(a.mp[l][e]), Goldilocks::toU64(b.mp[l][e])) << "sibling " << i << "/" << q;
        }
    }
    for (uint64_t i = 0; i < c.betas.size(); i++) expectEq(g.betas[i], c.betas[i], c.betas[i].size());
    for (uint64_t i = 0; i < c.ansCoeffs.size(); i++) { expectEq(g.ansCoeffs[i], c.ansCoeffs[i], c.ansCoeffs[i].size());
                                                        expectEq(g.shakeCoeffs[i], c.shakeCoeffs[i], c.shakeCoeffs[i].size()); }
    expectEq(g.finalPol, c.finalPol, c.finalPol.size());
    for (uint64_t i = 0; i < c.nonces.size(); i++) ASSERT_EQ(g.nonces[i], c.nonces[i]) << "nonce " << i;
}

void gpuMatchesCpuAndVerifies(const StirParams &p, uint64_t seed)
{
    std::vector<FE> f0; honestF0(f0, p, seed);
    Proof cpu = Stir::makeProof(p);
    TranscriptGL tc = freshTranscript(p);
    ProofTree<FE> stageQ(HASH_SIZE, p.numQueries[0], p.merkleTreeArity, p.lastLevelVerification);
    Stir::prove(cpu, p, f0.data(), tc, stageQ, nullptr, 0, grind);

    GpuRun g = proveOnGpu(p, f0);
    expectSameProof(g.proof, cpu);

    TranscriptGL tv = freshTranscript(p);
    std::string why;
    auto checkF0 = [&](uint64_t, uint64_t idx, const E3 &v) { return stir::equal(v, (const E3 &)f0[idx * 3]); };
    EXPECT_TRUE(Stir::verify(g.proof, p, tv, checkF0, powOk, &why)) << why;
}
} // namespace

TEST_F(StirGpuTest, driver_matches_cpu_default_params) { gpuMatchesCpuAndVerifies(testParams(), 7); }
TEST_F(StirGpuTest, hashed_final_pol_and_last_levels) { gpuMatchesCpuAndVerifies(testParams(true, 2), 8); }
TEST_F(StirGpuTest, single_iteration)
{
    StirParams p = testParams();
    p.logFoldingFactors = {3}; p.logDegrees = {9, 6}; p.logDomainSizes = {11, 10};
    p.numQueries = {12}; p.grindingBitsQueries = {2};
    gpuMatchesCpuAndVerifies(p, 9);
}
TEST_F(StirGpuTest, folding_factor_32)
{
    StirParams p = testParams();
    p.logFoldingFactors = {5, 2}; p.logDegrees = {12, 7, 5}; p.logDomainSizes = {14, 13, 12};
    p.numQueries = {30, 10}; p.grindingBitsQueries = {2, 2};
    gpuMatchesCpuAndVerifies(p, 10);
}
TEST_F(StirGpuTest, tiny_domain_duplicates)
{
    // t_0 = 12 draws from |L_0^k| = 32 cosets; seed 14 repeats three (seed 11 happens to draw 12 distinct).
    StirParams p = testParams();
    p.logFoldingFactors = {3, 2}; p.logDegrees = {7, 4, 2}; p.logDomainSizes = {8, 7, 6};
    p.numQueries = {12, 4}; p.grindingBitsQueries = {0, 0};
    gpuMatchesCpuAndVerifies(p, 14);
    // the case must actually have occurred: the last padded Âns coefficient is zero
    std::vector<FE> f0; honestF0(f0, p, 14);
    GpuRun g = proveOnGpu(p, f0);
    const auto &ans = g.proof.ansCoeffs[0];
    std::set<uint64_t> leaves;   // distinct T_0 openings = distinct coset indices
    for (auto &q : g.proof.trees[0].polQueries) leaves.insert(Goldilocks::toU64(q[0].v[0][0]));
    ASSERT_LT(leaves.size(), p.numQueries[0]);
    EXPECT_TRUE(Goldilocks::isZero(ans[ans.size() - 3]) && Goldilocks::isZero(ans[ans.size() - 1]));
}

// Grinding: every family's GPU nonce must be the minimal valid nonce, which the CPU provers rely on.
namespace
{
using GpuGrindFn = void (*)(uint64_t *, uint64_t *, const uint64_t *, uint32_t, cudaStream_t);
uint64_t firstLimbP2(const uint64_t *c, uint64_t n)
{
    Goldilocks::Element in[8] = {}, out[8];
    std::memcpy(in, c, 3 * sizeof(uint64_t)); in[3] = Goldilocks::fromU64(n);
    Poseidon2Goldilocks<8>::permute(out, in, Poseidon2Mode::Scalar);
    return Goldilocks::toU64(out[0]);
}
uint64_t firstLimbP1(const uint64_t *c, uint64_t n)
{
    Goldilocks::Element in[8] = {}, out[8];
    std::memcpy(in, c, 3 * sizeof(uint64_t)); in[3] = Goldilocks::fromU64(n);
    PoseidonGoldilocks<8>::permute(out, in, PoseidonMode::Scalar);
    return Goldilocks::toU64(out[0]);
}
uint64_t firstLimbB3(const uint64_t *c, uint64_t n)
{
    uint64_t in[8] = {c[0], c[1], c[2], n, 0, 0, 0, 0}, out[8];
    blake3core::permute8(in, out);
    return out[0];
}
uint64_t cpuMinimalNonce(uint64_t (*firstLimb)(const uint64_t *, uint64_t), const uint64_t *c, uint32_t bits)
{
    uint64_t n = 0;
    while (firstLimb(c, n) >= (uint64_t(1) << (64 - bits))) n++;
    return n;
}
// Runs the grind directly, or captured into a CUDA graph and replayed twice, as the FRI path does.
uint64_t gpuNonce(GpuGrindFn fn, const uint64_t *c, uint32_t bits, bool graph)
{
    DevBuf in(3), nonce(1), blocks(GRIND_NONCE_BLOCKS_MAX);
    CHECKCUDAERR(cudaMemcpy(in.p, c, 3 * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemset(nonce.p, 0x5A, sizeof(uint64_t)));   // stale value the grind must overwrite
    cudaStream_t s; CHECKCUDAERR(cudaStreamCreate(&s));
    if (graph) {
        cudaGraph_t g; cudaGraphExec_t ge;
        CHECKCUDAERR(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        fn((uint64_t *)nonce.p, (uint64_t *)blocks.p, (uint64_t *)in.p, bits, s);
        CHECKCUDAERR(cudaStreamEndCapture(s, &g));
        CHECKCUDAERR(cudaGraphInstantiate(&ge, g, 0));
        for (int r = 0; r < 2; r++) CHECKCUDAERR(cudaGraphLaunch(ge, s));
        CHECKCUDAERR(cudaStreamSynchronize(s));
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    } else {
        fn((uint64_t *)nonce.p, (uint64_t *)blocks.p, (uint64_t *)in.p, bits, s);
    }
    CHECKCUDAERR(cudaStreamSynchronize(s));
    CHECKCUDAERR(cudaStreamDestroy(s));
    return Goldilocks::toU64(nonce.down(1)[0]);
}
void checkMinimalNonces(GpuGrindFn fn, uint64_t (*firstLimb)(const uint64_t *, uint64_t), uint64_t seed)
{
    std::mt19937_64 rng(seed);
    for (uint32_t bits : {12u, 18u}) {
        for (int k = 0; k < 3; k++) {
            uint64_t c[3];
            for (auto &x : c) x = rng() % GOLDILOCKS_PRIME;
            const uint64_t want = cpuMinimalNonce(firstLimb, c, bits);
            EXPECT_EQ(gpuNonce(fn, c, bits, false), want) << "bits " << bits << " challenge " << k;
            EXPECT_EQ(gpuNonce(fn, c, bits, true), want) << "graph, bits " << bits << " challenge " << k;
        }
    }
}
} // namespace

TEST_F(StirGpuTest, grinding_minimal_nonce_poseidon2)
{
    int dev = 0; CHECKCUDAERR(cudaGetDevice(&dev)); uint32_t gpuId = (uint32_t)dev;
    Poseidon2GoldilocksGPUGrinding::initConstants(&gpuId, 1);
    checkMinimalNonces(Poseidon2GoldilocksGPUGrinding::grinding, firstLimbP2, 21);
}
TEST_F(StirGpuTest, grinding_minimal_nonce_poseidon1)
{
    int dev = 0; CHECKCUDAERR(cudaGetDevice(&dev)); uint32_t gpuId = (uint32_t)dev;
    PoseidonGoldilocksGPUGrinding::initConstants(&gpuId, 1);
    checkMinimalNonces(PoseidonGoldilocksGPUGrinding::grinding, firstLimbP1, 22);
}
TEST_F(StirGpuTest, grinding_minimal_nonce_blake3)
{
    checkMinimalNonces(Blake3GoldilocksGPU::grinding, firstLimbB3, 23);
}

// Query sampling (STIR and FRI): GPU getPermutations equals the CPU transcript's and leaves the same state.
// Blake3 runs in stirBenchGpu (*Sanity*): the hash family is fixed per process.
namespace
{
void checkPermutations(bool parallel)
{
    int dev = 0; CHECKCUDAERR(cudaGetDevice(&dev));
    uint32_t gpuId = (uint32_t)dev;
    TranscriptGL_GPU::init_const(&gpuId, 1, 4);
    // nBits 20/22/17/13 do not divide 63, so queries straddle field boundaries.
    for (auto [n, nBits] : std::vector<std::pair<uint64_t, uint64_t>>{{211, 20}, {53, 22}, {7, 17}, {300, 13}, {1, 21}}) {
        TranscriptGL cpu(4, true);
        cpu.put((Goldilocks::Element *)SEED, 4);
        std::vector<uint64_t> want(n);
        cpu.getPermutations(want.data(), n, nBits);
        uint64_t wantNext[3]; cpu.getField(wantNext);

        TranscriptGL_GPU gpu(4, true, 0, parallel);
        DevBuf seed(4), res(n), scratch((n * nBits + 62) / 63 + 1), next(3);
        seed.up(std::vector<FE>(SEED, SEED + 4));
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
} // namespace
TEST_F(StirGpuTest, permutations_match_cpu_poseidon2) { checkPermutations(true); }
TEST_F(StirGpuTest, permutations_match_cpu_poseidon2_serial) { checkPermutations(false); }
