// Equivalence tests for the DEEP/FRI polynomial kernels (starkpil/fri_expression.cuh).
//
// The oracle is a host implementation written straight from the mathematical
// definition; it was cross-checked bit-for-bit against the Horner kernel these
// folded kernels replaced, before that kernel was removed.
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include "cuda_utils.cuh"
#include "../../starkpil/fri_expression.cuh"

namespace {

// A synthetic instance small enough to reference-check on the host but shaped
// like the real thing: several opening points (including an empty one and a
// partial group of 4, to exercise the batched inversion), mixed dim 1/dim 3
// columns, and all three source buffers (cm / custom / fixed).
struct FriCase {
    uint64_t domainSize = 1024;
    uint64_t nBits = 10;
    std::vector<uint64_t> counts{3, 0, 5, 2, 4, 1};   // per opening point
    uint64_t nDistinctCols = 5;                       // pool the evals draw from, so the
                                                      // same column recurs across openings
    uint64_t stageCols() const { return nDistinctCols + FIELD_EXTENSION; }

    std::vector<EvalInfo> evalInfo;                   // flattened, opening-major
    std::vector<Goldilocks::Element> cmPols, customPols, fixedPols;
    std::vector<Goldilocks::Element> evals, xDivXSub, x, vf1, vf2;

    uint64_t nOpenings() const { return counts.size(); }
    uint64_t nEvals() const { uint64_t s = 0; for (auto c : counts) s += c; return s; }

    void build(uint64_t seed)
    {
        std::mt19937_64 rng(seed);
        auto rnd = [&]() { return Goldilocks::fromU64(rng() % GOLDILOCKS_PRIME); };

        uint64_t buf = stageCols() * domainSize;
        cmPols.resize(buf); customPols.resize(buf); fixedPols.resize(buf);
        for (uint64_t i = 0; i < buf; i++) { cmPols[i] = rnd(); customPols[i] = rnd(); fixedPols[i] = rnd(); }

        x.resize(domainSize);
        for (uint64_t i = 0; i < domainSize; i++) x[i] = rnd();

        // xi must differ from every x[r] or the denominator is zero.
        xDivXSub.resize(nOpenings() * FIELD_EXTENSION);
        for (uint64_t i = 0; i < xDivXSub.size(); i++) xDivXSub[i] = rnd();

        vf1.resize(FIELD_EXTENSION); vf2.resize(FIELD_EXTENSION);
        for (uint64_t i = 0; i < FIELD_EXTENSION; i++) { vf1[i] = rnd(); vf2[i] = rnd(); }

        evals.resize(nEvals() * FIELD_EXTENSION);
        for (uint64_t i = 0; i < evals.size(); i++) evals[i] = rnd();

        uint64_t evalPos = 0;
        for (uint64_t o = 0; o < nOpenings(); o++) {
            for (uint64_t j = 0; j < counts[o]; j++) {
                // Draw from a small pool so most columns are shared between openings,
                // which is what the grouped kernel exists to exploit.
                uint64_t col = (o * 3 + j * 7) % nDistinctCols;
                EvalInfo e{};
                e.type = col % 3;                     // cycle cm / custom / fixed
                e.offset = 0;
                e.dim = (col % 4 == 0) ? FIELD_EXTENSION : 1;
                e.stageCols = stageCols();
                e.stagePos = col;
                e.openingPos = o;
                e.evalPos = evalPos;
                evalInfo.push_back(e);
                evalPos++;
            }
        }
    }

    // Host reference, written straight from the mathematical definition:
    //   fri[r] = SUM_o vf1^(O-1-o) / (x[r] - xi_o) * SUM_j vf2^(n_o-1-j) * (p_j[r] - e_j)
    std::vector<Goldilocks::Element> reference() const
    {
        using F3 = Goldilocks3;
        std::vector<Goldilocks::Element> out(domainSize * FIELD_EXTENSION);
        const F3::Element &vf1e = *(F3::Element *)vf1.data();
        const F3::Element &vf2e = *(F3::Element *)vf2.data();

        for (uint64_t r = 0; r < domainSize; r++) {
            F3::Element fri;
            F3::zero(fri);
            uint64_t flat = 0;
            for (uint64_t o = 0; o < nOpenings(); o++) {
                F3::Element accum;
                F3::zero(accum);
                for (uint64_t j = 0; j < counts[o]; j++) {
                    const EvalInfo &e = evalInfo[flat + j];
                    const Goldilocks::Element *pol =
                        (e.type == 0) ? cmPols.data() : (e.type == 1) ? customPols.data() : fixedPols.data();
                    F3::Element term;
                    for (uint64_t d = 0; d < FIELD_EXTENSION; d++) {
                        term[d] = (d == 0 || e.dim == FIELD_EXTENSION)
                                      ? pol[e.offset + (e.stagePos + d) * domainSize + r]
                                      : Goldilocks::zero();
                    }
                    F3::Element &ev = *(F3::Element *)&evals[e.evalPos * FIELD_EXTENSION];
                    F3::sub(term, term, ev);
                    // vf2^(n_o-1-j): fold by Horner over j, same as the kernel
                    F3::mul(accum, accum, vf2e);
                    F3::add(accum, accum, term);
                }
                F3::Element &xi = *(F3::Element *)&xDivXSub[o * FIELD_EXTENSION];
                F3::Element num, den;
                num[0] = x[r]; num[1] = Goldilocks::zero(); num[2] = Goldilocks::zero();
                F3::sub(den, num, xi);
                F3::Element invDen;
                F3::inv(invDen, den);
                F3::mul(accum, accum, invDen);
                // vf1^(O-1-o): Horner over o
                F3::mul(fri, fri, vf1e);
                F3::add(fri, fri, accum);
                flat += counts[o];
            }
            for (uint64_t d = 0; d < FIELD_EXTENSION; d++) out[r * FIELD_EXTENSION + d] = fri[d];
        }
        return out;
    }
};

// Device-side mirror of a FriCase: allocates and uploads everything the kernels read.
struct FriDevice {
    gl64_t *fri = nullptr, *evals = nullptr, *vf1 = nullptr, *vf2 = nullptr;
    gl64_t *cm = nullptr, *custom = nullptr, *fixed = nullptr, *xDivXSub = nullptr, *x = nullptr;
    uint64_t *counts = nullptr;
    EvalInfo **evalInfoPerOpening = nullptr;
    std::vector<EvalInfo *> perOpening;

    template <typename T>
    static T *upload(const std::vector<T> &v)
    {
        T *d = nullptr;
        CHECKCUDAERR(cudaMalloc(&d, v.size() * sizeof(T)));
        CHECKCUDAERR(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
        return d;
    }

    explicit FriDevice(const FriCase &c)
    {
        CHECKCUDAERR(cudaMalloc(&fri, c.domainSize * FIELD_EXTENSION * sizeof(gl64_t)));
        evals = (gl64_t *)upload(c.evals);
        vf1 = (gl64_t *)upload(c.vf1);
        vf2 = (gl64_t *)upload(c.vf2);
        cm = (gl64_t *)upload(c.cmPols);
        custom = (gl64_t *)upload(c.customPols);
        fixed = (gl64_t *)upload(c.fixedPols);
        xDivXSub = (gl64_t *)upload(c.xDivXSub);
        x = (gl64_t *)upload(c.x);
        counts = upload(c.counts);

        uint64_t flat = 0;
        for (uint64_t o = 0; o < c.nOpenings(); o++) {
            std::vector<EvalInfo> slice(c.evalInfo.begin() + flat, c.evalInfo.begin() + flat + c.counts[o]);
            // cudaMalloc(0) yields nullptr, which the kernel never dereferences (count 0).
            perOpening.push_back(slice.empty() ? nullptr : upload(slice));
            flat += c.counts[o];
        }
        evalInfoPerOpening = upload(perOpening);
    }

    std::vector<Goldilocks::Element> download(uint64_t domainSize) const
    {
        std::vector<Goldilocks::Element> out(domainSize * FIELD_EXTENSION);
        CHECKCUDAERR(cudaMemcpy(out.data(), fri, out.size() * sizeof(gl64_t), cudaMemcpyDeviceToHost));
        return out;
    }

    ~FriDevice()
    {
        for (auto p : perOpening) cudaFree(p);
        for (void *p : {(void *)fri, (void *)evals, (void *)vf1, (void *)vf2, (void *)cm, (void *)custom,
                        (void *)fixed, (void *)xDivXSub, (void *)x, (void *)counts, (void *)evalInfoPerOpening})
            cudaFree(p);
    }
};

void expectEqual(const std::vector<Goldilocks::Element> &got, const std::vector<Goldilocks::Element> &want)
{
    ASSERT_EQ(got.size(), want.size());
    for (uint64_t i = 0; i < got.size(); i++) {
        ASSERT_EQ(Goldilocks::toU64(got[i]), Goldilocks::toU64(want[i])) << "element " << i;
    }
}

} // namespace

TEST(FriExpression, FoldedMatchesHostReference)
{
    FriCase c;
    c.build(0xbeef);
    FriDevice d(c);

    gl64_t *coef = nullptr, *k = nullptr;
    CHECKCUDAERR(cudaMalloc(&coef, c.nEvals() * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&k, c.nOpenings() * FIELD_EXTENSION * sizeof(gl64_t)));

    computeFRIFoldedConstants<<<1, 64>>>(c.nOpenings(), d.counts, d.evalInfoPerOpening, d.evals,
                                        d.vf1, d.vf2, coef, k);
    computeFRIExpressionFolded<<<4, 256>>>(c.domainSize, c.nBits, c.nOpenings(), d.fri, d.counts,
                                           d.evalInfoPerOpening, coef, k, d.cm, d.xDivXSub, d.x,
                                           d.fixed, d.custom);
    CHECKCUDAERR(cudaDeviceSynchronize());

    expectEqual(d.download(c.domainSize), c.reference());
    cudaFree(coef);
    cudaFree(k);
}

// Across the shapes that exercise the batched inversion (exact multiple of 4,
// partial group, single opening), empty openings, and a long vf2 power chain.
class FriExpressionShapes : public ::testing::TestWithParam<std::vector<uint64_t>> {};

TEST_P(FriExpressionShapes, FoldedMatchesHostReference)
{
    FriCase c;
    c.counts = GetParam();
    c.build(0x5eed + c.nOpenings());
    FriDevice d(c);

    gl64_t *coef = nullptr, *k = nullptr;
    CHECKCUDAERR(cudaMalloc(&coef, c.nEvals() * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&k, c.nOpenings() * FIELD_EXTENSION * sizeof(gl64_t)));
    computeFRIFoldedConstants<<<1, 64>>>(c.nOpenings(), d.counts, d.evalInfoPerOpening, d.evals,
                                        d.vf1, d.vf2, coef, k);
    computeFRIExpressionFolded<<<4, 256>>>(c.domainSize, c.nBits, c.nOpenings(), d.fri, d.counts,
                                          d.evalInfoPerOpening, coef, k, d.cm, d.xDivXSub, d.x,
                                          d.fixed, d.custom);
    CHECKCUDAERR(cudaDeviceSynchronize());

    expectEqual(d.download(c.domainSize), c.reference());
    cudaFree(coef);
    cudaFree(k);
}

INSTANTIATE_TEST_SUITE_P(Shapes, FriExpressionShapes,
    ::testing::Values(std::vector<uint64_t>{7},                       // single opening
                      std::vector<uint64_t>{4, 4, 4, 4},              // exact group of 4
                      std::vector<uint64_t>{1, 1, 1, 1, 1},           // partial group of 1
                      std::vector<uint64_t>{0, 0, 3},                 // leading empty openings
                      std::vector<uint64_t>{200, 1, 0, 37, 5, 5, 5},  // long vf2 chain
                      std::vector<uint64_t>{2, 3, 4, 5, 6, 7, 8, 9})); // two full groups

// ---------------------------------------------------------------------------
// Timing A/B at the real shapes of the zisk proving key
// ---------------------------------------------------------------------------
// Disabled by default; run with --gtest_also_run_disabled_tests.
namespace {

struct BenchShape {
    const char *name;
    uint64_t nBits, nBitsExt, nOpenings, nEvals, nCols;
};

// Each column is opened at a RUN of consecutive opening points, which is how a
// real eval map looks (a column read at prime -1 and 0 lands on adjacent
// openingPos), and is what decides how much reuse a group of G can capture.
std::vector<EvalInfo> benchEvalInfo(const BenchShape &sh, uint64_t stageCols)
{
    std::vector<EvalInfo> out;
    uint64_t base = sh.nEvals / sh.nCols, extra = sh.nEvals % sh.nCols;
    uint64_t evalPos = 0;
    for (uint64_t c = 0; c < sh.nCols && evalPos < sh.nEvals; c++) {
        uint64_t k = base + (c < extra ? 1 : 0);
        if (k == 0) k = 1;
        if (k > sh.nOpenings) k = sh.nOpenings;
        uint64_t start = (c * 7) % (sh.nOpenings - k + 1);
        for (uint64_t t = 0; t < k && evalPos < sh.nEvals; t++) {
            EvalInfo e{};
            e.type = 0;
            e.offset = 0;
            e.stagePos = c;
            e.stageCols = stageCols;
            e.dim = (c % 4 == 0) ? FIELD_EXTENSION : 1;
            e.openingPos = start + t;
            e.evalPos = evalPos++;
            out.push_back(e);
        }
    }
    return out;
}

void runBench(const BenchShape &sh)
{
    const uint64_t domainSize = 1ULL << sh.nBitsExt;
    const uint64_t stageCols = sh.nCols + FIELD_EXTENSION;
    std::vector<EvalInfo> ev = benchEvalInfo(sh, stageCols);
    const uint64_t nEvals = ev.size();

    std::vector<uint64_t> counts(sh.nOpenings, 0);
    for (const auto &e : ev) counts[e.openingPos]++;

    gl64_t *pols = nullptr, *fri = nullptr, *evals = nullptr, *x = nullptr, *xi = nullptr;
    gl64_t *vf1 = nullptr, *vf2 = nullptr, *coef = nullptr, *k = nullptr;
    uint64_t *dCounts = nullptr;
    EvalInfo **infoPerOpening = nullptr;
    const size_t polBytes = (size_t)stageCols * domainSize * sizeof(gl64_t);
    if (cudaMalloc(&pols, polBytes) != cudaSuccess) {
        cudaGetLastError();
        printf("[bench] %-14s SKIPPED (needs %zu MiB for the trace)\n", sh.name, polBytes >> 20);
        return;
    }
    CHECKCUDAERR(cudaMemset(pols, 1, polBytes));
    CHECKCUDAERR(cudaMalloc(&fri, domainSize * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&evals, nEvals * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&coef, nEvals * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&k, sh.nOpenings * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&x, domainSize * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&xi, sh.nOpenings * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&vf1, FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&vf2, FIELD_EXTENSION * sizeof(gl64_t)));

    std::mt19937_64 rng(7);
    auto uploadRandom = [&](gl64_t *dst, uint64_t n) {
        std::vector<uint64_t> h(n);
        for (auto &v : h) v = rng() % GOLDILOCKS_PRIME;
        CHECKCUDAERR(cudaMemcpy(dst, h.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    };
    uploadRandom(evals, nEvals * FIELD_EXTENSION);
    uploadRandom(x, domainSize);
    uploadRandom(xi, sh.nOpenings * FIELD_EXTENSION);
    uploadRandom(vf1, FIELD_EXTENSION);
    uploadRandom(vf2, FIELD_EXTENSION);
    CHECKCUDAERR(cudaMalloc(&dCounts, sh.nOpenings * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(dCounts, counts.data(), sh.nOpenings * sizeof(uint64_t), cudaMemcpyHostToDevice));

    std::vector<EvalInfo *> perOp;
    uint64_t flat = 0;
    std::vector<EvalInfo> sorted = ev;
    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const EvalInfo &a, const EvalInfo &b) { return a.openingPos < b.openingPos; });
    for (uint64_t o = 0; o < sh.nOpenings; o++) {
        EvalInfo *d = nullptr;
        if (counts[o]) {
            CHECKCUDAERR(cudaMalloc(&d, counts[o] * sizeof(EvalInfo)));
            CHECKCUDAERR(cudaMemcpy(d, sorted.data() + flat, counts[o] * sizeof(EvalInfo), cudaMemcpyHostToDevice));
        }
        perOp.push_back(d);
        flat += counts[o];
    }
    CHECKCUDAERR(cudaMalloc(&infoPerOpening, sh.nOpenings * sizeof(EvalInfo *)));
    CHECKCUDAERR(cudaMemcpy(infoPerOpening, perOp.data(), sh.nOpenings * sizeof(EvalInfo *), cudaMemcpyHostToDevice));

    const uint64_t nThreads = 256, nBlocks = domainSize / nThreads;
    const int reps = 5;
    auto timeMs = [&](auto &&launch) {
        launch();
        CHECKCUDAERR(cudaDeviceSynchronize());
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int i = 0; i < reps; i++) launch();
        cudaEventRecord(b);
        CHECKCUDAERR(cudaDeviceSynchronize());
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        cudaEventDestroy(a); cudaEventDestroy(b);
        return ms / reps;
    };

    float folded = timeMs([&] {
        computeFRIFoldedConstants<<<1, 64>>>(sh.nOpenings, dCounts, infoPerOpening, evals, vf1, vf2, coef, k);
        computeFRIExpressionFolded<<<nBlocks, nThreads>>>(domainSize, sh.nBits, sh.nOpenings, fri, dCounts,
                                                          infoPerOpening, coef, k, pols, xi, x, pols, pols);
    });

    printf("[bench] %-14s O=%-3lu evals=%-5lu cols=%-5lu  folded %8.3f ms\n",
           sh.name, sh.nOpenings, nEvals, sh.nCols, folded);
    fflush(stdout);

    for (auto p : perOp) cudaFree(p);
    for (void *p : {(void *)pols, (void *)fri, (void *)evals, (void *)coef, (void *)k, (void *)x,
                    (void *)xi, (void *)vf1, (void *)vf2, (void *)dCounts, (void *)infoPerOpening})
        cudaFree(p);
}

} // namespace

TEST(FriExpression, DISABLED_BenchZiskShapes)
{
    // Measured from /data/provingKey/zisk starkinfo.json (openingPoints, evMap,
    // distinct columns in the eval map).
    const BenchShape shapes[] = {
        {"Keccakf",     18, 19, 32, 5408, 2170},
        {"Sha256f",     18, 19, 87, 1266,  114},
        {"Poseidon",    17, 18, 17,  534,  316},
        {"compressor",  20, 22,  9,  181,  134},
        {"recursive2",  17, 20,  5,  135,  104},
        {"Main",        22, 23,  3,   59,   50},
    };
    for (const auto &sh : shapes) runBench(sh);
}

// Large opening counts: zisk Sha256f has 87 and examples/hashes Blake2b has 97,
// so the og loop runs 15-25 times and the vf1 exponent reaches its maximum. The
// end-to-end proofs available in-repo only reach 5 openings, so this is where
// that range is checked.
TEST(FriExpression, FoldedMatchesHostReferenceAtLargeOpeningCount)
{
    for (uint64_t O : {17, 32, 57, 73, 87, 97}) {
        FriCase c;
        c.counts.resize(O);
        for (uint64_t o = 0; o < O; o++) c.counts[o] = (o * 5 + 1) % 9;  // 0..8, empties included
        c.nDistinctCols = 11;
        c.build(0xba5e + O);
        FriDevice d(c);

        gl64_t *coef = nullptr, *k = nullptr;
        CHECKCUDAERR(cudaMalloc(&coef, c.nEvals() * FIELD_EXTENSION * sizeof(gl64_t)));
        CHECKCUDAERR(cudaMalloc(&k, O * FIELD_EXTENSION * sizeof(gl64_t)));
        computeFRIFoldedConstants<<<(O + 63) / 64, 64>>>(O, d.counts, d.evalInfoPerOpening, d.evals,
                                                        d.vf1, d.vf2, coef, k);
        computeFRIExpressionFolded<<<4, 256>>>(c.domainSize, c.nBits, O, d.fri, d.counts,
                                              d.evalInfoPerOpening, coef, k, d.cm, d.xDivXSub, d.x,
                                              d.fixed, d.custom);
        CHECKCUDAERR(cudaDeviceSynchronize());

        SCOPED_TRACE("nOpeningPoints = " + std::to_string(O));
        expectEqual(d.download(c.domainSize), c.reference());
        cudaFree(coef);
        cudaFree(k);
    }
}
