// Equivalence tests for the DEEP/FRI polynomial kernels (starkpil/fri_expression.cuh), against a host
// reference written straight from the definition: it inverts every x[r] - xi_o itself, so it also checks
// the shifted-denominator identity both kernels rely on.
#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <tuple>
#include <vector>
#include "cuda_utils.cuh"
#include "../../starkpil/fri_expression.cuh"

namespace {

using F3 = Goldilocks3;

// Shaped like a real instance: several openings (empty ones and partial groups of 4 included), mixed
// dim 1 / dim 3 columns shared between openings, all three source buffers, and the coset the kernels
// rely on: x[r] = s wExt^r and xi_o = xi w^o, w = wExt^(2^extendBits).
struct FriCase {
    uint64_t nBitsExt = 11, extendBits = 1;
    std::vector<uint64_t> counts{3, 0, 5, 2, 4, 1};   // terms per opening
    std::vector<int64_t> openings;                      // default -2, -1, 0, ...
    uint64_t nDistinctCols = 5;                         // the pool the terms draw columns from

    uint64_t domainSize() const { return 1ULL << nBitsExt; }
    uint64_t nOpenings() const { return counts.size(); }

    std::vector<FriTerm> terms;   // opening-major
    std::vector<uint64_t> termStart;
    std::vector<Goldilocks::Element> cm, custom, fixed, evals, xi, xis, x, vf1, vf2;   // xis: reference only

    void build(uint64_t seed)
    {
        std::mt19937_64 rng(seed);
        auto rnd = [&] { return Goldilocks::fromU64(rng() % GOLDILOCKS_PRIME); };
        const uint64_t n = domainSize(), buf = (nDistinctCols + FIELD_EXTENSION) * n;
        cm.resize(buf); custom.resize(buf); fixed.resize(buf);
        for (uint64_t i = 0; i < buf; i++) { cm[i] = rnd(); custom[i] = rnd(); fixed[i] = rnd(); }

        const Goldilocks::Element wExt = Goldilocks::w(nBitsExt), w = Goldilocks::w(nBitsExt - extendBits);
        x.resize(n);
        x[0] = Goldilocks::shift();
        for (uint64_t r = 1; r < n; r++) x[r] = Goldilocks::mul(x[r - 1], wExt);
        if (openings.empty())
            for (uint64_t o = 0; o < nOpenings(); o++) openings.push_back((int64_t)o - 2);
        xi = {rnd(), rnd(), rnd()};
        xis.resize(nOpenings() * FIELD_EXTENSION);
        for (uint64_t o = 0; o < nOpenings(); o++) {
            Goldilocks::Element wo = Goldilocks::pow(w, (uint64_t)std::abs(openings[o]));
            if (openings[o] < 0) wo = Goldilocks::inv(wo);
            for (uint64_t k = 0; k < FIELD_EXTENSION; k++) xis[o * FIELD_EXTENSION + k] = Goldilocks::mul(xi[k], wo);
        }

        vf1 = {rnd(), rnd(), rnd()};
        vf2 = {rnd(), rnd(), rnd()};

        uint64_t nTerms = 0;
        for (uint64_t c : counts) nTerms += c;
        termStart = {0};
        for (uint64_t o = 0; o < nOpenings(); o++) {
            for (uint64_t j = 0; j < counts[o]; j++) {
                const uint64_t col = (o * 3 + j * 7) % nDistinctCols;
                // evalPos reversed, so the kernels must follow it rather than the term index.
                terms.push_back(FriTerm{col * n, (uint32_t)(nTerms - 1 - terms.size()), (uint16_t)(col % 3),
                                        (uint16_t)(col % 4 == 0 ? FIELD_EXTENSION : 1)});
            }
            termStart.push_back(terms.size());
        }
        evals.resize(nTerms * FIELD_EXTENSION);
        for (auto &e : evals) e = rnd();
    }

    // fri[r] = SUM_o vf1^(O-1-o) / (x[r] - xi_o) * SUM_j vf2^(n_o-1-j) * (p_j[r] - e_j), by Horner.
    std::vector<Goldilocks::Element> reference() const
    {
        const uint64_t n = domainSize();
        std::vector<Goldilocks::Element> out(n * FIELD_EXTENSION);
        F3::Element vf1e = {vf1[0], vf1[1], vf1[2]}, vf2e = {vf2[0], vf2[1], vf2[2]};
        for (uint64_t r = 0; r < n; r++) {
            F3::Element fri;
            F3::zero(fri);
            for (uint64_t o = 0; o < nOpenings(); o++) {
                F3::Element accum;
                F3::zero(accum);
                for (uint64_t t = termStart[o]; t < termStart[o + 1]; t++) {
                    const FriTerm &m = terms[t];
                    const Goldilocks::Element *pol = (m.src == 0 ? cm : m.src == 1 ? custom : fixed).data() + m.col + r;
                    F3::Element term = {pol[0], m.dim == 1 ? Goldilocks::zero() : pol[n],
                                        m.dim == 1 ? Goldilocks::zero() : pol[2 * n]};
                    F3::sub(term, term, *(F3::Element *)&evals[m.evalPos * FIELD_EXTENSION]);
                    F3::mul(accum, accum, vf2e);
                    F3::add(accum, accum, term);
                }
                F3::Element den = {x[r], Goldilocks::zero(), Goldilocks::zero()}, inv;
                F3::sub(den, den, *(F3::Element *)&xis[o * FIELD_EXTENSION]);
                F3::inv(inv, den);
                F3::mul(accum, accum, inv);
                F3::mul(fri, fri, vf1e);
                F3::add(fri, fri, accum);
            }
            for (uint64_t k = 0; k < FIELD_EXTENSION; k++) out[r * FIELD_EXTENSION + k] = fri[k];
        }
        return out;
    }
};

template <typename T>
T *upload(const std::vector<T> &v)
{
    T *d = nullptr;
    CHECKCUDAERR(cudaMalloc(&d, v.size() * sizeof(T)));
    CHECKCUDAERR(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// Either kernel, as calculateFRIExpression launches them (256 threads).
std::vector<Goldilocks::Element> runKernel(const FriCase &c, bool shifted)
{
    const uint64_t n = c.domainSize(), O = c.nOpenings(), nThreads = 256;
    gl64_t *cm = (gl64_t *)upload(c.cm), *custom = (gl64_t *)upload(c.custom), *fixed = (gl64_t *)upload(c.fixed);
    gl64_t *evals = (gl64_t *)upload(c.evals), *xi = (gl64_t *)upload(c.xi), *x = (gl64_t *)upload(c.x);
    gl64_t *vf1 = (gl64_t *)upload(c.vf1), *vf2 = (gl64_t *)upload(c.vf2);
    int64_t *openings = upload(c.openings);
    uint64_t *termStart = upload(c.termStart);
    FriTerm *terms = c.terms.empty() ? nullptr : upload(c.terms);
    gl64_t *coef = nullptr, *k = nullptr, *fri = nullptr;
    CHECKCUDAERR(cudaMalloc(&coef, (c.terms.size() + 1) * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&k, O * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&fri, n * FIELD_EXTENSION * sizeof(gl64_t)));

    computeFRIFoldedConstants<<<(O + 63) / 64, 64>>>(O, openings, Goldilocks::w(c.nBitsExt - c.extendBits).fe,
                                                    termStart, terms, evals, vf1, vf2, coef, k);
    const int64_t oMin = *std::min_element(c.openings.begin(), c.openings.end());
    const int64_t oMax = *std::max_element(c.openings.begin(), c.openings.end());
    if (shifted) {
        const uint64_t window = friShiftedWindow(oMin, oMax, c.extendBits, nThreads);
        EXPECT_NE(window, 0u);
        computeFRIExpressionShifted<<<n / nThreads, nThreads, window * sizeof(Goldilocks3GPU::Element)>>>(
            n, c.extendBits, O, openings, oMax, window, termStart, terms, coef, k, cm, custom, fixed, xi, x, fri);
    } else {
        // Fewer blocks than rows / nThreads, so the grid-stride loop runs.
        computeFRIExpressionFolded<<<2, nThreads>>>(n, c.extendBits, O, openings, termStart, terms, coef, k, cm,
                                                   custom, fixed, xi, x, fri);
    }
    CHECKCUDAERR(cudaDeviceSynchronize());

    std::vector<Goldilocks::Element> out(n * FIELD_EXTENSION);
    CHECKCUDAERR(cudaMemcpy(out.data(), fri, out.size() * sizeof(gl64_t), cudaMemcpyDeviceToHost));
    for (void *p : {(void *)cm, (void *)custom, (void *)fixed, (void *)evals, (void *)xi, (void *)x, (void *)vf1,
                    (void *)vf2, (void *)openings, (void *)termStart, (void *)terms, (void *)coef, (void *)k, (void *)fri})
        cudaFree(p);
    return out;
}

void expectMatches(const FriCase &c, bool shifted)
{
    const auto got = runKernel(c, shifted), want = c.reference();
    ASSERT_EQ(got.size(), want.size());
    for (uint64_t i = 0; i < got.size(); i++)
        ASSERT_EQ(Goldilocks::toU64(got[i]), Goldilocks::toU64(want[i])) << "element " << i;
}

} // namespace

// Term counts (a single opening, exact and partial groups of 4, leading empties, a long vf2 chain) x
// kernel x blowup 1, 2, 4.
class FriExpressionShapes
    : public ::testing::TestWithParam<std::tuple<std::vector<uint64_t>, bool, uint64_t>> {};

TEST_P(FriExpressionShapes, MatchesHostReference)
{
    FriCase c;
    std::tie(c.counts, std::ignore, c.extendBits) = GetParam();
    c.build(0x5eed + c.nOpenings() + c.extendBits);
    expectMatches(c, std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(Shapes, FriExpressionShapes,
    ::testing::Combine(::testing::Values(std::vector<uint64_t>{7},
                                         std::vector<uint64_t>{4, 4, 4, 4},
                                         std::vector<uint64_t>{1, 1, 1, 1, 1},
                                         std::vector<uint64_t>{0, 0, 3},
                                         std::vector<uint64_t>{200, 1, 0, 37, 5, 5, 5},
                                         std::vector<uint64_t>{2, 3, 4, 5, 6, 7, 8, 9}),
                       ::testing::Bool(), ::testing::Values(0, 1, 2)));

// Openings with gaps and out of order, like the recursive compressors.
TEST(FriExpression, MatchesHostReferenceWithOpeningGaps)
{
    for (bool shifted : {false, true}) {
        FriCase c;
        c.extendBits = 2;
        c.counts = {2, 3, 1, 4, 2, 5};
        c.openings = {0, -5, -3, 1, 4, 9};
        c.build(0x6a95);
        SCOPED_TRACE(shifted ? "shifted" : "folded");
        expectMatches(c, shifted);
    }
}

// A range too wide for the shifted kernel's window: calculateFRIExpression takes the fallback.
TEST(FriExpression, FallbackMatchesWhenWindowDoesNotFit)
{
    FriCase c;
    c.counts = {2, 3, 1, 4, 2};
    c.openings = {-1000, -400, -3, 0, 5};
    c.build(0xfa11);
    ASSERT_EQ(friShiftedWindow(-1000, 5, c.extendBits, 256), 0u);
    expectMatches(c, false);
}

// Keccakf-like: openings -144..5, rows reaching around the domain.
TEST(FriExpression, MatchesHostReferenceWideOpeningRange)
{
    for (bool shifted : {false, true}) {
        FriCase c;
        c.counts.clear();
        for (int64_t o = -144; o <= 5; o++) {
            c.openings.push_back(o);
            c.counts.push_back((uint64_t)(o + 144) * 7 % 4);   // 0..3, empties included
        }
        c.nDistinctCols = 9;
        c.build(0xacc);
        SCOPED_TRACE(shifted ? "shifted" : "folded");
        expectMatches(c, shifted);
    }
}

// Up to 97 openings (examples/hashes Blake2b), where the vf1 exponent reaches its maximum.
TEST(FriExpression, MatchesHostReferenceAtLargeOpeningCount)
{
    for (uint64_t O : {17, 32, 57, 73, 87, 97}) {
        for (bool shifted : {false, true}) {
            FriCase c;
            c.counts.resize(O);
            for (uint64_t o = 0; o < O; o++) c.counts[o] = (o * 5 + 1) % 9;   // 0..8, empties included
            c.nDistinctCols = 11;
            c.build(0xba5e + O);
            SCOPED_TRACE("nOpeningPoints = " + std::to_string(O) + (shifted ? ", shifted" : ", folded"));
            expectMatches(c, shifted);
        }
    }
}

TEST(FriExpression, ShiftedWindowFitsSharedMemory)
{
    EXPECT_EQ(friShiftedWindow(-144, 5, 1, 256), 256u + 149 * 2);
    EXPECT_EQ(friShiftedWindow(0, 1792, 0, 256), 2048u);   // 2048 * 24 bytes = 48 KiB
    EXPECT_EQ(friShiftedWindow(0, 1793, 0, 256), 0u);
    EXPECT_EQ(friShiftedWindow(-1000, 0, 2, 256), 0u);
}

// ---------------------------------------------------------------------------
// Timing at the shapes of the zisk proving key. Disabled by default; run with
// --gtest_also_run_disabled_tests.
// ---------------------------------------------------------------------------
namespace {

struct BenchShape {
    const char *name;
    uint64_t nBits, nBitsExt, nOpenings, nEvals, nCols;
};

// Each column is opened at a run of consecutive openings, as in a real eval map.
void benchTerms(const BenchShape &sh, std::vector<FriTerm> &terms, std::vector<uint64_t> &termStart)
{
    const uint64_t n = 1ULL << sh.nBitsExt, base = sh.nEvals / sh.nCols, extra = sh.nEvals % sh.nCols;
    std::vector<std::vector<FriTerm>> byOpening(sh.nOpenings);
    uint64_t evalPos = 0;
    for (uint64_t c = 0; c < sh.nCols && evalPos < sh.nEvals; c++) {
        const uint64_t k = std::min(std::max<uint64_t>(base + (c < extra ? 1 : 0), 1), sh.nOpenings);
        const uint64_t start = (c * 7) % (sh.nOpenings - k + 1);
        for (uint64_t t = 0; t < k && evalPos < sh.nEvals; t++)
            byOpening[start + t].push_back(FriTerm{c * n, (uint32_t)evalPos++, 0, (uint16_t)(c % 4 == 0 ? FIELD_EXTENSION : 1)});
    }
    termStart = {0};
    for (const auto &o : byOpening) {
        terms.insert(terms.end(), o.begin(), o.end());
        termStart.push_back(terms.size());
    }
}

void runBench(const BenchShape &sh)
{
    const uint64_t n = 1ULL << sh.nBitsExt, extendBits = sh.nBitsExt - sh.nBits, nThreads = 256;
    std::vector<FriTerm> terms;
    std::vector<uint64_t> termStart;
    benchTerms(sh, terms, termStart);
    // Openings as in the keys: a run ending at +5 (Main: -1..1).
    std::vector<int64_t> ops(sh.nOpenings);
    const int64_t top = std::min<int64_t>(5, (int64_t)(sh.nOpenings - 1) / 2);
    for (uint64_t o = 0; o < sh.nOpenings; o++) ops[o] = (int64_t)o - (int64_t)(sh.nOpenings - 1) + top;

    gl64_t *pols = nullptr;
    const size_t polBytes = (size_t)(sh.nCols + FIELD_EXTENSION) * n * sizeof(gl64_t);
    if (cudaMalloc(&pols, polBytes) != cudaSuccess) {
        cudaGetLastError();
        printf("[bench] %-14s SKIPPED (needs %zu MiB for the trace)\n", sh.name, polBytes >> 20);
        return;
    }
    CHECKCUDAERR(cudaMemset(pols, 1, polBytes));
    std::mt19937_64 rng(7);
    auto random = [&](uint64_t count) {
        std::vector<uint64_t> h(count);
        for (auto &v : h) v = rng() % GOLDILOCKS_PRIME;
        return (gl64_t *)upload(h);
    };
    gl64_t *evals = random(sh.nEvals * FIELD_EXTENSION), *x = random(n), *xi = random(FIELD_EXTENSION);
    gl64_t *vf1 = random(FIELD_EXTENSION), *vf2 = random(FIELD_EXTENSION);
    gl64_t *coef = nullptr, *k = nullptr, *fri = nullptr;
    CHECKCUDAERR(cudaMalloc(&coef, sh.nEvals * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&k, sh.nOpenings * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&fri, n * FIELD_EXTENSION * sizeof(gl64_t)));
    int64_t *dOps = upload(ops);
    uint64_t *dStart = upload(termStart);
    FriTerm *dTerms = upload(terms);
    const uint64_t wTrace = Goldilocks::w(sh.nBits).fe;
    const uint64_t window = friShiftedWindow(ops.front(), ops.back(), extendBits, nThreads);

    auto timeMs = [&](auto &&launch) {
        const int reps = 5;
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
    auto constants = [&] {
        computeFRIFoldedConstants<<<(sh.nOpenings + 63) / 64, 64>>>(sh.nOpenings, dOps, wTrace, dStart, dTerms, evals,
                                                                   vf1, vf2, coef, k);
    };
    const float folded = timeMs([&] {
        constants();
        computeFRIExpressionFolded<<<n / nThreads, nThreads>>>(n, extendBits, sh.nOpenings, dOps, dStart, dTerms, coef, k,
                                                              pols, pols, pols, xi, x, fri);
    });
    const float shifted = window == 0 ? 0.f : timeMs([&] {
        constants();
        computeFRIExpressionShifted<<<n / nThreads, nThreads, window * sizeof(Goldilocks3GPU::Element)>>>(
            n, extendBits, sh.nOpenings, dOps, ops.back(), window, dStart, dTerms, coef, k, pols, pols, pols, xi, x, fri);
    });
    printf("[bench] %-14s O=%-3lu evals=%-5lu cols=%-5lu  folded %8.3f ms  shifted %8.3f ms (window %lu)\n", sh.name,
           sh.nOpenings, sh.nEvals, sh.nCols, folded, shifted, window);
    fflush(stdout);

    for (void *p : {(void *)pols, (void *)evals, (void *)x, (void *)xi, (void *)vf1, (void *)vf2, (void *)coef,
                    (void *)k, (void *)fri, (void *)dOps, (void *)dStart, (void *)dTerms})
        cudaFree(p);
}

} // namespace

TEST(FriExpression, DISABLED_BenchZiskShapes)
{
    // From the zisk proving key's starkinfo.json (openingPoints, evMap, distinct columns in the eval map).
    const BenchShape shapes[] = {
        {"Keccakf",     21, 22, 150, 3635, 520},
        {"compressor",  20, 21,  66, 1047, 388},
        {"recursive2",  19, 21,  67, 1200, 341},
        {"Main",        22, 23,   3,  194, 185},
    };
    for (const auto &sh : shapes) runBench(sh);
}
