// Equivalence tests for the DEEP/FRI polynomial kernels (starkpil/fri_expression.cuh), against a host
// reference written straight from the definition: it inverts every x[r] - xi_o itself, so it also checks
// the shifted-denominator identity the kernels rely on.
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
    std::vector<Goldilocks::Element> cm, custom, fixed, evals, xi, xis, x, vf1, vf2;   // xis, x: reference only

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
                                        (uint16_t)(col % 4 == 0 ? FIELD_EXTENSION : 1), (uint32_t)col});
            }
            termStart.push_back(terms.size());
        }
        evals.resize(nTerms * FIELD_EXTENSION);
        for (auto &e : evals) e = rnd();
    }

    // fri[r] = SUM_o vf1^e_o / (x[r] - xi_o) * SUM_j vf2^(c_j) * (p_j[r] - e_j), e_o counting later openings with terms.
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
                    F3::Element p = {Goldilocks::one(), Goldilocks::zero(), Goldilocks::zero()};
                    for (uint32_t e = 0; e < m.vf2Exp; e++) F3::mul(p, p, vf2e);
                    F3::mul(term, term, p);
                    F3::add(accum, accum, term);
                }
                F3::Element den = {x[r], Goldilocks::zero(), Goldilocks::zero()}, inv;
                F3::sub(den, den, *(F3::Element *)&xis[o * FIELD_EXTENSION]);
                F3::inv(inv, den);
                F3::mul(accum, accum, inv);
                if (termStart[o + 1] == termStart[o]) continue;   // fri_poly.rs skips openings without evaluations
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

// As calculateFRIExpression launches them, nThreads rows per block (friThreads unless given).
std::vector<Goldilocks::Element> runKernel(const FriCase &c, uint32_t nThreads = 0)
{
    const uint64_t n = c.domainSize(), O = c.nOpenings();
    FriWindow w = friWindow(c.openings, c.extendBits, nThreads == 0 ? n : nThreads);
    EXPECT_NE(w.nThreads, 0u);
    const Goldilocks::Element wExt = Goldilocks::w(c.nBitsExt), wStep = Goldilocks::pow(wExt, w.nThreads);
    for (FriSegment &s : w.segments) s.w = Goldilocks::pow(wExt, (uint64_t)s.offset & (n - 1)).fe;
    uint64_t nPols = 0;
    for (const FriTerm &t : c.terms) nPols = std::max<uint64_t>(nPols, t.vf2Exp + 1);
    const FriColGroups g = buildFriColGroups(c.terms, c.termStart, nPols);

    gl64_t *cm = (gl64_t *)upload(c.cm), *custom = (gl64_t *)upload(c.custom), *fixed = (gl64_t *)upload(c.fixed);
    gl64_t *evals = (gl64_t *)upload(c.evals), *xi = (gl64_t *)upload(c.xi);
    gl64_t *vf1 = (gl64_t *)upload(c.vf1), *vf2 = (gl64_t *)upload(c.vf2);
    int64_t *openings = upload(c.openings);
    uint64_t *termStart = upload(c.termStart);
    FriTerm *terms = c.terms.empty() ? nullptr : upload(c.terms);
    FriTerm *dCols = g.cols.empty() ? nullptr : upload(g.cols);
    uint32_t *dColStart = upload(g.colStart), *dOpStart = upload(g.opStart);
    uint32_t *dOpGroups = g.opGroups.empty() ? nullptr : upload(g.opGroups);
    FriSegment *dSegs = upload(w.segments);
    uint32_t *dOpBase = upload(w.opBase);
    gl64_t *b = nullptr, *a = nullptr, *k = nullptr, *fri = nullptr;
    CHECKCUDAERR(cudaMalloc(&b, (nPols + 1) * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&a, O * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&k, O * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&fri, n * FIELD_EXTENSION * sizeof(gl64_t)));

    computeFRIConstants(O, nPols, openings, Goldilocks::w(c.nBitsExt - c.extendBits).fe, termStart, terms, evals, vf1, vf2,
                        b, a, k);
    computeFRIExpression<<<n / w.nThreads, w.nThreads, friSharedBytes(w.size)>>>(
        n, O, w.size, w.segments.size(), dSegs, dOpBase, Goldilocks::shift().fe, wExt.fe, wStep.fe, Goldilocks::inv(wStep).fe, xi,
        g.colStart.size() - 1, dColStart, dCols, dOpStart, dOpGroups, b, a, k, cm, custom, fixed, fri);
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());

    std::vector<Goldilocks::Element> out(n * FIELD_EXTENSION);
    CHECKCUDAERR(cudaMemcpy(out.data(), fri, out.size() * sizeof(gl64_t), cudaMemcpyDeviceToHost));
    for (void *p : {(void *)cm, (void *)custom, (void *)fixed, (void *)evals, (void *)xi, (void *)vf1, (void *)vf2,
                    (void *)openings, (void *)termStart, (void *)terms, (void *)dCols, (void *)dColStart,
                    (void *)dOpStart, (void *)dOpGroups, (void *)dSegs, (void *)dOpBase, (void *)b, (void *)a, (void *)k, (void *)fri})
        cudaFree(p);
    return out;
}

void expectMatches(const FriCase &c, uint32_t nThreads = 0)
{
    const auto got = runKernel(c, nThreads), want = c.reference();
    ASSERT_EQ(got.size(), want.size());
    for (uint64_t i = 0; i < got.size(); i++)
        ASSERT_EQ(Goldilocks::toU64(got[i]), Goldilocks::toU64(want[i])) << "element " << i;
}

} // namespace

// Term counts (a single opening, empty openings, a long vf2 chain) x blowup 1, 2, 4.
class FriExpressionShapes : public ::testing::TestWithParam<std::tuple<std::vector<uint64_t>, uint64_t>> {};

TEST_P(FriExpressionShapes, MatchesHostReference)
{
    FriCase c;
    std::tie(c.counts, c.extendBits) = GetParam();
    c.build(0x5eed + c.nOpenings() + c.extendBits);
    expectMatches(c);
}

INSTANTIATE_TEST_SUITE_P(Shapes, FriExpressionShapes,
    ::testing::Combine(::testing::Values(std::vector<uint64_t>{7},
                                         std::vector<uint64_t>{4, 4, 4, 4},
                                         std::vector<uint64_t>{1, 1, 1, 1, 1},
                                         std::vector<uint64_t>{0, 0, 3},
                                         std::vector<uint64_t>{200, 1, 0, 37, 5, 5, 5},
                                         std::vector<uint64_t>{2, 3, 4, 5, 6, 7, 8, 9}),
                       ::testing::Values(0, 1, 2)));

// Blocks from one row (a window of D per thread) to 256.
TEST(FriExpression, MatchesHostReferenceAtAnyBlockSize)
{
    for (uint32_t nThreads : {1, 32, 256}) {
        FriCase c;
        c.build(0xd0 + nThreads);
        SCOPED_TRACE("nThreads = " + std::to_string(nThreads));
        expectMatches(c, nThreads);
    }
}

// Openings with gaps, out of order and far apart, like the recursive compressors.
TEST(FriExpression, MatchesHostReferenceWithOpeningGaps)
{
    for (auto ops : {std::vector<int64_t>{0, -5, -3, 1, 4, 9}, std::vector<int64_t>{-1000, -400, -3, 0, 5, 1},
                     std::vector<int64_t>{100000, -3, 0, -100000, 1, 7}}) {
        FriCase c;
        c.extendBits = 2;
        c.counts = {2, 3, 1, 4, 2, 5};
        c.openings = ops;
        c.build(0x6a95 + ops[0]);
        expectMatches(c);
    }
}

// Keccakf-like: openings -144..5, rows reaching around the domain.
TEST(FriExpression, MatchesHostReferenceWideOpeningRange)
{
    FriCase c;
    c.counts.clear();
    for (int64_t o = -144; o <= 5; o++) {
        c.openings.push_back(o);
        c.counts.push_back((uint64_t)(o + 144) * 7 % 4);   // 0..3, empties included
    }
    c.nDistinctCols = 9;
    c.build(0xacc);
    expectMatches(c);
}

// Contiguous openings take one segment; far apart ones a segment each, so the range does not matter.
TEST(FriExpression, WindowSegments)
{
    std::vector<int64_t> keccak;
    for (int64_t o = -134; o <= 5; o++) keccak.push_back(o);
    FriWindow w = friWindow(keccak, 1, 1 << 22);
    EXPECT_EQ(w.nThreads, 256u);
    EXPECT_EQ(w.segments.size(), 1u);
    EXPECT_EQ(w.size, 256u + 139 * 2);
    EXPECT_EQ(w.opBase[0], 139u * 2);   // opening -134 reads the latest rows
    EXPECT_EQ(friWindow({0, 1792}, 0, 1 << 22).nThreads, 256u);
    w = friWindow({-100000, 0, 1, 100000}, 2, 1 << 22);
    EXPECT_EQ(w.nThreads, 256u);
    EXPECT_EQ(w.segments.size(), 3u);
    EXPECT_EQ(w.size, 3 * 256u + 4);
    std::vector<int64_t> sparse;
    for (int64_t o = 0; o < 64; o++) sparse.push_back(o * 1000);
    EXPECT_EQ(friWindow(sparse, 1, 1 << 22).nThreads, 16u);   // 64 * 32 rows * 24 bytes = 48 KiB, plus the block's x
    sparse.pop_back();
    EXPECT_EQ(friWindow(sparse, 1, 1 << 22).nThreads, 32u);
}

// A window right at the shared-memory limit launches.
TEST(FriExpression, MatchesHostReferenceAtTheSharedMemoryLimit)
{
    FriCase c;
    c.counts.clear();
    for (int64_t o = 0; o < 63; o++) {
        c.openings.push_back(o * 1000);
        c.counts.push_back(1 + o % 3);
    }
    c.build(0x5a3);
    const FriWindow w = friWindow(c.openings, c.extendBits, c.domainSize());
    ASSERT_EQ(w.nThreads, 32u);
    ASSERT_GT(friSharedBytes(w.size), 47u * 1024);
    expectMatches(c);
}

// More groups than one batch of S_G (FRI_GROUP_BATCH).
TEST(FriExpression, MatchesHostReferenceWithManyGroups)
{
    FriCase c;
    c.counts.resize(24);
    for (uint64_t o = 0; o < 24; o++) c.counts[o] = (o * 7 + 3) % 13;
    c.nDistinctCols = 97;
    c.build(0x9a0);
    uint64_t nPols = 0;
    for (const FriTerm &t : c.terms) nPols = std::max<uint64_t>(nPols, t.vf2Exp + 1);
    ASSERT_GT(buildFriColGroups(c.terms, c.termStart, nPols).colStart.size() - 1, FRI_GROUP_BATCH + 1u);
    expectMatches(c);
}

// Up to 97 openings (examples/hashes Blake2b), where the vf1 exponent reaches its maximum.
TEST(FriExpression, MatchesHostReferenceAtLargeOpeningCount)
{
    for (uint64_t O : {17, 32, 57, 73, 87, 97}) {
        FriCase c;
        c.counts.resize(O);
        for (uint64_t o = 0; o < O; o++) c.counts[o] = (o * 5 + 1) % 9;   // 0..8, empties included
        c.nDistinctCols = 11;
        c.build(0xba5e + O);
        SCOPED_TRACE("nOpeningPoints = " + std::to_string(O));
        expectMatches(c);
    }
}

