// Equivalence tests for the DEEP polynomial kernels (starkpil/deep_expression.cuh) against deep/deep.hpp, the
// host DEEP: it inverts every x[r] - xi_o itself, so it also checks the shifted-denominator identity the kernels use.
#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <tuple>
#include <vector>
#include "cuda_utils.cuh"
#include "deep_expression.cuh"
#include "deep.hpp"

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
    std::vector<Goldilocks::Element> cm, custom, fixed, evals, xi, x, vf1, vf2;   // x: reference only

    void build(uint64_t seed)
    {
        std::mt19937_64 rng(seed);
        auto rnd = [&] { return Goldilocks::fromU64(rng() % GOLDILOCKS_PRIME); };
        const uint64_t n = domainSize(), buf = (nDistinctCols + FIELD_EXTENSION) * n;
        cm.resize(buf); custom.resize(buf); fixed.resize(buf);
        for (uint64_t i = 0; i < buf; i++) { cm[i] = rnd(); custom[i] = rnd(); fixed[i] = rnd(); }

        const Goldilocks::Element wExt = Goldilocks::w(nBitsExt);
        x.resize(n);
        x[0] = Goldilocks::shift();
        for (uint64_t r = 1; r < n; r++) x[r] = Goldilocks::mul(x[r - 1], wExt);
        if (openings.empty())
            for (uint64_t o = 0; o < nOpenings(); o++) openings.push_back((int64_t)o - 2);
        xi = {rnd(), rnd(), rnd()};

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

    // deep.hpp's host DEEP (the CPU prover's and verifier's) over the same terms: GPU against CPU.
    std::vector<Goldilocks::Element> reference() const
    {
        const uint64_t n = domainSize(), O = nOpenings();
        DeepTables t{0, 1, 2, 0, openings, {}, termStart};
        for (const FriTerm &m : terms) {
            t.terms.push_back(DeepTerm{m.evalPos, m.vf2Exp, EvMap::eType::cm, 0, 0, 0, 0, m.dim});
            t.nPols = std::max<uint32_t>(t.nPols, m.vf2Exp + 1);
        }
        std::vector<Goldilocks::Element> ch(xi);
        ch.insert(ch.end(), vf1.begin(), vf1.end());
        ch.insert(ch.end(), vf2.begin(), vf2.end());
        const DeepConsts k = deepConstants(t, ch.data(), evals.data(), nBitsExt - extendBits);
        std::vector<Goldilocks::Element> out(n * FIELD_EXTENSION);
        DeepF3Vec inv(O);
        for (uint64_t r = 0; r < n; r++) {
            deepDenominators(k, x[r], &inv[0]);
            for (uint64_t o = 0; o < O; o++) F3::inv(inv[o], inv[o]);
            auto get = [&](uint64_t j, Goldilocks::Element *v) {
                const FriTerm &m = terms[j];
                const Goldilocks::Element *pol = (m.src == 0 ? cm : m.src == 1 ? custom : fixed).data() + m.col + r;
                for (uint64_t d = 0; d < m.dim; d++) v[d] = pol[d * n];
            };
            deepEvalPoint(t, k, &inv[0], get, *(F3::Element *)&out[r * FIELD_EXTENSION]);
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

// As calculateFRIExpression launches them: the denominators at their own shape, then nThreads rows per block.
std::vector<Goldilocks::Element> runKernel(const FriCase &c, uint32_t nThreads = 256)
{
    const uint64_t n = c.domainSize(), O = c.nOpenings();
    const uint32_t denThreads = friDenominatorThreads(n);
    const Goldilocks::Element wExt = Goldilocks::w(c.nBitsExt), wDen = Goldilocks::pow(wExt, denThreads);
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
    gl64_t *b = nullptr, *a = nullptr, *k = nullptr, *fri = nullptr, *den = nullptr;
    CHECKCUDAERR(cudaMalloc(&den, n * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&b, (nPols + 1) * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&a, O * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&k, O * FIELD_EXTENSION * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc(&fri, n * FIELD_EXTENSION * sizeof(gl64_t)));

    computeFRIConstants(O, nPols, openings, Goldilocks::w(c.nBitsExt - c.extendBits).fe, termStart, terms, evals, vf1, vf2,
                        b, a, k);
    computeFRIDenominators<<<friDenominatorBlocks(n, denThreads), denThreads>>>(
        n, Goldilocks::shift().fe, wExt.fe, wDen.fe, Goldilocks::inv(wDen).fe, xi, den);
    CHECKCUDAERR(cudaGetLastError());
    computeFRIExpression<<<n / nThreads, nThreads>>>(
        n, O, c.extendBits, openings, den, g.colStart.size() - 1, dColStart, dCols, dOpStart, dOpGroups, b, a, k, cm, custom,
        fixed, fri);
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());

    std::vector<Goldilocks::Element> out(n * FIELD_EXTENSION);
    CHECKCUDAERR(cudaMemcpy(out.data(), fri, out.size() * sizeof(gl64_t), cudaMemcpyDeviceToHost));
    for (void *p : {(void *)cm, (void *)custom, (void *)fixed, (void *)evals, (void *)xi, (void *)vf1, (void *)vf2,
                    (void *)openings, (void *)termStart, (void *)terms, (void *)dCols, (void *)dColStart,
                    (void *)dOpStart, (void *)dOpGroups, (void *)den, (void *)b, (void *)a, (void *)k, (void *)fri})
        cudaFree(p);
    return out;
}

void expectMatches(const FriCase &c, uint32_t nThreads = 256)
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

// Blocks from one row to 256.
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

// More groups than one batch of S_G (FRI_GROUP_BATCH).
TEST(FriExpression, MatchesHostReferenceWithManyGroups)
{
    FriCase c;
    c.counts.resize(48);
    for (uint64_t o = 0; o < 48; o++) c.counts[o] = (o * 7 + 3) % 13;
    c.nDistinctCols = 97;   // 52 groups: two batches, the second partial
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

