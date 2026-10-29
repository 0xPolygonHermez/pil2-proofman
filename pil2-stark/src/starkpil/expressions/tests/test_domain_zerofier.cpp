// Domain zerofiers: the prover's periodic tables and the verifier's point value, against the
// definition Zi_b(x) = (x^N - 1) / prod_{h in H_b} (x - g^h), H_b the rows boundary b holds on.
#include <gtest/gtest.h>
#include <array>
#include <fstream>
#include <sstream>
#include "setup_ctx.hpp"

namespace {

const uint64_t NBITS = 6;
const uint64_t NBITS_EXT = 7;

Boundary everyRow() { Boundary b; b.name = "everyRow"; return b; }

Boundary domain(uint64_t cycleBits, std::vector<uint64_t> offsets, bool complement) {
    Boundary b;
    b.name = "domain";
    b.cycleBits = cycleBits;
    b.offsets = offsets;
    b.complement = complement;
    return b;
}

// FIRST, LAST, !FIRST, LIMITS_16, ODD, !ODD.
StarkInfo starkInfo() {
    StarkInfo si;
    si.starkStruct.nBits = NBITS;
    si.starkStruct.nBitsExt = NBITS_EXT;
    si.boundaries = {everyRow(), domain(6, {0}, false), domain(6, {63}, false), domain(6, {0}, true),
                     domain(4, {0, 15}, false), domain(1, {1}, false), domain(1, {1}, true)};
    si.ziSize = ziLayout(si.boundaries, NBITS, NBITS_EXT, si.ziOffsets, si.ziMasks);
    return si;
}

Goldilocks::Element extX(uint64_t row) {
    return Goldilocks::shift() * Goldilocks::exp(Goldilocks::w(NBITS_EXT), row);
}

Goldilocks::Element bruteForce(const Boundary &b, Goldilocks::Element x) {
    uint64_t N = 1 << NBITS;
    Goldilocks::Element zh = Goldilocks::exp(x, N) - Goldilocks::one();
    if (b.name == "everyRow") return Goldilocks::inv(zh);
    Goldilocks::Element den = Goldilocks::one();
    for (uint64_t h = 0; h < N; ++h) {
        if (inDomain(b, h, NBITS)) den = den * (x - Goldilocks::exp(Goldilocks::w(NBITS), h));
    }
    return zh * Goldilocks::inv(den);
}

// The definition over the cubic extension, for a point off the base field.
void bruteForce3(const Boundary &b, Goldilocks3::Element &z, Goldilocks3::Element &res) {
    uint64_t N = 1 << NBITS;
    Goldilocks3::Element zh = {Goldilocks::one(), Goldilocks::zero(), Goldilocks::zero()};
    for (uint64_t i = 0; i < N; ++i) Goldilocks3::mul(zh, zh, z);
    Goldilocks3::sub(zh, zh, Goldilocks::one());
    if (b.name == "everyRow") { Goldilocks3::inv(&res, &zh); return; }
    Goldilocks3::Element den = {Goldilocks::one(), Goldilocks::zero(), Goldilocks::zero()};
    for (uint64_t h = 0; h < N; ++h) {
        if (!inDomain(b, h, NBITS)) continue;
        Goldilocks3::Element f;
        Goldilocks3::sub(f, z, Goldilocks::exp(Goldilocks::w(NBITS), h));
        Goldilocks3::mul(den, den, f);
    }
    Goldilocks3::inv(&den, &den);
    Goldilocks3::mul(res, zh, den);
}

} // namespace

TEST(DomainZerofier, InDomain) {
    Boundary limits = domain(4, {0, 15}, false);
    Boundary notLimits = domain(4, {0, 15}, true);
    for (uint64_t row : {0, 15, 16, 31}) {
        EXPECT_TRUE(inDomain(limits, row, 10)) << row;
        EXPECT_FALSE(inDomain(notLimits, row, 10)) << row;
    }
    EXPECT_FALSE(inDomain(limits, 1, 10));
    EXPECT_TRUE(inDomain(notLimits, 1, 10));
    EXPECT_EQ(periodBits(limits, 10, 11), 5u);
    EXPECT_EQ(periodBits(everyRow(), 10, 11), 1u);
}

TEST(DomainZerofier, TableMatchesDefinition) {
    StarkInfo si = starkInfo();
    ProverHelpers ph(si, false);
    uint64_t NExt = 1 << NBITS_EXT;
    for (uint64_t b = 0; b < si.boundaries.size(); ++b) {
        for (uint64_t row = 0; row < NExt; ++row) {
            Goldilocks::Element got = ph.zi[si.ziOffsets[b] + (row & si.ziMasks[b])];
            ASSERT_EQ(Goldilocks::toU64(got), Goldilocks::toU64(bruteForce(si.boundaries[b], extX(row))))
                << "boundary " << b << " row " << row;
        }
    }
}

TEST(DomainZerofier, PointMatchesTable) {
    StarkInfo si = starkInfo();
    ProverHelpers table(si, false);
    Goldilocks::Element z[3] = {extX(5), Goldilocks::zero(), Goldilocks::zero()};
    ProverHelpers point(si, z);
    EXPECT_FALSE(point.zeroDenominator);
    for (uint64_t b = 0; b < si.boundaries.size(); ++b) {
        EXPECT_EQ(Goldilocks::toU64(point.zi[b * FIELD_EXTENSION]), Goldilocks::toU64(table.zi[si.ziOffsets[b] + (5 & si.ziMasks[b])])) << "boundary " << b;
        EXPECT_EQ(Goldilocks::toU64(point.zi[b * FIELD_EXTENSION + 1]), 0u) << "boundary " << b;
        EXPECT_EQ(Goldilocks::toU64(point.zi[b * FIELD_EXTENSION + 2]), 0u) << "boundary " << b;
    }
}

TEST(DomainZerofier, PointOnTheDomainIsRejected) {
    StarkInfo si = starkInfo();
    // g^3 is a root of ODD's V_O
    Goldilocks::Element z[3] = {Goldilocks::exp(Goldilocks::w(NBITS), 3), Goldilocks::zero(), Goldilocks::zero()};
    ProverHelpers point(si, z);
    EXPECT_TRUE(point.zeroDenominator);
}

TEST(DomainZerofier, PointWithHighLimbsMatchesDefinition) {
    StarkInfo si = starkInfo();
    Goldilocks::Element z[3] = {Goldilocks::fromU64(0x1234567890abcdefULL), Goldilocks::fromU64(0x0fedcba987654321ULL), Goldilocks::fromU64(42)};
    ProverHelpers point(si, z);
    EXPECT_FALSE(point.zeroDenominator);
    for (uint64_t b = 0; b < si.boundaries.size(); ++b) {
        Goldilocks3::Element want;
        bruteForce3(si.boundaries[b], *(Goldilocks3::Element *)z, want);
        for (uint64_t e = 0; e < FIELD_EXTENSION; ++e) {
            EXPECT_EQ(Goldilocks::toU64(point.zi[b * FIELD_EXTENSION + e]), Goldilocks::toU64(want[e])) << "boundary " << b << " limb " << e;
        }
    }
}

TEST(DomainZerofier, ValidateBoundaries) {
    StarkInfo si = starkInfo();
    EXPECT_EQ(validateBoundaries(si.boundaries, NBITS), "");
    EXPECT_EQ(validateBoundaries({}, NBITS), "boundary 0 must be everyRow");
    EXPECT_EQ(validateBoundaries({domain(1, {1}, false)}, NBITS), "boundary 0 must be everyRow");
    EXPECT_EQ(validateBoundaries({everyRow(), domain(4, {3, 3}, false)}, NBITS), "boundary 1: offsets must be strictly increasing");
    EXPECT_EQ(validateBoundaries({everyRow(), domain(4, {15, 0}, false)}, NBITS), "boundary 1: offsets must be strictly increasing");
    EXPECT_EQ(validateBoundaries({everyRow(), domain(4, {16}, false)}, NBITS), "boundary 1: offset 16 is outside the cycle");
    EXPECT_EQ(validateBoundaries({everyRow(), domain(7, {0}, false)}, NBITS), "boundary 1: cycle 2^7 exceeds N=2^6");
    EXPECT_EQ(validateBoundaries({everyRow(), domain(4, {}, false)}, NBITS), "boundary 1: a domain without offsets");
    Boundary frame; frame.name = "everyFrame";
    EXPECT_EQ(validateBoundaries({everyRow(), frame}, NBITS), "boundary 1: unknown boundary everyFrame");
}

// The native Rust verifier's vectors (verifier/src/domain.rs reads them): ProverHelpers(si, z) at each z.
// Writes the file DOMAIN_ZI_VECTORS names when set, else checks DOMAIN_ZI_VECTORS_FILE (set by the Makefile).
TEST(DomainZerofier, VerifierVectors) {
    StarkInfo si = starkInfo();
    json out;
    out["nBits"] = NBITS;
    for (const Boundary &b : si.boundaries) {
        json jb = {{"name", b.name}};
        if (b.name == "domain") {
            jb["cycleBits"] = b.cycleBits;
            jb["offsets"] = b.offsets;
            jb["complement"] = b.complement;
        }
        out["boundaries"].push_back(jb);
    }
    std::vector<std::array<uint64_t, 3>> points = {
        {Goldilocks::toU64(extX(5)), 0, 0},
        {0x1234567890abcdefULL, 0x0fedcba987654321ULL, 42},
        {0xffffffff00000000ULL, 0x8000000000000001ULL, 0xdeadbeefcafebabeULL},
        {Goldilocks::toU64(Goldilocks::exp(Goldilocks::w(NBITS), 3)), 0, 0},
    };
    for (const auto &p : points) {
        Goldilocks::Element z[3] = {Goldilocks::fromU64(p[0]), Goldilocks::fromU64(p[1]), Goldilocks::fromU64(p[2])};
        ProverHelpers point(si, z);
        json jp = {{"xi", {Goldilocks::toU64(z[0]), Goldilocks::toU64(z[1]), Goldilocks::toU64(z[2])}}};
        if (point.zeroDenominator) {
            jp["reject"] = true;
        } else {
            for (uint64_t b = 0; b < si.boundaries.size(); ++b) {
                jp["zi"].push_back({Goldilocks::toU64(point.zi[b * FIELD_EXTENSION]), Goldilocks::toU64(point.zi[b * FIELD_EXTENSION + 1]), Goldilocks::toU64(point.zi[b * FIELD_EXTENSION + 2])});
            }
        }
        out["points"].push_back(jp);
    }
    ASSERT_TRUE(out["points"][3].contains("reject"));
    std::string text = out.dump(1) + "\n";
    if (const char *path = std::getenv("DOMAIN_ZI_VECTORS")) {
        std::ofstream(path) << text;
        return;
    }
    std::ifstream in(DOMAIN_ZI_VECTORS_FILE);
    ASSERT_TRUE(in.good()) << "cannot read " << DOMAIN_ZI_VECTORS_FILE;
    std::stringstream checked;
    checked << in.rdbuf();
    EXPECT_EQ(checked.str(), text) << "regenerate with DOMAIN_ZI_VECTORS=" << DOMAIN_ZI_VECTORS_FILE;
}
