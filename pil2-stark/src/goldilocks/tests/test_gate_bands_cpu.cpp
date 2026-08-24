// The trace expander carries its own copy of each hash permutation: it needs the mid-round
// state, which the library's `permute` does not expose. If the two drift, every recursive proof
// builds a trace whose interiors disagree with its boundary, and the failure only surfaces
// layers away as an invalid proof.
//
// These tests pin the expander's copies against the library call the witness gate makes, mode
// Auto so whichever backend this build selected is what gets checked.

#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon_goldilocks.hpp"
#include "../src/poseidon2_goldilocks.hpp"

#include "../../starkpil/gate_bands.hpp"
#include "../../starkpil/gate_bands_poseidon1.hpp"
#include "../../starkpil/gate_bands_poseidon2.hpp"

using GL = Goldilocks::Element;
static constexpr int W = 16;
static constexpr int ROUNDS = 200;

namespace {

std::vector<uint64_t> randomState(std::mt19937_64 &rng)
{
    std::vector<uint64_t> in(W);
    for (int i = 0; i < W; i++) in[i] = rng() % GOLDILOCKS_PRIME;
    return in;
}

// in[0..4] leads at 4-group slot `key`, the rest shift down. Mirrors order_by_key in
// setup/circom/poseidon_gate.cpp.
void orderByKey(GL *state, const uint64_t *in, uint64_t key)
{
    int src = 1;
    for (int g = 0; g < 4; g++) {
        const int from = (g == (int)key) ? 0 : src++;
        for (int i = 0; i < 4; i++) state[g * 4 + i] = Goldilocks::fromU64(in[from * 4 + i]);
    }
}

}  // namespace

TEST(GateBands, Poseidon1SnapshotsMatchTheGate)
{
    std::mt19937_64 rng(0xB1A11E);
    uint64_t im[gate_bands::POS_IM_GROUPS * W], out[W];

    for (int t = 0; t < ROUNDS; t++) {
        const std::vector<uint64_t> in = randomState(rng);
        for (uint64_t key = 0; key < 4; key++) {
            const bool compression = (t % 2) == 0;

            GL state[W], expected[W];
            if (compression) orderByKey(state, in.data(), key);
            else for (int i = 0; i < W; i++) state[i] = Goldilocks::fromU64(in[i]);
            PoseidonGoldilocks<W>::permute(expected, state, PoseidonMode::Auto);

            gate_bands::poseidon1::snapshots(im, out, in.data(), key, compression);

            for (int i = 0; i < W; i++) {
                ASSERT_EQ(out[i], Goldilocks::toU64(expected[i]))
                    << "Poseidon1 t=" << t << " key=" << key << " compression=" << compression
                    << " limb=" << i;
            }
            if (!compression) break;  // key is a don't-care for the sponge shape
        }
    }
}

TEST(GateBands, Poseidon2SnapshotsMatchTheGate)
{
    std::mt19937_64 rng(0xB2A22E);
    uint64_t im[gate_bands::POS_IM_GROUPS * W], out[W];

    for (int t = 0; t < ROUNDS; t++) {
        const std::vector<uint64_t> in = randomState(rng);
        for (uint64_t key = 0; key < 4; key++) {
            const bool compression = (t % 2) == 0;

            GL state[W], expected[W];
            if (compression) orderByKey(state, in.data(), key);
            else for (int i = 0; i < W; i++) state[i] = Goldilocks::fromU64(in[i]);
            Poseidon2Goldilocks<W>::permute(expected, state, Poseidon2Mode::Auto);

            gate_bands::poseidon2::snapshots(im, out, in.data(), key, compression);

            for (int i = 0; i < W; i++) {
                ASSERT_EQ(out[i], Goldilocks::toU64(expected[i]))
                    << "Poseidon2 t=" << t << " key=" << key << " compression=" << compression
                    << " limb=" << i;
            }
            if (!compression) break;
        }
    }
}

// The snapshot buffer is filled by a bare index++ with no bound, and the row layouts index
// into it. Pin the exact fill so a round-count change cannot overrun it or leave a group
// unwritten.
TEST(GateBands, SnapshotsFillEveryGroupExactly)
{
    constexpr int N = gate_bands::POS_IM_GROUPS * W;
    const std::vector<uint64_t> in(W, 7);
    uint64_t out[W];

    for (int family = 0; family < 2; family++) {
        uint64_t im[N];
        for (int i = 0; i < N; i++) im[i] = 0xDEADBEEFDEADBEEFull;
        if (family == 0) gate_bands::poseidon1::snapshots(im, out, in.data(), 0, false);
        else             gate_bands::poseidon2::snapshots(im, out, in.data(), 0, false);

        for (int i = 0; i < N; i++) {
            ASSERT_NE(im[i], 0xDEADBEEFDEADBEEFull)
                << "family=" << family << " word " << i << " was never written";
        }
        // im[5] and im[7] are 11 partial-round anchors plus 5 words of zero pad.
        for (int g : {5, 7})
            for (int i = 11; i < W; i++)
                ASSERT_EQ(im[g * W + i], 0u) << "family=" << family << " im[" << g << "][" << i << "]";
    }
}

// The band section is parsed out of a buffer whose length is not otherwise checked, so the
// reader has to refuse a truncated, mis-headered, or unknown-version one.
TEST(GateBands, BandSectionParsingRejectsMalformedBuffers)
{
    const uint64_t V = gate_bands::GATE_BAND_FORMAT_VERSION;

    // No tail past the map: a pre-bands exec file, which must read as Absent, not an error.
    std::vector<uint64_t> prefixOnly{0, 1, 5, 5};   // nAdds=0, nSMap=1, nCols=2
    auto v = gate_bands::band_section(prefixOnly.data(), 2, prefixOnly.size());
    ASSERT_EQ((int)v.status, (int)gate_bands::BandSection::Absent);
    ASSERT_EQ(v.n, 0u);

    // Version word with no count behind it.
    std::vector<uint64_t> versionOnly{0, 1, 5, 5, V};
    ASSERT_EQ((int)gate_bands::band_section(versionOnly.data(), 2, versionOnly.size()).status,
              (int)gate_bands::BandSection::Malformed);

    // Count present but the pairs are missing.
    std::vector<uint64_t> truncated{0, 1, 5, 5, V, 3, 0, 1};   // claims 3 bands, carries 1
    ASSERT_EQ((int)gate_bands::band_section(truncated.data(), 2, truncated.size()).status,
              (int)gate_bands::BandSection::Malformed);

    // Header whose map cannot fit the buffer.
    std::vector<uint64_t> lying{0, 1000, 5, 5, V, 1, 0, 1};
    ASSERT_EQ((int)gate_bands::band_section(lying.data(), 2, lying.size()).status,
              (int)gate_bands::BandSection::Absent);

    // A section from a newer setup is refused, not misparsed.
    std::vector<uint64_t> newer{0, 1, 5, 5, V + 1, 1, 0, 1};
    auto nv = gate_bands::band_section(newer.data(), 2, newer.size());
    ASSERT_EQ((int)nv.status, (int)gate_bands::BandSection::UnsupportedVersion);
    ASSERT_EQ(nv.version, V + 1);

    // A well-formed section reads back exactly.
    std::vector<uint64_t> good{0, 1, 5, 5, V, 2, 0, GB_POSEIDON1_COMPRESSOR_SPONGE,
                               10, GB_POSEIDON2_AGGREGATION_COMPRESSION};
    v = gate_bands::band_section(good.data(), 2, good.size());
    ASSERT_EQ((int)v.status, (int)gate_bands::BandSection::Ok);
    ASSERT_EQ(v.n, 2u);
    ASSERT_EQ(v.bands[0], 0u);
    ASSERT_EQ(v.bands[3], (uint64_t)GB_POSEIDON2_AGGREGATION_COMPRESSION);

    // Kinds this build does not know are rejected rather than expanded with a guess, and so
    // is a band that would run off the end of the trace.
    ASSERT_EQ(gate_bands::first_bad_band(v.bands, v.n, 64), v.n);
    std::vector<uint64_t> unknownKind{0, 99};
    ASSERT_EQ(gate_bands::first_bad_band(unknownKind.data(), 1, 64), 0u);
    std::vector<uint64_t> pastEnd{60, GB_POSEIDON1_COMPRESSOR_SPONGE};  // 10 rows from row 60
    ASSERT_EQ(gate_bands::first_bad_band(pastEnd.data(), 1, 64), 0u);
    ASSERT_EQ(gate_bands::first_bad_band(pastEnd.data(), 1, 70), 1u);
}
