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
#include "../../starkpil/gate_bands_blake3.hpp"
#include "../../starkpil/gate_bands_cpu.hpp"

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
// reader has to refuse a truncated, mis-headered, or unknown-version one. Where the section
// starts now comes from the exec header, so a header this build cannot read has to be refused
// too rather than guessed at.
TEST(GateBands, BandSectionParsingRejectsMalformedBuffers)
{
    const uint64_t V = gate_bands::GATE_BAND_FORMAT_VERSION;
    const uint64_t H = exec_layout::EXEC_MAGIC | exec_layout::EXEC_FORMAT_VERSION;
    // nAdds=0, a 1x2 map -> one packed word, so the section starts at word 5.
    const uint64_t M = 0x0000000500000004ull;   // two u32 entries, arbitrary

    // No tail past the map: nothing to expand, which must read as Absent rather than an error.
    std::vector<uint64_t> prefixOnly{H, 0, 1, 2, M};
    auto v = gate_bands::band_section(prefixOnly.data(), prefixOnly.size());
    ASSERT_EQ((int)v.status, (int)gate_bands::BandSection::Absent);
    ASSERT_EQ(v.n, 0u);

    // Version word with no count behind it.
    std::vector<uint64_t> versionOnly{H, 0, 1, 2, M, V};
    ASSERT_EQ((int)gate_bands::band_section(versionOnly.data(), versionOnly.size()).status,
              (int)gate_bands::BandSection::Malformed);

    // Count present but the pairs are missing.
    std::vector<uint64_t> truncated{H, 0, 1, 2, M, V, 3, 0, 1};   // claims 3 bands, carries 1
    ASSERT_EQ((int)gate_bands::band_section(truncated.data(), truncated.size()).status,
              (int)gate_bands::BandSection::Malformed);

    // Header whose map cannot fit the buffer.
    std::vector<uint64_t> lying{H, 0, 1000, 2, M, V, 1, 0, 1};
    ASSERT_EQ((int)gate_bands::band_section(lying.data(), lying.size()).status,
              (int)gate_bands::BandSection::Absent);

    // A buffer with no exec header at all -- the pre-magic layout -- is refused, not parsed
    // with the offsets this build happens to use.
    std::vector<uint64_t> headerless{0, 1, 5, 5, V, 1, 0, 1};
    ASSERT_EQ((int)gate_bands::band_section(headerless.data(), headerless.size()).status,
              (int)gate_bands::BandSection::Malformed);

    // An exec file from a newer setup: the section's offset is unknown, so it is refused as an
    // exec-format problem rather than misread at this build's offset.
    const uint64_t newerExec = exec_layout::EXEC_MAGIC | (exec_layout::EXEC_FORMAT_VERSION + 1);
    std::vector<uint64_t> newerFile{newerExec, 0, 1, 2, M, V, 1, 0, 1};
    auto ev = gate_bands::band_section(newerFile.data(), newerFile.size());
    ASSERT_EQ((int)ev.status, (int)gate_bands::BandSection::UnsupportedExecFormat);
    ASSERT_EQ(ev.version, exec_layout::EXEC_FORMAT_VERSION + 1);

    // A section from a newer setup is refused, not misparsed.
    std::vector<uint64_t> newer{H, 0, 1, 2, M, V + 1, 1, 0, 1};
    auto nv = gate_bands::band_section(newer.data(), newer.size());
    ASSERT_EQ((int)nv.status, (int)gate_bands::BandSection::UnsupportedVersion);
    ASSERT_EQ(nv.version, V + 1);

    // Dimensions whose own offsets overflow describe no real buffer. That is a corrupt header,
    // not a version this build refuses -- only one of the two is worth regenerating a key over.
    std::vector<uint64_t> overflowing{H, 0, UINT64_MAX, 2, M, V, 1, 0, 1};
    auto qv = gate_bands::band_section(overflowing.data(), overflowing.size());
    ASSERT_EQ((int)qv.status, (int)gate_bands::BandSection::Malformed);

    // An odd entry count leaves half a word unused, and the section still starts on the next
    // whole word. Losing that round-up reads the map's last word as the section's version.
    std::vector<uint64_t> oddMap{H, 0, 1, 1, 7, V, 1, 0, 4, GB_POSEIDON1_AGGREGATION_SPONGE, 0};
    auto ov = gate_bands::band_section(oddMap.data(), oddMap.size());
    ASSERT_EQ((int)ov.status, (int)gate_bands::BandSection::Ok);
    ASSERT_EQ(ov.n, 1u);
    ASSERT_EQ(ov.bands[0], 4u);

    // A count the buffer cannot hold is refused rather than read off the end. Three words per band
    // since format version 2, so two bands need six -- this gives five.
    std::vector<uint64_t> shortSection{H, 0, 1, 2, M, V, 2, 0, 0, GB_POSEIDON1_COMPRESSOR_SPONGE, 0, 10};
    ASSERT_EQ((int)gate_bands::band_section(shortSection.data(), shortSection.size()).status,
              (int)gate_bands::BandSection::Malformed);

    // A well-formed section reads back exactly: (row, kind, payload) per band. The payload carries
    // a per-block constant the expander cannot get off the witness trace -- BLAKE3's `flags`.
    std::vector<uint64_t> good{H, 0, 1, 2, M, V, 2, 4, 0, GB_POSEIDON1_COMPRESSOR_SPONGE, 0,
                               10, GB_POSEIDON2_AGGREGATION_COMPRESSION, 0xB3};
    v = gate_bands::band_section(good.data(), good.size());
    ASSERT_EQ((int)v.status, (int)gate_bands::BandSection::Ok);
    ASSERT_EQ(v.n, 2u);
    ASSERT_EQ(v.aux, 4u) << "the per-air aux word -- BLAKE3's LANES";
    ASSERT_EQ(v.bands[0], 0u);
    ASSERT_EQ(v.bands[2], 0u) << "payload of the first band";
    // The second band's KIND, not just its row: reading kind at the wrong stride is a real slip
    // and every assertion that only checks rows passes straight through it.
    ASSERT_EQ(v.bands[3], 10u);
    ASSERT_EQ(v.bands[4], (uint64_t)GB_POSEIDON2_AGGREGATION_COMPRESSION);
    ASSERT_EQ(v.bands[5], 0xB3u) << "payload of the second band";

    // first_bad_band reports the offending band's row AND kind, both at the triple stride.
    std::vector<uint64_t> beyond{H, 0, 1, 2, M, V, 1, 0, 9000, GB_POSEIDON2_AGGREGATION_SPONGE, 0};
    auto pv = gate_bands::band_section(beyond.data(), beyond.size());
    ASSERT_EQ((int)pv.status, (int)gate_bands::BandSection::Ok);
    ASSERT_EQ(gate_bands::first_bad_band(pv.bands, pv.n, 100), 0u) << "a band past the trace";

    // Kinds this build does not know are rejected rather than expanded with a guess, and so
    // is a band that would run off the end of the trace.
    ASSERT_EQ(gate_bands::first_bad_band(v.bands, v.n, 64), v.n);
    std::vector<uint64_t> unknownKind{0, 99};
    ASSERT_EQ(gate_bands::first_bad_band(unknownKind.data(), 1, 64), 0u);
    std::vector<uint64_t> pastEnd{60, GB_POSEIDON1_COMPRESSOR_SPONGE};  // 10 rows from row 60
    ASSERT_EQ(gate_bands::first_bad_band(pastEnd.data(), 1, 64), 0u);
    ASSERT_EQ(gate_bands::first_bad_band(pastEnd.data(), 1, 70), 1u);
}

// ─── BLAKE3 ──────────────────────────────────────────────────────────────────

/// The expander has to agree with the air's column order, and that order is whatever the compiler
/// generates -- not what the PIL looks like it says. These offsets come from the compiled
/// Compressor.starkinfo.json's cmPolsMap, which is the only authority on it. Both the per-lane width
/// and the position of the boundary group move whenever the air adds or folds away a column, and
/// neither shows up as a compile error, so pinning the offsets is the point.
TEST(GateBandsBlake3, LayoutMatchesTheGeneratedTrace)
{
    const auto L = gate_bands::blake3::layout(4);
    // The boundary columns come FIRST, right after the band: the air declares them before it calls
    // blake3Lanes, because that call binds them and PIL2 wants an argument declared before it is
    // passed. Reading them from the tail instead put every interior cell one group off.
    EXPECT_EQ(L.dinv, 18u);      EXPECT_EQ(L.vbTopHi, 22u);
    EXPECT_EQ(L.outBytes, 34u);
    EXPECT_EQ(L.va, 50u);        EXPECT_EQ(L.vb, 58u);
    EXPECT_EQ(L.vd, 74u);
    EXPECT_EQ(L.x, 90u);         EXPECT_EQ(L.y, 98u);
    EXPECT_EQ(L.va_p, 106u);     EXPECT_EQ(L.vd_p, 122u);
    EXPECT_EQ(L.vc_p, 138u);     EXPECT_EQ(L.vb_p_s, 154u);
    EXPECT_EQ(L.va_pp, 186u);    EXPECT_EQ(L.vd_pp, 202u);
    EXPECT_EQ(L.vc_pp, 218u);    EXPECT_EQ(L.vb_pp_xor, 234u);
    EXPECT_EQ(L.vb_pp_t, 250u);
    EXPECT_EQ(L.mul_table, 254u); EXPECT_EQ(L.mul_range, 255u);
    EXPECT_EQ(gate_bands::blake3::stage1_cols(4), 256u);

    // and it must scale, since LANES is an air parameter
    EXPECT_EQ(gate_bands::blake3::stage1_cols(1), 79u);
    EXPECT_EQ(gate_bands::blake3::stage1_cols(8), 492u);
}

/// The reference permutation is written independently of setup/circom/blake3_gate.cpp so the two
/// can be differenced. Held here to BLAKE3's own published vector rather than to that gate: an
/// agreement between two copies of the same mistake proves nothing.
TEST(GateBandsBlake3, ReferenceMatchesTheSpecVector)
{
    namespace b3 = gate_bands::blake3;
    b3::BlockInputs in{};
    for (int i = 0; i < 8; i++) in.cv[i] = b3::iv(i);
    in.blockLen = 0;
    in.counterLo = 0;
    in.flags = 1 | 2 | 8;  // CHUNK_START | CHUNK_END | ROOT

    uint32_t v[16];
    in.initial_state(v);
    for (int r = 0; r < b3::ROUNDS; r++) {
        for (int g = 0; g < b3::G_PER_ROUND; g++) {
            const int ia = b3::g_idx(g, 0), ib = b3::g_idx(g, 1);
            const int ic = b3::g_idx(g, 2), id = b3::g_idx(g, 3);
            const b3::GTrace t = b3::g_step(v[ia], v[ib], v[ic], v[id],
                                            in.block[b3::sigma(r, 2 * g)],
                                            in.block[b3::sigma(r, 2 * g + 1)]);
            v[ia] = t.a2; v[ib] = t.b2; v[ic] = t.c2; v[id] = t.d2;
        }
    }
    // BLAKE3("") = af1349b9f5f9a1a6 a0404dea36dcc949 9bcb25c9adc112b7 cc9a93cae41f3262
    const uint32_t expected[8] = {0xb94913af, 0xa6a1f9f5, 0xea4d40a0, 0x49c9dc36,
                                  0xc925cb9b, 0xb712c1ad, 0xca939acc, 0x62321fe4};
    for (int i = 0; i < 8; i++) EXPECT_EQ(v[i] ^ v[i + 8], expected[i]) << "digest word " << i;
}

/// table_row and table_out have to describe the SAME table blake3Tables builds, or the
/// multiplicities land on the wrong rows -- which no per-air check would catch, only the global
/// grand-sum constraint.
TEST(GateBandsBlake3, TableRowAndOutputAgreeWithTheAirsTable)
{
    namespace b3 = gate_bands::blake3;
    using b3::table_out;
    using b3::table_row;
    // A cycles fastest, then B, then ROTATION -- so a rot-12 row sits a whole 2^16 above rot 0.
    EXPECT_EQ(table_row(0, 0, 0), 0u);
    EXPECT_EQ(table_row(0xFF, 0, 0), 0xFFu);
    EXPECT_EQ(table_row(0, 1, 0), 0x100u);
    EXPECT_EQ(table_row(0, 0, 12), 1u << 16);
    EXPECT_LT(table_row(0xFF, 0xFF, 12), b3::TABLE_SIZE);

    // rot 0 is a plain byte XOR with nothing in the second output
    for (uint32_t a = 0; a < 256; a += 37) {
        for (uint32_t b = 0; b < 256; b += 41) {
            uint8_t c0, c1;
            table_out((uint8_t)a, (uint8_t)b, 0, c0, c1);
            EXPECT_EQ(c0, (uint8_t)(a ^ b));
            EXPECT_EQ(c1, 0);
        }
    }

    // and rot 12 splits the rotated byte across two output slots, which recombine to ROTR12
    for (uint32_t a = 0; a < 256; a += 53) {
        uint8_t c0, c1;
        table_out((uint8_t)a, 0, 12, c0, c1);
        const uint32_t rotated = (a >> 12) | (a << 20);
        EXPECT_EQ(c0, (uint8_t)((rotated >> 16) & 0xFF));
        EXPECT_EQ(c1, (uint8_t)((rotated >> 24) & 0xFF));
    }
}

/// The round trip that matters: place a Blake3Node's boundary, expand the block, and read the
/// digest back out of the feedforward columns. Held against a BLAKE3 computed straight from the
/// same inputs -- if the expander and the air's layout disagree by one cell, this is what says so.
TEST(GateBandsBlake3, ExpandedBlockReproducesTheDigest)
{
    namespace b3 = gate_bands::blake3;
    constexpr uint64_t LANES = 2;
    const uint64_t nCols = b3::stage1_cols(LANES);
    std::vector<Goldilocks::Element> trace(b3::CLOCKS * nCols, Goldilocks::zero());

    // Two lanes with different inputs, so a lane-indexing slip cannot pass.
    uint64_t input[LANES][8];
    for (uint64_t l = 0; l < LANES; l++) {
        for (int i = 0; i < 8; i++) {
            input[l][i] = 0x9E3779B97F4A7C15ull * (uint64_t)(l * 8 + i + 1) % 0xFFFFFFFF00000001ull;
            trace[(0 + l) * nCols + i] = Goldilocks::fromU64(input[l][i]);
        }
        trace[(0 + l) * nCols + 8] = Goldilocks::zero();  // key = 0
    }

    b3::Multiplicities mul;
    b3::HostSink sink{mul};
    b3::expand_block(trace.data(), nCols, 0, LANES, b3::Kind::Node, 11, sink);

    for (uint64_t l = 0; l < LANES; l++) {
        // what BLAKE3 says, from the same eight words
        b3::BlockInputs in{};
        for (int i = 0; i < 8; i++) in.cv[i] = b3::iv(i);
        for (int i = 0; i < 8; i++) {
            in.block[2 * i] = (uint32_t)(input[l][i] & 0xFFFFFFFFull);
            in.block[2 * i + 1] = (uint32_t)(input[l][i] >> 32);
        }
        in.blockLen = 64;
        in.counterLo = 0;
        in.flags = 11;

        uint32_t v[16];
        in.initial_state(v);
        for (int r = 0; r < b3::ROUNDS; r++) {
            for (int g = 0; g < b3::G_PER_ROUND; g++) {
                const int ia = b3::g_idx(g, 0), ib = b3::g_idx(g, 1);
                const int ic = b3::g_idx(g, 2), id = b3::g_idx(g, 3);
                const b3::GTrace t = b3::g_step(v[ia], v[ib], v[ic], v[id],
                                                in.block[b3::sigma(r, 2 * g)],
                                                in.block[b3::sigma(r, 2 * g + 1)]);
                v[ia] = t.a2; v[ib] = t.b2; v[ic] = t.c2; v[id] = t.d2;
            }
        }

        // and what the expanded trace holds: out[i] at clock 40+i, four bytes wide
        const auto L = b3::layout(LANES);
        for (int i = 0; i < 8; i++) {
            uint32_t got = 0;
            for (int b = 0; b < 4; b++) {
                const uint64_t row = (uint64_t)(b3::CLOCKS - 16 + i);
                got |= (uint32_t)Goldilocks::toU64(trace[row * nCols + L.outBytes + l * 4 + b]) << (8 * b);
            }
            EXPECT_EQ(got, v[i] ^ v[i + 8]) << "lane " << l << " out[" << i << "]";
        }
    }

    // Every lookup the block makes must have been counted: 16 XOR per lane per row, 4 range checks
    // per lane per row for x/y, 8 folded va limbs per lane, and 4 feedforward XOR per lane per
    // output row. A miscount does not fail per-air verification -- only the global constraint.
    uint64_t tableTotal = 0, rangeTotal = 0;
    for (uint64_t c : mul.table) tableTotal += c;
    for (uint64_t c : mul.range) rangeTotal += c;
    EXPECT_EQ(tableTotal, LANES * (16 * b3::CLOCKS + 4 * 16)) << "XOR table lookups";
    EXPECT_EQ(rangeTotal, LANES * 6 * b3::CLOCKS) << "range lookups: va, x and y, two limbs each, every row";
}

/// expand_gate_bands must report the offending band's row AND kind, both read at the triple
/// stride. Only the row was covered before, and a kind read at the old stride of 2 passed every
/// assertion there was -- it would have shipped a diagnostic naming the wrong gate shape.
TEST(GateBands, UnexpandableBandReportsBothRowAndKind)
{
    const uint64_t H = exec_layout::EXEC_MAGIC | exec_layout::EXEC_FORMAT_VERSION;
    const uint64_t V = gate_bands::GATE_BAND_FORMAT_VERSION;
    // header, no adds, a 1x2 map (one word), then: version, count, aux, and two bands. The second
    // sits past the trace, so it is the one reported.
    std::vector<uint64_t> exec{H, 0, 1, 2, 0x00000002'00000001ull, V, 2, 0,
                               0,   (uint64_t)GB_POSEIDON1_COMPRESSOR_SPONGE,      0,
                               900, (uint64_t)GB_POSEIDON2_AGGREGATION_COMPRESSION, 0};

    std::vector<Goldilocks::Element> trace(64 * 48, Goldilocks::zero());
    auto res = gate_bands::expand_gate_bands(trace.data(), exec.data(), 48, exec.size(), 64);

    ASSERT_EQ((int)res.status, (int)gate_bands::ExpandStatus::UnexpandableBand);
    EXPECT_EQ(res.badBand, 1u);
    EXPECT_EQ(res.row, 900u);
    EXPECT_EQ(res.kind, (uint64_t)GB_POSEIDON2_AGGREGATION_COMPRESSION) << "kind at the triple stride";
}

/// The whole CPU path for a BLAKE3 air: exec buffer in, expanded trace and multiplicities out.
/// LANES arrives in the section's aux word rather than being derived from the column count -- it is
/// a setup parameter, and that arithmetic would mean nothing for an air of another family.
TEST(GateBandsBlake3, ExpandGateBandsFillsInteriorsAndMultiplicities)
{
    namespace b3 = gate_bands::blake3;
    constexpr uint64_t LANES = 2, BLOCKS = 3;
    const uint64_t nCols = b3::stage1_cols(LANES);
    const uint64_t nRows = 1u << 18;  // must hold the 2^17 table rows the multiplicities land on

    const uint64_t H = exec_layout::EXEC_MAGIC | exec_layout::EXEC_FORMAT_VERSION;
    const uint64_t V = gate_bands::GATE_BAND_FORMAT_VERSION;
    // one 1x2 map word, then version, count, aux=LANES, then three bands of the three kinds
    std::vector<uint64_t> exec{H, 0, 1, 2, 0x00000002'00000001ull, V, BLOCKS, LANES};
    const uint64_t kinds[BLOCKS] = {GB_BLAKE3_NODE, GB_BLAKE3_COMPRESS_CHUNK, GB_BLAKE3_COMPRESS_PARENT};
    for (uint64_t k = 0; k < BLOCKS; k++) {
        exec.push_back(k * b3::CLOCKS);
        exec.push_back(kinds[k]);
        exec.push_back(11);  // flags
    }

    std::vector<Goldilocks::Element> trace(nRows * nCols, Goldilocks::zero());
    // Boundary cells for every lane of every block, distinct so an indexing slip cannot pass.
    for (uint64_t k = 0; k < BLOCKS; k++) {
        for (uint64_t l = 0; l < LANES; l++) {
            const uint64_t row = k * b3::CLOCKS + l;
            for (int j = 0; j < 18; j++) {
                trace[row * nCols + j] =
                    Goldilocks::fromU64(0x9E3779B97F4A7C15ull * (k * 32 + l * 18 + j + 1) %
                                        0xFFFFFFFF00000001ull);
            }
        }
    }

    auto res = gate_bands::expand_gate_bands(trace.data(), exec.data(), nCols, exec.size(), nRows);
    ASSERT_EQ((int)res.status, (int)gate_bands::ExpandStatus::Ok);
    EXPECT_EQ(res.nBands, BLOCKS);

    // Interiors are filled: the permutation columns of the last row of each block cannot all be
    // zero unless nothing ran there.
    const auto L = b3::layout(LANES);
    for (uint64_t k = 0; k < BLOCKS; k++) {
        const uint64_t last = k * b3::CLOCKS + b3::CLOCKS - 1;
        bool any = false;
        for (uint64_t c = L.va; c < L.mul_table && !any; c++) any = Goldilocks::toU64(trace[last * nCols + c]) != 0;
        EXPECT_TRUE(any) << "block " << k << " interior is empty";
    }

    // And the multiplicities landed, summing to what three blocks of two lanes actually look up.
    uint64_t tableTotal = 0, rangeTotal = 0;
    for (uint64_t i = 0; i < b3::TABLE_SIZE; i++) tableTotal += Goldilocks::toU64(trace[i * nCols + L.mul_table]);
    for (uint64_t i = 0; i < b3::RANGE_SIZE; i++) rangeTotal += Goldilocks::toU64(trace[i * nCols + L.mul_range]);
    EXPECT_EQ(tableTotal, BLOCKS * LANES * (16 * b3::CLOCKS + 4 * 16));
    EXPECT_EQ(rangeTotal, BLOCKS * LANES * 6 * b3::CLOCKS);
}

/// A BLAKE3 air whose section forgot LANES must be refused, not expanded at lanes = 0 -- which
/// would silently fill nothing and leave the interiors zero.
TEST(GateBandsBlake3, MissingLanesIsRefused)
{
    const uint64_t H = exec_layout::EXEC_MAGIC | exec_layout::EXEC_FORMAT_VERSION;
    const uint64_t V = gate_bands::GATE_BAND_FORMAT_VERSION;
    std::vector<uint64_t> exec{H, 0, 1, 2, 0x00000002'00000001ull, V, 1, 0,
                               0, (uint64_t)GB_BLAKE3_NODE, 11};
    std::vector<Goldilocks::Element> trace(64 * 143, Goldilocks::zero());
    auto res = gate_bands::expand_gate_bands(trace.data(), exec.data(), 143, exec.size(), 64);
    EXPECT_EQ((int)res.status, (int)gate_bands::ExpandStatus::MalformedSection);
}
