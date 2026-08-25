// The witness-to-trace gather. It is the one reader of the exec file's map, and a mistake in it
// is silent: the trace comes out well-formed but wrong, and the proof only fails to verify a
// recursion layer later.
//
// The map covers a live extent rather than the whole trace, so what is asserted here is mostly
// what happens *outside* it -- the dead columns a gate band fills instead, the padding rows past
// the map, and the unused-signal sentinel.

#include <gtest/gtest.h>
#include <array>
#include <vector>

#include "../src/goldilocks_base_field.hpp"
#include "../../starkpil/exec_file.hpp"

using GL = Goldilocks::Element;

namespace {

// An exec buffer with the given additions and a row-major map of u32 entries.
std::vector<uint64_t> makeExec(const std::vector<std::array<uint64_t, 4>> &adds,
                               const std::vector<std::vector<uint32_t>> &map)
{
    const uint64_t rows = map.size();
    const uint64_t cols = rows > 0 ? map[0].size() : 0;
    const uint64_t mapWords = (rows * cols + 1) / 2;

    std::vector<uint64_t> exec(exec_layout::HEADER_WORDS + adds.size() * 4 + mapWords, 0);
    exec[0] = exec_layout::EXEC_MAGIC | exec_layout::EXEC_FORMAT_VERSION;
    exec[1] = adds.size();
    exec[2] = rows;
    exec[3] = cols;
    for (size_t i = 0; i < adds.size(); i++)
        for (int j = 0; j < 4; j++) exec[exec_layout::HEADER_WORDS + i * 4 + j] = adds[i][j];

    const uint64_t mapAt = exec_layout::HEADER_WORDS + adds.size() * 4;
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t c = 0; c < cols; c++) {
            const uint64_t entry = r * cols + c;
            exec[mapAt + entry / 2] |= (uint64_t)map[r][c] << (32 * (entry % 2));
        }
    }
    return exec;
}

}  // namespace

// Cells the map does not reach must come out zero: the columns past its width, which the gate-band
// expander fills afterwards, and the rows past its height, which are the power-of-two padding.
TEST(ExecFile, ZeroesEveryCellOutsideTheMapsExtent)
{
    constexpr uint64_t N = 6;          // trace rows
    constexpr uint64_t NCOLS = 5;      // trace columns
    // 3x2 map: the trace is wider and taller than what the map covers.
    std::vector<uint64_t> exec = makeExec({}, {{1, 2}, {3, 0}, {0, 4}});

    std::vector<GL> witness(N * NCOLS, Goldilocks::fromU64(0xBADBADBAD));
    // The buffer runs past sizeWitness, where the additions go -- the shape every real caller
    // allocates (see Setup::get_circom_witness_size).
    std::vector<GL> circomWitness(8);
    for (uint64_t i = 0; i < circomWitness.size(); i++) circomWitness[i] = Goldilocks::fromU64(100 + i);

    getCommitedPols(circomWitness.data(), exec.data(), witness.data(), nullptr,
                    /*sizeWitness=*/6, N, /*nPublics=*/0, NCOLS);

    const uint64_t expected[N][NCOLS] = {
        {101, 102, 0, 0, 0},
        {103,   0, 0, 0, 0},   // entry 0 is the unused-signal sentinel, not witness[0]
        {  0, 104, 0, 0, 0},
        {  0,   0, 0, 0, 0},   // rows past the map's height
        {  0,   0, 0, 0, 0},
        {  0,   0, 0, 0, 0},
    };
    for (uint64_t i = 0; i < N; i++)
        for (uint64_t j = 0; j < NCOLS; j++)
            ASSERT_EQ(Goldilocks::toU64(witness[i * NCOLS + j]), expected[i][j]) << "row " << i << " col " << j;
}

// Additions land past the circuit's own witness and the map reaches them by index, so they have
// to be computed before the gather reads them -- and each may read an earlier one's output.
TEST(ExecFile, AdditionsAreVisibleToTheGatherAndMayChain)
{
    constexpr uint64_t sizeWitness = 4;
    // w[i] = i, so w[4] = w[1]*2 + w[2]*3 = 2 + 6 = 8, and w[5] = w[4]*1 + w[3]*10 = 8 + 30 = 38,
    // which only holds if the second addition sees the first one's output.
    std::vector<uint64_t> exec = makeExec({{{1, 2, 2, 3}}, {{4, 3, 1, 10}}}, {{5, 4}});

    std::vector<GL> circomWitness(sizeWitness + 2);
    for (uint64_t i = 0; i < sizeWitness; i++) circomWitness[i] = Goldilocks::fromU64(i);
    std::vector<GL> witness(2, Goldilocks::fromU64(0xBAD));

    getCommitedPols(circomWitness.data(), exec.data(), witness.data(), nullptr, sizeWitness,
                    /*N=*/1, /*nPublics=*/0, /*nCommitedPols=*/2);

    ASSERT_EQ(Goldilocks::toU64(circomWitness[4]), 8u);
    ASSERT_EQ(Goldilocks::toU64(circomWitness[5]), 38u);
    ASSERT_EQ(Goldilocks::toU64(witness[0]), 38u) << "map entry 5 must see the second addition";
    ASSERT_EQ(Goldilocks::toU64(witness[1]), 8u) << "map entry 4 must see the first addition";
}

// Publics come off the front of the circom witness, offset by the constant-one signal.
TEST(ExecFile, PublicsSkipTheConstantOneSignal)
{
    std::vector<uint64_t> exec = makeExec({}, {{1}});
    std::vector<GL> circomWitness(4);
    for (uint64_t i = 0; i < 4; i++) circomWitness[i] = Goldilocks::fromU64(70 + i);
    std::vector<GL> witness(1), publics(3, Goldilocks::fromU64(0xBAD));

    getCommitedPols(circomWitness.data(), exec.data(), witness.data(), publics.data(),
                    /*sizeWitness=*/3, /*N=*/1, /*nPublics=*/3, /*nCommitedPols=*/1);

    ASSERT_EQ(Goldilocks::toU64(publics[0]), 71u);
    ASSERT_EQ(Goldilocks::toU64(publics[1]), 72u);
    ASSERT_EQ(Goldilocks::toU64(publics[2]), 73u);
}

// A key this build cannot read must not be gathered at this build's offsets. There is no error
// channel here -- the loader owns that -- so the requirement is that it stays inside both
// buffers and produces a trace the AIR rejects, not a plausible wrong one.
TEST(ExecFile, AnUnreadableHeaderYieldsAnEmptyTrace)
{
    constexpr uint64_t N = 3, NCOLS = 2;
    std::vector<uint64_t> exec = makeExec({}, {{1, 2}, {3, 4}, {5, 6}});

    const uint64_t badHeaders[] = {0, exec_layout::EXEC_MAGIC | (exec_layout::EXEC_FORMAT_VERSION + 1)};
    for (uint64_t bad : badHeaders) {
        exec[0] = bad;
        std::vector<GL> witness(N * NCOLS, Goldilocks::fromU64(0xBAD));
        std::vector<GL> circomWitness(8, Goldilocks::fromU64(9));

        getCommitedPols(circomWitness.data(), exec.data(), witness.data(), nullptr,
                        /*sizeWitness=*/6, N, /*nPublics=*/0, NCOLS);

        for (uint64_t i = 0; i < N * NCOLS; i++)
            ASSERT_EQ(Goldilocks::toU64(witness[i]), 0u) << "header " << bad << " cell " << i;
    }
}

// A map wider or taller than the trace is a corrupt key. The loader rejects it, but the gather
// must not walk off the trace if one reaches it anyway.
TEST(ExecFile, ClampsAMapLargerThanTheTrace)
{
    std::vector<uint64_t> exec = makeExec({}, {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    std::vector<GL> circomWitness(16);
    for (uint64_t i = 0; i < 16; i++) circomWitness[i] = Goldilocks::fromU64(200 + i);

    // Trace is 2 rows x 2 cols; the map claims 3 x 3.
    std::vector<GL> witness(4, Goldilocks::fromU64(0xBAD));
    getCommitedPols(circomWitness.data(), exec.data(), witness.data(), nullptr,
                    /*sizeWitness=*/12, /*N=*/2, /*nPublics=*/0, /*nCommitedPols=*/2);

    ASSERT_EQ(Goldilocks::toU64(witness[0]), 201u);
    ASSERT_EQ(Goldilocks::toU64(witness[1]), 202u);
    ASSERT_EQ(Goldilocks::toU64(witness[2]), 204u);   // row 1 keeps the map's stride of 3
    ASSERT_EQ(Goldilocks::toU64(witness[3]), 205u);
}

// The compressor's real shape: a 46-column trace whose map is 30 wide, because a[30..45] is the
// gate band's chain slot -- written by the expander afterwards, never by the map. Those columns
// must come out zero on every row, band or not, and no cell may keep the pooled buffer's old
// contents: the trace comes from a pool, so an unwritten cell is stale data from a previous proof,
// not a zero.
TEST(ExecFile, LeavesNoCellUnwrittenAtTheCompressorsShape)
{
    constexpr uint64_t N = 40, NCOLS = 46, MAPCOLS = 30, MAPROWS = 32;
    const GL poison = Goldilocks::fromU64(0xFEEDFACECAFEBEEDull);

    // Every mapped cell carries a distinct signal index, so a misread stride shows up as a wrong
    // value rather than a coincidentally-right one.
    std::vector<std::vector<uint32_t>> map(MAPROWS, std::vector<uint32_t>(MAPCOLS));
    uint32_t next = 1;
    for (uint64_t r = 0; r < MAPROWS; r++)
        for (uint64_t c = 0; c < MAPCOLS; c++)
            map[r][c] = (r % 7 == 3 && c % 5 == 2) ? 0 : next++;   // some unused-signal sentinels
    std::vector<uint64_t> exec = makeExec({}, map);

    std::vector<GL> circomWitness(next + 4);
    for (uint64_t i = 0; i < circomWitness.size(); i++) circomWitness[i] = Goldilocks::fromU64(1000 + i);
    std::vector<GL> witness(N * NCOLS, poison);

    getCommitedPols(circomWitness.data(), exec.data(), witness.data(), nullptr,
                    /*sizeWitness=*/next, N, /*nPublics=*/0, NCOLS);

    for (uint64_t i = 0; i < N; i++) {
        for (uint64_t j = 0; j < NCOLS; j++) {
            const GL got = witness[i * NCOLS + j];
            ASSERT_NE(Goldilocks::toU64(got), Goldilocks::toU64(poison))
                << "row " << i << " col " << j << " was never written -- it still holds pool garbage";
            uint64_t want = 0;
            if (i < MAPROWS && j < MAPCOLS && map[i][j] != 0) want = 1000 + map[i][j];
            ASSERT_EQ(Goldilocks::toU64(got), want) << "row " << i << " col " << j;
        }
    }
}
