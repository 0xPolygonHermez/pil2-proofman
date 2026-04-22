#include "../src/platform.hpp"

#include <gtest/gtest.h>

#if PIL2_HAS_METAL

#include "../src/metal/metal_context.hpp"
#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon2_goldilocks.hpp"
#include "../src/poseidon2_goldilocks_constants.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace {

constexpr uint64_t GL_P = 0xFFFFFFFF00000001ULL;

std::vector<uint64_t> load_C8_u64() {
    std::vector<uint64_t> out(86);
    for (size_t i = 0; i < 86; ++i) out[i] = Goldilocks::toU64(Poseidon2GoldilocksConstants::C8[i]);
    return out;
}
std::vector<uint64_t> load_D8_u64() {
    std::vector<uint64_t> out(8);
    for (size_t i = 0; i < 8; ++i) out[i] = Goldilocks::toU64(Poseidon2GoldilocksConstants::D8[i]);
    return out;
}

std::vector<uint64_t> load_C12_u64() {
    std::vector<uint64_t> out(118);
    for (size_t i = 0; i < 118; ++i) out[i] = Goldilocks::toU64(Poseidon2GoldilocksConstants::C12[i]);
    return out;
}
std::vector<uint64_t> load_D12_u64() {
    std::vector<uint64_t> out(12);
    for (size_t i = 0; i < 12; ++i) out[i] = Goldilocks::toU64(Poseidon2GoldilocksConstants::D12[i]);
    return out;
}
std::vector<uint64_t> load_C16_u64() {
    std::vector<uint64_t> out(150);
    for (size_t i = 0; i < 150; ++i) out[i] = Goldilocks::toU64(Poseidon2GoldilocksConstants::C16[i]);
    return out;
}
std::vector<uint64_t> load_D16_u64() {
    std::vector<uint64_t> out(16);
    for (size_t i = 0; i < 16; ++i) out[i] = Goldilocks::toU64(Poseidon2GoldilocksConstants::D16[i]);
    return out;
}

} // namespace

TEST(MetalMerkle, W8Arity2_N16) {
    // Binary Merkle tree over 16 leaves, each leaf = RATE=4 u64 input.
    // Scalar reference: Poseidon2Goldilocks<8>::merkletree_seq with
    // num_cols=RATE=4, dim=1, arity=2. Full tree: 16 leaves + 8 + 4 + 2 + 1
    // = 31 nodes × CAPACITY=4 u64s = 124 u64s.
    constexpr uint64_t NUM_ROWS = 16;
    constexpr uint64_t RATE     = 4;
    constexpr uint64_t CAP      = 4;
    constexpr uint64_t TREE_ELS = (2 * NUM_ROWS - 1) * CAP;

    std::mt19937_64 rng(0xC800DEEDULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> input_u64(NUM_ROWS * RATE);
    for (auto& x : input_u64) x = dist(rng);

    // CPU reference
    std::vector<Goldilocks::Element> cpu_input(NUM_ROWS * RATE);
    for (size_t i = 0; i < cpu_input.size(); ++i) cpu_input[i] = Goldilocks::fromU64(input_u64[i]);
    std::vector<Goldilocks::Element> cpu_tree(TREE_ELS, Goldilocks::zero());
    Poseidon2Goldilocks<8>::merkletree(cpu_tree.data(),
                                       cpu_input.data(),
                                       /*num_cols=*/RATE,
                                       /*num_rows=*/NUM_ROWS,
                                       /*arity=*/2,
                                       Poseidon2Mode::Scalar,
                                       /*num_threads=*/1,
                                       /*dim=*/1);

    std::vector<uint64_t> cpu_tree_u64(TREE_ELS);
    for (size_t i = 0; i < TREE_ELS; ++i) cpu_tree_u64[i] = Goldilocks::toU64(cpu_tree[i]);

    // GPU
    std::vector<uint64_t> gpu_tree(TREE_ELS, 0);
    auto C8 = load_C8_u64();
    auto D8 = load_D8_u64();
    pil2::metal::merkletree_poseidon2_w8_metal(pil2::metal::get_context(),
                                               gpu_tree.data(),
                                               input_u64.data(),
                                               NUM_ROWS,
                                               C8.data(), D8.data());

    for (size_t i = 0; i < TREE_ELS; ++i) {
        ASSERT_EQ(gpu_tree[i], cpu_tree_u64[i])
            << "Merkle tree mismatch at flat-i=" << i
            << " level " << (i < NUM_ROWS * CAP ? "leaves"
                             : i < (NUM_ROWS + NUM_ROWS / 2) * CAP ? "l1"
                             : "deeper");
    }

    // Explicit spotcheck: root is the final CAPACITY u64s.
    const size_t root_base = (TREE_ELS - CAP);
    for (size_t i = 0; i < CAP; ++i) {
        EXPECT_EQ(gpu_tree[root_base + i], cpu_tree_u64[root_base + i])
            << "root mismatch at i=" << i;
    }
}

TEST(MetalMerkle, W8Arity2_N64) {
    // Larger tree — 64 leaves, 127 nodes total. Exercises 6 Merkle levels.
    constexpr uint64_t NUM_ROWS = 64;
    constexpr uint64_t RATE     = 4;
    constexpr uint64_t CAP      = 4;
    constexpr uint64_t TREE_ELS = (2 * NUM_ROWS - 1) * CAP;

    std::mt19937_64 rng(0xC801B16ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> input_u64(NUM_ROWS * RATE);
    for (auto& x : input_u64) x = dist(rng);

    std::vector<Goldilocks::Element> cpu_input(NUM_ROWS * RATE);
    for (size_t i = 0; i < cpu_input.size(); ++i) cpu_input[i] = Goldilocks::fromU64(input_u64[i]);
    std::vector<Goldilocks::Element> cpu_tree(TREE_ELS, Goldilocks::zero());
    Poseidon2Goldilocks<8>::merkletree(cpu_tree.data(), cpu_input.data(),
                                       RATE, NUM_ROWS, 2,
                                       Poseidon2Mode::Scalar, 1, 1);

    std::vector<uint64_t> cpu_tree_u64(TREE_ELS);
    for (size_t i = 0; i < TREE_ELS; ++i) cpu_tree_u64[i] = Goldilocks::toU64(cpu_tree[i]);

    std::vector<uint64_t> gpu_tree(TREE_ELS, 0);
    auto C8 = load_C8_u64();
    auto D8 = load_D8_u64();
    pil2::metal::merkletree_poseidon2_w8_metal(pil2::metal::get_context(),
                                               gpu_tree.data(), input_u64.data(),
                                               NUM_ROWS, C8.data(), D8.data());

    for (size_t i = 0; i < TREE_ELS; ++i) {
        ASSERT_EQ(gpu_tree[i], cpu_tree_u64[i])
            << "N=64 Merkle tree mismatch at flat-i=" << i;
    }
}

TEST(MetalMerkle, W12Arity3_N27) {
    constexpr uint64_t NUM_ROWS = 27;       // 3^3 → 3 levels
    constexpr uint64_t RATE     = 8;
    constexpr uint64_t CAP      = 4;
    constexpr uint64_t ARITY    = 3;
    const     uint64_t TREE_ELS = ((ARITY * NUM_ROWS - 1) / (ARITY - 1)) * CAP;

    std::mt19937_64 rng(0xC712BEEFULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> input_u64(NUM_ROWS * RATE);
    for (auto& x : input_u64) x = dist(rng);

    std::vector<Goldilocks::Element> cpu_input(NUM_ROWS * RATE);
    for (size_t i = 0; i < cpu_input.size(); ++i) cpu_input[i] = Goldilocks::fromU64(input_u64[i]);
    std::vector<Goldilocks::Element> cpu_tree(TREE_ELS, Goldilocks::zero());
    Poseidon2Goldilocks<12>::merkletree(cpu_tree.data(), cpu_input.data(),
                                        RATE, NUM_ROWS, ARITY,
                                        Poseidon2Mode::Scalar, 1, 1);

    std::vector<uint64_t> cpu_tree_u64(TREE_ELS);
    for (size_t i = 0; i < TREE_ELS; ++i) cpu_tree_u64[i] = Goldilocks::toU64(cpu_tree[i]);

    std::vector<uint64_t> gpu_tree(TREE_ELS, 0);
    auto C12 = load_C12_u64();
    auto D12 = load_D12_u64();
    pil2::metal::merkletree_poseidon2_w12_metal(pil2::metal::get_context(),
                                                gpu_tree.data(), input_u64.data(),
                                                NUM_ROWS, C12.data(), D12.data());

    for (size_t i = 0; i < TREE_ELS; ++i) {
        ASSERT_EQ(gpu_tree[i], cpu_tree_u64[i])
            << "W=12 arity-3 Merkle mismatch at flat-i=" << i;
    }
}

TEST(MetalMerkle, W16Arity4_N64) {
    constexpr uint64_t NUM_ROWS = 64;       // 4^3 → 3 levels
    constexpr uint64_t RATE     = 12;
    constexpr uint64_t CAP      = 4;
    constexpr uint64_t ARITY    = 4;
    const     uint64_t TREE_ELS = ((ARITY * NUM_ROWS - 1) / (ARITY - 1)) * CAP;

    std::mt19937_64 rng(0xC716BEEFULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> input_u64(NUM_ROWS * RATE);
    for (auto& x : input_u64) x = dist(rng);

    std::vector<Goldilocks::Element> cpu_input(NUM_ROWS * RATE);
    for (size_t i = 0; i < cpu_input.size(); ++i) cpu_input[i] = Goldilocks::fromU64(input_u64[i]);
    std::vector<Goldilocks::Element> cpu_tree(TREE_ELS, Goldilocks::zero());
    Poseidon2Goldilocks<16>::merkletree(cpu_tree.data(), cpu_input.data(),
                                        RATE, NUM_ROWS, ARITY,
                                        Poseidon2Mode::Scalar, 1, 1);

    std::vector<uint64_t> cpu_tree_u64(TREE_ELS);
    for (size_t i = 0; i < TREE_ELS; ++i) cpu_tree_u64[i] = Goldilocks::toU64(cpu_tree[i]);

    std::vector<uint64_t> gpu_tree(TREE_ELS, 0);
    auto C16 = load_C16_u64();
    auto D16 = load_D16_u64();
    pil2::metal::merkletree_poseidon2_w16_metal(pil2::metal::get_context(),
                                                gpu_tree.data(), input_u64.data(),
                                                NUM_ROWS, C16.data(), D16.data());

    for (size_t i = 0; i < TREE_ELS; ++i) {
        ASSERT_EQ(gpu_tree[i], cpu_tree_u64[i])
            << "W=16 arity-4 Merkle mismatch at flat-i=" << i;
    }
}

// W=16 arity-4 Merkle over leaves with multi-column input, validating the
// sponge-absorb leaf kernel pose2_linear_hash_w16 against the CPU
// linear_hash_seq reference. Three shapes exercise the three control-flow
// branches of the sponge loop:
//   * num_cols == CAPACITY (4)       — single partial-fill permute
//   * num_cols == RATE (12)          — single full-fill permute, matches
//                                      the existing RATE-only Merkle path
//   * num_cols == 37 (non-multiple)  — multi-block absorb (4 permutes per
//                                      row: 12 + 12 + 12 + 1)
struct MetalMerkleMulticolCase {
    uint64_t num_rows;
    uint64_t num_cols;
    const char* label;
};

class MetalMerkleMulticol
    : public ::testing::TestWithParam<MetalMerkleMulticolCase> {};

TEST_P(MetalMerkleMulticol, W16Arity4) {
    const auto param = GetParam();
    constexpr uint64_t ARITY = 4;
    constexpr uint64_t CAP   = 4;
    const uint64_t NUM_ROWS  = param.num_rows;
    const uint64_t NUM_COLS  = param.num_cols;
    // Total nodes including per-level padding. The pow-of-arity closed form
    // (arity*num_rows - 1) / (arity - 1) undercounts when padding kicks in;
    // match MerkleTreeGL::getNumNodes exactly so the buffer fits both the
    // CPU reference and the Metal output when num_rows is not pow-of-arity.
    uint64_t total_nodes = NUM_ROWS;
    {
        uint64_t level_n = NUM_ROWS;
        while (level_n > 1) {
            uint64_t extra = (ARITY - (level_n % ARITY)) % ARITY;
            total_nodes += extra;
            uint64_t next = (level_n + ARITY - 1) / ARITY;
            total_nodes += next;
            level_n = next;
        }
    }
    const uint64_t TREE_ELS = total_nodes * CAP;

    std::mt19937_64 rng(0xC716C015ULL ^ NUM_COLS);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    std::vector<uint64_t> input_u64(NUM_ROWS * NUM_COLS);
    for (auto& x : input_u64) x = dist(rng);

    std::vector<Goldilocks::Element> cpu_input(NUM_ROWS * NUM_COLS);
    for (size_t i = 0; i < cpu_input.size(); ++i) cpu_input[i] = Goldilocks::fromU64(input_u64[i]);
    std::vector<Goldilocks::Element> cpu_tree(TREE_ELS, Goldilocks::zero());
    Poseidon2Goldilocks<16>::merkletree(cpu_tree.data(), cpu_input.data(),
                                        NUM_COLS, NUM_ROWS, ARITY,
                                        Poseidon2Mode::Scalar, 1, 1);

    std::vector<uint64_t> cpu_tree_u64(TREE_ELS);
    for (size_t i = 0; i < TREE_ELS; ++i) cpu_tree_u64[i] = Goldilocks::toU64(cpu_tree[i]);

    std::vector<uint64_t> gpu_tree(TREE_ELS, 0);
    auto C16 = load_C16_u64();
    auto D16 = load_D16_u64();
    pil2::metal::merkletree_poseidon2_w16_cols_metal(pil2::metal::get_context(),
                                                     gpu_tree.data(), input_u64.data(),
                                                     NUM_COLS, NUM_ROWS,
                                                     C16.data(), D16.data());

    for (size_t i = 0; i < TREE_ELS; ++i) {
        ASSERT_EQ(gpu_tree[i], cpu_tree_u64[i])
            << param.label << " mismatch at flat-i=" << i
            << " (num_cols=" << NUM_COLS << ")";
    }
}

INSTANTIATE_TEST_SUITE_P(
    SpongeShapes, MetalMerkleMulticol,
    ::testing::Values(
        MetalMerkleMulticolCase{64,  4,  "num_cols_eq_CAPACITY"},
        MetalMerkleMulticolCase{64, 12,  "num_cols_eq_RATE"},
        MetalMerkleMulticolCase{64, 37,  "num_cols_non_multiple_of_RATE"},
        // num_rows non-power-of-4 shapes exercise the per-level zero-
        // padding mirror of Poseidon2Goldilocks<16>::merkletree_seq. FRI
        // Merkle trees in fibonacci-square hit these (currentBits = odd ->
        // num_rows = 2^17, 2^11, 2^5 for FibonacciSquare) so this covers
        // the pow-of-4 fallback path.
        MetalMerkleMulticolCase{10,  12, "npof4_small_n10_extra2"},
        MetalMerkleMulticolCase{2048, 8, "npof4_n2048_extra_at_each_level"}));

// W=12 arity-3 multi-col: same sponge-absorb pattern as W=16 but with
// RATE=8. Covers num_cols < RATE, == RATE, > RATE (non-multiple), and
// non-pow-of-arity num_rows.
TEST(MetalMerkle, W12Arity3_MultiCol_N27_Cols8)  {
    constexpr uint64_t NUM_ROWS = 27;   // 3^3
    constexpr uint64_t NUM_COLS = 8;    // == RATE
    constexpr uint64_t ARITY    = 3, CAP = 4;
    uint64_t total = NUM_ROWS;
    { uint64_t l = NUM_ROWS; while (l > 1) { uint64_t e = (ARITY - l % ARITY) % ARITY; total += e; uint64_t n = (l + ARITY - 1)/ARITY; total += n; l = n; } }
    const uint64_t TREE_ELS = total * CAP;
    std::mt19937_64 rng(0xC712C0ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);
    std::vector<uint64_t> input_u64(NUM_ROWS * NUM_COLS);
    for (auto& x : input_u64) x = dist(rng);
    std::vector<Goldilocks::Element> cpu_input(NUM_ROWS * NUM_COLS), cpu_tree(TREE_ELS, Goldilocks::zero());
    for (size_t i = 0; i < cpu_input.size(); ++i) cpu_input[i] = Goldilocks::fromU64(input_u64[i]);
    Poseidon2Goldilocks<12>::merkletree(cpu_tree.data(), cpu_input.data(), NUM_COLS, NUM_ROWS, ARITY, Poseidon2Mode::Scalar, 1, 1);
    std::vector<uint64_t> cpu_u64(TREE_ELS), gpu_tree(TREE_ELS, 0);
    for (size_t i = 0; i < TREE_ELS; ++i) cpu_u64[i] = Goldilocks::toU64(cpu_tree[i]);
    auto C12 = load_C12_u64(); auto D12 = load_D12_u64();
    pil2::metal::merkletree_poseidon2_w12_cols_metal(pil2::metal::get_context(),
                                                     gpu_tree.data(), input_u64.data(),
                                                     NUM_COLS, NUM_ROWS, C12.data(), D12.data());
    for (size_t i = 0; i < TREE_ELS; ++i) {
        ASSERT_EQ(gpu_tree[i], cpu_u64[i]) << "W12 cols mismatch at " << i;
    }
}

TEST(MetalMerkle, W12Arity3_MultiCol_N20_Cols29) {
    // n=20 is NOT a power of 3 → padding exercised per level.
    constexpr uint64_t NUM_ROWS = 20, NUM_COLS = 29;
    constexpr uint64_t ARITY    = 3, CAP = 4;
    uint64_t total = NUM_ROWS;
    { uint64_t l = NUM_ROWS; while (l > 1) { uint64_t e = (ARITY - l % ARITY) % ARITY; total += e; uint64_t n = (l + ARITY - 1)/ARITY; total += n; l = n; } }
    const uint64_t TREE_ELS = total * CAP;
    std::mt19937_64 rng(0xC71220ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);
    std::vector<uint64_t> input_u64(NUM_ROWS * NUM_COLS);
    for (auto& x : input_u64) x = dist(rng);
    std::vector<Goldilocks::Element> cpu_input(NUM_ROWS * NUM_COLS), cpu_tree(TREE_ELS, Goldilocks::zero());
    for (size_t i = 0; i < cpu_input.size(); ++i) cpu_input[i] = Goldilocks::fromU64(input_u64[i]);
    Poseidon2Goldilocks<12>::merkletree(cpu_tree.data(), cpu_input.data(), NUM_COLS, NUM_ROWS, ARITY, Poseidon2Mode::Scalar, 1, 1);
    std::vector<uint64_t> cpu_u64(TREE_ELS), gpu_tree(TREE_ELS, 0);
    for (size_t i = 0; i < TREE_ELS; ++i) cpu_u64[i] = Goldilocks::toU64(cpu_tree[i]);
    auto C12 = load_C12_u64(); auto D12 = load_D12_u64();
    pil2::metal::merkletree_poseidon2_w12_cols_metal(pil2::metal::get_context(),
                                                     gpu_tree.data(), input_u64.data(),
                                                     NUM_COLS, NUM_ROWS, C12.data(), D12.data());
    for (size_t i = 0; i < TREE_ELS; ++i) {
        ASSERT_EQ(gpu_tree[i], cpu_u64[i]) << "W12 non-pow-3 mismatch at " << i;
    }
}

TEST(MetalMerkle, W8Arity2_MultiCol_N64_Cols17) {
    // n=64 is pow of 2, num_cols=17 > RATE=4 so multi-absorb per leaf.
    constexpr uint64_t NUM_ROWS = 64, NUM_COLS = 17;
    constexpr uint64_t ARITY    = 2, CAP = 4;
    uint64_t total = NUM_ROWS;
    { uint64_t l = NUM_ROWS; while (l > 1) { uint64_t e = (ARITY - l % ARITY) % ARITY; total += e; uint64_t n = (l + ARITY - 1)/ARITY; total += n; l = n; } }
    const uint64_t TREE_ELS = total * CAP;
    std::mt19937_64 rng(0xC78020ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);
    std::vector<uint64_t> input_u64(NUM_ROWS * NUM_COLS);
    for (auto& x : input_u64) x = dist(rng);
    std::vector<Goldilocks::Element> cpu_input(NUM_ROWS * NUM_COLS), cpu_tree(TREE_ELS, Goldilocks::zero());
    for (size_t i = 0; i < cpu_input.size(); ++i) cpu_input[i] = Goldilocks::fromU64(input_u64[i]);
    Poseidon2Goldilocks<8>::merkletree(cpu_tree.data(), cpu_input.data(), NUM_COLS, NUM_ROWS, ARITY, Poseidon2Mode::Scalar, 1, 1);
    std::vector<uint64_t> cpu_u64(TREE_ELS), gpu_tree(TREE_ELS, 0);
    for (size_t i = 0; i < TREE_ELS; ++i) cpu_u64[i] = Goldilocks::toU64(cpu_tree[i]);
    auto C8 = load_C8_u64(); auto D8 = load_D8_u64();
    pil2::metal::merkletree_poseidon2_w8_cols_metal(pil2::metal::get_context(),
                                                    gpu_tree.data(), input_u64.data(),
                                                    NUM_COLS, NUM_ROWS, C8.data(), D8.data());
    for (size_t i = 0; i < TREE_ELS; ++i) {
        ASSERT_EQ(gpu_tree[i], cpu_u64[i]) << "W8 cols mismatch at " << i;
    }
}

#else

TEST(MetalMerkle, SkippedBuildFlag) {
    GTEST_SKIP() << "PIL2_HAS_METAL=0 at compile time";
}

#endif // PIL2_HAS_METAL
