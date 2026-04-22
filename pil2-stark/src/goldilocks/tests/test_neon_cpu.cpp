// Bit-exact equivalence tests for the NEON Goldilocks / Poseidon2 NEON path
// against the scalar reference. Skipped on non-NEON builds.
//
// Two layers of coverage:
//   1. Goldilocks_neon field-op helpers (gl_add, gl_sub, gl_mul) — random
//      pairs vs scalar Goldilocks::add/sub/mul.
//   2. Poseidon2Goldilocks<W>::permute_neon vs ::permute_seq for W=8 — both
//      a random sweep and the published PERMUTE_W8_GOLDEN gold value.

#include "test_helpers.hpp"

#include "../src/platform.hpp"

#if PIL2_HAS_NEON

#include <arm_neon.h>
#include <cstdint>
#include <random>

#include "../src/poseidon2_goldilocks.hpp"
#include "../src/poseidon2_goldilocks_neon.hpp"
#include "../src/goldilocks_base_field.hpp"

namespace {

// Stable PRNG so tests are reproducible across runs.
std::mt19937_64 make_rng(uint64_t seed = 0xC0FFEE'DEADBEEFULL) {
    return std::mt19937_64{seed};
}

uint64_t random_field(std::mt19937_64& rng) {
    // Sample a value in [0, p) so canonical-form preconditions hold.
    constexpr uint64_t P = 0xFFFFFFFF00000001ULL;
    return rng() % P;
}

}  // namespace

// ---------------------------------------------------------------------------
// Goldilocks_neon field-op helpers vs scalar Goldilocks
// ---------------------------------------------------------------------------

TEST(Goldilocks_neon, gl_add_matches_scalar_random_sweep) {
    auto rng = make_rng();
    for (int iter = 0; iter < 1024; ++iter) {
        uint64_t a0 = random_field(rng), a1 = random_field(rng);
        uint64_t b0 = random_field(rng), b1 = random_field(rng);
        uint64x2_t va = vsetq_lane_u64(a1, vsetq_lane_u64(a0, vdupq_n_u64(0), 0), 1);
        uint64x2_t vb = vsetq_lane_u64(b1, vsetq_lane_u64(b0, vdupq_n_u64(0), 0), 1);
        uint64x2_t vr = Goldilocks_neon::gl_add(va, vb);
        Goldilocks::Element ea, eb, er0, er1;
        ea.fe = a0; eb.fe = b0; Goldilocks::add(er0, ea, eb);
        ea.fe = a1; eb.fe = b1; Goldilocks::add(er1, ea, eb);
        EXPECT_EQ(vgetq_lane_u64(vr, 0), er0.fe) << "lane 0 iter " << iter;
        EXPECT_EQ(vgetq_lane_u64(vr, 1), er1.fe) << "lane 1 iter " << iter;
    }
}

TEST(Goldilocks_neon, gl_sub_matches_scalar_random_sweep) {
    auto rng = make_rng(0xBADC0DE);
    for (int iter = 0; iter < 1024; ++iter) {
        uint64_t a0 = random_field(rng), a1 = random_field(rng);
        uint64_t b0 = random_field(rng), b1 = random_field(rng);
        uint64x2_t va = vsetq_lane_u64(a1, vsetq_lane_u64(a0, vdupq_n_u64(0), 0), 1);
        uint64x2_t vb = vsetq_lane_u64(b1, vsetq_lane_u64(b0, vdupq_n_u64(0), 0), 1);
        uint64x2_t vr = Goldilocks_neon::gl_sub(va, vb);
        Goldilocks::Element ea, eb, er0, er1;
        ea.fe = a0; eb.fe = b0; Goldilocks::sub(er0, ea, eb);
        ea.fe = a1; eb.fe = b1; Goldilocks::sub(er1, ea, eb);
        EXPECT_EQ(vgetq_lane_u64(vr, 0), er0.fe) << "lane 0 iter " << iter;
        EXPECT_EQ(vgetq_lane_u64(vr, 1), er1.fe) << "lane 1 iter " << iter;
    }
}

TEST(Goldilocks_neon, gl_mul_matches_scalar_random_sweep) {
    auto rng = make_rng(0xFEEDFACE);
    for (int iter = 0; iter < 1024; ++iter) {
        uint64_t a0 = random_field(rng), a1 = random_field(rng);
        uint64_t b0 = random_field(rng), b1 = random_field(rng);
        uint64x2_t va = vsetq_lane_u64(a1, vsetq_lane_u64(a0, vdupq_n_u64(0), 0), 1);
        uint64x2_t vb = vsetq_lane_u64(b1, vsetq_lane_u64(b0, vdupq_n_u64(0), 0), 1);
        uint64x2_t vr = Goldilocks_neon::gl_mul(va, vb);
        Goldilocks::Element ea, eb, er0, er1;
        ea.fe = a0; eb.fe = b0; Goldilocks::mul(er0, ea, eb);
        ea.fe = a1; eb.fe = b1; Goldilocks::mul(er1, ea, eb);
        EXPECT_EQ(vgetq_lane_u64(vr, 0), er0.fe) << "lane 0 iter " << iter;
        EXPECT_EQ(vgetq_lane_u64(vr, 1), er1.fe) << "lane 1 iter " << iter;
    }
}

TEST(Goldilocks_neon, gl_mul_pure_matches_gl_mul_random_sweep) {
    auto rng = make_rng(0xDEADF00D);
    for (int iter = 0; iter < 1024; ++iter) {
        uint64_t a0 = random_field(rng), a1 = random_field(rng);
        uint64_t b0 = random_field(rng), b1 = random_field(rng);
        uint64x2_t va = vsetq_lane_u64(a1, vsetq_lane_u64(a0, vdupq_n_u64(0), 0), 1);
        uint64x2_t vb = vsetq_lane_u64(b1, vsetq_lane_u64(b0, vdupq_n_u64(0), 0), 1);
        uint64x2_t vr_asm  = Goldilocks_neon::gl_mul(va, vb);
        uint64x2_t vr_pure = Goldilocks_neon::gl_mul_pure(va, vb);
        EXPECT_EQ(vgetq_lane_u64(vr_asm, 0), vgetq_lane_u64(vr_pure, 0)) << "lane 0 iter " << iter;
        EXPECT_EQ(vgetq_lane_u64(vr_asm, 1), vgetq_lane_u64(vr_pure, 1)) << "lane 1 iter " << iter;
    }
}

TEST(Goldilocks_neon, gl_add_modulus_boundary_matches_scalar) {
    // Each case checks that NEON gl_add returns the exact same representation
    // as scalar Goldilocks::add — including non-canonical results (the scalar
    // path is allowed to return values in [0, 2p), and downstream Poseidon2
    // arithmetic is robust to that). We compute the expected value via the
    // scalar reference so the test stays in sync with the scalar contract.
    constexpr uint64_t P = 0xFFFFFFFF00000001ULL;
    const uint64_t boundary[] = {0, 1, 2, P / 2, P - 2, P - 1};
    for (uint64_t a : boundary) {
        for (uint64_t b : boundary) {
            uint64x2_t va = vdupq_n_u64(a);
            uint64x2_t vb = vdupq_n_u64(b);
            uint64x2_t vr = Goldilocks_neon::gl_add(va, vb);
            Goldilocks::Element ea, eb, er;
            ea.fe = a; eb.fe = b; Goldilocks::add(er, ea, eb);
            EXPECT_EQ(vgetq_lane_u64(vr, 0), er.fe)
                << std::hex << "a=0x" << a << " b=0x" << b;
        }
    }
}

// ---------------------------------------------------------------------------
// Poseidon2Goldilocks<W>::permute_neon vs ::permute_seq
// ---------------------------------------------------------------------------

TEST(Poseidon2Neon_W8, permute_neon_matches_permute_seq_random) {
    auto rng = make_rng(0x11223344'55667788ULL);
    for (int iter = 0; iter < 64; ++iter) {
        Goldilocks::Element input[8];
        for (int i = 0; i < 8; ++i) input[i].fe = random_field(rng);

        Goldilocks::Element out_seq[8], out_neon[8];
        Poseidon2Goldilocks<8>::permute(out_seq,  input, Poseidon2Mode::Scalar);
        Poseidon2Goldilocks<8>::permute(out_neon, input, Poseidon2Mode::Neon);

        for (int i = 0; i < 8; ++i) {
            EXPECT_EQ(out_neon[i].fe, out_seq[i].fe)
                << "iter " << iter << " element " << i;
        }
    }
}

TEST(Poseidon2Neon_W8, permute_neon_matches_permute_seq_zero_input) {
    Goldilocks::Element input[8] = {};
    Goldilocks::Element out_seq[8], out_neon[8];
    Poseidon2Goldilocks<8>::permute(out_seq,  input, Poseidon2Mode::Scalar);
    Poseidon2Goldilocks<8>::permute(out_neon, input, Poseidon2Mode::Neon);
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(out_neon[i].fe, out_seq[i].fe) << "element " << i;
    }
}

// Gold-value gate: permute_neon against the published PERMUTE_W8_GOLDEN and
// compress_neon against COMPRESS_W8_GOLDEN. Input: [0, 1, ..., W-1] (the
// canonical input the gold values were captured against in test_helpers.hpp).
TEST(Poseidon2Neon_W8, permute_neon_matches_PERMUTE_W8_GOLDEN) {
    Goldilocks::Element input[8];
    for (int i = 0; i < 8; ++i) input[i].fe = i;
    Goldilocks::Element out[8];
    Poseidon2Goldilocks<8>::permute(out, input, Poseidon2Mode::Neon);
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(out[i].fe, GoldilocksTestData::PERMUTE_W8_GOLDEN[i])
            << "element " << i;
    }
}

TEST(Poseidon2Neon_W8, compress_neon_matches_COMPRESS_W8_GOLDEN) {
    Goldilocks::Element input[8];
    for (int i = 0; i < 8; ++i) input[i].fe = i;
    Goldilocks::Element state[Poseidon2Goldilocks<8>::CAPACITY];
    Poseidon2Goldilocks<8>::compress(state, input, Poseidon2Mode::Neon);
    for (int i = 0; i < (int)Poseidon2Goldilocks<8>::CAPACITY; ++i) {
        EXPECT_EQ(state[i].fe, GoldilocksTestData::COMPRESS_W8_GOLDEN[i])
            << "element " << i;
    }
}

// W=4 — grinding hot path (single permutation in a leading-zero search loop).
TEST(Poseidon2Neon_W4, permute_neon_matches_PERMUTE_W4_GOLDEN) {
    Goldilocks::Element input[4];
    for (int i = 0; i < 4; ++i) input[i].fe = i;
    Goldilocks::Element out[4];
    Poseidon2Goldilocks<4>::permute(out, input, Poseidon2Mode::Neon);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(out[i].fe, GoldilocksTestData::PERMUTE_W4_GOLDEN[i])
            << "element " << i;
    }
}

TEST(Poseidon2Neon_W4, compress_neon_matches_COMPRESS_W4_GOLDEN) {
    Goldilocks::Element input[4];
    for (int i = 0; i < 4; ++i) input[i].fe = i;
    Goldilocks::Element state[Poseidon2Goldilocks<4>::CAPACITY];
    Poseidon2Goldilocks<4>::compress(state, input, Poseidon2Mode::Neon);
    for (int i = 0; i < (int)Poseidon2Goldilocks<4>::CAPACITY; ++i) {
        EXPECT_EQ(state[i].fe, GoldilocksTestData::COMPRESS_W4_GOLDEN[i])
            << "element " << i;
    }
}

TEST(Poseidon2Neon_W4, permute_neon_matches_permute_seq_random) {
    auto rng = make_rng(0x44444444'aaaaaaaaULL);
    for (int iter = 0; iter < 32; ++iter) {
        Goldilocks::Element input[4];
        for (int i = 0; i < 4; ++i) input[i].fe = random_field(rng);
        Goldilocks::Element out_seq[4], out_neon[4];
        Poseidon2Goldilocks<4>::permute(out_seq,  input, Poseidon2Mode::Scalar);
        Poseidon2Goldilocks<4>::permute(out_neon, input, Poseidon2Mode::Neon);
        for (int i = 0; i < 4; ++i)
            EXPECT_EQ(out_neon[i].fe, out_seq[i].fe)
                << "iter " << iter << " element " << i;
    }
}

// W=12 / W=16 don't have real-NEON test cases: permute_neon is a
// compile-time error at those widths because it regresses vs scalar
// (Auto mode routes W=12 to Scalar and W=16 to the scalar-unrolled
// ScalarUnrolledW16 variant tested below).

TEST(Poseidon2Neon_W16ScalarUnrolled, permute_matches_scalar_random) {
    auto rng = make_rng(0x16c0ffee'78787878ULL);
    for (int iter = 0; iter < 64; ++iter) {
        Goldilocks::Element input[16];
        for (int i = 0; i < 16; ++i) input[i].fe = random_field(rng);
        Goldilocks::Element out_seq[16], out_candidate[16];
        Poseidon2Goldilocks<16>::permute(out_seq, input, Poseidon2Mode::Scalar);
        Poseidon2Goldilocks<16>::permute(out_candidate, input, Poseidon2Mode::ScalarUnrolledW16);
        for (int i = 0; i < 16; ++i)
            EXPECT_EQ(out_candidate[i].fe, out_seq[i].fe)
                << "iter " << iter << " element " << i;
    }
}

TEST(Poseidon2Neon_W16ScalarUnrolled, permute_matches_PERMUTE_W16_GOLDEN) {
    Goldilocks::Element input[16];
    for (int i = 0; i < 16; ++i) input[i].fe = i;
    Goldilocks::Element out[16];
    Poseidon2Goldilocks<16>::permute(out, input, Poseidon2Mode::ScalarUnrolledW16);
    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(out[i].fe, GoldilocksTestData::PERMUTE_W16_GOLDEN[i])
            << "element " << i;
    }
}

TEST(Poseidon2Neon_W16ScalarUnrolled, compress_matches_COMPRESS_W16_GOLDEN) {
    Goldilocks::Element input[16];
    for (int i = 0; i < 16; ++i) input[i].fe = i;
    Goldilocks::Element state[Poseidon2Goldilocks<16>::CAPACITY];
    Poseidon2Goldilocks<16>::compress(state, input, Poseidon2Mode::ScalarUnrolledW16);
    for (int i = 0; i < (int)Poseidon2Goldilocks<16>::CAPACITY; ++i) {
        EXPECT_EQ(state[i].fe, GoldilocksTestData::COMPRESS_W16_GOLDEN[i])
            << "element " << i;
    }
}

TEST(Poseidon2Neon_W16ScalarUnrolled, linear_hash_matches_scalar_random) {
    auto rng = make_rng(0x16c0ffee'11112222ULL);
    for (uint64_t size : {1ULL, 4ULL, 12ULL, 13ULL, 24ULL, 36ULL, 56ULL}) {
        std::vector<Goldilocks::Element> input(size);
        for (auto& e : input) e.fe = random_field(rng);
        Goldilocks::Element out_seq[4], out_candidate[4];
        Poseidon2Goldilocks<16>::linearHash(out_seq, input.data(), size, Poseidon2Mode::Scalar);
        Poseidon2Goldilocks<16>::linearHash(out_candidate, input.data(), size, Poseidon2Mode::ScalarUnrolledW16);
        for (int i = 0; i < 4; ++i)
            EXPECT_EQ(out_candidate[i].fe, out_seq[i].fe)
                << "size " << size << " element " << i;
    }
}

TEST(Poseidon2Neon_W16ScalarUnrolled, merkletree_matches_scalar_random) {
    auto rng = make_rng(0x16c0ffee'55556666ULL);
    constexpr uint64_t num_cols = 36;
    constexpr uint64_t num_rows = 257;
    constexpr uint64_t arity = 4;
    std::vector<Goldilocks::Element> input(num_rows * num_cols);
    for (auto& e : input) e.fe = random_field(rng);

    auto tree_size = [&]() {
        uint64_t total = num_rows;
        uint64_t pending = num_rows;
        while (pending > 1) {
            uint64_t extra = (arity - (pending % arity)) % arity;
            total += extra;
            pending = (pending + (arity - 1)) / arity;
            total += pending;
        }
        return total;
    }();

    std::vector<Goldilocks::Element> tree_seq(tree_size * 4, Goldilocks::Element{0});
    std::vector<Goldilocks::Element> tree_candidate(tree_size * 4, Goldilocks::Element{0});
    Poseidon2Goldilocks<16>::merkletree(tree_seq.data(), input.data(), num_cols, num_rows, arity, Poseidon2Mode::Scalar, 0, 1);
    Poseidon2Goldilocks<16>::merkletree(tree_candidate.data(), input.data(), num_cols, num_rows, arity, Poseidon2Mode::ScalarUnrolledW16, 0, 1);
    ASSERT_EQ(tree_candidate.size(), tree_seq.size());
    for (size_t i = 0; i < tree_seq.size(); ++i)
        EXPECT_EQ(tree_candidate[i].fe, tree_seq[i].fe) << "tree element " << i;
}

// Merkletree mode-equivalence: Scalar vs Neon produce the same tree for a
// range of (num_rows, num_cols, arity) shapes. This is the full-chain gate
// — linear_hash_neon inside, compress_neon for internal nodes.
namespace {

template <uint32_t W>
void check_merkletree_mode_equivalence(uint64_t num_cols, uint64_t num_rows, uint64_t arity,
                                       uint64_t seed) {
    auto rng = make_rng(seed);
    std::vector<Goldilocks::Element> input(num_rows * num_cols);
    for (auto& e : input) e.fe = random_field(rng);

    // Tree size: sum of per-level rounded counts × CAPACITY, built the same way
    // as merkletree_seq / merkletree_neon expect.
    auto tree_size = [&]() {
        uint64_t total = num_rows;
        uint64_t pending = num_rows;
        while (pending > 1) {
            uint64_t extra = (arity - (pending % arity)) % arity;
            total += extra;
            uint64_t next = (pending + arity - 1) / arity;
            total += next;
            pending = next;
        }
        return total;
    };
    const uint64_t tsize = tree_size();
    std::vector<Goldilocks::Element> tree_seq (tsize * Poseidon2Goldilocks<W>::CAPACITY, Goldilocks::Element{0});
    std::vector<Goldilocks::Element> tree_neon(tsize * Poseidon2Goldilocks<W>::CAPACITY, Goldilocks::Element{0});

    Poseidon2Goldilocks<W>::merkletree(tree_seq.data(),  input.data(), num_cols, num_rows, arity, Poseidon2Mode::Scalar, 0, 1);
    Poseidon2Goldilocks<W>::merkletree(tree_neon.data(), input.data(), num_cols, num_rows, arity, Poseidon2Mode::Neon,   0, 1);

    for (size_t i = 0; i < tree_seq.size(); ++i) {
        EXPECT_EQ(tree_seq[i].fe, tree_neon[i].fe)
            << "W=" << W << " rows=" << num_rows << " cols=" << num_cols
            << " arity=" << arity << " tree[" << i << "]";
        if (tree_seq[i].fe != tree_neon[i].fe) return;  // stop after first diff
    }
}

}  // namespace

TEST(Poseidon2Neon_merkletree, W8_arity2) {
    check_merkletree_mode_equivalence<8>(/*ncols=*/24, /*nrows=*/64,  /*arity=*/2, 0xaa);
    check_merkletree_mode_equivalence<8>(/*ncols=*/36, /*nrows=*/128, /*arity=*/2, 0xbb);
}
// W=12 / W=16 merkletree via Mode::Neon is gated off — see permute_neon
// comment: Auto routes those widths to Scalar / ScalarUnrolledW16.

// Batch (Mode::NeonBatch): tree built via NeonBatch must equal scalar tree.
namespace {
template <uint32_t W>
void check_merkletree_batch_mode_equivalence(uint64_t num_cols, uint64_t num_rows,
                                             uint64_t arity, uint64_t seed) {
    auto rng = make_rng(seed);
    std::vector<Goldilocks::Element> input(num_rows * num_cols);
    for (auto& e : input) e.fe = random_field(rng);

    auto tree_size = [&]() {
        uint64_t total = num_rows;
        uint64_t pending = num_rows;
        while (pending > 1) {
            uint64_t extra = (arity - (pending % arity)) % arity;
            total += extra;
            uint64_t next = (pending + arity - 1) / arity;
            total += next;
            pending = next;
        }
        return total;
    };
    const uint64_t tsize = tree_size();
    std::vector<Goldilocks::Element> tree_seq  (tsize * Poseidon2Goldilocks<W>::CAPACITY, Goldilocks::Element{0});
    std::vector<Goldilocks::Element> tree_batch(tsize * Poseidon2Goldilocks<W>::CAPACITY, Goldilocks::Element{0});

    Poseidon2Goldilocks<W>::merkletree(tree_seq.data(),   input.data(), num_cols, num_rows, arity, Poseidon2Mode::Scalar,    0, 1);
    Poseidon2Goldilocks<W>::merkletree(tree_batch.data(), input.data(), num_cols, num_rows, arity, Poseidon2Mode::NeonBatch, 0, 1);

    for (size_t i = 0; i < tree_seq.size(); ++i) {
        EXPECT_EQ(tree_seq[i].fe, tree_batch[i].fe)
            << "W=" << W << " rows=" << num_rows << " cols=" << num_cols
            << " arity=" << arity << " tree[" << i << "]";
        if (tree_seq[i].fe != tree_batch[i].fe) return;
    }
}
}  // namespace

TEST(Poseidon2NeonBatch_merkletree, W8_arity2) {
    // Even and odd row counts (odd exercises the scalar tail in
    // merkletree_batch_neon's `num_rows - i < 2` branch).
    check_merkletree_batch_mode_equivalence<8>(/*ncols=*/24, /*nrows=*/64,  /*arity=*/2, 0xa1);
    check_merkletree_batch_mode_equivalence<8>(/*ncols=*/36, /*nrows=*/65,  /*arity=*/2, 0xa2);
    check_merkletree_batch_mode_equivalence<8>(/*ncols=*/24, /*nrows=*/128, /*arity=*/2, 0xa3);
}
// W=12 / W=16 merkletree via Mode::NeonBatch is gated off — see
// merkletree() dispatch: Auto routes those widths to Scalar / ScalarUnrolledW16.

#endif  // PIL2_HAS_NEON
