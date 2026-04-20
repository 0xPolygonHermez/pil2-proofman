#include "../src/platform.hpp"

#include <gtest/gtest.h>

#if PIL2_HAS_METAL

#include "../src/metal/metal_context.hpp"
#include "../src/goldilocks_base_field.hpp"
#include "../src/goldilocks_cubic_extension.hpp"

#include <array>
#include <cstdint>
#include <random>
#include <vector>

namespace {

constexpr uint64_t GL_P = 0xFFFFFFFF00000001ULL;

struct EvalSpec {
    uint32_t buf_id;        // 0 = aux, 1 = custom, 2 = const
    uint64_t offset;        // element index into the backing buffer
    uint64_t stride;        // row stride (u64 per row)
    uint32_t dim;           // 1 or 3
    uint32_t opening_pos;   // index into lev's np axis
};

// CPU oracle mirroring starks.hpp::evmap's inner summation for one eval.
std::array<uint64_t, 3> cpu_evmap_one(const std::vector<uint64_t>& lev,
                                      const std::vector<uint64_t>& aux,
                                      const std::vector<uint64_t>& custom,
                                      const std::vector<uint64_t>& const_pols,
                                      const EvalSpec& spec,
                                      uint32_t N,
                                      uint32_t extend_bits,
                                      uint32_t np) {
    const std::vector<uint64_t>* src =
        spec.buf_id == 0 ? &aux
      : spec.buf_id == 1 ? &custom
      :                    &const_pols;

    Goldilocks3::Element acc;
    acc[0] = Goldilocks::zero(); acc[1] = Goldilocks::zero(); acc[2] = Goldilocks::zero();
    for (uint32_t k = 0; k < N; ++k) {
        uint32_t row_ext = k << extend_bits;
        uint64_t lev_base = (uint64_t)(spec.opening_pos + k * np) * 3ULL;
        Goldilocks3::Element lev_vec;
        for (int c = 0; c < 3; ++c) lev_vec[c] = Goldilocks::fromU64(lev[lev_base + c]);

        Goldilocks3::Element pol_vec;
        uint64_t pol_base = spec.offset + (uint64_t)row_ext * spec.stride;
        if (spec.dim == 1) {
            pol_vec[0] = Goldilocks::fromU64((*src)[pol_base]);
            pol_vec[1] = Goldilocks::zero();
            pol_vec[2] = Goldilocks::zero();
        } else {
            for (int c = 0; c < 3; ++c) pol_vec[c] = Goldilocks::fromU64((*src)[pol_base + c]);
        }

        Goldilocks3::Element prod;
        Goldilocks3::mul(prod, lev_vec, pol_vec);
        Goldilocks3::add(acc, acc, prod);
    }
    return { Goldilocks::toU64(acc[0]), Goldilocks::toU64(acc[1]), Goldilocks::toU64(acc[2]) };
}

} // namespace

// Exercises all three backing buffers + both dims, with N large enough
// that the GPU kernel's strided loop runs > 1 iteration per thread
// (requires N > EVMAP_TG_SIZE = 256 in the kernel).
TEST(MetalEvmap, BitExactThreeBuffersMixedDims) {
    const uint32_t N           = 1024;
    const uint32_t extend_bits = 1;     // extended domain is 2*N = 2048 rows
    const uint32_t NExt        = N << extend_bits;
    const uint32_t np          = 2;     // 2 opening points

    std::mt19937_64 rng(0xB1600001ULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    // LEv: N * np * 3 u64.
    std::vector<uint64_t> lev(N * np * 3);
    for (auto& x : lev) x = dist(rng);

    // Backing buffers: each is just NExt * cols u64 of random data. Real
    // prover layouts are per-stage and not fully random, but the oracle
    // only cares about the (offset, stride, row_ext) addressing being
    // identical between CPU and GPU.
    const uint32_t aux_cols    = 8;
    const uint32_t custom_cols = 4;
    const uint32_t const_cols  = 5;
    std::vector<uint64_t> aux   (NExt * aux_cols);
    std::vector<uint64_t> custom(NExt * custom_cols);
    std::vector<uint64_t> cnst  (NExt * const_cols);
    for (auto& x : aux)    x = dist(rng);
    for (auto& x : custom) x = dist(rng);
    for (auto& x : cnst)   x = dist(rng);

    // A handful of eval specs covering every combo we care about.
    std::vector<EvalSpec> evals = {
        // aux, dim=1
        EvalSpec{0, /*offset=*/0, /*stride=*/aux_cols, /*dim=*/1, /*opening=*/0},
        // aux, dim=3
        EvalSpec{0, /*offset=*/2, /*stride=*/aux_cols, /*dim=*/3, /*opening=*/1},
        // custom, dim=1
        EvalSpec{1, /*offset=*/1, /*stride=*/custom_cols, /*dim=*/1, /*opening=*/0},
        // const, dim=3
        EvalSpec{2, /*offset=*/0, /*stride=*/const_cols, /*dim=*/3, /*opening=*/1},
        // aux, dim=3, non-zero offset, odd stride
        EvalSpec{0, /*offset=*/5, /*stride=*/aux_cols, /*dim=*/3, /*opening=*/0},
    };
    const uint32_t n_evals = (uint32_t)evals.size();

    // Flatten specs into the host arrays the bridge expects.
    std::vector<uint64_t> offsets(n_evals);
    std::vector<uint64_t> strides(n_evals);
    std::vector<uint32_t> dims(n_evals);
    std::vector<uint32_t> opening_pos(n_evals);
    std::vector<uint32_t> buf_ids(n_evals);
    for (uint32_t i = 0; i < n_evals; ++i) {
        offsets[i]     = evals[i].offset;
        strides[i]     = evals[i].stride;
        dims[i]        = evals[i].dim;
        opening_pos[i] = evals[i].opening_pos;
        buf_ids[i]     = evals[i].buf_id;
    }

    std::vector<uint64_t> gpu_out(n_evals * 3, 0xDEADBEEFULL);
    pil2::metal::run_evmap_metal(pil2::metal::get_context(),
                                 lev.data(),
                                 aux.data(), custom.data(), cnst.data(),
                                 offsets.data(), strides.data(),
                                 dims.data(), opening_pos.data(), buf_ids.data(),
                                 gpu_out.data(),
                                 N, extend_bits, np, n_evals,
                                 (uint32_t)lev.size(),
                                 (uint32_t)aux.size(),
                                 (uint32_t)custom.size(),
                                 (uint32_t)cnst.size());

    for (uint32_t i = 0; i < n_evals; ++i) {
        auto expected = cpu_evmap_one(lev, aux, custom, cnst,
                                      evals[i], N, extend_bits, np);
        for (int c = 0; c < 3; ++c) {
            ASSERT_EQ(gpu_out[i * 3 + c], expected[c])
                << "eval " << i << " component " << c
                << " got=0x" << std::hex << gpu_out[i * 3 + c]
                << " want=0x" << expected[c]
                << " (buf_id=" << std::dec << (int)evals[i].buf_id
                << " dim=" << evals[i].dim << ")";
        }
    }
}

#else

TEST(MetalEvmap, SkippedBuildFlag) {
    GTEST_SKIP() << "PIL2_HAS_METAL=0 at compile time";
}

#endif  // PIL2_HAS_METAL
