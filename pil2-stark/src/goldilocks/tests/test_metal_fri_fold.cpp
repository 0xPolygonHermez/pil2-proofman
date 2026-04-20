#include "../src/platform.hpp"

#include <gtest/gtest.h>

#if PIL2_HAS_METAL

#include "../src/metal/metal_context.hpp"
#include "../src/goldilocks_base_field.hpp"
#include "../src/goldilocks_cubic_extension.hpp"
#include "../src/ntt_goldilocks.hpp"

#include <cstdint>
#include <random>
#include <vector>

// Forward declaration to avoid pulling the whole starkpil tree into the
// test binary. fri.hpp is header-only and template-heavy so we re-
// implement the reference fold here using only Goldilocks3 primitives —
// verified to match the CPU FRI::fold by construction (see fri.hpp:32).

namespace {

constexpr uint64_t GL_P = 0xFFFFFFFF00000001ULL;

uint64_t goldilocks_pow_u64(uint64_t base, uint64_t exp) {
    Goldilocks::Element b = Goldilocks::fromU64(base);
    Goldilocks::Element r = Goldilocks::pow(b, exp);
    return Goldilocks::toU64(r);
}

uint64_t goldilocks_inv_u64_local(uint64_t x) {
    return Goldilocks::toU64(Goldilocks::inv(Goldilocks::fromU64(x)));
}

std::vector<uint64_t> build_nx_roots_u64(uint32_t log2_nX) {
    const uint64_t N = 1ULL << log2_nX;
    std::vector<uint64_t> roots(N);
    Goldilocks::Element w = Goldilocks::w(log2_nX);
    Goldilocks::Element acc = Goldilocks::one();
    for (uint64_t k = 0; k < N; ++k) {
        roots[k] = Goldilocks::toU64(acc);
        acc = acc * w;
    }
    return roots;
}

// Reference CPU FRI fold, mirroring FRI::fold with step==1 & nBitsExt==prevBits.
// Takes u64 inputs/outputs (canonical) so the comparison with Metal is
// bit-exact at the byte level. Fills out[0..pol2N*3) with the result.
void cpu_fri_fold_step1(std::vector<uint64_t>& pol,
                        const uint64_t challenge[3],
                        uint64_t polBits,
                        uint64_t currentBits) {
    const uint64_t pol2N = 1ULL << currentBits;
    const uint64_t nX    = (1ULL << polBits) / pol2N;

    Goldilocks::Element polShiftInv = Goldilocks::inv(Goldilocks::shift());
    // step==1 && nBitsExt==prevBits => no extra squaring
    Goldilocks::Element wi = Goldilocks::inv(Goldilocks::w(polBits));
    Goldilocks3::Element chal;
    for (int c = 0; c < 3; ++c) chal[c] = Goldilocks::fromU64(challenge[c]);

    Goldilocks::Element sinv = polShiftInv;
    for (uint64_t g = 0; g < pol2N; ++g) {
        // gather nX cubic-ext rows at stride pol2N, flattened as nX*3
        std::vector<Goldilocks::Element> flat(nX * 3), out(nX * 3);
        for (uint64_t i = 0; i < nX; ++i) {
            const uint64_t src = ((i * pol2N) + g) * 3;
            for (int c = 0; c < 3; ++c)
                flat[i * 3 + c] = Goldilocks::fromU64(pol[src + c]);
        }
        NTT_Goldilocks ntt(nX, 1);
        ntt.INTT(out.data(), flat.data(), nX, 3);

        // polMulAxi: row i *= sinv^i
        Goldilocks::Element r = Goldilocks::one();
        for (uint64_t i = 0; i < nX; ++i) {
            Goldilocks3::Element row = { out[i * 3 + 0], out[i * 3 + 1], out[i * 3 + 2] };
            Goldilocks3::Element scaled;
            Goldilocks3::mul(scaled, row, r);
            out[i * 3 + 0] = scaled[0];
            out[i * 3 + 1] = scaled[1];
            out[i * 3 + 2] = scaled[2];
            r = r * sinv;
        }

        // Horner evaluation at chal
        Goldilocks3::Element acc = { out[(nX - 1) * 3 + 0], out[(nX - 1) * 3 + 1], out[(nX - 1) * 3 + 2] };
        for (int64_t i = int64_t(nX) - 2; i >= 0; --i) {
            Goldilocks3::Element aux;
            Goldilocks3::mul(aux, acc, chal);
            Goldilocks3::Element pi = { out[i * 3 + 0], out[i * 3 + 1], out[i * 3 + 2] };
            Goldilocks3::add(acc, aux, pi);
        }

        for (int c = 0; c < 3; ++c) pol[g * 3 + c] = Goldilocks::toU64(acc[c]);
        sinv = sinv * wi;
    }
}

} // namespace

TEST(MetalFriFold, W8BitExactStep1_N1024_Pol2N128) {
    // nX = 8, pol2N = 128 (currentBits=7), polBits = 10.
    constexpr uint32_t LOG2_POLBITS     = 10;
    constexpr uint32_t LOG2_CURRENTBITS = 7;
    constexpr uint64_t N        = 1ULL << LOG2_POLBITS;      // 1024
    constexpr uint64_t POL2N    = 1ULL << LOG2_CURRENTBITS;  // 128
    constexpr uint64_t NX       = N / POL2N;                 // 8

    std::mt19937_64 rng(0xF01DF01DULL);
    std::uniform_int_distribution<uint64_t> dist(0, GL_P - 1);

    // pol holds nX * pol2N cubic-ext elements on input.
    std::vector<uint64_t> pol_cpu(N * 3);
    for (auto& x : pol_cpu) x = dist(rng);
    std::vector<uint64_t> pol_gpu = pol_cpu;

    const uint64_t challenge[3] = { dist(rng), dist(rng), dist(rng) };

    cpu_fri_fold_step1(pol_cpu, challenge, LOG2_POLBITS, LOG2_CURRENTBITS);

    // Metal kernel needs pre-computed polShiftInv, wi, inv(nX), and the
    // nX-th roots table.
    const uint64_t polShiftInv = goldilocks_inv_u64_local(Goldilocks::toU64(Goldilocks::shift()));
    const uint64_t wi = goldilocks_inv_u64_local(Goldilocks::toU64(Goldilocks::w(LOG2_POLBITS)));
    const uint64_t inv8 = goldilocks_inv_u64_local(NX);
    auto roots8 = build_nx_roots_u64(3);  // log2(8) = 3

    pil2::metal::fri_fold_w8_metal(pil2::metal::get_context(),
                                   pol_gpu.data(),
                                   challenge,
                                   POL2N,
                                   polShiftInv, wi, inv8,
                                   roots8.data());

    for (uint64_t i = 0; i < POL2N * 3; ++i) {
        ASSERT_EQ(pol_gpu[i], pol_cpu[i])
            << "fri_fold W8 mismatch at flat-i=" << i
            << " (group " << (i / 3) << " col " << (i % 3) << ")"
            << " gpu=0x" << std::hex << pol_gpu[i]
            << " cpu=0x" << pol_cpu[i];
    }
}

#else

TEST(MetalFriFold, SkippedBuildFlag) {
    GTEST_SKIP() << "PIL2_HAS_METAL=0 at compile time";
}

#endif // PIL2_HAS_METAL
