#ifndef POSEIDON2_GOLDILOCKS_NEON
#define POSEIDON2_GOLDILOCKS_NEON

#include "platform.hpp"
#include "poseidon2_goldilocks.hpp"
#include "goldilocks_base_field.hpp"
#include "goldilocks_neon.hpp"

#if PIL2_HAS_NEON

#include <arm_neon.h>
#include <cstdint>

// ============================================================================
// Poseidon2-level NEON primitive bodies. Forward declarations of permute_neon
// / compress_neon live in poseidon2_goldilocks.hpp under #if PIL2_HAS_NEON.
//
// Implementation notes:
// - State of W elements lives in W/2 NEON registers (uint64x2_t).
// - matmul_external_neon is a NEON-store -> scalar-call -> NEON-load wrapper
//   around the existing scalar matmul. The math is purely cheap adds; the
//   round bottleneck is pow7 (which IS in NEON), so the punt is fine for
//   now and keeps the matmul bit-exact-by-construction.
// - The partial-round loop mirrors the AVX path's "track aux + state0 in
//   scalar; let st[0] lane 0 hold the wrong value; fix it once at the end"
//   trick to avoid per-iteration NEON<->scalar roundtrips.
// ============================================================================

// ---- Element-wise primitives ---------------------------------------------

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::permute_neon(
    Goldilocks::Element *output, const Goldilocks::Element *input)
{
    // NEON wins only at W=4 (~9%) and W=8 (~5%). At W=12/W=16 it
    // regresses vs scalar because matmul_external punts to scalar and
    // the NEON-store / scalar-call / NEON-load overhead dominates. Auto
    // mode dispatches W=12 to Scalar and W=16 to the scalar-unrolled
    // ScalarUnrolledW16 variant; the `if constexpr` below keeps the full NEON
    // body out of explicit template instantiations of W=12/W=16 so
    // they compile without pulling in unreachable code.
    if constexpr (SPONGE_WIDTH_T != 4 && SPONGE_WIDTH_T != 8) {
        abortMode("permute_neon (W=12/W=16 unsupported)", Poseidon2Mode::Neon);
    } else {
    const Goldilocks::Element* C =
        SPONGE_WIDTH == 4 ? Poseidon2GoldilocksConstants::C4
                          : Poseidon2GoldilocksConstants::C8;
    const Goldilocks::Element* D =
        SPONGE_WIDTH == 4 ? Poseidon2GoldilocksConstants::D4
                          : Poseidon2GoldilocksConstants::D8;

    // Lambdas factor out the inline NEON helpers without dragging the W
    // template parameter into a free-function dispatch.
    constexpr uint32_t HALF_W = SPONGE_WIDTH >> 1;

    // matmul_external punts to the scalar implementation: NEON-store,
    // scalar-call, NEON-load. gl_add takes 7 NEON ops vs ~3 scalar ops,
    // and Apple Silicon has 8 integer ALUs vs 4 NEON ALUs, so scalar
    // wins for this cross-element add-heavy code. Inline NEON variants
    // tried and lost by 13–14% on M4 Pro, 2026-04-17.
    auto matmul_external_neon = [](uint64x2_t st[HALF_W]) {
        Goldilocks::Element scratch[SPONGE_WIDTH];
        for (uint32_t i = 0; i < HALF_W; ++i)
            Goldilocks_neon::store(&scratch[i << 1], st[i]);
        Poseidon2Goldilocks<SPONGE_WIDTH_T>::matmul_external_(scratch);
        for (uint32_t i = 0; i < HALF_W; ++i)
            st[i] = Goldilocks_neon::load(&scratch[i << 1]);
    };

    // Fused (state + C)^7 element-wise, paired across two regs for ILP.
    // C_ is a compile-time constant table with canonical values, so use
    // gl_add_c (4 ops) instead of gl_add (7 ops) — saves 3 NEON ops per add.
    auto pow7add_neon = [](uint64x2_t st[HALF_W],
                            const Goldilocks::Element C_[SPONGE_WIDTH]) {
        for (uint32_t i = 0; i < HALF_W; i += 2) {
            uint64x2_t c0 = Goldilocks_neon::load(&C_[(i + 0) << 1]);
            uint64x2_t c1 = Goldilocks_neon::load(&C_[(i + 1) << 1]);
            uint64x2_t a0 = Goldilocks_neon::gl_add_c(st[i + 0], c0);
            uint64x2_t a1 = Goldilocks_neon::gl_add_c(st[i + 1], c1);
            // Inline pow7 with two independent chains so the two integer-mul
            // pipes on Apple Silicon stay busy each cycle.
            uint64x2_t pw2_0 = Goldilocks_neon::gl_square(a0);
            uint64x2_t pw2_1 = Goldilocks_neon::gl_square(a1);
            uint64x2_t pw4_0 = Goldilocks_neon::gl_square(pw2_0);
            uint64x2_t pw4_1 = Goldilocks_neon::gl_square(pw2_1);
            uint64x2_t pw3_0 = Goldilocks_neon::gl_mul(a0, pw2_0);
            uint64x2_t pw3_1 = Goldilocks_neon::gl_mul(a1, pw2_1);
            st[i + 0] = Goldilocks_neon::gl_mul(pw3_0, pw4_0);
            st[i + 1] = Goldilocks_neon::gl_mul(pw3_1, pw4_1);
        }
    };

    // Load state into W/2 NEON regs.
    std::memcpy(output, input, SPONGE_WIDTH * sizeof(Goldilocks::Element));
    uint64x2_t st[HALF_W];
    for (uint32_t i = 0; i < HALF_W; ++i)
        st[i] = Goldilocks_neon::load(&output[i << 1]);

    // Initial M_E.
    matmul_external_neon(st);

    // First half full rounds.
    for (uint32_t r = 0; r < HALF_N_FULL_ROUNDS; ++r) {
        pow7add_neon(st, &C[r * SPONGE_WIDTH]);
        matmul_external_neon(st);
    }

    // Partial rounds. Track state[0] and aux in scalar; let st[0] lane 0 drift
    // to the wrong value and fix it once at the end (matches AVX trick).
    Goldilocks::Element aux_lanes[2];
    Goldilocks_neon::store(aux_lanes, st[0]);
    Goldilocks::Element state0 = aux_lanes[0];
    Goldilocks::Element aux    = state0;

    for (uint32_t r = 0; r < N_PARTIAL_ROUNDS; ++r) {
        // Sum across all lanes of all NEON regs.
        uint64x2_t partial = st[0];
        for (uint32_t i = 1; i < HALF_W; ++i)
            partial = Goldilocks_neon::gl_add(partial, st[i]);
        Goldilocks::Element partial_lanes[2];
        Goldilocks_neon::store(partial_lanes, partial);
        Goldilocks::Element sum = partial_lanes[0] + partial_lanes[1];
        sum = sum - aux;                                            // exclude old state[0]
        state0 = state0 + C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + r];
        Poseidon2Goldilocks<SPONGE_WIDTH_T>::pow7(state0);          // partial-round S-box
        sum = sum + state0;                                         // include new state[0]

        // Pre-canonicalise sum once (one scalar cmp+sub) so we can use
        // gl_add_c (4 ops) instead of gl_add (7 ops) inside the loop.
        // Saves 3 NEON ops × HALF_W × N_PARTIAL_ROUNDS per hash.
        uint64_t sum_canon = sum.fe;
        if (sum_canon >= Goldilocks_neon::P) sum_canon -= Goldilocks_neon::P;
        uint64x2_t scalar = Goldilocks_neon::splat(sum_canon);
        for (uint32_t i = 0; i < HALF_W; ++i) {
            uint64x2_t d = Goldilocks_neon::load(&D[i << 1]);
            st[i] = Goldilocks_neon::gl_mul(st[i], d);
            st[i] = Goldilocks_neon::gl_add_c(st[i], scalar);
        }

        // st[0] lane 0 now holds aux*D[0] + sum (wrong; the right value is
        // state0*D[0] + sum). Update both trackers in lockstep.
        state0 = state0 * D[0] + sum;
        aux    = aux    * D[0] + sum;
    }

    // Patch the wrong lane-0 of st[0] to the correct state[0].
    Goldilocks_neon::store(aux_lanes, st[0]);
    aux_lanes[0] = state0;
    st[0] = Goldilocks_neon::load(aux_lanes);

    // Second half full rounds.
    for (uint32_t r = 0; r < HALF_N_FULL_ROUNDS; ++r) {
        pow7add_neon(st, &C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH]);
        matmul_external_neon(st);
    }

    // Store result.
    for (uint32_t i = 0; i < HALF_W; ++i)
        Goldilocks_neon::store(&output[i << 1], st[i]);
    }  // end else (W == 4 || W == 8)
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::permute_w16_scalar_unrolled(
    Goldilocks::Element (&state)[SPONGE_WIDTH],
    const Goldilocks::Element (&input)[SPONGE_WIDTH])
{
    if constexpr (SPONGE_WIDTH_T != 16) {
        abortMode("permute_w16_scalar_unrolled", Poseidon2Mode::ScalarUnrolledW16);
    } else {

    const Goldilocks::Element* C = Poseidon2GoldilocksConstants::C16;
    const Goldilocks::Element* D = Poseidon2GoldilocksConstants::D16;

    auto pow7_local = [](Goldilocks::Element &x) {
        Goldilocks::Element x2 = x * x;
        Goldilocks::Element x3 = x * x2;
        Goldilocks::Element x4 = x2 * x2;
        x = x3 * x4;
    };

    auto pow7add16 = [&pow7_local](Goldilocks::Element *x, const Goldilocks::Element *c) {
#pragma clang loop unroll(full)
        for (uint32_t i = 0; i < 16; ++i) {
            Goldilocks::Element xi = x[i] + c[i];
            pow7_local(xi);
            x[i] = xi;
        }
    };

    auto matmul_m4_local = [](Goldilocks::Element *x) {
        Goldilocks::Element t0 = x[0] + x[1];
        Goldilocks::Element t1 = x[2] + x[3];
        Goldilocks::Element t2 = x[1] + x[1] + t1;
        Goldilocks::Element t3 = x[3] + x[3] + t0;
        Goldilocks::Element t1_2 = t1 + t1;
        Goldilocks::Element t0_2 = t0 + t0;
        Goldilocks::Element t4 = t1_2 + t1_2 + t3;
        Goldilocks::Element t5 = t0_2 + t0_2 + t2;
        Goldilocks::Element t6 = t3 + t5;
        Goldilocks::Element t7 = t2 + t4;
        x[0] = t6;
        x[1] = t5;
        x[2] = t7;
        x[3] = t4;
    };

    auto matmul_external16 = [&matmul_m4_local](Goldilocks::Element *x) {
        matmul_m4_local(&x[0]);
        matmul_m4_local(&x[4]);
        matmul_m4_local(&x[8]);
        matmul_m4_local(&x[12]);

        Goldilocks::Element s0 = x[0] + x[4] + x[8] + x[12];
        Goldilocks::Element s1 = x[1] + x[5] + x[9] + x[13];
        Goldilocks::Element s2 = x[2] + x[6] + x[10] + x[14];
        Goldilocks::Element s3 = x[3] + x[7] + x[11] + x[15];

        x[0] = x[0] + s0;   x[1] = x[1] + s1;   x[2] = x[2] + s2;   x[3] = x[3] + s3;
        x[4] = x[4] + s0;   x[5] = x[5] + s1;   x[6] = x[6] + s2;   x[7] = x[7] + s3;
        x[8] = x[8] + s0;   x[9] = x[9] + s1;   x[10] = x[10] + s2; x[11] = x[11] + s3;
        x[12] = x[12] + s0; x[13] = x[13] + s1; x[14] = x[14] + s2; x[15] = x[15] + s3;
    };

    std::memcpy(state, input, 16 * sizeof(Goldilocks::Element));

    matmul_external16(state);

    pow7add16(state, &C[0]);
    matmul_external16(state);
    pow7add16(state, &C[16]);
    matmul_external16(state);
    pow7add16(state, &C[32]);
    matmul_external16(state);
    pow7add16(state, &C[48]);
    matmul_external16(state);

#pragma clang loop unroll(full)
    for (uint32_t r = 0; r < N_PARTIAL_ROUNDS; ++r) {
        state[0] = state[0] + C[HALF_N_FULL_ROUNDS * SPONGE_WIDTH + r];
        pow7_local(state[0]);

        Goldilocks::Element sum = Goldilocks::zero();
        sum = sum + state[0];  sum = sum + state[1];  sum = sum + state[2];  sum = sum + state[3];
        sum = sum + state[4];  sum = sum + state[5];  sum = sum + state[6];  sum = sum + state[7];
        sum = sum + state[8];  sum = sum + state[9];  sum = sum + state[10]; sum = sum + state[11];
        sum = sum + state[12]; sum = sum + state[13]; sum = sum + state[14]; sum = sum + state[15];

        state[0] = state[0] * D[0] + sum;   state[1] = state[1] * D[1] + sum;
        state[2] = state[2] * D[2] + sum;   state[3] = state[3] * D[3] + sum;
        state[4] = state[4] * D[4] + sum;   state[5] = state[5] * D[5] + sum;
        state[6] = state[6] * D[6] + sum;   state[7] = state[7] * D[7] + sum;
        state[8] = state[8] * D[8] + sum;   state[9] = state[9] * D[9] + sum;
        state[10] = state[10] * D[10] + sum; state[11] = state[11] * D[11] + sum;
        state[12] = state[12] * D[12] + sum; state[13] = state[13] * D[13] + sum;
        state[14] = state[14] * D[14] + sum; state[15] = state[15] * D[15] + sum;
    }

    pow7add16(state, &C[86]);
    matmul_external16(state);
    pow7add16(state, &C[102]);
    matmul_external16(state);
    pow7add16(state, &C[118]);
    matmul_external16(state);
    pow7add16(state, &C[134]);
    matmul_external16(state);
    }
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::compress_w16_scalar_unrolled(
    Goldilocks::Element (&state)[CAPACITY],
    const Goldilocks::Element (&input)[SPONGE_WIDTH])
{
    if constexpr (SPONGE_WIDTH_T != 16) {
        abortMode("compress_w16_scalar_unrolled", Poseidon2Mode::ScalarUnrolledW16);
    } else {
    Goldilocks::Element aux[SPONGE_WIDTH];
    permute_w16_scalar_unrolled((Goldilocks::Element(&)[SPONGE_WIDTH])aux, input);
    std::memcpy(state, aux, CAPACITY * sizeof(Goldilocks::Element));
    }
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::linear_hash_w16_scalar_unrolled(
    Goldilocks::Element *output, Goldilocks::Element *input, uint64_t size)
{
    if constexpr (SPONGE_WIDTH_T != 16) {
        abortMode("linear_hash_w16_scalar_unrolled", Poseidon2Mode::ScalarUnrolledW16);
    } else {
    uint64_t remaining = size;
    Goldilocks::Element state[SPONGE_WIDTH];

    while (remaining)
    {
        if (remaining == size)
        {
            memset(state + RATE, 0, CAPACITY * sizeof(Goldilocks::Element));
        }
        else
        {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
            std::memcpy(state + RATE, state, CAPACITY * sizeof(Goldilocks::Element));
#pragma GCC diagnostic pop
        }

        uint64_t n = (remaining < RATE) ? remaining : RATE;
        memset(&state[n], 0, (RATE - n) * sizeof(Goldilocks::Element));
        std::memcpy(state, input + (size - remaining), n * sizeof(Goldilocks::Element));
        permute_w16_scalar_unrolled((Goldilocks::Element(&)[SPONGE_WIDTH])state,
                                   (const Goldilocks::Element(&)[SPONGE_WIDTH])state);
        remaining -= n;
    }
    if (size > 0)
    {
        std::memcpy(output, state, CAPACITY * sizeof(Goldilocks::Element));
    }
    else
    {
        memset(output, 0, CAPACITY * sizeof(Goldilocks::Element));
    }
    }
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::merkletree_w16_scalar_unrolled(
    Goldilocks::Element *tree, Goldilocks::Element *input,
    uint64_t num_cols, uint64_t num_rows, uint64_t arity,
    int num_threads, uint64_t dim)
{
    if constexpr (SPONGE_WIDTH_T != 16) {
        abortMode("merkletree_w16_scalar_unrolled", Poseidon2Mode::ScalarUnrolledW16);
    } else {
    if (num_rows == 0)
    {
        return;
    }

    Goldilocks::Element *cursor = tree;
    if (num_threads == 0)
        num_threads = omp_get_max_threads();

#pragma omp parallel for num_threads(num_threads)
    for (uint64_t i = 0; i < num_rows; i++)
    {
        linear_hash_w16_scalar_unrolled(&cursor[i * CAPACITY], &input[i * num_cols * dim], num_cols * dim);
    }

    uint64_t pending = num_rows;
    uint64_t nextN = (pending + (arity - 1)) / arity;
    uint64_t nextIndex = 0;

    while (pending > 1)
    {
        uint64_t extraZeros = (arity - (pending % arity)) % arity;
        if (extraZeros > 0)
        {
            std::memset(&cursor[nextIndex + pending * CAPACITY], 0, extraZeros * CAPACITY * sizeof(Goldilocks::Element));
        }

#pragma omp parallel for num_threads(num_threads)
        for (uint64_t i = 0; i < nextN; i++)
        {
            Goldilocks::Element pol_input[SPONGE_WIDTH];
            memset(pol_input, 0, SPONGE_WIDTH * sizeof(Goldilocks::Element));
            std::memcpy(pol_input, &cursor[nextIndex + i * SPONGE_WIDTH], SPONGE_WIDTH * sizeof(Goldilocks::Element));
            compress_w16_scalar_unrolled((Goldilocks::Element(&)[CAPACITY])cursor[nextIndex + (pending + extraZeros + i) * CAPACITY],
                                         (const Goldilocks::Element(&)[SPONGE_WIDTH])pol_input);
        }

        nextIndex += (pending + extraZeros) * CAPACITY;
        pending = (pending + (arity - 1)) / arity;
        nextN = (pending + (arity - 1)) / arity;
    }
    }
}

// ============================================================================
// NEON 2-sponge BATCH primitives (Part 5 Task 36).
//
// Layout: state[2 * SPONGE_WIDTH] = {sp0.x0..sp0.x{W-1}, sp1.x0..sp1.x{W-1}}
//   — two sponges back-to-back, mirroring AVX BATCH's 4-sponges contract
//   but with 2 sponges (NEON has 2 lanes per uint64x2_t vs AVX2's 4).
//
// Per-element regs: st[k] = {sp0.x_k, sp1.x_k}. With this layout, every
// Poseidon2 op (M4, pow7add, partial-round D-mul) becomes element-wise
// NEON across regs — NO lane shuffles within a reg, ever. This is the
// pattern that breaks the W=12/W=16 single-sponge regression: the matmul
// algebra (which mixes elements within a sponge) becomes adds across
// element-indexed regs, which NEON does in parallel for both sponges.
// ============================================================================

namespace Poseidon2Neon_batch {

// Strided load: gather lane 0 = base[0], lane 1 = base[stride]. Used to
// pick element_k from each of the 2 back-to-back sponges in `state[]`.
static inline uint64x2_t load_strided_2(const Goldilocks::Element* base, uint64_t stride) {
    uint64x2_t r = vsetq_lane_u64(base[0].fe, vdupq_n_u64(0), 0);
    return vsetq_lane_u64(base[stride].fe, r, 1);
}

static inline void store_strided_2(Goldilocks::Element* base, uint64_t stride, uint64x2_t v) {
    base[0].fe       = vgetq_lane_u64(v, 0);
    base[stride].fe  = vgetq_lane_u64(v, 1);
}

}  // namespace Poseidon2Neon_batch

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::permute_batch_neon(
    Goldilocks::Element *state, const Goldilocks::Element *input)
{
    namespace N = Goldilocks_neon;
    namespace B = Poseidon2Neon_batch;

    // Per-width round constants (same arrays as scalar / single-sponge).
    const Goldilocks::Element* C =
        SPONGE_WIDTH ==  4 ? Poseidon2GoldilocksConstants::C4  :
        SPONGE_WIDTH ==  8 ? Poseidon2GoldilocksConstants::C8  :
        SPONGE_WIDTH == 12 ? Poseidon2GoldilocksConstants::C12 :
                             Poseidon2GoldilocksConstants::C16;
    const Goldilocks::Element* D =
        SPONGE_WIDTH ==  4 ? Poseidon2GoldilocksConstants::D4  :
        SPONGE_WIDTH ==  8 ? Poseidon2GoldilocksConstants::D8  :
        SPONGE_WIDTH == 12 ? Poseidon2GoldilocksConstants::D12 :
                             Poseidon2GoldilocksConstants::D16;

    constexpr uint32_t W = SPONGE_WIDTH;

    // Lambdas factor the batch primitives without polluting the class scope.
    auto matmul_m4_batch = [](uint64x2_t& s0, uint64x2_t& s1, uint64x2_t& s2, uint64x2_t& s3) {
        // Same M4 algebra as scalar matmul_m4_, but each "scalar add" is a
        // NEON gl_add — both sponges processed in parallel with no shuffle.
        uint64x2_t t0 = N::gl_add(s0, s1);
        uint64x2_t t1 = N::gl_add(s2, s3);
        uint64x2_t two_s1 = N::gl_add(s1, s1);
        uint64x2_t t2 = N::gl_add(two_s1, t1);
        uint64x2_t two_s3 = N::gl_add(s3, s3);
        uint64x2_t t3 = N::gl_add(two_s3, t0);
        uint64x2_t t1_2 = N::gl_add(t1, t1);
        uint64x2_t t0_2 = N::gl_add(t0, t0);
        uint64x2_t t4 = N::gl_add(N::gl_add(t1_2, t1_2), t3);
        uint64x2_t t5 = N::gl_add(N::gl_add(t0_2, t0_2), t2);
        uint64x2_t t6 = N::gl_add(t3, t5);
        uint64x2_t t7 = N::gl_add(t2, t4);
        s0 = t6;  s1 = t5;  s2 = t7;  s3 = t4;
    };

    auto matmul_external_batch = [&matmul_m4_batch](uint64x2_t* x) {
        for (uint32_t i = 0; i < W; i += 4)
            matmul_m4_batch(x[i], x[i + 1], x[i + 2], x[i + 3]);
        if constexpr (W > 4) {
            uint64x2_t stored[4];
            stored[0] = N::gl_add(x[0], x[4]);
            stored[1] = N::gl_add(x[1], x[5]);
            stored[2] = N::gl_add(x[2], x[6]);
            stored[3] = N::gl_add(x[3], x[7]);
            for (uint32_t i = 8; i < W; i += 4) {
                stored[0] = N::gl_add(stored[0], x[i]);
                stored[1] = N::gl_add(stored[1], x[i + 1]);
                stored[2] = N::gl_add(stored[2], x[i + 2]);
                stored[3] = N::gl_add(stored[3], x[i + 3]);
            }
            for (uint32_t i = 0; i < W; ++i)
                x[i] = N::gl_add(x[i], stored[i % 4]);
        }
    };

    // Fused (state + C)^7 element-wise across both sponges.
    // C_ is canonical → use gl_add_c (4 ops, saves 3 vs gl_add).
    auto pow7add_batch = [](uint64x2_t* x, const Goldilocks::Element C_[W]) {
        for (uint32_t i = 0; i < W; ++i) {
            uint64x2_t c  = N::splat(C_[i].fe);
            uint64x2_t s  = N::gl_add_c(x[i], c);
            uint64x2_t s2 = N::gl_square(s);
            uint64x2_t s4 = N::gl_square(s2);
            uint64x2_t s3 = N::gl_mul(s, s2);
            x[i] = N::gl_mul(s3, s4);
        }
    };

    auto element_pow7_batch = [](uint64x2_t& x) {
        uint64x2_t pw2 = N::gl_square(x);
        uint64x2_t pw4 = N::gl_square(pw2);
        uint64x2_t pw3 = N::gl_mul(x, pw2);
        x = N::gl_mul(pw3, pw4);
    };

    // ---- Load 2 sponges into W NEON regs (one reg per element index) ----
    std::memcpy(state, input, 2 * W * sizeof(Goldilocks::Element));
    uint64x2_t st[W];
    for (uint32_t i = 0; i < W; ++i)
        st[i] = B::load_strided_2(&state[i], W);

    // Initial M_E.
    matmul_external_batch(st);

    // First half full rounds.
    for (uint32_t r = 0; r < HALF_N_FULL_ROUNDS; ++r) {
        pow7add_batch(st, &C[r * W]);
        matmul_external_batch(st);
    }

    // Partial rounds — both sponges' state[0] live in lane 0 / 1 of st[0],
    // so element_pow7 across both lanes does both partial S-boxes at once.
    uint64x2_t d[W];
    for (uint32_t i = 0; i < W; ++i)
        d[i] = N::splat(D[i].fe);

    for (uint32_t r = 0; r < N_PARTIAL_ROUNDS; ++r) {
        // c is canonical (compile-time table) → gl_add_c (4 ops).
        uint64x2_t c = N::splat(C[HALF_N_FULL_ROUNDS * W + r].fe);
        st[0] = N::gl_add_c(st[0], c);
        element_pow7_batch(st[0]);
        uint64x2_t sum = N::splat(0);
        for (uint32_t i = 0; i < W; ++i)
            sum = N::gl_add(sum, st[i]);
        for (uint32_t i = 0; i < W; ++i) {
            st[i] = N::gl_mul(st[i], d[i]);
            st[i] = N::gl_add(st[i], sum);
        }
    }

    // Second half full rounds.
    for (uint32_t r = 0; r < HALF_N_FULL_ROUNDS; ++r) {
        pow7add_batch(st, &C[HALF_N_FULL_ROUNDS * W + N_PARTIAL_ROUNDS + r * W]);
        matmul_external_batch(st);
    }

    // Store result back to {sp0, sp1} consecutive layout.
    for (uint32_t i = 0; i < W; ++i)
        B::store_strided_2(&state[i], W, st[i]);
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::compress_batch_neon(
    Goldilocks::Element (&state)[2 * CAPACITY],
    Goldilocks::Element const (&input)[2 * SPONGE_WIDTH])
{
    Goldilocks::Element aux[2 * SPONGE_WIDTH];
    permute_batch_neon(aux, input);
    // First CAPACITY elements of each permuted sponge.
    std::memcpy(&state[0],          &aux[0],            CAPACITY * sizeof(Goldilocks::Element));
    std::memcpy(&state[CAPACITY],   &aux[SPONGE_WIDTH], CAPACITY * sizeof(Goldilocks::Element));
}

template<uint32_t SPONGE_WIDTH_T>
inline void Poseidon2Goldilocks<SPONGE_WIDTH_T>::compress_neon(
    Goldilocks::Element (&state)[CAPACITY],
    Goldilocks::Element const (&input)[SPONGE_WIDTH])
{
    Goldilocks::Element aux[SPONGE_WIDTH];
    permute_neon(aux, input);
    std::memcpy(state, aux, CAPACITY * sizeof(Goldilocks::Element));
}

#endif  // PIL2_HAS_NEON
#endif  // POSEIDON2_GOLDILOCKS_NEON
