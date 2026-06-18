#ifndef POSEIDON2_GOLDILOCKS
#define POSEIDON2_GOLDILOCKS

#include <vector>
#include "poseidon2_goldilocks_constants.hpp"
#include "goldilocks_base_field.hpp"
#ifdef __AVX2__
    #include <immintrin.h>
#endif

#define WIDTH 16

inline void pow7(Goldilocks::Element &x)
{
    Goldilocks::Element x2 = x * x;
    Goldilocks::Element x3 = x * x2;
    Goldilocks::Element x4 = x2 * x2;
    x = x3 * x4;
};

inline void add_(Goldilocks::Element &x, const Goldilocks::Element *st)
{
    for (int i = 0; i < WIDTH; ++i)
    {
        x = x + st[i];
    }
}
inline void prodadd_(Goldilocks::Element *x, const Goldilocks::Element *D, const Goldilocks::Element &sum)
{
    for (int i = 0; i < WIDTH; ++i)
    {
        x[i] = x[i]*D[i] + sum;
    }
}

inline void pow7add_(Goldilocks::Element *x, const Goldilocks::Element *C)
{
    Goldilocks::Element x2[WIDTH], x3[WIDTH], x4[WIDTH];
    
    for (int i = 0; i < WIDTH; ++i)
    {
        Goldilocks::Element xi = x[i] + C[i];
        x2[i] = xi * xi;
        x3[i] = xi * x2[i];
        x4[i] = x2[i] * x2[i];
        x[i] = x3[i] * x4[i];
    }
};

inline void matmul_m4_(Goldilocks::Element *x) {
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
}

inline void matmul_external_(Goldilocks::Element *x) {
    for (int i = 0; i < WIDTH/4; ++i) {
        matmul_m4_(&x[i*4]);
    }
    
    Goldilocks::Element stored[4] = {Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero()};

    for(int i = 0; i < 4; ++i) {
        for (int j = 0; j < WIDTH/4; ++j) {
            stored[i] = stored[i] + x[j*4 + i];
        }
    }
    
    for (int i = 0; i < WIDTH; ++i)
    {
        x[i] = x[i] + stored[i % 4];
    }
}

#ifdef __AVX2__
const __m256i zero = _mm256_setzero_si256();

inline void add_avx_small(__m256i st[], const Goldilocks::Element C_small[])
{
    size_t num_vectors = WIDTH / 4;
    for (size_t i = 0; i < num_vectors; i++)
    {
        __m256i c;
        Goldilocks::load_avx(c, &(C_small[4*i]));
        Goldilocks::add_avx_b_small(st[i], st[i], c);
    }
}

inline void matmul_external_avx(__m256i st[])
{
    assert(WIDTH == 12 || WIDTH == 16);
# if WIDTH == 12 
    __m256i t0_ = _mm256_permute2f128_si256(st[0], st[2], 0b00100000);
    __m256i t1_ = _mm256_permute2f128_si256(st[1], zero, 0b00100000);
    __m256i t2_ = _mm256_permute2f128_si256(st[0], st[2], 0b00110001);
    __m256i t3_ = _mm256_permute2f128_si256(st[1], zero, 0b00110001);
    __m256i x0 = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t0_), _mm256_castsi256_pd(t1_)));
    __m256i x1 = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t0_), _mm256_castsi256_pd(t1_)));
    __m256i x2 = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t2_), _mm256_castsi256_pd(t3_)));
    __m256i x3 = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t2_), _mm256_castsi256_pd(t3_)));
#else
    __m256i t0_ = _mm256_permute2f128_si256(st[0], st[2], 0b00100000);
    __m256i t1_ = _mm256_permute2f128_si256(st[1], st[3], 0b00100000);
    __m256i t2_ = _mm256_permute2f128_si256(st[0], st[2], 0b00110001);
    __m256i t3_ = _mm256_permute2f128_si256(st[1], st[3], 0b00110001);
    __m256i x0 = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t0_), _mm256_castsi256_pd(t1_)));
    __m256i x1 = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t0_), _mm256_castsi256_pd(t1_)));
    __m256i x2 = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t2_), _mm256_castsi256_pd(t3_)));
    __m256i x3 = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t2_), _mm256_castsi256_pd(t3_)));
#endif

   __m256i t0, t0_2, t1, t1_2, t2, t3, t4, t5, t6, t7;
    Goldilocks::add_avx(t0, x0, x1);
    Goldilocks::add_avx(t1, x2, x3);
    Goldilocks::add_avx(t2, x1, x1);
    Goldilocks::add_avx(t2, t2, t1);
    Goldilocks::add_avx(t3, x3, x3);
    Goldilocks::add_avx(t3, t3, t0);
    Goldilocks::add_avx(t1_2, t1, t1);
    Goldilocks::add_avx(t0_2, t0, t0);
    Goldilocks::add_avx(t4, t1_2, t1_2);
    Goldilocks::add_avx(t4, t4, t3);
    Goldilocks::add_avx(t5, t0_2, t0_2);
    Goldilocks::add_avx(t5, t5, t2);
    Goldilocks::add_avx(t6, t3, t5);
    Goldilocks::add_avx(t7, t2, t4);

#if SPONGE_WIDTH == 12
    t0_ = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t6), _mm256_castsi256_pd(t5)));
    t1_ = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t6), _mm256_castsi256_pd(t5)));
    t2_ = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t7), _mm256_castsi256_pd(t4)));
    t3_ = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t7), _mm256_castsi256_pd(t4)));

    // Step 2: Reverse _mm256_permute2f128_si256
    st[0] = _mm256_permute2f128_si256(t0_, t2_, 0b00100000); // Combine low halves
    st[2] = _mm256_permute2f128_si256(t0_, t2_, 0b00110001); // Combine high halves
    st[1] = _mm256_permute2f128_si256(t1_, t3_, 0b00100000); // Combine low halves
#else
    t0_ = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t6), _mm256_castsi256_pd(t5)));
    t1_ = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t6), _mm256_castsi256_pd(t5)));
    t2_ = _mm256_castpd_si256(_mm256_unpacklo_pd(_mm256_castsi256_pd(t7), _mm256_castsi256_pd(t4)));
    t3_ = _mm256_castpd_si256(_mm256_unpackhi_pd(_mm256_castsi256_pd(t7), _mm256_castsi256_pd(t4)));

    // Step 2: Reverse _mm256_permute2f128_si256
    st[0] = _mm256_permute2f128_si256(t0_, t2_, 0b00100000); // Combine low halves
    st[2] = _mm256_permute2f128_si256(t0_, t2_, 0b00110001); // Combine high halves
    st[1] = _mm256_permute2f128_si256(t1_, t3_, 0b00100000); // Combine low halves
    st[3] = _mm256_permute2f128_si256(t1_, t3_, 0b00110001); // Combine high halves
#endif

    __m256i stored;
    Goldilocks::add_avx(stored, st[0], st[1]);
    for(int i = 2; i < (WIDTH >> 2); i++) {
        Goldilocks::add_avx(stored, stored, st[i]);            
    }

    for(int i = 0; i < (WIDTH >> 2); i++) {
        Goldilocks::add_avx(st[i], st[i], stored);
    }
}


inline void pow7_avx(__m256i st[])
{
    __m256i pw2[WIDTH/4], pw3[WIDTH/4], pw4[WIDTH/4];

    // pw2 = st^2
    for (size_t i = 0; i < WIDTH/4; i++)
        Goldilocks::square_avx(pw2[i], st[i]);

    // pw4 = pw2^2 = st^4
    for (size_t i = 0; i < WIDTH/4; i++)
        Goldilocks::square_avx(pw4[i], pw2[i]);

    // pw3 = st * pw2 = st^3
    for (size_t i = 0; i < WIDTH/4; i++)
        Goldilocks::mult_avx(pw3[i], st[i], pw2[i]);

    // st = pw3 * pw4 = st^7
    for (size_t i = 0; i < WIDTH/4; i++)
        Goldilocks::mult_avx(st[i], pw3[i], pw4[i]);
}

#endif

void Poseidon2(Goldilocks::Element *state, uint64_t* im)
{   
    const Goldilocks::Element *RC = WIDTH == 12 ? Poseidon2GoldilocksConstants::C12 : Poseidon2GoldilocksConstants::C16;
    const Goldilocks::Element *D = WIDTH == 12 ? Poseidon2GoldilocksConstants::D16 : Poseidon2GoldilocksConstants::D16;

    uint64_t index = 0;
#ifdef __AVX2__
    __m256i st[WIDTH/4];
    for (int i = 0; i < WIDTH/4; i++) {
        Goldilocks::load_avx(st[i], &(state[4*i]));
    }

    matmul_external_avx(st);

    for (int i = 0; i < WIDTH/4; i++) {
        Goldilocks::store_avx(&(state[4*i]), st[i]);
    }
    
    for (int i = 0; i < WIDTH; i++) {
        im[index++] = Goldilocks::toU64(state[i]);
    }

    for (int r = 0; r < 4; r++)
    {
        add_avx_small(st, &(RC[WIDTH * r]));
        pow7_avx(st);
        matmul_external_avx(st);

        for (int i = 0; i < WIDTH/4; i++) {
            Goldilocks::store_avx(&(state[4*i]), st[i]);
        }
        
        for (int i = 0; i < WIDTH; i++) {
            im[index++] = Goldilocks::toU64(state[i]);
        }
    }
    
    Goldilocks::store_avx(&(state[0]), st[0]);
    Goldilocks::Element state0_ = state[0];

    __m256i d[WIDTH/4];
    for (int i = 0; i < WIDTH/4; i++) {
        Goldilocks::load_avx(d[i], &(D[4*i]));
    }

    __m256i part_sum;
    Goldilocks::Element partial_sum[4];
    Goldilocks::Element aux = state0_;
    for (int r = 0; r < 22; r++)
    {
        Goldilocks::add_avx(part_sum, st[0], st[1]);
        Goldilocks::add_avx(part_sum, part_sum, st[2]);
        Goldilocks::add_avx(part_sum, part_sum, st[3]);
        Goldilocks::store_avx(partial_sum, part_sum);
        Goldilocks::Element sum = partial_sum[0] + partial_sum[1] + partial_sum[2] + partial_sum[3];
        sum = sum - aux;

        im[index++] = Goldilocks::toU64(state0_);
        state0_ = state0_ + RC[4 * WIDTH + r];
        pow7(state0_);
        sum = sum + state0_;    
            
        __m256i scalar1 = _mm256_set1_epi64x(sum.fe);
        for (int i = 0; i < WIDTH/4; i++) {
            Goldilocks::mult_avx(st[i], st[i], d[i]);
            Goldilocks::add_avx(st[i], st[i], scalar1);
        }
        state0_ = state0_ * D[0] + sum;
        aux = aux * D[0] + sum;
        if (r == 10 || r == 21) {
            for (int i = 11; i < WIDTH; i++) {
                im[index++] = 0;
            }
            
            for (int i = 0; i < WIDTH/4; i++) {
                Goldilocks::store_avx(&(state[4*i]), st[i]);
            }
            
            im[index++] = Goldilocks::toU64(state0_);
            for (int i = 1; i < WIDTH; i++) {
                im[index++] = Goldilocks::toU64(state[i]);
            }
        }
    }

    Goldilocks::store_avx(&(state[0]), st[0]);
    state[0] = state0_;
    Goldilocks::load_avx(st[0], &(state[0]));

    for (int r = 0; r < 4; r++)
    {
        add_avx_small(st, &(RC[4 * WIDTH + 22 + WIDTH * r]));
        pow7_avx(st);
        
        matmul_external_avx(st);

        if (r < 3) {
            for (int i = 0; i < WIDTH/4; i++) {
                Goldilocks::store_avx(&(state[4*i]), st[i]);
            }
            
            for (int i = 0; i < WIDTH; i++) {
                im[index++] = Goldilocks::toU64(state[i]);
            }
        }
    }
    
    for (int i = 0; i < WIDTH/4; i++) {
        Goldilocks::store_avx(&(state[4*i]), st[i]);
    }
#else
    matmul_external_(state);
    
    for(uint64_t i = 0; i < WIDTH; ++i) {
        im[index++] = Goldilocks::toU64(state[i]);
    }

    for (int r = 0; r < 4; r++)
    {
        pow7add_(state, &(RC[WIDTH * r]));
        matmul_external_(state);
        for(uint64_t i = 0; i < WIDTH; ++i) {
            im[index++] = Goldilocks::toU64(state[i]);
        }
    }

    for (int r = 0; r < 22; r++)
    {
        im[index++] = Goldilocks::toU64(state[0]);
        state[0] = state[0] + RC[4 * WIDTH + r];
        pow7(state[0]);
        Goldilocks::Element sum_ = Goldilocks::zero();
        add_(sum_, state);
        prodadd_(state, D, sum_);
        if (r == 10 || r == 21) {
            for (int i = 11; i < WIDTH; i++) {
                im[index++] = 0;
            }
            for (int i = 0; i < WIDTH; i++) {
                im[index++] = Goldilocks::toU64(state[i]);
            }
        }
    }

    for (int r = 0; r < 4; r++)
    {
        pow7add_(state, &(RC[4 * WIDTH + 22 + r * WIDTH]));
        matmul_external_(state);
        if(r < 3) {
            for(uint64_t i = 0; i < WIDTH; ++i) {
                im[index++] = Goldilocks::toU64(state[i]);
            }
        }
    }
#endif
}


void Poseidon2_16(uint64_t *im,uint *size_im,uint64_t *out, uint* size_out,uint64_t *in, uint *size_in)
{
    Goldilocks::Element state[16];
    for(uint64_t i = 0; i < 16; ++i) {
        state[i] = Goldilocks::fromU64(in[i]);
    }
    Poseidon2(state, im);

    for(uint64_t i = 0; i < WIDTH; ++i) {
        out[i] = Goldilocks::toU64(state[i]);
    }
}

void CustPoseidon2_16(uint64_t *im,uint *size_im,uint64_t *out, uint* size_out,uint64_t *in, uint *size_in,uint64_t *key, uint *size_key)
{   
    Goldilocks::Element state[16];
    if (key[0] == 0 && key[1] == 0) {
        for(uint64_t i = 0; i < 16; ++i) {
            state[i] = Goldilocks::fromU64(in[i]);
        }
    } else if (key[0] == 1 && key[1] == 0) {
        state[0] = Goldilocks::fromU64(in[4]);
        state[1] = Goldilocks::fromU64(in[5]);
        state[2] = Goldilocks::fromU64(in[6]);
        state[3] = Goldilocks::fromU64(in[7]);
        state[4] = Goldilocks::fromU64(in[0]);
        state[5] = Goldilocks::fromU64(in[1]);
        state[6] = Goldilocks::fromU64(in[2]);
        state[7] = Goldilocks::fromU64(in[3]);
        state[8] = Goldilocks::fromU64(in[8]);
        state[9] = Goldilocks::fromU64(in[9]);
        state[10] = Goldilocks::fromU64(in[10]);
        state[11] = Goldilocks::fromU64(in[11]);
        state[12] = Goldilocks::fromU64(in[12]);
        state[13] = Goldilocks::fromU64(in[13]);
        state[14] = Goldilocks::fromU64(in[14]);
        state[15] = Goldilocks::fromU64(in[15]);
    } else if (key[0] == 0 && key[1] == 1) {
        state[0] = Goldilocks::fromU64(in[4]);
        state[1] = Goldilocks::fromU64(in[5]);
        state[2] = Goldilocks::fromU64(in[6]);
        state[3] = Goldilocks::fromU64(in[7]);
        state[4] = Goldilocks::fromU64(in[8]);
        state[5] = Goldilocks::fromU64(in[9]);
        state[6] = Goldilocks::fromU64(in[10]);
        state[7] = Goldilocks::fromU64(in[11]);
        state[8] = Goldilocks::fromU64(in[0]);
        state[9] = Goldilocks::fromU64(in[1]);
        state[10] = Goldilocks::fromU64(in[2]);
        state[11] = Goldilocks::fromU64(in[3]);
        state[12] = Goldilocks::fromU64(in[12]);
        state[13] = Goldilocks::fromU64(in[13]);
        state[14] = Goldilocks::fromU64(in[14]);
        state[15] = Goldilocks::fromU64(in[15]);
    } else {
        state[0] = Goldilocks::fromU64(in[4]);
        state[1] = Goldilocks::fromU64(in[5]);
        state[2] = Goldilocks::fromU64(in[6]);
        state[3] = Goldilocks::fromU64(in[7]);
        state[4] = Goldilocks::fromU64(in[8]);
        state[5] = Goldilocks::fromU64(in[9]);
        state[6] = Goldilocks::fromU64(in[10]);
        state[7] = Goldilocks::fromU64(in[11]);
        state[8] = Goldilocks::fromU64(in[12]);
        state[9] = Goldilocks::fromU64(in[13]);
        state[10] = Goldilocks::fromU64(in[14]);
        state[11] = Goldilocks::fromU64(in[15]);
        state[12] = Goldilocks::fromU64(in[0]);
        state[13] = Goldilocks::fromU64(in[1]);
        state[14] = Goldilocks::fromU64(in[2]);
        state[15] = Goldilocks::fromU64(in[3]);
    }

    Poseidon2(state, im);

    for(uint64_t i = 0; i < WIDTH; ++i) {
        out[i] = Goldilocks::toU64(state[i]);
    }
}
#endif