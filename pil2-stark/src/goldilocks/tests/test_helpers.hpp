#ifndef TEST_HELPERS_HPP
#define TEST_HELPERS_HPP

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <cstring>

#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon2_goldilocks.hpp"
#include "../src/goldilocks_tooling.hpp"
#include "../src/ntt_goldilocks.hpp"

#ifdef __AVX2__
#include <immintrin.h>
#endif

// Golden hash values — cryptographic ground truth. Input=[0,1,...,W-1].
namespace GoldilocksTestData {

// permuteTrunc() golden: first CAPACITY elements of full permutation output
constexpr uint64_t PERMUTE_TRUNC_W4_GOLDEN[4]  = { 0x758085b0af0a16aa, 0x85141acc29c479de,
                                           0x50127371e2b77ae5, 0xefee3a8033630029 };
constexpr uint64_t PERMUTE_TRUNC_W8_GOLDEN[4]  = { 0xc5fb1cfe0b4697bb, 0x4a4a32ff849af473,
                                           0xd2fd266077f8efba, 0xf4ad9b74e833916d };
constexpr uint64_t PERMUTE_TRUNC_W12_GOLDEN[4] = { 0x01eaef96bdf1c0c1, 0x1f0d2cc525b2540c,
                                           0x6282c1dfe1e0358d, 0xe780d721f698e1e6 };
constexpr uint64_t PERMUTE_TRUNC_W16_GOLDEN[4] = { 0x85c54702470d9756, 0xaa53c7a7d52d9898,
                                           0x285128096efb0dd7, 0xf3fde5edd3050ac8 };

// permute() golden: full SPONGE_WIDTH output (input=[0,1,...,W-1])
constexpr uint64_t PERMUTE_W4_GOLDEN[4]  = { 0x758085b0af0a16aa, 0x85141acc29c479de,
                                                0x50127371e2b77ae5, 0xefee3a8033630029 };
constexpr uint64_t PERMUTE_W8_GOLDEN[8]  = { 0xc5fb1cfe0b4697bb, 0x4a4a32ff849af473,
                                                0xd2fd266077f8efba, 0xf4ad9b74e833916d,
                                                0xe6648eb0acc11463, 0x8d5529a930d75194,
                                                0xe8c993aa10da6c90, 0xa73104a95b68031c };
constexpr uint64_t PERMUTE_W12_GOLDEN[12] = { 0x01eaef96bdf1c0c1, 0x1f0d2cc525b2540c,
                                                 0x6282c1dfe1e0358d, 0xe780d721f698e1e6,
                                                 0x280c0b6f753d833b, 0x1b942dd5023156ab,
                                                 0x43f0df3fcccb8398, 0xe8e8190585489025,
                                                 0x56bdbf72f77ada22, 0x7911c32bf9dcd705,
                                                 0xec467926508fbe67, 0x6a50450ddf85a6ed };
constexpr uint64_t PERMUTE_W16_GOLDEN[16] = { 0x85c54702470d9756, 0xaa53c7a7d52d9898,
                                                 0x285128096efb0dd7, 0xf3fde5edd3050ac8,
                                                 0xc7b65efd040df908, 0x4be3f6c467f57ae9,
                                                 0x274e9a67b41754fb, 0x0f7d39cd5de94dac,
                                                 0xd0224b9794d0b78c, 0x372f6139570042e1,
                                                 0xce6e8a93dc4ec26c, 0xace65e30a4daf7af,
                                                 0x016f2824cc1ba3db, 0x2e8f3af37c434dec,
                                                 0xc80831bb6e09da01, 0x3a7d670bf1a86ee8 };

// linearHash() golden: input=[1,2,...,20], size=20
constexpr uint64_t LINEAR_HASH_W8_GOLDEN[4]  = { 0x507b25d95a257597, 0x830676083370cbfc,
                                                  0xb00f1c8d58001898, 0x77b4b74befb91a9e };
constexpr uint64_t LINEAR_HASH_W12_GOLDEN[4] = { 0x61f0ba9a1d28a9de, 0x3a208a00a0edc70a,
                                                  0x69f46857c1c44e77, 0x9853053f65cb2381 };
constexpr uint64_t LINEAR_HASH_W16_GOLDEN[4] = { 0x3fd2692fa8954a7e, 0x3a53be1d99d262a6,
                                                  0xa51078560d69a26e, 0xf6a0e953979b19d2 };

// merkletreeReduce() golden: 3 digests [1..12], arity=3, W=12
constexpr uint64_t MERKLETREE_REDUCE_W12_AR3_GOLDEN[4] = { 0x22bef4680f2270e2, 0xd2ac860d60a049ad,
                                                            0x5ae1a12c645b38db, 0x6e2acfa658c210e3 };

} // namespace GoldilocksTestData

#endif // TEST_HELPERS_HPP
