#ifndef TEST_HELPERS_HPP
#define TEST_HELPERS_HPP

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <cstring>

#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon2_goldilocks.hpp"
#include "../src/ntt_goldilocks.hpp"
#include "../src/merklehash_goldilocks.hpp"

#ifdef __AVX2__
#include <immintrin.h>
#endif

// Golden hash values for Poseidon2 hash(input=[0,1,...,W-1]).
// These are the cryptographic ground truth — must never change.
namespace GoldilocksTestData {

constexpr uint64_t HASH_W4_GOLDEN[4]  = { 0x758085b0af0a16aa, 0x85141acc29c479de,
                                           0x50127371e2b77ae5, 0xefee3a8033630029 };
constexpr uint64_t HASH_W8_GOLDEN[4]  = { 0xc5fb1cfe0b4697bb, 0x4a4a32ff849af473,
                                           0xd2fd266077f8efba, 0xf4ad9b74e833916d };
constexpr uint64_t HASH_W12_GOLDEN[4] = { 0x01eaef96bdf1c0c1, 0x1f0d2cc525b2540c,
                                           0x6282c1dfe1e0358d, 0xe780d721f698e1e6 };
constexpr uint64_t HASH_W16_GOLDEN[4] = { 0x85c54702470d9756, 0xaa53c7a7d52d9898,
                                           0x285128096efb0dd7, 0xf3fde5edd3050ac8 };

} // namespace GoldilocksTestData

#endif // TEST_HELPERS_HPP
