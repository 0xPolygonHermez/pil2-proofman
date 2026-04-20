#ifndef PIL2_MERKLE_METAL_BRIDGE_HPP
#define PIL2_MERKLE_METAL_BRIDGE_HPP

#include "../platform.hpp"

#if PIL2_HAS_METAL

#include "metal_c_api.h"
#include "../goldilocks_base_field.hpp"
#include "../poseidon2_goldilocks_constants.hpp"
#include <cstdint>
#include <vector>

namespace pil2 { namespace metal {

// Adapters from Poseidon2Goldilocks<W>::merkletree(nodes, source, num_cols,
// num_rows, arity) to pil2_metal_merkletree_w{8,12,16}_cols. Build the C
// and D round-constant tables from Poseidon2GoldilocksConstants as
// canonical u64 and hand them to the C API. Bit-exact with the CPU
// reference for any num_rows >= 1 (kernel handles per-level zero-padding
// to match Poseidon2Goldilocks<W>::merkletree_seq).
inline void merkletree_w16_via_metal(Goldilocks::Element* tree,
                                     const Goldilocks::Element* source,
                                     uint64_t num_cols,
                                     uint64_t num_rows) {
    static_assert(sizeof(Goldilocks::Element) == sizeof(uint64_t),
                  "Goldilocks::Element must be layout-compatible with uint64_t");
    std::vector<uint64_t> C(150);
    std::vector<uint64_t> D(16);
    for (uint64_t i = 0; i < 150; ++i) C[i] = Goldilocks::toU64(Poseidon2GoldilocksConstants::C16[i]);
    for (uint64_t i = 0; i < 16;  ++i) D[i] = Goldilocks::toU64(Poseidon2GoldilocksConstants::D16[i]);
    pil2_metal_merkletree_w16_cols(
        reinterpret_cast<uint64_t*>(tree),
        reinterpret_cast<const uint64_t*>(source),
        num_cols, num_rows, C.data(), D.data());
}

inline void merkletree_w12_via_metal(Goldilocks::Element* tree,
                                     const Goldilocks::Element* source,
                                     uint64_t num_cols,
                                     uint64_t num_rows) {
    std::vector<uint64_t> C(118);
    std::vector<uint64_t> D(12);
    for (uint64_t i = 0; i < 118; ++i) C[i] = Goldilocks::toU64(Poseidon2GoldilocksConstants::C12[i]);
    for (uint64_t i = 0; i < 12;  ++i) D[i] = Goldilocks::toU64(Poseidon2GoldilocksConstants::D12[i]);
    pil2_metal_merkletree_w12_cols(
        reinterpret_cast<uint64_t*>(tree),
        reinterpret_cast<const uint64_t*>(source),
        num_cols, num_rows, C.data(), D.data());
}

inline void merkletree_w8_via_metal(Goldilocks::Element* tree,
                                    const Goldilocks::Element* source,
                                    uint64_t num_cols,
                                    uint64_t num_rows) {
    std::vector<uint64_t> C(86);
    std::vector<uint64_t> D(8);
    for (uint64_t i = 0; i < 86; ++i) C[i] = Goldilocks::toU64(Poseidon2GoldilocksConstants::C8[i]);
    for (uint64_t i = 0; i < 8;  ++i) D[i] = Goldilocks::toU64(Poseidon2GoldilocksConstants::D8[i]);
    pil2_metal_merkletree_w8_cols(
        reinterpret_cast<uint64_t*>(tree),
        reinterpret_cast<const uint64_t*>(source),
        num_cols, num_rows, C.data(), D.data());
}

}} // namespace pil2::metal

#endif // PIL2_HAS_METAL
#endif // PIL2_MERKLE_METAL_BRIDGE_HPP
