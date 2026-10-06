#ifndef PILFFLONK_MSM_HPP
#define PILFFLONK_MSM_HPP

#include <cstdint>

#include "alt_bn128.hpp"

namespace PilFflonk {

// Σ_i scalars[i]·bases[i] over BN128 G1, for n canonical (< r, not Montgomery) little-endian
// scalars of 4 limbs and affine bases in ffiasm's Montgomery form: a Pippenger with signed digits
// whose buckets are affine, added in batches with one inversion each. The same point as ffiasm's
// multiMulByScalar, on all of OpenMP's threads.
AltBn128::Engine::G1Point msm(const AltBn128::Engine::G1PointAffine *bases, const uint64_t *scalars, uint64_t n);

} // namespace PilFflonk

#endif
