#ifndef GOLDILOCKS_HASH_CONFIG_HPP
#define GOLDILOCKS_HASH_CONFIG_HPP

// ---------------------------------------------------------------------------
// Goldilocks hash family configuration -- single source of truth.
//
//   USE_DM_HASH       Default Davies-Meyer mode for every Poseidon{1,2}{,GPU}
//                     class template (used as `bool DM_T = USE_DM_HASH`).
//                     Each Poseidon class header repeats this define under
//                     `#ifndef USE_DM_HASH` as a safety fallback for
//                     standalone test/bench TUs that do not pull in this
//                     header. The two values MUST match if flipped.
//
//   STARK_POSEIDON1   When defined, selects the Poseidon1. When undefined, 
//                     the Poseidon2 family is used everywhere.
//                     Set via the build (-DSTARK_POSEIDON1).
//
//   GoldilocksGrinding  Type alias resolved from the active family. Use it
//                       at call sites that need to be family-agnostic
//                       (e.g. the grinding helper in the public API).
//
// Flipping either USE_DM_HASH or STARK_POSEIDON1 is a HARD FORK of the proof
// system -- every prior merkle root, transcript and proving key becomes
// invalid.
// ---------------------------------------------------------------------------

#ifndef USE_DM_HASH
#define USE_DM_HASH true
#endif

#ifdef STARK_POSEIDON1
#include "poseidon_goldilocks.hpp"
using GoldilocksGrinding = PoseidonGoldilocks<8>;
#else
#include "poseidon2_goldilocks.hpp"
using GoldilocksGrinding = Poseidon2GoldilocksGrinding;
#endif

#endif  // GOLDILOCKS_HASH_CONFIG_HPP
