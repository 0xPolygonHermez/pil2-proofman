#ifndef SPPARK_LDE_CUH
#define SPPARK_LDE_CUH

#include <cstdint>

// sppark-backed NTT primitives for the FLAT column-major trace layout (Layout::ColMajor).
//
// Defined in sppark_lde.cu, an ISOLATED TU (sppark's headers clash with ntt_goldilocks.cuh), so callers
// see only these C-linkage prototypes.
//
// All operate on the prover arena in flat column-major form (column c of an N-row section is base + c*N,
// sppark's native per-column format -- no transpose needed). Each host-synchronizes before returning, so
// they are illegal to issue mid CUDA-graph-capture (callers gate on resolveLayout).
extern "C" {

// LDE of nCols flat columns: iNTT(N) -> coset-spread -> NTT(Next), column c written to d_dst at c*Next.
// preserve_src keeps the small-domain src intact (the iNTT is otherwise in place); preserve_scratch is
// an optional N-element caller buffer (disjoint from src/dst) for the iNTT copy -- null = internal buffer.
void sppark_lde_flat(void* d_dst, void* d_src, uint32_t lg_n, uint32_t lg_next, uint32_t nCols,
                     bool preserve_src, void* preserve_scratch);

// computeQ: iNTT(ext) each q column -> coset shift + zero-pad into cmQ -> NTT(ext) each cmQ column.
// q and cmQ are disjoint flat regions of the arena at off_q / off_cmQ (in elements).
void sppark_computeq_flat(void* d_aux, uint64_t off_cmQ, uint64_t off_q,
                          uint32_t qDeg, uint32_t qDim, uint64_t shiftIn,
                          uint32_t lg_n, uint32_t lg_next, uint32_t nCols);

// In-place INTT of nCols flat columns of 2^lg_n rows (column c at d_data + c*2^lg_n). Used for the LEv
// vector (the native flat NTT path has no col<->row transpose and would mangle column-major LEv).
void sppark_intt_flat(void* d_data, uint32_t lg_n, uint32_t nCols);

} // extern "C"

#endif // SPPARK_LDE_CUH
