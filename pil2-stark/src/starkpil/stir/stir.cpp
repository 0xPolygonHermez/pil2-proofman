// Explicit instantiation of the STIR prover for the Goldilocks hash family, so the whole
// template is type-checked in the library build even before it is wired into `gen_proof`.
#include "stir.hpp"

template class STIR<Goldilocks::Element>;
template struct StirProof<Goldilocks::Element>;
