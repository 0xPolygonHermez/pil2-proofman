#ifndef ZKGLOBALS_HPP
#define ZKGLOBALS_HPP

#include "goldilocks_base_field.hpp"
#include "poseidon2_goldilocks.hpp"
#include "ffiasm/fec.hpp"
#include "ffiasm/fnec.hpp"
#include "ffiasm/fr.hpp"
#include "ffiasm/fq.hpp"

extern Goldilocks fr;
extern RawFecP fec;
extern RawFnecP fnec;
extern RawFrP bn128;
extern RawFqP fq;

#endif