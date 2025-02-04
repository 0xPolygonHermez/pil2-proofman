#ifndef ZKGLOBALS_HPP
#define ZKGLOBALS_HPP

#include "goldilocks_base_field.hpp"
#include "poseidon2_goldilocks.hpp"
#include "ffiasm/fec.hpp"
#include "ffiasm/fnec.hpp"
#include "ffiasm/fr.hpp"
#include "ffiasm/fq.hpp"

extern Goldilocks fr;
extern Poseidon2Goldilocks poseidon;
extern RawFec fec;
extern RawFnec fnec;
extern RawFr bn128;
extern RawFq fq;

#endif