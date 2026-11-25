#ifndef STEPS_HPP
#define STEPS_HPP

#pragma once 

#include <cstdint>

struct StepsParams
{
    Goldilocks::Element *trace;
    Goldilocks::Element *aux_trace;
    Goldilocks::Element *publicInputs;
    Goldilocks::Element *proofValues;
    Goldilocks::Element *challenges;
    Goldilocks::Element *airgroupValues;
    Goldilocks::Element *airValues;
    Goldilocks::Element *evals;
    Goldilocks::Element *pConstPolsAddress;
    Goldilocks::Element *pConstPolsExtendedTreeAddress;
    Goldilocks::Element *pCustomCommitsFixed;
    // Operations staged in `trace` when this air's witness comes from a GPU kernel:
    // `trace` then holds the kernel's inputs, not a trace, and this is how many.
    // 0 for every ordinary air.
    uint64_t witnessOps;
};

// Stamped into every generated .exps.so (exps_abi); bump on any StepsParams or exps_* launcher change.
#define EXPS_ABI_VERSION 1ull
#define EXPS_ABI ((EXPS_ABI_VERSION << 32) | sizeof(StepsParams))

#endif