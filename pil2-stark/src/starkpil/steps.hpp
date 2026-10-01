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
    Goldilocks::Element *xDivXSub;
    Goldilocks::Element *pConstPolsAddress;
    Goldilocks::Element *pConstPolsExtendedTreeAddress;
    Goldilocks::Element *pCustomCommitsFixed;
    // Operations staged in `trace` when this air's witness comes from a GPU kernel:
    // `trace` then holds the kernel's inputs, not a trace, and this is how many.
    // 0 for every ordinary air. Appended last so the layout stays ABI-compatible.
    uint64_t witnessOps;
};

#endif