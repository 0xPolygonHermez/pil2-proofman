#ifndef LIB_API_INTERNAL_H
#define LIB_API_INTERNAL_H
#include "starks_api.hpp"

extern ProofDoneCallback proof_done_callback;

struct DeviceCommitBuffersCPU
{
    uint64_t airgroupId;
    uint64_t airId;
    std::string proofType;
};

#endif