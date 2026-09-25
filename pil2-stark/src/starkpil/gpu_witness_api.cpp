// C entry points for the GPU witness-kernel registry.
//
// Compiled into BOTH the CPU and GPU libraries because Rust calls these unconditionally. On the
// CPU backend nothing is registered and every lookup returns null.

#include <cstdint>

#include "gpu_witness.hpp"

extern "C" {

/// Declare that `airgroupId:airId`'s stage-1 witness comes from `fill` rather
/// than a host upload. `emits` is a `GpuWitnessLayout`.
void gpu_witness_register(uint64_t airgroupId, uint64_t airId, uint64_t bytesPerOp, int emits,
                          GpuWitnessFillFn fill) {
    gpu_witness_register_impl(airgroupId, airId, bytesPerOp, emits, fill);
}

/// Forget every registration, so a later prover in the same process starts clean.
void gpu_witness_clear() {
    gpu_witness_clear_impl();
}

/// How many airs are registered. The startup check reports this back.
uint64_t gpu_witness_count() {
    return gpu_witness_count_impl();
}

/// Whether this air's witness is produced on the device.
int gpu_witness_is_registered(uint64_t airgroupId, uint64_t airId) {
    return gpu_witness_for(airgroupId, airId) != nullptr ? 1 : 0;
}

}  // extern "C"
