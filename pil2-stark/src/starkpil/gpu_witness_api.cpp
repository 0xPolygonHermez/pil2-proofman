// C entry points for the GPU witness-kernel registry.
//
// Compiled into BOTH the CPU and GPU libraries because Rust calls these unconditionally. On the
// CPU backend nothing is registered and every lookup returns null.

#include <cstdint>

#include "gpu_witness.hpp"

extern "C" {

/// Declare that `airgroupId:airId`'s stage-1 witness comes from `fill` rather
/// than a host upload. `emits` is a `GpuWitnessLayout`.
void gpu_witness_register(void *d_buffers, uint64_t airgroupId, uint64_t airId, uint64_t bytesPerOp,
                          int emits, GpuWitnessFillFn fill) {
    gpu_witness_register_impl(d_buffers, airgroupId, airId, bytesPerOp, emits, fill);
}

/// Forget this prover's registrations.
void gpu_witness_clear(void *d_buffers) {
    gpu_witness_clear_impl(d_buffers);
}

/// How many airs are registered. The startup check reports this back.
uint64_t gpu_witness_count(void *d_buffers) {
    return gpu_witness_count_impl(d_buffers);
}

/// Whether this air's witness is produced on the device.
int gpu_witness_is_registered(void *d_buffers, uint64_t airgroupId, uint64_t airId) {
    GpuWitnessAirReg r;
    return gpu_witness_for(d_buffers, airgroupId, airId, &r) ? 1 : 0;
}

}  // extern "C"
