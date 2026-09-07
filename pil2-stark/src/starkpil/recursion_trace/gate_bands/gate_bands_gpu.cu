// GPU expander: the same reconstruction as gate_bands_cpu.hpp, on device.
//
// This file is the C ABI and the dispatch; the reconstruction lives per family in
// gate_bands_<family>_gpu.cuh. Adding a family means a back-end and an arm in each switch below.
//
// One translation unit on purpose. The build compiles every .cu whole-program (no -rdc), so a
// family's __constant__ tables exist once per includer -- split across two .cu files, the upload
// would fill one copy and the kernel read another, silently, with no link error. Splitting the
// FILES is what buys the separation; splitting the TU would cost correctness.

#include <cstdint>
#include "cuda_utils.cuh"
#include "gate_bands.hpp"
#include "gate_bands_blake3_gpu.cuh"
#include "gate_bands_poseidon_gpu.cuh"

// Dense per-stream scratch a family's expansion needs, in words. Sized from the band list's family
// so a caller allocates without knowing which one it has; 0 means the family needs none.
extern "C" uint64_t gateBandScratchWordsGPU(uint64_t family) {
    switch ((gate_bands::Family)family) {
        case gate_bands::Family::Blake3:   return gate_bands::blake3_gpu::SCRATCH_WORDS;
        case gate_bands::Family::Poseidon: return gate_bands::poseidon_gpu::SCRATCH_WORDS;
        default: return 0;
    }
}

// Uploads a family's constant tables to the current device. They are __constant__, so this is
// per-device state: the caller selects the device first. Idempotent, so repeating it per air is
// harmless, but it must not race work already reading the tables on that device.
extern "C" void uploadGateBandConstantsGPU(uint64_t family) {
    if ((gate_bands::Family)family == gate_bands::Family::Poseidon) gate_bands::poseidon_gpu::upload_constants();
    // BLAKE3's constants are small enough to be materialised in the kernel; nothing to upload.
}

// Expands every band in place on the device trace. The band list is already device-resident,
// uploaded with the air's setup, so this is launch-only -- no allocation, no sync -- and sits
// stream-ordered behind the trace copy.
//
// `family` and `aux` are decided host-side at setup: a band list is one family (load_device_setup
// rejects a mixed one), and aux carries setup parameters no kernel can recover from the trace.
// `d_scratch` is the caller's per-stream buffer of gateBandScratchWordsGPU(family) words.
extern "C" void expandGateBandsGPU(uint64_t *d_trace, uint64_t nCols, uint64_t nRows,
                                   const uint64_t *d_bands, uint64_t nBands, uint64_t aux,
                                   uint64_t family, uint64_t *d_scratch, void *stream_) {
    if (nBands == 0) return;
    const cudaStream_t stream = (cudaStream_t)stream_;
    switch ((gate_bands::Family)family) {
        case gate_bands::Family::Poseidon:
            gate_bands::poseidon_gpu::launch(d_trace, nCols, nRows, d_bands, nBands, aux, d_scratch, stream);
            break;
        case gate_bands::Family::Blake3:
            gate_bands::blake3_gpu::launch(d_trace, nCols, nRows, d_bands, nBands, aux, d_scratch, stream);
            break;
        default:
            // load_device_setup refuses None-with-bands and Mixed, so reaching here is a caller bug.
            fprintf(stderr, "expandGateBandsGPU: %llu bands with no owning family (%llu)\n",
                    (unsigned long long)nBands, (unsigned long long)family);
            fflush(stderr);
            exit(-1);
    }
}
