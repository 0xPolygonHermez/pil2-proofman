#ifndef NTT_LDE_HELPER_CUH
#define NTT_LDE_HELPER_CUH

#include <cuda_runtime.h>

// NTTGoldilocksGPU::ldeColMajor on `stream` splits its batches over a second stream once the owner of
// `stream` attaches one. The helper is created here, on the current device and at the stream's priority,
// so nothing is created inside an LDE (which may run under graph capture). Attach a long-lived stream
// after creating it and detach before destroying it; a stream without a helper runs its LDEs alone.
// Default-stream handles are refused: they name a different stream in every thread.
void nttAttachLdeHelper(cudaStream_t stream);
void nttDetachLdeHelper(cudaStream_t stream);

#endif
