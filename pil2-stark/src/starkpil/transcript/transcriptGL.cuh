
#ifndef TRANSCRIPT_GPU_CLASS
#define TRANSCRIPT_GPU_CLASS

#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "poseidon2_goldilocks.hpp"
#include "zklog.hpp"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include <memory>
#include "blake3_core.hpp"
#include "sha256_core.hpp"

// blake3 is not a sponge, so it uses none of the state/pending/out buffers and
// gets its own kernels rather than a branch inside the Poseidon ones. Its work
// is scalar, so `parallel` makes no difference and every kernel is <<<1,1>>>.
struct Blake3TrxStateGPU
{
    blake3core::Hasher h;
    uint64_t xof[8];
    uint32_t offset;   // words consumed from xof
    uint32_t ob;       // which output block xof holds
    uint32_t valid;
};

__global__ void _b3Reset(Blake3TrxStateGPU* b3);
__global__ void _b3Add(Goldilocks::Element* input, uint64_t size, Blake3TrxStateGPU* b3);
__global__ void _b3Add2(Goldilocks::Element* input1, uint64_t size1,
                        Goldilocks::Element* input2, uint64_t size2, Blake3TrxStateGPU* b3);
__global__ void _b3GetField(uint64_t* output, Blake3TrxStateGPU* b3);
__global__ void _b3GetState(Goldilocks::Element* output, uint64_t nOutputs, Blake3TrxStateGPU* b3);
__global__ void _b3GetPermutations(uint64_t* res, uint64_t n, uint64_t nBits,
                                   Goldilocks::Element* fields, Blake3TrxStateGPU* b3);

// sha256, same shape as the blake3 state. `out` holds 4 words (a digest) against blake3's 8-word
// XOF block, and `ctr` is the squeeze counter.
struct Sha256TrxStateGPU
{
    sha256core::Hasher h;
    uint64_t out[4];
    uint32_t offset;   // words consumed from out
    uint64_t ctr;      // which squeeze block out holds
    uint32_t valid;
};

__global__ void _shaReset(Sha256TrxStateGPU* sha);
__global__ void _shaAdd(Goldilocks::Element* input, uint64_t size, Sha256TrxStateGPU* sha);
__global__ void _shaAdd2(Goldilocks::Element* input1, uint64_t size1,
                         Goldilocks::Element* input2, uint64_t size2, Sha256TrxStateGPU* sha);
__global__ void _shaGetField(uint64_t* output, Sha256TrxStateGPU* sha);
__global__ void _shaGetState(Goldilocks::Element* output, uint64_t nOutputs, Sha256TrxStateGPU* sha);
__global__ void _shaGetPermutations(uint64_t* res, uint64_t n, uint64_t nBits,
                                    Goldilocks::Element* fields, Sha256TrxStateGPU* sha);

__device__ void _updateState(Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity);
__device__ Goldilocks::Element _getFields1(Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity);
__global__ void _add(Goldilocks::Element* input, uint64_t size, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity);
__global__ void _getField(uint64_t* output, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity);
__global__ void __getState(Goldilocks::Element* output, uint64_t nOutputs, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity);
__global__ void __getPermutations(uint64_t *res, uint64_t n, uint64_t nBits, Goldilocks::Element* fields, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity, uint8_t hashFamily);

/// One transcript construction, host side. The kernels stay per family -- device virtuals would
/// put a vtable in device memory for no gain -- but the dispatch happens ONCE, in the constructor,
/// instead of at every call. Each impl owns only its own device state.
class TranscriptImplGL_GPU
{
public:
    virtual ~TranscriptImplGL_GPU() = default;
    virtual void reset(cudaStream_t stream) = 0;
    virtual void put(Goldilocks::Element *input, uint64_t size, cudaStream_t stream) = 0;
    virtual void put2(Goldilocks::Element *input1, uint64_t size1,
                      Goldilocks::Element *input2, uint64_t size2, cudaStream_t stream) = 0;
    virtual void getField(uint64_t *output, cudaStream_t stream) = 0;
    virtual void getState(Goldilocks::Element *output, uint64_t nOutputs, cudaStream_t stream) = 0;
    virtual void getPermutations(uint64_t *res, uint64_t n, uint64_t nBits,
                                 Goldilocks::Element *scratch, cudaStream_t stream) = 0;
};

class TranscriptGL_GPU
{
private:
    std::unique_ptr<TranscriptImplGL_GPU> impl;
    uint32_t transcriptStateSize;

public:
    TranscriptGL_GPU(uint64_t arity, bool custom, cudaStream_t stream, bool parallel = true);
    ~TranscriptGL_GPU();

    void reset(cudaStream_t stream) { impl->reset(stream); }
    void put(Goldilocks::Element *input, uint64_t size, cudaStream_t stream) { impl->put(input, size, stream); }
    void put2(Goldilocks::Element *input1, uint64_t size1, Goldilocks::Element *input2, uint64_t size2,
              cudaStream_t stream) { impl->put2(input1, size1, input2, size2, stream); }
    void getField(uint64_t *output, cudaStream_t stream) { impl->getField(output, stream); }
    void getState(Goldilocks::Element *output, cudaStream_t stream) { impl->getState(output, transcriptStateSize, stream); }
    void getState(Goldilocks::Element *output, uint64_t nOutputs, cudaStream_t stream) { impl->getState(output, nOutputs, stream); }
    void getPermutations(uint64_t *res, uint64_t n, uint64_t nBits, Goldilocks::Element *perm_scratch,
                         cudaStream_t stream) { impl->getPermutations(res, n, nBits, perm_scratch, stream); }

    static void init_const(uint32_t *gpu_ids, uint32_t num_gpu_ids, uint32_t arity_init);
};

#endif