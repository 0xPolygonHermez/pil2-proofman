
#ifndef TRANSCRIPT_GPU_CLASS
#define TRANSCRIPT_GPU_CLASS

#include "goldilocks_base_field.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "poseidon2_goldilocks.hpp"
#include "zklog.hpp"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "blake3_core.hpp"

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

__device__ void _updateState(Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity);
__device__ Goldilocks::Element _getFields1(Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity);
__global__ void _add(Goldilocks::Element* input, uint64_t size, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity);
__global__ void _getField(uint64_t* output, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity);
__global__ void __getState(Goldilocks::Element* output, uint64_t nOutputs, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity);
__global__ void __getPermutations(uint64_t *res, uint64_t n, uint64_t nBits, Goldilocks::Element* fields, Goldilocks::Element* state, Goldilocks::Element* pending, Goldilocks::Element* out, uint* pending_cursor, uint* out_cursor, uint32_t arity, uint8_t hashFamily);

class TranscriptGL_GPU
{
    

public:

    uint32_t arity;
    uint32_t transcriptStateSize;
    uint32_t transcriptPendingSize;
    uint32_t transcriptOutSize;

    Goldilocks::Element* state;
    Goldilocks::Element* pending;
    Goldilocks::Element* out;

    uint *pending_cursor;
    uint *out_cursor;

    Blake3TrxStateGPU *b3;

    bool parallel;

    TranscriptGL_GPU(uint64_t arity, bool custom, cudaStream_t stream, bool parallel = true);
    ~TranscriptGL_GPU()
    {
        CHECKCUDAERR(cudaFree(state));
        CHECKCUDAERR(cudaFree(pending));
        CHECKCUDAERR(cudaFree(out));
        CHECKCUDAERR(cudaFree(pending_cursor));
        CHECKCUDAERR(cudaFree(out_cursor));
        CHECKCUDAERR(cudaFree(b3));
    }
    
    void reset(cudaStream_t stream) {
        cudaMemsetAsync(state, 0, transcriptOutSize * sizeof(Goldilocks::Element), stream);
        cudaMemsetAsync(pending, 0, transcriptPendingSize * sizeof(Goldilocks::Element), stream);
        cudaMemsetAsync(out, 0, transcriptOutSize * sizeof(Goldilocks::Element), stream);
        cudaMemsetAsync(pending_cursor, 0, sizeof(uint), stream);
        cudaMemsetAsync(out_cursor, 0, sizeof(uint), stream);
        _b3Reset<<<1, 1, 0, stream>>>(b3);   // Hasher::init is a device function
    };

    void put(Goldilocks::Element *input, uint64_t size, cudaStream_t stream);
    void put2(Goldilocks::Element *input1, uint64_t size1, Goldilocks::Element *input2, uint64_t size2, cudaStream_t stream);
    void getField(uint64_t *output, cudaStream_t stream);
    void getState(Goldilocks::Element* output, cudaStream_t stream);
    void getState(Goldilocks::Element* output, uint64_t nOutputs, cudaStream_t stream);
    void getPermutations(uint64_t *res, uint64_t n, uint64_t nBits, Goldilocks::Element* perm_scratch, cudaStream_t stream);
    static void init_const(uint32_t* gpu_ids, uint32_t num_gpu_ids, uint32_t arity_init);
    
};

#endif