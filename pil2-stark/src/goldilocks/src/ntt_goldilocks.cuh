#ifndef NTT_GOLDILOCKS_GPU
#define NTT_GOLDILOCKS_GPU

#include "gl64_tooling.cuh"
#include "cuda_utils.cuh"
#include <cuda_runtime.h>
#include <sys/time.h>
#include "ntt_goldilocks.hpp"
#include "gpu_timer.cuh"

__device__ __constant__ uint64_t omegas[33] = {
    1,
    18446744069414584320ULL,
    281474976710656ULL,
    16777216ULL,
    4096ULL,
    64ULL,
    8ULL,
    2198989700608ULL,
    4404853092538523347ULL,
    6434636298004421797ULL,
    4255134452441852017ULL,
    9113133275150391358ULL,
    4355325209153869931ULL,
    4308460244895131701ULL,
    7126024226993609386ULL,
    1873558160482552414ULL,
    8167150655112846419ULL,
    5718075921287398682ULL,
    3411401055030829696ULL,
    8982441859486529725ULL,
    1971462654193939361ULL,
    6553637399136210105ULL,
    8124823329697072476ULL,
    5936499541590631774ULL,
    2709866199236980323ULL,
    8877499657461974390ULL,
    3757607247483852735ULL,
    4969973714567017225ULL,
    2147253751702802259ULL,
    2530564950562219707ULL,
    1905180297017055339ULL,
    3524815499551269279ULL,
    7277203076849721926ULL,
};

__device__ __constant__ uint64_t omegas_inv[33] = {
    0x1,
    0xffffffff00000000,
    0xfffeffff00000001,
    0xfffffeff00000101,
    0xffefffff00100001,
    0xfbffffff04000001,
    0xdfffffff20000001,
    0x3fffbfffc0,
    0x7f4949dce07bf05d,
    0x4bd6bb172e15d48c,
    0x38bc97652b54c741,
    0x553a9b711648c890,
    0x55da9bb68958caa,
    0xa0a62f8f0bb8e2b6,
    0x276fd7ae450aee4b,
    0x7b687b64f5de658f,
    0x7de5776cbda187e9,
    0xd2199b156a6f3b06,
    0xd01c8acd8ea0e8c0,
    0x4f38b2439950a4cf,
    0x5987c395dd5dfdcf,
    0x46cf3d56125452b1,
    0x909c4b1a44a69ccb,
    0xc188678a32a54199,
    0xf3650f9ddfcaffa8,
    0xe8ef0e3e40a92655,
    0x7c8abec072bb46a6,
    0xe0bfc17d5c5a7a04,
    0x4c6b8a5a0b79f23a,
    0x6b4d20533ce584fe,
    0xe5cceae468a70ec2,
    0x8958579f296dac7a,
    0x16d265893b5b7e85,
};

__device__ __constant__ uint64_t domain_size_inverse[33] = {
    0x0000000000000001, // 1^{-1}
    0x7fffffff80000001, // 2^{-1}
    0xbfffffff40000001, // (1 << 2)^{-1}
    0xdfffffff20000001, // (1 << 3)^{-1}
    0xefffffff10000001,
    0xf7ffffff08000001,
    0xfbffffff04000001,
    0xfdffffff02000001,
    0xfeffffff01000001,
    0xff7fffff00800001,
    0xffbfffff00400001,
    0xffdfffff00200001,
    0xffefffff00100001,
    0xfff7ffff00080001,
    0xfffbffff00040001,
    0xfffdffff00020001,
    0xfffeffff00010001,
    0xffff7fff00008001,
    0xffffbfff00004001,
    0xffffdfff00002001,
    0xffffefff00001001,
    0xfffff7ff00000801,
    0xfffffbff00000401,
    0xfffffdff00000201,
    0xfffffeff00000101,
    0xffffff7f00000081,
    0xffffffbf00000041,
    0xffffffdf00000021,
    0xffffffef00000011,
    0xfffffff700000009,
    0xfffffffb00000005,
    0xfffffffd00000003,
    0xfffffffe00000002, // (1 << 32)^{-1}
};

class gl64_t;

class NTT_Goldilocks_GPU : public NTT_Goldilocks {
public:
    using NTT_Goldilocks::NTT_Goldilocks;

    NTT_Goldilocks_GPU(uint64_t maxLogDomainSize_, uint32_t nGPUs_input = 0, uint32_t* gpu_ids = nullptr)
       : NTT_Goldilocks() {
        init_twiddle_factors_and_r(maxLogDomainSize_, nGPUs_input, gpu_ids);
    }

    void LDE_MerkleTree_GPU_inplace(Goldilocks::Element *d_tree, gl64_t* d_dst_ntt, uint64_t offset_dst_ntt,
                                    gl64_t* d_src_ntt, uint64_t offset_src_ntt, u_int64_t n_bits,
                                    u_int64_t n_bits_ext, u_int64_t ncols, TimerGPU &timer, cudaStream_t stream);

    void computeQ_inplace(Goldilocks::Element *d_tree, uint64_t offset_cmQ, uint64_t offset_q,
                          uint64_t qDeg, uint64_t qDim, Goldilocks::Element shiftIn, uint64_t N,
                          uint64_t n_bits_ext, uint64_t nCols, gl64_t *d_aux_trace,
                          uint64_t offset_helper, TimerGPU &timer, cudaStream_t stream);

    void INTT_inplace(uint64_t data_offset, u_int64_t n_bits, u_int64_t ncols,
                      gl64_t* d_aux_trace, uint64_t offset_helper,
                      gl64_t* d_data, cudaStream_t stream);

    static void init_twiddle_factors_and_r(uint64_t maxLogDomainSize_, uint32_t nGPUs_input = 0, uint32_t* gpu_ids = nullptr);

    // IMPORTANT: Memory management is manual. Call free_twiddle_factors_and_r() explicitly
    // at application shutdown to release GPU memory. Twiddle factors persist across
    // instance creation/destruction to avoid recomputation overhead.
    static void free_twiddle_factors_and_r();

private:
    
    static uint64_t maxLogDomainSize;
    static uint32_t nGPUs_available;
    static gl64_t **d_fwd_twiddle_factors;
    static gl64_t **d_inv_twiddle_factors;
    static gl64_t **d_r;

};

#endif