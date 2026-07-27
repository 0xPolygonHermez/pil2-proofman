// sppark-backed NTT primitives (LDE / computeQ / INTT) for the FLAT column-major layout.
// Contracts are in sppark_lde.cuh. Isolated TU (sppark headers clash with ntt_goldilocks.cuh);
// reached by subclassing NTT, exposed via extern "C".
//
// Compiled -DGOLDILOCKS_ZISK so sppark's roots == prover Goldilocks::W AND the LDE coset generator is
// SHIFT=7 (external/sppark/ntt/parameters/goldilocks.h) — correct by construction, no runtime re-seed.
// All work runs on the CALLER's stream (no private stream / event handoff), so ordering is plain
// program order — the same guarantee the native ColMajorTiled path has.

#include <ff/goldilocks.hpp>   // fr_t == gl64_t
#include <ntt/ntt.cuh>
#include <ntt/parameters.cuh>
#include <util/gpu_t.cuh>
#include "goldilocks_trace_layout.cuh"
#include "sppark_lde.cuh"

#include <cassert>

// Non-cooperative replacement for sppark's LDE_launch: with DISTINCT in/out, one thread per idx writes
// r*7^bit_rev(idx) to out[idx<<lg_blowup] and zeroes the rest of the blowup group. Bit-exact with LDE_launch.
__global__ void spk_ldeSpread(fr_t *out, const fr_t *in,
                              const fr_t (*gen_powers)[WINDOW_SIZE],
                              uint32_t lg_domain_size, uint32_t lg_blowup)
{
    uint64_t domain_size = (uint64_t)1 << lg_domain_size;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= domain_size)
        return;

    uint32_t blowup = 1u << lg_blowup;
    fr_t r = in[idx];

    index_t pow = bit_rev((index_t)idx, lg_domain_size);
    r = r * get_intermediate_root(pow, gen_powers);

    uint64_t base = idx << lg_blowup;
    out[base] = r;
    fr_t zero;
    zero.zero();
    for (uint32_t j = 1; j < blowup; j++)
        out[base + j] = zero;
}

class SpparkLDE : public NTT {
public:
    // Coset-LDE: iNTT(small) -> spread+coset -> NTT(ext). spread_in (disjoint from d_buf) holds the
    // small domain for the plain spk_ldeSpread; if null, fall back to sppark's cooperative LDE_launch.
    static void run(stream_t &stream, fr_t *d_buf, uint32_t lg_domain_size, uint32_t lg_blowup,
                    fr_t *spread_in = nullptr)
    {
        size_t domain_size = (size_t)1 << lg_domain_size;
        size_t ext_domain_size = domain_size << lg_blowup;
        fr_t *ext_domain_data = &d_buf[0];
        fr_t *domain_data = spread_in ? spread_in : &d_buf[ext_domain_size - domain_size];

        NTT_internal(domain_data, lg_domain_size,
                     InputOutputOrder::NR, Direction::inverse, Type::standard, stream);

        const auto gen_powers = NTTParameters::all()[stream].partial_group_gen_powers;

        if (spread_in) {
            uint32_t threads = 256;
            uint32_t blocks = (uint32_t)((domain_size + threads - 1) / threads);
            spk_ldeSpread<<<blocks, threads, 0, stream>>>(ext_domain_data, domain_data, gen_powers,
                                                          lg_domain_size, lg_blowup);
            CUDA_OK(cudaGetLastError());
        } else {
            LDE_launch(stream, ext_domain_data, domain_data, gen_powers, lg_domain_size, lg_blowup);
        }

        NTT_internal(ext_domain_data, lg_domain_size + lg_blowup,
                     InputOutputOrder::RN, Direction::forward, Type::standard, stream);
    }

    // Plain full-domain NTT/INTT (natural in/out) on a single contiguous column. Matches the prover's
    // nttDit (bit-reversal + DIT = NN order) on the same root.
    static void ntt_inplace(stream_t &stream, fr_t *d_col, uint32_t lg, bool inverse)
    {
        NTT_internal(d_col, lg, InputOutputOrder::NN,
                     inverse ? Direction::inverse : Direction::forward, Type::standard, stream);
    }
};

// Resolve sppark's LOGICAL gpu id for the caller's stream (and cudaSetDevice it). The device comes
// from the STREAM itself (cudaStreamGetDevice), not the thread-ambient device, so multi-GPU is correct
// regardless of caller device state. select_gpu maps the cuda ordinal -> sppark logical id (they
// differ when sppark filters a device); NTTParameters::all()[id] is keyed by that logical id.
static int sp_device_id(cudaStream_t caller)
{
    int cuda_dev = -1;
    CUDA_OK(cudaStreamGetDevice(caller, &cuda_dev));
    const gpu_t &gpu = select_gpu(cuda_dev);
    if (gpu.cid() != cuda_dev) {
        fprintf(stderr, "[sppark_lde] stream's CUDA device %d not in sppark's GPU list\n", cuda_dev);
        abort();
    }
    return gpu.id();
}

// Per column c: iNTT(N) -> spk_ldeSpread (coset) into dst[c]. When preserve_src (callers reread
// cm1/const/custom commits) the iNTT runs on a COPY in preserve_scratch (a free N-slice, e.g. the mt
// region); callers must supply it when preserve_src is set.
extern "C" void sppark_lde_flat(void *d_dst_v, void *d_src_v,
                                uint32_t lg_n, uint32_t lg_next, uint32_t nCols,
                                bool preserve_src, void *preserve_scratch, void *caller_stream)
{
    cudaStream_t cs = (cudaStream_t)caller_stream;
    stream_t s(cs, sp_device_id(cs));  // run on the caller's stream (adopt, no destroy)
    gl64_t *d_dst = reinterpret_cast<gl64_t *>(d_dst_v);
    gl64_t *d_src = reinterpret_cast<gl64_t *>(d_src_v);
    gl64_t *col_scratch = reinterpret_cast<gl64_t *>(preserve_scratch);
    uint32_t lg_blowup = lg_next - lg_n;
    uint64_t N = 1ull << lg_n;
    uint64_t Next = 1ull << lg_next;
    assert(!preserve_src || col_scratch);  // preserve_src requires a caller-provided scratch slice

    for (uint32_t c = 0; c < nCols; c++) {
        fr_t *spread_in;
        if (preserve_src) {
            CUDA_OK(cudaMemcpyAsync(col_scratch, d_src + (uint64_t)c * N, N * sizeof(gl64_t),
                                    cudaMemcpyDeviceToDevice, s));
            spread_in = (fr_t *)col_scratch;
        } else {
            spread_in = (fr_t *)(d_src + (uint64_t)c * N);
        }
        SpparkLDE::run(s, (fr_t *)(d_dst + (uint64_t)c * Next), lg_n, lg_blowup, spread_in);
    }
}

// Flat coset + spread for computeQ. Reads qDim flat columns of q-coefficients (length Next, only the
// first N meaningful after iNTT of a degree-<N poly) and writes qDeg*qDim flat columns of cmQ:
//   cmQ[p*qDim + k][row] = q[k][row + p*N] * shiftIn^p   for row < N
//   cmQ[p*qDim + k][row] = 0                             for row >= N
// shiftIn is a BASE-FIELD scalar, so the cubic-extension element scales component-wise.
// Launch: <<<ceil(Next/256), 256>>>
__global__ void spk_cosetFlat(const gl64_t *q, gl64_t *cmQ, uint32_t N, uint32_t Next,
                              uint32_t qDeg, uint32_t qDim, uint64_t shiftIn)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Next)
        return;
    gl64_t shift = gl64_t(shiftIn);
    gl64_t s = gl64_t(uint64_t(1));
    for (uint32_t p = 0; p < qDeg; p++) {
        for (uint32_t k = 0; k < qDim; k++) {
            uint32_t outcol = p * qDim + k;
            gl64_t v = (row < N) ? q[(uint64_t)k * Next + (row + (uint64_t)p * N)] * s
                                 : gl64_t(uint64_t(0));
            cmQ[(uint64_t)outcol * Next + row] = v;
        }
        s = shift * s;
    }
}


// iNTT each q column -> spk_cosetFlat (coset shift + zero-pad) -> NTT each cmQ column, all in place on
// disjoint flat regions (q at off_q, cmQ at off_cmQ).
extern "C" void sppark_computeq_flat(void *d_aux_v, uint64_t off_cmQ, uint64_t off_q,
                                     uint32_t qDeg, uint32_t qDim, uint64_t shiftIn,
                                     uint32_t lg_n, uint32_t lg_next, uint32_t nCols,
                                     void *caller_stream)
{
    cudaStream_t cs = (cudaStream_t)caller_stream;
    stream_t s(cs, sp_device_id(cs));  // run on the caller's stream (adopt, no destroy)
    gl64_t *d_aux = reinterpret_cast<gl64_t *>(d_aux_v);
    gl64_t *d_q = d_aux + off_q;       // flat, qDim cols, Next rows
    gl64_t *d_cmQ = d_aux + off_cmQ;   // flat, nCols cols, Next rows
    uint32_t N = 1u << lg_n;
    uint32_t Next = 1u << lg_next;

    dim3 threads(256);
    uint32_t gridExt = (Next + threads.x - 1) / threads.x;
    // 1. iNTT each q column over the ext domain, in place.
    for (uint32_t c = 0; c < qDim; c++)
        SpparkLDE::ntt_inplace(s, (fr_t *)(d_q + (uint64_t)c * Next), lg_next, /*inverse=*/true);
    // 2. coset shift + zero-pad: q (flat) -> cmQ (flat, disjoint region).
    spk_cosetFlat<<<gridExt, threads, 0, s>>>(d_q, d_cmQ, N, Next, qDeg, qDim, shiftIn);
    CUDA_OK(cudaGetLastError());
    // 3. NTT each cmQ column over the ext domain, in place.
    for (uint32_t c = 0; c < nCols; c++)
        SpparkLDE::ntt_inplace(s, (fr_t *)(d_cmQ + (uint64_t)c * Next), lg_next, /*inverse=*/false);
}

// Per-column in-place INTT of nCols flat columns of N rows (used for the LEv vector).
extern "C" void sppark_intt_flat(void *d_data_v, uint32_t lg_n, uint32_t nCols, void *caller_stream)
{
    cudaStream_t cs = (cudaStream_t)caller_stream;
    stream_t s(cs, sp_device_id(cs));  // run on the caller's stream (adopt, no destroy)
    gl64_t *d_data = reinterpret_cast<gl64_t *>(d_data_v);
    uint64_t N = 1ull << lg_n;
    for (uint32_t c = 0; c < nCols; c++)
        SpparkLDE::ntt_inplace(s, (fr_t *)(d_data + (uint64_t)c * N), lg_n, /*inverse=*/true);
}
