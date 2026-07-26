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

// ============================================================================
// BATCHED (column-chunked) flat LDE.
//
// sppark's stock driver transforms one column per kernel launch. At small domains that
// starves the GPU (measured: 0.09 waves/SM at 2^17 on a 170-SM part), and the serial
// per-column chain was the dominant NTT cost in recursive proofs. The two kernels below
// are verbatim transcriptions of sppark's _CT_NTT / _GS_NTT (mixed-radix, register-
// resident butterflies, warp-shuffle exchange -- ntt/kernels/{ct,gs}_mixed_radix_narrow.cu,
// Apache-2.0) with ONE change: gridDim.y selects the column, so a single launch covers a
// CHUNK of columns.
//
// Chunks are sized to L2 (~60%): the 2-3 steps of one transform reuse the whole domain
// between launches, and that reuse only lands in cache if chunk_cols * domain_bytes fits.
// Small domains -> wide chunks (occupancy win, e.g. 2^17: ~50 cols); large domains ->
// 1-2 cols (already occupancy-saturated, keeps sppark's step-to-step L2 hit intact).
// ============================================================================

template<class fr_t>
__global__ void spk_copyToTops(fr_t *dst_base, const fr_t *src_base,
                               uint32_t lg_n, size_t dst_stride)
{
    const size_t N = (size_t)1 << lg_n;
    const fr_t *src = src_base + (size_t)blockIdx.y * N;
    fr_t *dst = dst_base + (size_t)blockIdx.y * dst_stride + (dst_stride - N);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < N;
         i += (size_t)gridDim.x * blockDim.x)
        dst[i] = src[i];
}

template<int z_count, bool coalesced = false, class fr_t>
__launch_bounds__(768, 1) __global__
void _CT_NTT_b(const unsigned int radix, const unsigned int lg_domain_size,
               const unsigned int stage, const unsigned int iterations,
               fr_t* d_base, const size_t col_stride,
               const fr_t (*d_partial_twiddles)[WINDOW_SIZE],
               const fr_t (*d_plus_one_twiddles)[1024],
               const fr_t* d_radix6_twiddles, const fr_t* d_radixX_twiddles,
               bool is_intt, const fr_t d_domain_size_inverse)
{
    fr_t* d_inout = d_base + (size_t)blockIdx.y * col_stride;   // <- column select (only change)
    extern __shared__ int xchg_raw[];
    fr_t* shared_exchange = reinterpret_cast<fr_t*>(xchg_raw);

    index_t tid = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    const index_t diff_mask = (1 << (iterations - 1)) - 1;
    const index_t inp_mask = ((index_t)1 << stage) - 1;
    const index_t out_mask = ((index_t)1 << (stage + iterations - 1)) - 1;

    const index_t tiz = (tid & ~diff_mask) * z_count + (tid & diff_mask);
    const index_t thread_ntt_pos = (tiz >> (iterations - 1)) & inp_mask;

    index_t idx0 = (tiz & ~out_mask) | ((tiz << stage) & out_mask);
    idx0 = idx0 * 2 + thread_ntt_pos;
    index_t idx1 = idx0 + ((index_t)1 << stage);

    fr_t r[2][z_count];

    if (coalesced) {
        coalesced_load<z_count>(r[0], d_inout, idx0, stage + 1);
        coalesced_load<z_count>(r[1], d_inout, idx1, stage + 1);
        transpose<z_count>(r[0]);
        __syncwarp();
        transpose<z_count>(r[1]);
    } else {
        unsigned int z_shift = inp_mask==0 ? iterations : 0;
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            r[0][z] = d_inout[idx0 + (z << z_shift)];
            r[1][z] = d_inout[idx1 + (z << z_shift)];
        }
    }

    if (stage != 0) {
        unsigned int thread_ntt_idx = (tiz & diff_mask) * 2;
        unsigned int nbits = MAX_LG_DOMAIN_SIZE - stage;
        index_t idx0r = bit_rev(thread_ntt_idx, nbits);
        index_t root_idx0 = idx0r * thread_ntt_pos;
        index_t root_idx1 = root_idx0 + (thread_ntt_pos << (nbits - 1));

        fr_t first_root, second_root;
        get_intermediate_roots(first_root, second_root,
                               root_idx0, root_idx1, d_partial_twiddles);
        r[0][0] = r[0][0] * first_root;
        r[1][0] = r[1][0] * second_root;

        if (z_count > 1) {
            unsigned int off = nbits >= 10 ? (nbits - 10) : 0;
            unsigned int scale = nbits >= 10 ? 0 : (10 - nbits);

            thread_ntt_idx <<= scale;
            fr_t first_root_z = d_plus_one_twiddles[off][thread_ntt_idx];
            fr_t second_root_z = d_plus_one_twiddles[off][thread_ntt_idx + (1<<scale)];

            #pragma unroll
            for (int z = 1; z < z_count; z++) {
                first_root *= first_root_z;
                second_root *= second_root_z;
                r[0][z] = r[0][z] * first_root;
                r[1][z] = r[1][z] * second_root;
            }
        }
    }

    #pragma unroll
    for (int z = 0; z < z_count; z++) {
        fr_t t = r[1][z];
        r[1][z] = r[0][z] - t;
        r[0][z] = r[0][z] + t;
    }

    #pragma unroll 1
    for (unsigned int s = 1; s < min(iterations, 6u); s++) {
        unsigned int laneMask = 1 << (s - 1);
        unsigned int thrdMask = (1 << s) - 1;
        unsigned int rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

        fr_t root = d_radix6_twiddles[rank << (6 - (s + 1))];

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            fr_t t = fr_t::csel(r[1][z], r[0][z], pos);

            t.shfl_bfly(laneMask);

            r[0][z] = fr_t::csel(r[0][z], t, pos);
            r[1][z] = fr_t::csel(t, r[1][z], pos);

            t = root * r[1][z];
            r[1][z] = r[0][z] - t;
            r[0][z] = r[0][z] + t;
        }
    }

    #pragma unroll 1
    for (unsigned int s = 6; s < iterations; s++) {
        unsigned int laneMask = 1 << (s - 1);
        unsigned int thrdMask = (1 << s) - 1;
        unsigned int rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

        fr_t root = d_radixX_twiddles[rank << (radix - (s + 1))];

        fr_t (*xchg)[z_count] = reinterpret_cast<decltype(xchg)>(shared_exchange);

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            fr_t t = fr_t::csel(r[1][z], r[0][z], pos);
            xchg[threadIdx.x][z] = t;
        }

        __syncthreads();

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            fr_t t = xchg[threadIdx.x ^ laneMask][z];

            r[0][z] = fr_t::csel(r[0][z], t, pos);
            r[1][z] = fr_t::csel(t, r[1][z], pos);

            t = root * r[1][z];
            r[1][z] = r[0][z] - t;
            r[0][z] = t + r[0][z];
        }

        __syncthreads();
    }

    if (is_intt && (stage + iterations) == lg_domain_size) {
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            r[0][z] = r[0][z] * d_domain_size_inverse;
            r[1][z] = r[1][z] * d_domain_size_inverse;
        }
    }

    index_t mask = (index_t)((1 << iterations) - 1) << stage;
    index_t rotw = idx0 & mask;
    rotw = (rotw >> 1) | (rotw << (iterations - 1));
    idx0 = (idx0 & ~mask) | (rotw & mask);
    rotw = idx1 & mask;
    rotw = (rotw >> 1) | (rotw << (iterations - 1));
    idx1 = (idx1 & ~mask) | (rotw & mask);

    if (coalesced) {
        transpose<z_count>(r[0]);
        __syncwarp();
        transpose<z_count>(r[1]);
        coalesced_store<z_count>(d_inout, idx0, r[0], stage);
        coalesced_store<z_count>(d_inout, idx1, r[1], stage);
    } else {
        unsigned int z_shift = inp_mask==0 ? iterations : 0;
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            d_inout[idx0 + (z << z_shift)] = r[0][z];
            d_inout[idx1 + (z << z_shift)] = r[1][z];
        }
    }
}

template<int z_count, bool coalesced = false, class fr_t>
__launch_bounds__(768, 1) __global__
void _GS_NTT_b(const unsigned int radix, const unsigned int lg_domain_size,
               const unsigned int stage, const unsigned int iterations,
               fr_t* d_base, const size_t col_stride,
               const fr_t (*d_partial_twiddles)[WINDOW_SIZE],
               const fr_t (*d_plus_one_twiddles)[1024],
               const fr_t* d_radix6_twiddles, const fr_t* d_radixX_twiddles,
               bool is_intt, const fr_t d_domain_size_inverse)
{
    fr_t* d_inout = d_base + (size_t)blockIdx.y * col_stride;   // <- column select (only change)
    extern __shared__ int xchg_raw[];
    fr_t* shared_exchange = reinterpret_cast<fr_t*>(xchg_raw);

    index_t tid = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    const index_t diff_mask = (1 << (iterations - 1)) - 1;
    const index_t inp_mask = ((index_t)1 << (stage - 1)) - 1;
    const index_t out_mask = ((index_t)1 << (stage - iterations)) - 1;

    const index_t tiz = (tid & ~diff_mask) * z_count + (tid & diff_mask);

    index_t idx0 = (tiz & ~inp_mask) * 2;
    idx0 += (tiz << (stage - iterations)) & inp_mask;
    idx0 += (tiz >> (iterations - 1)) & out_mask;
    index_t idx1 = idx0 + ((index_t)1 << (stage - 1));

    fr_t r[2][z_count];

    if (coalesced) {
        coalesced_load<z_count>(r[0], d_inout, idx0, stage - iterations);
        coalesced_load<z_count>(r[1], d_inout, idx1, stage - iterations);
        transpose<z_count>(r[0]);
        __syncwarp();
        transpose<z_count>(r[1]);
    } else {
        unsigned int z_shift = out_mask==0 ? iterations : 0;
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            r[0][z] = d_inout[idx0 + (z << z_shift)];
            r[1][z] = d_inout[idx1 + (z << z_shift)];
        }
    }

    #pragma unroll 1
    for (unsigned int s = iterations; --s >= 6;) {
        unsigned int laneMask = 1 << (s - 1);
        unsigned int thrdMask = (1 << s) - 1;
        unsigned int rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

        fr_t root = d_radixX_twiddles[rank << (radix - (s + 1))];

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            fr_t t = root * (r[0][z] - r[1][z]);
            r[0][z] = r[0][z] + r[1][z];
            r[1][z] = t;
        }

        __syncthreads();

        fr_t (*xchg)[z_count] = reinterpret_cast<decltype(xchg)>(shared_exchange);

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            fr_t t = fr_t::csel(r[1][z], r[0][z], pos);
            xchg[threadIdx.x][z] = t;
        }

        __syncthreads();

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            fr_t t = xchg[threadIdx.x ^ laneMask][z];
            r[0][z] = fr_t::csel(r[0][z], t, pos);
            r[1][z] = fr_t::csel(t, r[1][z], pos);
        }
    }

    #pragma unroll 1
    for (unsigned int s = min(iterations, 6u); --s >= 1;) {
        unsigned int laneMask = 1 << (s - 1);
        unsigned int thrdMask = (1 << s) - 1;
        unsigned int rank = threadIdx.x & thrdMask;
        bool pos = rank < laneMask;

        fr_t root = d_radix6_twiddles[rank << (6 - (s + 1))];

        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            fr_t t = root * (r[0][z] - r[1][z]);
            r[0][z] = r[0][z] + r[1][z];
            r[1][z] = t;

            t = fr_t::csel(r[1][z], r[0][z], pos);

            t.shfl_bfly(laneMask);

            r[0][z] = fr_t::csel(r[0][z], t, pos);
            r[1][z] = fr_t::csel(t, r[1][z], pos);
        }
    }

    #pragma unroll
    for (int z = 0; z < z_count; z++) {
        fr_t t = r[0][z] - r[1][z];
        r[0][z] = r[0][z] + r[1][z];
        r[1][z] = t;
    }

    if (stage - iterations != 0) {
        index_t thread_ntt_pos = (tiz & inp_mask) >> (iterations - 1);
        unsigned int thread_ntt_idx = (tiz & diff_mask) * 2;
        unsigned int nbits = MAX_LG_DOMAIN_SIZE - (stage - iterations);
        index_t idx0r = bit_rev(thread_ntt_idx, nbits);
        index_t root_idx0 = idx0r * thread_ntt_pos;
        index_t root_idx1 = root_idx0 + (thread_ntt_pos << (nbits - 1));

        fr_t first_root, second_root;
        get_intermediate_roots(first_root, second_root,
                               root_idx0, root_idx1, d_partial_twiddles);
        r[0][0] = r[0][0] * first_root;
        r[1][0] = r[1][0] * second_root;

        if (z_count > 1) {
            unsigned int off = nbits >= 10 ? (nbits - 10) : 0;
            unsigned int scale = nbits >= 10 ? 0 : (10 - nbits);

            thread_ntt_idx <<= scale;
            fr_t first_root_z = d_plus_one_twiddles[off][thread_ntt_idx];
            fr_t second_root_z = d_plus_one_twiddles[off][thread_ntt_idx + (1<<scale)];

            #pragma unroll
            for (int z = 1; z < z_count; z++) {
                first_root *= first_root_z;
                second_root *= second_root_z;
                r[0][z] = r[0][z] * first_root;
                r[1][z] = r[1][z] * second_root;
            }
        }
    }

    if (is_intt && stage == iterations) {
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            r[0][z] = r[0][z] * d_domain_size_inverse;
            r[1][z] = r[1][z] * d_domain_size_inverse;
        }
    }

    index_t mask = (index_t)((1 << iterations) - 1) << (stage - iterations);
    index_t rotw = idx0 & mask;
    rotw = (rotw << 1) | (rotw >> (iterations - 1));
    idx0 = (idx0 & ~mask) | (rotw & mask);
    rotw = idx1 & mask;
    rotw = (rotw << 1) | (rotw >> (iterations - 1));
    idx1 = (idx1 & ~mask) | (rotw & mask);

    if (coalesced) {
        transpose<z_count>(r[0]);
        __syncwarp();
        transpose<z_count>(r[1]);
        coalesced_store<z_count>(d_inout, idx0, r[0], stage - iterations + 1);
        coalesced_store<z_count>(d_inout, idx1, r[1], stage - iterations + 1);
    } else {
        unsigned int z_shift = out_mask==0 ? iterations : 0;
        #pragma unroll
        for (int z = 0; z < z_count; z++) {
            d_inout[idx0 + (z << z_shift)] = r[0][z];
            d_inout[idx1 + (z << z_shift)] = r[1][z];
        }
    }
}

// Batched launchers: same step-config math as sppark's CT_launcher / GS_launcher, with
// gridDim.y = ncols and the column stride threaded through.
namespace {

struct CTBatched {
    fr_t* base; size_t col_stride; unsigned ncols;
    int lg; bool is_intt; const NTTParameters& p; cudaStream_t s;
    int stage = 0;

    void step(int iterations)
    {
        assert(iterations <= 10);
        const int radix = iterations < 6 ? 6 : iterations;

        index_t num_threads = (index_t)1 << (lg - 1);
        index_t block_size = 1 << (radix - 1);
        index_t num_blocks;
        block_size = (num_threads <= block_size) ? num_threads : block_size;
        num_blocks = (num_threads + block_size - 1) / block_size;

        const int Z_COUNT = 256/8/sizeof(fr_t);
        size_t shared_sz = sizeof(fr_t) << (radix - 1);

        #define NTT_ARGUMENTS_B radix, (unsigned)lg, (unsigned)stage, (unsigned)iterations, \
                base, col_stride, p.partial_twiddles, p.plus_one_twiddles, \
                p.twiddles[0], p.twiddles[radix-6], is_intt, domain_size_inverse[lg]
        if (num_blocks < Z_COUNT)
            _CT_NTT_b<1><<<dim3(num_blocks, ncols), block_size, shared_sz, s>>>(NTT_ARGUMENTS_B);
        else if (stage == 0 || lg < 12)
            _CT_NTT_b<Z_COUNT><<<dim3(num_blocks/Z_COUNT, ncols), block_size, Z_COUNT*shared_sz, s>>>(NTT_ARGUMENTS_B);
        else
            _CT_NTT_b<Z_COUNT, true><<<dim3(num_blocks/Z_COUNT, ncols), block_size, Z_COUNT*shared_sz, s>>>(NTT_ARGUMENTS_B);
        #undef NTT_ARGUMENTS_B
        CUDA_OK(cudaGetLastError());
        stage += iterations;
    }

    // Step split copied from sppark NTT::CT_NTT (forward, RN order -> CT algorithm).
    void run()
    {
        if (lg <= 10) { step(lg); }
        else if (lg <= 18) { int st = lg/2; step(st + lg%2); step(st); }
        else if (lg <= 30) { int st = lg/3, rem = lg%3;
            step(st); step(st + (lg == 29 ? 1 : 0)); step(st + (lg == 29 ? 1 : rem)); }
        else { assert(false); }
    }
};

struct GSBatched {
    fr_t* base; size_t col_stride; unsigned ncols;
    int lg; bool is_intt; const NTTParameters& p; cudaStream_t s;
    int stage;

    void step(int iterations)
    {
        assert(iterations <= 10);
        const int radix = iterations < 6 ? 6 : iterations;

        index_t num_threads = (index_t)1 << (lg - 1);
        index_t block_size = 1 << (radix - 1);
        index_t num_blocks;
        block_size = (num_threads <= block_size) ? num_threads : block_size;
        num_blocks = (num_threads + block_size - 1) / block_size;

        const int Z_COUNT = 256/8/sizeof(fr_t);
        size_t shared_sz = sizeof(fr_t) << (radix - 1);

        #define NTT_ARGUMENTS_B radix, (unsigned)lg, (unsigned)stage, (unsigned)iterations, \
                base, col_stride, p.partial_twiddles, p.plus_one_twiddles, \
                p.twiddles[0], p.twiddles[radix-6], is_intt, domain_size_inverse[lg]
        if (num_blocks < Z_COUNT)
            _GS_NTT_b<1><<<dim3(num_blocks, ncols), block_size, shared_sz, s>>>(NTT_ARGUMENTS_B);
        else if (stage == iterations || lg < 12)
            _GS_NTT_b<Z_COUNT><<<dim3(num_blocks/Z_COUNT, ncols), block_size, Z_COUNT*shared_sz, s>>>(NTT_ARGUMENTS_B);
        else
            _GS_NTT_b<Z_COUNT, true><<<dim3(num_blocks/Z_COUNT, ncols), block_size, Z_COUNT*shared_sz, s>>>(NTT_ARGUMENTS_B);
        #undef NTT_ARGUMENTS_B
        CUDA_OK(cudaGetLastError());
        stage -= iterations;
    }

    // Step split copied from sppark NTT::GS_NTT (inverse, NR order -> GS algorithm).
    void run()
    {
        stage = lg;
        if (lg <= 10) { step(lg); }
        else if (lg <= 18) { int st = lg/2; step(st); step(st + lg%2); }
        else if (lg <= 30) { int st = lg/3, rem = lg%3;
            step(st + (lg == 29 ? 1 : rem)); step(st + (lg == 29 ? 1 : 0)); step(st); }
        else { assert(false); }
    }
};

} // anonymous namespace

// Batched flat LDE. Same per-column semantics/results as sppark_lde_flat (preserve_src
// path), different launch shape:
//   1. batched copy of the chunk's src columns into the TOP N of each dst column block
//      (src never written -> preserve_src holds by construction)
//   2. batched GS-iNTT (NR order) on the tops
//   3. per column: top -> scratch, spk_ldeSpread(scratch -> full block)  [reuses the
//      existing kernel + the caller's single-N scratch contract]
//   4. batched CT-NTT (RN order) on the full extended blocks
extern "C" void sppark_lde_flat_batched(void *d_dst_v, void *d_src_v,
                                        uint32_t lg_n, uint32_t lg_next, uint32_t nCols,
                                        bool preserve_src, void *preserve_scratch, void *caller_stream)
{
    if (!preserve_src || nCols == 0) {   // batched staging relies on the dst-top trick
        sppark_lde_flat(d_dst_v, d_src_v, lg_n, lg_next, nCols, preserve_src, preserve_scratch, caller_stream);
        return;
    }
    assert(preserve_scratch);

    cudaStream_t cs = (cudaStream_t)caller_stream;
    int dev = sp_device_id(cs);
    gl64_t *d_dst = reinterpret_cast<gl64_t *>(d_dst_v);
    gl64_t *d_src = reinterpret_cast<gl64_t *>(d_src_v);
    gl64_t *scratch = reinterpret_cast<gl64_t *>(preserve_scratch);
    size_t N = (size_t)1 << lg_n;
    size_t Next = (size_t)1 << lg_next;

    // L2-sized column chunk: the 2-3 steps of a transform reuse the whole domain, and that
    // reuse only stays in L2 if the chunk fits (~60% budget leaves room for twiddles).
    static int l2Bytes = -1;
    if (l2Bytes < 0 && cudaDeviceGetAttribute(&l2Bytes, cudaDevAttrL2CacheSize, dev) != cudaSuccess)
        l2Bytes = 32 << 20;
    uint32_t chunk = (uint32_t)(((size_t)l2Bytes * 3 / 5) / (Next * sizeof(gl64_t)));
    if (chunk > nCols) chunk = nCols;
    // A 1-column chunk is the stock serial shape, but with an extra top->scratch staging
    // copy per column -- measured 4-5% slower at 2^22. Batching must earn its staging cost.
    if (chunk < 2) {
        sppark_lde_flat(d_dst_v, d_src_v, lg_n, lg_next, nCols, preserve_src, preserve_scratch, caller_stream);
        return;
    }

    const NTTParameters& pf = NTTParameters::all(false)[dev];
    const NTTParameters& pi = NTTParameters::all(true)[dev];
    const auto gen_powers = pf.partial_group_gen_powers;

    for (uint32_t c0 = 0; c0 < nCols; c0 += chunk) {
        uint32_t nc = (c0 + chunk <= nCols) ? chunk : (nCols - c0);
        gl64_t *dchunk = d_dst + (size_t)c0 * Next;

        {   // 1) src columns -> dst block tops
            uint32_t thr = 256;
            uint32_t blk = (uint32_t)std::min<size_t>((N + thr - 1) / thr, 4096);
            spk_copyToTops<<<dim3(blk, nc), thr, 0, cs>>>((fr_t *)dchunk, (const fr_t *)(d_src + (size_t)c0 * N), lg_n, Next);
            CUDA_OK(cudaGetLastError());
        }

        // 2) batched iNTT on the tops
        GSBatched{(fr_t *)(dchunk + (Next - N)), Next, nc, (int)lg_n, true, pi, cs}.run();

        // 3) spread each column: top -> scratch -> full block (top gets overwritten by the
        //    spread, so it must be copied out first; scratch reuse is safe, same stream)
        for (uint32_t c = 0; c < nc; c++) {
            gl64_t *blockp = dchunk + (size_t)c * Next;
            CUDA_OK(cudaMemcpyAsync(scratch, blockp + (Next - N), N * sizeof(gl64_t),
                                    cudaMemcpyDeviceToDevice, cs));
            uint32_t thr = 256;
            uint32_t blk = (uint32_t)((N + thr - 1) / thr);
            spk_ldeSpread<<<blk, thr, 0, cs>>>((fr_t *)blockp, (const fr_t *)scratch, gen_powers,
                                               lg_n, lg_next - lg_n);
            CUDA_OK(cudaGetLastError());
        }

        // 4) batched forward NTT on the full extended blocks
        CTBatched{(fr_t *)dchunk, Next, nc, (int)lg_next, false, pf, cs}.run();
    }
}
