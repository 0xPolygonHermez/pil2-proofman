// sppark-backed NTT primitives (LDE / computeQ / INTT) for the FLAT column-major layout.
// Prototypes + per-function contracts are in sppark_lde.cuh.
//
// Isolated TU: sppark's NTT headers clash with ntt_goldilocks.cuh, so sppark is compiled only here and
// reached via subclassing NTT (no submodule edit); callers see only the extern "C" prototypes.
//
// Roots: compiled -DGOLDILOCKS_PLONKY2 (Makefile NVCCFLAGS_GL) -> sppark's roots == prover Goldilocks::W,
// so all NTT twiddles match with no re-seed. Only the LDE coset generator differs (PLONKY2 group_gen
// 0xc65c.. vs prover SHIFT 7), so we re-seed just partial_group_gen_powers from 7 (ensure_prover_coset).

#include <ff/goldilocks.hpp>   // fr_t == gl64_t
#include <ntt/ntt.cuh>
#include <ntt/parameters.cuh>  // NTTParameters + generate_*_twiddles kernels
#include <util/gpu_t.cuh>
#include "goldilocks_trace_layout.cuh"   // getBufferOffset (column-major accessor)
#include "sppark_lde.cuh"                // extern "C" prototypes (decl/def check)

#include <mutex>

// =============================================================================
// sppark NTT access (subclass to reach protected primitives; no submodule edit)
// =============================================================================

// Non-cooperative replacement for sppark's LDE_launch (whose grid-sync exists only for in/out overlap).
// With DISTINCT in/out, one thread per idx writes r*7^bit_rev(idx) to out[idx<<lg_blowup] and zeroes the
// rest of the blowup group. Reuses sppark's get_intermediate_root + bit_rev -> bit-exact with LDE_launch.
// Launch: <<<ceil(domain_size/256), 256>>>
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
    // Coset-LDE: iNTT(small) -> spread+coset -> NTT(ext). d_buf holds ext_size elements.
    // spread_in (length domain_size, disjoint from d_buf) holds the small domain so spread in/out don't
    // overlap and we use the plain spk_ldeSpread; without it the small domain aliases d_buf's tail and we
    // fall back to sppark's cooperative LDE_launch.
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

    // Plain full-domain NTT/INTT (natural in/out, no coset) on a single contiguous column.
    // Matches the prover's nttDit (bit-reversal + DIT = NN order) on the same root.
    static void ntt_inplace(stream_t &stream, fr_t *d_col, uint32_t lg, bool inverse)
    {
        NTT_internal(d_col, lg, InputOutputOrder::NN,
                     inverse ? Direction::inverse : Direction::forward, Type::standard, stream);
    }
};

// ---- Coset-generator override (re-seed partial_group_gen_powers from SHIFT=7; see top of file) ----

#define GL_MOD 0xffffffff00000001ULL

static inline uint64_t gl_mulmod(uint64_t a, uint64_t b)
{
    return (uint64_t)(((__uint128_t)a * b) % GL_MOD);
}

// b^e mod p (used only to compute 7^-1 = 7^(p-2) for the inverse-direction coset table).
static uint64_t gl_powmod(uint64_t b, uint64_t e)
{
    uint64_t r = 1;
    b %= GL_MOD;
    while (e) {
        if (e & 1)
            r = gl_mulmod(r, b);
        b = gl_mulmod(b, b);
        e >>= 1;
    }
    return r;
}

static void override_coset_to_prover_shift(NTTParameters &p, const gpu_t &gpu, bool inverse)
{
    const uint64_t SHIFT = 7;
    uint64_t coset = inverse ? gl_powmod(SHIFT, GL_MOD - 2) : SHIFT;   // 7 or 7^-1
    generate_partial_twiddles<<<WINDOW_SIZE / 32, 32, 0, gpu[0]>>>(
        p.partial_group_gen_powers, fr_t(coset));
    CUDA_OK(cudaGetLastError());
    gpu[0].sync();
}

// Align sppark's coset generator with the prover's SHIFT for this device's NTTParameters. Caller
// guards against re-running per device (SpparkDevice::coset_done).
static void ensure_prover_coset(const gpu_t &gpu)
{
    override_coset_to_prover_shift(const_cast<NTTParameters &>(NTTParameters::all(false)[gpu.id()]), gpu, false);
    override_coset_to_prover_shift(const_cast<NTTParameters &>(NTTParameters::all(true)[gpu.id()]), gpu, true);
}

// =============================================================================
// Per-device persistent sppark state (stream + scratch + coset), serialized per device
// =============================================================================

// One per sppark device, created lazily and owned for the process lifetime (sppark's twiddle tables
// persist across calls). Each entry point holds `lock` for its whole duration, so the stream and the
// grow-only scratch are reused safely across concurrent host threads on the same GPU.
struct SpparkDevice {
    std::mutex lock;
    stream_t  *stream = nullptr;
    gl64_t    *scratch = nullptr;      // grow-only per-column iNTT workspace (preserve_src fallback)
    size_t     scratch_bytes = 0;
    bool       coset_done = false;     // SHIFT=7 coset override applied to this device's tables
};

#ifndef SPPARK_MAX_GPUS
#define SPPARK_MAX_GPUS 32
#endif
static SpparkDevice  g_dev[SPPARK_MAX_GPUS];
static std::mutex    g_dev_init_mutex;   // guards lazy stream creation

// SpparkDevice for the caller's current GPU, inited on first use. Does NOT take dev.lock (callers do).
static SpparkDevice &sp_device()
{
    // select_gpu(-1) resolves the current cuda device to sppark's LOGICAL gpu id (and cudaSetDevice's it).
    // We key on that id because sppark indexes NTTParameters::all()[id] / stream_t(id) by logical id,
    // which differs from the cuda ordinal when sppark filters out a device.
    int cuda_dev = -1;
    CUDA_OK(cudaGetDevice(&cuda_dev));
    const gpu_t &gpu = select_gpu(-1);
    if (gpu.cid() != cuda_dev) {
        fprintf(stderr, "[sppark_lde] current CUDA device %d is not in sppark's GPU list "
                        "(got logical id %d on cuda device %d); sppark cannot run here\n",
                cuda_dev, gpu.id(), gpu.cid());
        abort();
    }
    int dev = gpu.id();
    if (dev < 0 || dev >= SPPARK_MAX_GPUS) {
        fprintf(stderr, "[sppark_lde] sppark gpu id %d out of range [0,%d)\n", dev, SPPARK_MAX_GPUS);
        abort();
    }
    SpparkDevice &d = g_dev[dev];
    if (d.stream == nullptr) {
        std::lock_guard<std::mutex> g(g_dev_init_mutex);
        if (d.stream == nullptr) {
            (void)NTTParameters::all(false);
            (void)NTTParameters::all(true);
            if (!d.coset_done) {
                ensure_prover_coset(gpu);   // align this device's LDE coset with SHIFT=7
                d.coset_done = true;
            }
            d.stream = new stream_t(dev);   // stream on this (logical) device
        }
    }
    return d;
}

static gl64_t *ensure_scratch(gl64_t *&buf, size_t &cap, size_t need)
{
    if (need > cap) {
        if (buf)
            (void)cudaFree(buf);
        CUDA_OK(cudaMalloc(&buf, need));
        cap = need;
    }
    return buf;
}

// =============================================================================
// Entry points (declared in sppark_lde.cuh; see there for the full contracts)
// =============================================================================

// Per column c: iNTT(N) -> spk_ldeSpread (coset) into dst[c]. To keep src intact when preserve_src
// (callers reread cm1/const/custom commits), the iNTT runs on a COPY in col_scratch: preserve_scratch if
// given (a free arena N-slice, e.g. the mt region -> no alloc), else this device's grow-only dev.scratch.
extern "C" void sppark_lde_flat(void *d_dst_v, void *d_src_v,
                                uint32_t lg_n, uint32_t lg_next, uint32_t nCols,
                                bool preserve_src, void *preserve_scratch)
{
    SpparkDevice &dev = sp_device();
    std::lock_guard<std::mutex> g(dev.lock);   // serialize sppark on this device (stream + scratch reuse)
    stream_t &s = *dev.stream;
    gl64_t *d_dst = reinterpret_cast<gl64_t *>(d_dst_v);
    gl64_t *d_src = reinterpret_cast<gl64_t *>(d_src_v);
    uint32_t lg_blowup = lg_next - lg_n;
    uint64_t N = 1ull << lg_n;
    uint64_t Next = 1ull << lg_next;

    gl64_t *col_scratch = nullptr;
    if (preserve_src) {
        col_scratch = preserve_scratch
            ? reinterpret_cast<gl64_t *>(preserve_scratch)
            : ensure_scratch(dev.scratch, dev.scratch_bytes, (size_t)N * sizeof(gl64_t));
    }

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
    s.sync();
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
                                     uint32_t lg_n, uint32_t lg_next, uint32_t nCols)
{
    SpparkDevice &dev = sp_device();
    std::lock_guard<std::mutex> g(dev.lock);
    stream_t &s = *dev.stream;
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
    s.sync();
}

// Per-column in-place INTT of nCols flat columns of N rows (used for the LEv vector).
extern "C" void sppark_intt_flat(void *d_data_v, uint32_t lg_n, uint32_t nCols)
{
    SpparkDevice &dev = sp_device();
    std::lock_guard<std::mutex> g(dev.lock);
    stream_t &s = *dev.stream;
    gl64_t *d_data = reinterpret_cast<gl64_t *>(d_data_v);
    uint64_t N = 1ull << lg_n;
    for (uint32_t c = 0; c < nCols; c++)
        SpparkLDE::ntt_inplace(s, (fr_t *)(d_data + (uint64_t)c * N), lg_n, /*inverse=*/true);
    s.sync();
}