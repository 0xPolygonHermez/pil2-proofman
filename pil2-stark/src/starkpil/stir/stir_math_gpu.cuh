#ifndef STIR_MATH_GPU_CUH
#define STIR_MATH_GPU_CUH
// Device twins of stir_math.hpp. Include from ONE library TU only (kernels are defined here).
#include "gl64_t.cuh"
#include "goldilocks_cubic_extension.cuh"
#include "ntt_goldilocks.cuh"
#include "cuda_utils.cuh"
#include "starks_gpu.cuh"

namespace stirgpu
{
using E3 = Goldilocks3GPU::Element;

__device__ __forceinline__ gl64_t powBase(gl64_t b, uint64_t e)
{
    gl64_t r(uint64_t(1));
    while (e) { if (e & 1) r *= b; b *= b; e >>= 1; }
    return r;
}
__device__ __forceinline__ void ld(E3 &o, const gl64_t *p) { o[0] = p[0]; o[1] = p[1]; o[2] = p[2]; }
__device__ __forceinline__ void st(gl64_t *p, const E3 &v) { p[0] = v[0]; p[1] = v[1]; p[2] = v[2]; }
__device__ __forceinline__ bool eq(const E3 &a, const E3 &b) { return (a[0] - b[0]).is_zero() && (a[1] - b[1]).is_zero() && (a[2] - b[2]).is_zero(); }   // partially-reduced limbs: compare via is_zero

// Leaf m, member j <- point m + j·nLeaves (stir.hpp commit). Source interleaved or column-major.
static __global__ void cosetLeavesK(gl64_t *dst, const gl64_t *src, uint64_t N, uint64_t logK, bool colMajor)
{
    uint64_t p = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;   // destination point index
    if (p >= N) return;
    uint64_t k = 1ull << logK, nLeaves = N >> logK, m = p >> logK, j = p & (k - 1);
    uint64_t srcPt = m + j * nLeaves;
    for (int c = 0; c < 3; c++) dst[p * 3 + c] = colMajor ? src[c * N + srcPt] : src[srcPt * 3 + c];
}

static __global__ void interleavedToColMajorK(gl64_t *dst, const gl64_t *src, uint64_t N)
{
    uint64_t r = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (r >= N) return;
    for (int c = 0; c < 3; c++) dst[c * N + r] = src[r * 3 + c];
}

// Column j (< N) scaled by s^j; columns >= nKeep zeroed (padding).
static __global__ void scalePowersK(gl64_t *colMajor, uint64_t N, uint64_t nKeep, uint64_t s_)
{
    uint64_t j = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (j >= N) return;
    gl64_t f = j < nKeep ? powBase(gl64_t(s_), j) : gl64_t(uint64_t(0));
    for (int c = 0; c < 3; c++) colMajor[c * N + j] *= f;
}

// From interleaved coefficients (nCoeffs) to a zero-padded, shift-scaled column-major buffer.
static __global__ void loadCoeffsScaledK(gl64_t *colMajor, const gl64_t *coeffs, uint64_t N, uint64_t nCoeffs, uint64_t s_)
{
    uint64_t j = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (j >= N) return;
    gl64_t f = powBase(gl64_t(s_), j);
    for (int c = 0; c < 3; c++) colMajor[c * N + j] = j < nCoeffs ? coeffs[j * 3 + c] * f : gl64_t(uint64_t(0));
}

static __global__ void colMajorPrefixToInterleavedK(gl64_t *dst, const gl64_t *colMajor, uint64_t N, uint64_t n)
{
    uint64_t j = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (j >= n) return;
    for (int c = 0; c < 3; c++) dst[j * 3 + c] = colMajor[c * N + j];
}

inline dim3 grid1(uint64_t n, uint32_t t = 256) { return dim3((uint32_t)((n + t - 1) / t)); }

inline void cosetLeaves(gl64_t *dst, const gl64_t *src, uint64_t logN, uint64_t logK, bool srcColMajor, cudaStream_t stream)
{
    uint64_t N = 1ull << logN;
    cosetLeavesK<<<grid1(N), 256, 0, stream>>>(dst, src, N, logK, srcColMajor);
    CHECKCUDAERR(cudaGetLastError());
}

inline void evalsToCoeffs(gl64_t *dstInterleaved, uint64_t nKeep, gl64_t *scratch, const gl64_t *srcInterleaved,
                          uint64_t logN, Goldilocks::Element shift, cudaStream_t stream)
{
    uint64_t N = 1ull << logN;
    interleavedToColMajorK<<<grid1(N), 256, 0, stream>>>(scratch, srcInterleaved, N);
    NTTGoldilocksGPU ntt;
    ntt.INTT(scratch, logN, FIELD_EXTENSION, stream);
    scalePowersK<<<grid1(N), 256, 0, stream>>>(scratch, N, N, Goldilocks::inv(shift).fe);
    colMajorPrefixToInterleavedK<<<grid1(nKeep), 256, 0, stream>>>(dstInterleaved, scratch, N, nKeep);
    CHECKCUDAERR(cudaGetLastError());
}

inline void coeffsToEvalsColMajor(gl64_t *dstColMajor, const gl64_t *coeffsInterleaved, uint64_t nCoeffs,
                                  uint64_t logN, Goldilocks::Element shift, cudaStream_t stream)
{
    uint64_t N = 1ull << logN;
    loadCoeffsScaledK<<<grid1(N), 256, 0, stream>>>(dstColMajor, coeffsInterleaved, N, nCoeffs, shift.fe);
    NTTGoldilocksGPU ntt;
    ntt.nttColMajor(dstColMajor, logN, FIELD_EXTENSION, stream);   // NTT() asserts logN >= 8
    CHECKCUDAERR(cudaGetLastError());
}

// STIR's L_i is shift·<w> (not shift^(2^j)·<w> as in FRI), so fold_inplace is called with
// nBitsExt = logL: its first sub-step then uses shift^-1 and later ones shift^-(2^bits).
inline void foldStir(gl64_t *f, const gl64_t *d_rFold, uint64_t logL, uint64_t logK, TimerGPU &timer, cudaStream_t stream)
{
    fold_inplace(1, 0, (Goldilocks::Element *)d_rFold, logL, logL, logL - logK, f, timer, stream);
}

constexpr uint32_t G_THREADS = 256;

// out = Σ c_j x^j: thread t sums the stride-T subsequence by Horner in x^T, scales by x^t, then block-reduces.
static __global__ void hornerK(gl64_t *out, const gl64_t *coeffs, uint64_t n, const gl64_t *xp)
{
    __shared__ gl64_t red[G_THREADS * 3];
    uint32_t t = threadIdx.x;
    E3 xT, xt, b;
    ld(b, xp); Goldilocks3GPU::pow(b, G_THREADS, xT);
    ld(b, xp); Goldilocks3GPU::pow(b, t, xt);
    E3 acc; Goldilocks3GPU::zero(acc);
    if (t < n) {
        int64_t last = (int64_t)t + (int64_t)(((n - 1 - t) / G_THREADS) * G_THREADS);
        for (int64_t j = last; j >= (int64_t)t; j -= G_THREADS) {
            E3 c; ld(c, &coeffs[j * 3]);
            Goldilocks3GPU::mul(acc, acc, xT);
            Goldilocks3GPU::add(acc, acc, c);
        }
    }
    Goldilocks3GPU::mul(acc, acc, xt);
    st(&red[t * 3], acc);
    __syncthreads();
    for (uint32_t w = G_THREADS / 2; w > 0; w >>= 1) {
        if (t < w) { E3 a, c; ld(a, &red[t * 3]); ld(c, &red[(t + w) * 3]); Goldilocks3GPU::add(a, a, c); st(&red[t * 3], a); }
        __syncthreads();
    }
    if (t == 0) { out[0] = red[0]; out[1] = red[1]; out[2] = red[2]; }
}
inline void horner(gl64_t *d_out, const gl64_t *d_coeffs, uint64_t n, const gl64_t *d_x, cudaStream_t stream)
{
    hornerK<<<1, G_THREADS, 0, stream>>>(d_out, d_coeffs, n, d_x);
    CHECKCUDAERR(cudaGetLastError());
}

// Candidate c < s is r_out[c], else the shift point of idx[c - s]; it survives unless an earlier
// candidate equals it, and survivors are compacted in candidate order (QuotientContext::add).
static __global__ void buildGK(gl64_t *pts, gl64_t *vals, uint64_t *nG, const gl64_t *rOut, const gl64_t *beta, uint64_t s,
                               const uint64_t *idx, uint64_t t, const gl64_t *fold, uint64_t shiftK_, uint64_t w_)
{
    extern __shared__ gl64_t sm[];   // total E3 candidates, then total u32 keep/offset, then blockDim u32 counts
    const uint64_t total = s + t;
    gl64_t *cand = sm;
    uint32_t *keep = (uint32_t *)(sm + total * 3), *cnt = keep + total;
    for (uint64_t i = threadIdx.x; i < total; i += blockDim.x) {
        E3 p;
        if (i < s) ld(p, &rOut[i * 3]);
        else { p[0] = gl64_t(shiftK_) * powBase(gl64_t(w_), idx[i - s]); p[1] = gl64_t(uint64_t(0)); p[2] = gl64_t(uint64_t(0)); }
        st(&cand[i * 3], p);
    }
    __syncthreads();
    // Contiguous chunk per thread, so a scan over the per-thread counts preserves candidate order.
    const uint64_t chunk = (total + blockDim.x - 1) / blockDim.x, lo = threadIdx.x * chunk, hi = min(total, lo + chunk);
    uint32_t c = 0;
    for (uint64_t i = lo; i < hi; i++) {
        E3 p; ld(p, &cand[i * 3]);
        uint32_t k = 1;
        for (uint64_t j = 0; j < i && k; j++) { E3 q; ld(q, &cand[j * 3]); if (eq(q, p)) k = 0; }
        keep[i] = k; c += k;
    }
    cnt[threadIdx.x] = c;
    __syncthreads();
    for (uint32_t off = 1; off < blockDim.x; off <<= 1) {   // inclusive Hillis-Steele scan
        uint32_t v = threadIdx.x >= off ? cnt[threadIdx.x - off] : 0;
        __syncthreads();
        cnt[threadIdx.x] += v;
        __syncthreads();
    }
    uint32_t at = cnt[threadIdx.x] - c;
    for (uint64_t i = lo; i < hi; i++) {
        if (!keep[i]) continue;
        E3 v;
        if (i < s) ld(v, &beta[i * 3]); else ld(v, &fold[idx[i - s] * 3]);
        E3 p; ld(p, &cand[i * 3]);
        st(&pts[at * 3], p); st(&vals[at * 3], v); at++;
    }
    if (threadIdx.x == blockDim.x - 1) *nG = cnt[threadIdx.x];
}
// w() is host-only, so the root of unity is passed in.
inline void buildG(gl64_t *d_points, gl64_t *d_values, uint64_t *d_nG, const gl64_t *d_rOut, const gl64_t *d_beta, uint64_t s,
                   const uint64_t *d_idx, uint64_t t, const gl64_t *d_fold, Goldilocks::Element shiftK, uint64_t logLk, cudaStream_t stream)
{
    constexpr uint32_t threads = 256;
    const uint64_t total = s + t, bytes = total * 3 * sizeof(gl64_t) + (total + threads) * sizeof(uint32_t);
    if (bytes > 48 * 1024)
        CHECKCUDAERR(cudaFuncSetAttribute(buildGK, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
    buildGK<<<1, threads, bytes, stream>>>(d_points, d_values, d_nG, d_rOut, d_beta, s, d_idx, t, d_fold, shiftK.fe, Goldilocks::w(logLk).fe);
    CHECKCUDAERR(cudaGetLastError());
}

constexpr uint32_t R_THREADS = 128;

// Block-wide E3 sum; the result is valid in thread 0 only.
__device__ __forceinline__ void blockSumE3(E3 &r, const E3 &v, gl64_t *red)
{
    const uint32_t t = threadIdx.x;
    st(&red[t * 3], v);
    __syncthreads();
    for (uint32_t w = R_THREADS / 2; w > 0; w >>= 1) {
        if (t < w) { E3 a, b; ld(a, &red[t * 3]); ld(b, &red[(t + w) * 3]); Goldilocks3GPU::add(a, a, b); st(&red[t * 3], a); }
        __syncthreads();
    }
    ld(r, red);
}

// Z(X) = Π_{j<n} (X - a_j), n+1 coefficients, one parallel step per root (double buffer in shared memory).
static __global__ void vanishingPolyK(gl64_t *zOut, uint64_t nPad, const gl64_t *pts, const uint64_t *nGp)
{
    extern __shared__ gl64_t zs[];   // 2 x (nPad+1) E3 for Z, then the n roots (a global load per step would serialize)
    const uint64_t n = *nGp;
    gl64_t *cur = zs, *nxt = zs + (nPad + 1) * 3, *A = zs + 2 * (nPad + 1) * 3;
    for (uint64_t m = threadIdx.x; m < n * 3; m += blockDim.x) A[m] = pts[m];
    for (uint64_t m = threadIdx.x; m <= nPad; m += blockDim.x) {
        E3 v; if (m == 0) Goldilocks3GPU::one(v); else Goldilocks3GPU::zero(v);
        st(&cur[m * 3], v);
    }
    __syncthreads();
    for (uint64_t j = 0; j < n; j++) {
        E3 a; ld(a, &A[j * 3]);
        for (uint64_t m = threadIdx.x; m <= j + 1; m += blockDim.x) {   // deg(cur) = j: cur[j+1] is not written yet
            E3 lo, hi, r;
            if (m > 0) ld(lo, &cur[(m - 1) * 3]); else Goldilocks3GPU::zero(lo);
            if (m <= j) { ld(hi, &cur[m * 3]); Goldilocks3GPU::mul(hi, hi, a); } else Goldilocks3GPU::zero(hi);
            Goldilocks3GPU::sub(r, lo, hi);
            st(&nxt[m * 3], r);
        }
        __syncthreads();
        gl64_t *t = cur; cur = nxt; nxt = t;
    }
    for (uint64_t m = threadIdx.x; m <= nPad; m += blockDim.x) {
        E3 v; if (m <= n) ld(v, &cur[m * 3]); else Goldilocks3GPU::zero(v);
        st(&zOut[m * 3], v);
    }
}

// Block i: c_i = y_i / Z'(a_i); strided Horner in a^R over d_k = (k+1)·Z_{k+1}, k < n.
static __global__ void baryWeightsK(gl64_t *c, const gl64_t *z, const gl64_t *pts, const gl64_t *vals, const uint64_t *nGp)
{
    __shared__ gl64_t red[R_THREADS * 3];
    const uint64_t n = *nGp, i = blockIdx.x, t = threadIdx.x;
    if (i >= n) return;
    E3 b, aR, at; ld(b, &pts[i * 3]); Goldilocks3GPU::pow(b, R_THREADS, aR);
    ld(b, &pts[i * 3]); Goldilocks3GPU::pow(b, t, at);
    E3 acc; Goldilocks3GPU::zero(acc);
    if (t < n) {
        int64_t last = (int64_t)t + (int64_t)(((n - 1 - t) / R_THREADS) * R_THREADS);
        for (int64_t k = last; k >= (int64_t)t; k -= R_THREADS) {
            E3 d; ld(d, &z[(k + 1) * 3]); Goldilocks3GPU::mul(d, d, (uint64_t)(k + 1));
            Goldilocks3GPU::mul(acc, acc, aR);
            Goldilocks3GPU::add(acc, acc, d);
        }
    }
    Goldilocks3GPU::mul(acc, acc, at);
    E3 dz; blockSumE3(dz, acc, red);
    if (t == 0) { E3 y; ld(y, &vals[i * 3]); Goldilocks3GPU::inv(dz, dz); Goldilocks3GPU::mul(y, y, dz); st(&c[i * 3], y); }
}

// Block e: T_e = Σ_i c_i·a_i^e.
static __global__ void weightedPowerSumsK(gl64_t *T, const gl64_t *c, const gl64_t *pts, const uint64_t *nGp)
{
    __shared__ gl64_t red[R_THREADS * 3];
    const uint64_t n = *nGp, e = blockIdx.x;
    if (e >= n) return;
    E3 acc; Goldilocks3GPU::zero(acc);
    for (uint64_t i = threadIdx.x; i < n; i += R_THREADS) {
        E3 a, ae, ci; ld(a, &pts[i * 3]); Goldilocks3GPU::pow(a, e, ae); ld(ci, &c[i * 3]);
        Goldilocks3GPU::mul(ae, ae, ci); Goldilocks3GPU::add(acc, acc, ae);
    }
    E3 r; blockSumE3(r, acc, red);
    if (threadIdx.x == 0) st(&T[e * 3], r);
}

// Block j: Âns_j = Σ_{e<n-j} Z_{j+1+e}·T_e, from Z(X)/(X - a_i) = Σ_j X^j Σ_{m>j} Z_m a_i^{m-1-j}; zero for j >= n.
static __global__ void ansFromZK(gl64_t *coeffs, const gl64_t *z, const gl64_t *T, const uint64_t *nGp)
{
    __shared__ gl64_t red[R_THREADS * 3];
    const uint64_t n = *nGp, j = blockIdx.x;
    E3 acc; Goldilocks3GPU::zero(acc);
    for (uint64_t e = threadIdx.x; j + 1 + e <= n; e += R_THREADS) {
        E3 zm, te; ld(zm, &z[(j + 1 + e) * 3]); ld(te, &T[e * 3]);
        Goldilocks3GPU::mul(zm, zm, te); Goldilocks3GPU::add(acc, acc, zm);
    }
    E3 r; blockSumE3(r, acc, red);
    if (threadIdx.x == 0) st(&coeffs[j * 3], r);
}

// Lagrange form via Z: no sequential inversion chain. Scratch: tmp nPad E3 (c_i), z nPad+1 E3, T nPad E3.
inline void interpolate(gl64_t *d_coeffs, uint64_t nPad, const gl64_t *d_points, const gl64_t *d_values, gl64_t *d_tmp,
                        gl64_t *d_z, gl64_t *d_T, const uint64_t *d_nG, cudaStream_t stream)
{
    const uint64_t bytes = (3 * nPad + 2) * 3 * sizeof(gl64_t);
    if (bytes > 48 * 1024)
        CHECKCUDAERR(cudaFuncSetAttribute(vanishingPolyK, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
    vanishingPolyK<<<1, 256, bytes, stream>>>(d_z, nPad, d_points, d_nG);
    if (nPad > 0) {
        baryWeightsK<<<(uint32_t)nPad, R_THREADS, 0, stream>>>(d_tmp, d_z, d_points, d_values, d_nG);
        weightedPowerSumsK<<<(uint32_t)nPad, R_THREADS, 0, stream>>>(d_T, d_tmp, d_points, d_nG);
        ansFromZK<<<(uint32_t)nPad, R_THREADS, 0, stream>>>(d_coeffs, d_z, d_T, d_nG);
    }
    CHECKCUDAERR(cudaGetLastError());
}

// Block e: S_e = Σ_i a_i^e, only e < n-1 is used.
static __global__ void powerSumsK(gl64_t *S, const gl64_t *pts, const uint64_t *nGp)
{
    __shared__ gl64_t red[R_THREADS * 3];
    const uint64_t n = *nGp, e = blockIdx.x;
    if (e + 1 >= n) return;
    E3 acc; Goldilocks3GPU::zero(acc);
    for (uint64_t i = threadIdx.x; i < n; i += R_THREADS) {
        E3 a, ae; ld(a, &pts[i * 3]); Goldilocks3GPU::pow(a, e, ae); Goldilocks3GPU::add(acc, acc, ae);
    }
    E3 r; blockSumE3(r, acc, red);
    if (threadIdx.x == 0) st(&S[e * 3], r);
}

// Block j: Sh_j = Σ_e ans_{j+1+e}·S_e (synthetic division summed over G); zero for j >= n-1.
static __global__ void shakeFromSK(gl64_t *out, const gl64_t *ans, const gl64_t *S, const uint64_t *nGp)
{
    __shared__ gl64_t red[R_THREADS * 3];
    const uint64_t n = *nGp, j = blockIdx.x;
    E3 acc; Goldilocks3GPU::zero(acc);
    for (uint64_t e = threadIdx.x; j + 1 + e < n; e += R_THREADS) {
        E3 p, sv; ld(p, &ans[(j + 1 + e) * 3]); ld(sv, &S[e * 3]);
        Goldilocks3GPU::mul(p, p, sv); Goldilocks3GPU::add(acc, acc, p);
    }
    E3 r; blockSumE3(r, acc, red);
    if (threadIdx.x == 0) st(&out[j * 3], r);
}
inline void shake(gl64_t *d_out, uint64_t nPad, const gl64_t *d_ans, const gl64_t *d_points, gl64_t *d_powerSums,
                  const uint64_t *d_nG, cudaStream_t stream)
{
    if (nPad == 0) return;
    powerSumsK<<<(uint32_t)nPad, R_THREADS, 0, stream>>>(d_powerSums, d_points, d_nG);
    shakeFromSK<<<(uint32_t)nPad, R_THREADS, 0, stream>>>(d_out, d_ans, d_powerSums, d_nG);
    CHECKCUDAERR(cudaGetLastError());
}

// Columns 0..2 <- Âns (nPad coeffs), 3..5 <- Z (nPad+1 coeffs), each scaled by shift^j and zero-padded to N.
static __global__ void loadAnsZScaledK(gl64_t *colMajor6, const gl64_t *ans, const gl64_t *z, uint64_t N, uint64_t nPad, uint64_t s_)
{
    uint64_t j = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (j >= N) return;
    gl64_t f = j <= nPad ? powBase(gl64_t(s_), j) : gl64_t(uint64_t(0));
    for (int c = 0; c < 3; c++) {
        colMajor6[c * N + j] = j < nPad ? ans[j * 3 + c] * f : gl64_t(uint64_t(0));
        colMajor6[(3 + c) * N + j] = j <= nPad ? z[j * 3 + c] * f : gl64_t(uint64_t(0));
    }
}

// In place on column-major g: f = (g - Âns(x))*(1 - y^{n+1}) / (Z(x)*(1 - y)), y = r_comb*x
// (or *(n+1)/Z(x) when y = 1); the same value stir_math.hpp computes in two inversions.
static __global__ void quotientDegCorK(gl64_t *g, const gl64_t *evals6, uint64_t N, uint64_t shift_, uint64_t w_,
                                       const uint64_t *nGp, const gl64_t *rCombP)
{
    __shared__ gl64_t rcN[3];   // r_comb^{n+1}, so each point only needs x^{n+1} in the base field
    const uint64_t n = *nGp;
    E3 rc; ld(rc, rCombP);
    if (threadIdx.x == 0) { E3 b, r; ld(b, rCombP); Goldilocks3GPU::pow(b, n + 1, r); st(rcN, r); }   // pow overwrites its base
    __syncthreads();

    uint64_t m = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (m >= N) return;
    gl64_t x = gl64_t(shift_) * powBase(gl64_t(w_), m);
    E3 q, den;
    for (int c = 0; c < 3; c++) { q[c] = g[c * N + m] - evals6[c * N + m]; den[c] = evals6[(3 + c) * N + m]; }
    E3 y; for (int c = 0; c < 3; c++) y[c] = rc[c] * x;
    E3 one; Goldilocks3GPU::one(one);
    if (eq(y, one)) {
        Goldilocks3GPU::inv(den, den);
        Goldilocks3GPU::mul(q, q, den);
        gl64_t k(n + 1);
        for (int c = 0; c < 3; c++) q[c] = q[c] * k;
    } else {
        gl64_t xn = powBase(x, n + 1);
        E3 num; for (int c = 0; c < 3; c++) num[c] = one[c] - rcN[c] * xn;
        E3 omy; Goldilocks3GPU::sub(omy, one, y);
        Goldilocks3GPU::mul(den, den, omy);
        Goldilocks3GPU::inv(den, den);
        Goldilocks3GPU::mul(q, q, num);
        Goldilocks3GPU::mul(q, q, den);
    }
    for (int c = 0; c < 3; c++) g[c * N + m] = q[c];
}
// f (interleaved, out) also holds the 6·|L| Âns/Z evaluations as scratch; gColMajor is clobbered.
inline void quotientDegCor(gl64_t *f, gl64_t *gColMajor, uint64_t logL, Goldilocks::Element shift, const gl64_t *d_ans,
                           const gl64_t *d_z, uint64_t nPad, const uint64_t *d_nG, const gl64_t *d_rComb, cudaStream_t stream)
{
    const uint64_t N = 1ull << logL;
    loadAnsZScaledK<<<grid1(N), 256, 0, stream>>>(f, d_ans, d_z, N, nPad, shift.fe);
    NTTGoldilocksGPU ntt;
    ntt.nttColMajor(f, logL, 2 * FIELD_EXTENSION, stream);
    quotientDegCorK<<<grid1(N), 256, 0, stream>>>(gColMajor, f, N, shift.fe, Goldilocks::w(logL).fe, d_nG, d_rComb);
    colMajorPrefixToInterleavedK<<<grid1(N), 256, 0, stream>>>(f, gColMajor, N, N);
    CHECKCUDAERR(cudaGetLastError());
}

// Never clears the flag: the driver zeroes it at proof start.
static __global__ void oodFlagK(uint64_t *flag, const gl64_t *rOut, uint64_t s, uint64_t shiftInv_, uint64_t logL)
{
    for (uint64_t i = 0; i < s; i++) {
        E3 x; ld(x, &rOut[i * 3]);
        if (x[1].is_zero() && x[2].is_zero() && (powBase(x[0] * gl64_t(shiftInv_), 1ull << logL) - gl64_t(uint64_t(1))).is_zero())
            *flag = 1;
    }
}
inline void oodFlag(uint64_t *d_flag, const gl64_t *d_rOut, uint64_t s, Goldilocks::Element shift, uint64_t logL, cudaStream_t stream)
{
    oodFlagK<<<1, 1, 0, stream>>>(d_flag, d_rOut, s, Goldilocks::inv(shift).fe, logL);
    CHECKCUDAERR(cudaGetLastError());
}

// Coset index = raw & (nLeaves - 1), into a separate buffer so the raw round-1 indices survive.
static __global__ void modQueriesK(uint64_t *dst, const uint64_t *src, uint64_t n, uint64_t mask)
{
    uint64_t q = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (q < n) dst[q] = src[q] & mask;
}
inline void modQueries(uint64_t *dst, const uint64_t *src, uint64_t n, uint64_t nLeaves, cudaStream_t stream)
{
    modQueriesK<<<grid1(n), 256, 0, stream>>>(dst, src, n, nLeaves - 1);
    CHECKCUDAERR(cudaGetLastError());
}
} // namespace stirgpu
#endif
