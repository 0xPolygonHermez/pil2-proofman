// CPU/GPU bit-identity for sha256_core.hpp. The header is compiled twice and the
// __CUDA_ARCH__ accessors around IV/K take different paths to the same constants,
// so "it is one header" is not proof they agree. Runs the CPU vectors on device.
// A disagreement here is a proof that fails to verify with nothing pointing at
// the hash, so this gates everything above the core.

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "sha256_core.hpp"
#include "sha256_core_vectors.hpp"

namespace V = sha256_vectors;

#define CUDA_ASSERT(call)                                                      \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        ASSERT_EQ(_e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(_e); \
    } while (0)

// One thread per case; each writes 4 digest words.
__global__ void shaLeafKernel(const uint64_t *in, const uint32_t *lens,
                              const uint32_t *offsets, uint64_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sha256core::hash_le64(in + offsets[i], lens[i], out + i * 4);
}

__global__ void shaNodeKernel(const uint64_t *in, const uint32_t *lens,
                              const uint32_t *offsets, uint64_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sha256core::node_hash(in + offsets[i], lens[i], out + i * 4);
}

__global__ void shaGrindKernel(const uint64_t *in, uint64_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sha256core::grind_hash(in + i * 8, out + i * 4);
}

__global__ void shaTrxKernel(const uint64_t *in, const uint32_t *lens,
                             const uint32_t *offsets, const uint64_t *counters,
                             uint64_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sha256core::Hasher h;
    h.init();
    h.absorb(in + offsets[i], lens[i]);
    h.squeeze(counters[i], out + i * 4);
}

// Flatten a set of variable-length inputs into one device buffer.
struct FlatInput
{
    std::vector<uint64_t> data;
    std::vector<uint32_t> lens;
    std::vector<uint32_t> offsets;

    void add(const uint64_t *in, uint32_t n)
    {
        offsets.push_back((uint32_t)data.size());
        lens.push_back(n);
        for (uint32_t i = 0; i < n; ++i) data.push_back(in[i]);
        if (n == 0) data.push_back(0ull);   // keep offsets valid for empty inputs
    }
};

TEST(Sha256CoreGPU, LeafHashMatchesTheCpuVectors)
{
    FlatInput fi;
    std::vector<uint64_t> expect;
    for (int c = 0; c < V::leaf_case_count; ++c)
    {
        fi.add(V::leaf_cases[c].in, V::leaf_cases[c].n);
        for (int i = 0; i < 4; ++i) expect.push_back(V::leaf_cases[c].out[i]);
    }
    const int n = (int)fi.lens.size();

    uint64_t *d_in, *d_out;
    uint32_t *d_lens, *d_offs;
    CUDA_ASSERT(cudaMalloc(&d_in, fi.data.size() * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMalloc(&d_lens, n * sizeof(uint32_t)));
    CUDA_ASSERT(cudaMalloc(&d_offs, n * sizeof(uint32_t)));
    CUDA_ASSERT(cudaMalloc(&d_out, n * 4 * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMemcpy(d_in, fi.data.data(), fi.data.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_lens, fi.lens.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_offs, fi.offsets.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));

    shaLeafKernel<<<(n + 63) / 64, 64>>>(d_in, d_lens, d_offs, d_out, n);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    std::vector<uint64_t> got(n * 4);
    CUDA_ASSERT(cudaMemcpy(got.data(), d_out, n * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    for (int c = 0; c < n; ++c)
        for (int i = 0; i < 4; ++i)
            EXPECT_EQ(got[c * 4 + i], expect[c * 4 + i])
                << "hash_le64: device disagrees at case " << c << " word " << i
                << " (len " << fi.lens[c] << ")";

    CUDA_ASSERT(cudaFree(d_in));
    CUDA_ASSERT(cudaFree(d_lens));
    CUDA_ASSERT(cudaFree(d_offs));
    CUDA_ASSERT(cudaFree(d_out));
}

TEST(Sha256CoreGPU, NodeHashMatchesTheCpuVectors)
{
    FlatInput fi;
    std::vector<uint64_t> expect;
    for (int c = 0; c < V::node_case_count; ++c)
    {
        fi.add(V::node_cases[c].in, V::node_cases[c].n);
        for (int i = 0; i < 4; ++i) expect.push_back(V::node_cases[c].out[i]);
    }
    const int n = (int)fi.lens.size();

    uint64_t *d_in, *d_out;
    uint32_t *d_lens, *d_offs;
    CUDA_ASSERT(cudaMalloc(&d_in, fi.data.size() * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMalloc(&d_lens, n * sizeof(uint32_t)));
    CUDA_ASSERT(cudaMalloc(&d_offs, n * sizeof(uint32_t)));
    CUDA_ASSERT(cudaMalloc(&d_out, n * 4 * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMemcpy(d_in, fi.data.data(), fi.data.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_lens, fi.lens.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_offs, fi.offsets.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));

    shaNodeKernel<<<1, 64>>>(d_in, d_lens, d_offs, d_out, n);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    std::vector<uint64_t> got(n * 4);
    CUDA_ASSERT(cudaMemcpy(got.data(), d_out, n * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    for (int c = 0; c < n; ++c)
        for (int i = 0; i < 4; ++i)
            EXPECT_EQ(got[c * 4 + i], expect[c * 4 + i])
                << "node_hash: device disagrees at case " << c << " word " << i;

    CUDA_ASSERT(cudaFree(d_in));
    CUDA_ASSERT(cudaFree(d_lens));
    CUDA_ASSERT(cudaFree(d_offs));
    CUDA_ASSERT(cudaFree(d_out));
}

TEST(Sha256CoreGPU, GrindHashMatchesTheCpuVectors)
{
    const int n = V::grind_case_count;
    std::vector<uint64_t> in, expect;
    for (int c = 0; c < n; ++c)
    {
        for (int i = 0; i < 8; ++i) in.push_back(V::grind_cases[c].in[i]);
        for (int i = 0; i < 4; ++i) expect.push_back(V::grind_cases[c].out[i]);
    }
    uint64_t *d_in, *d_out;
    CUDA_ASSERT(cudaMalloc(&d_in, in.size() * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMalloc(&d_out, n * 4 * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMemcpy(d_in, in.data(), in.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    shaGrindKernel<<<1, 64>>>(d_in, d_out, n);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
    std::vector<uint64_t> got(n * 4);
    CUDA_ASSERT(cudaMemcpy(got.data(), d_out, n * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    for (int c = 0; c < n; ++c)
        for (int i = 0; i < 4; ++i)
            EXPECT_EQ(got[c * 4 + i], expect[c * 4 + i]) << "grind_hash case " << c << " word " << i;
    CUDA_ASSERT(cudaFree(d_in));
    CUDA_ASSERT(cudaFree(d_out));
}

/// The transcript is the piece blake3 had to redesign, so its device path gets
/// the same scrutiny: the Hasher struct lives in registers/local memory on the
/// device and its buffering must behave identically.
TEST(Sha256CoreGPU, TranscriptSqueezeMatchesTheCpuVectors)
{
    const int n = V::trx_case_count;
    FlatInput fi;
    std::vector<uint64_t> counters, expect;
    for (int c = 0; c < n; ++c)
    {
        fi.add(V::trx_cases[c].in, V::trx_cases[c].n);
        counters.push_back(V::trx_cases[c].counter);
        for (int i = 0; i < 4; ++i) expect.push_back(V::trx_cases[c].out[i]);
    }

    uint64_t *d_in, *d_out, *d_ctr;
    uint32_t *d_lens, *d_offs;
    CUDA_ASSERT(cudaMalloc(&d_in, fi.data.size() * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMalloc(&d_lens, n * sizeof(uint32_t)));
    CUDA_ASSERT(cudaMalloc(&d_offs, n * sizeof(uint32_t)));
    CUDA_ASSERT(cudaMalloc(&d_ctr, n * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMalloc(&d_out, n * 4 * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMemcpy(d_in, fi.data.data(), fi.data.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_lens, fi.lens.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_offs, fi.offsets.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_ctr, counters.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));

    shaTrxKernel<<<1, 64>>>(d_in, d_lens, d_offs, d_ctr, d_out, n);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    std::vector<uint64_t> got(n * 4);
    CUDA_ASSERT(cudaMemcpy(got.data(), d_out, n * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    for (int c = 0; c < n; ++c)
        for (int i = 0; i < 4; ++i)
            EXPECT_EQ(got[c * 4 + i], expect[c * 4 + i])
                << "Hasher::squeeze case " << c << " word " << i << " (len " << fi.lens[c] << ")";

    CUDA_ASSERT(cudaFree(d_in));
    CUDA_ASSERT(cudaFree(d_lens));
    CUDA_ASSERT(cudaFree(d_offs));
    CUDA_ASSERT(cudaFree(d_ctr));
    CUDA_ASSERT(cudaFree(d_out));
}

// Prover-class equivalence: the leaf gather (GPU through getBufferOffset, CPU
// contiguous), the arity-2 reduction and the tree layout. The whole tree is
// compared, not just the root.

#include "sha256_goldilocks.hpp"
#include "sha256_goldilocks.cuh"
#include "goldilocks_tooling.hpp"   // getTreeNumElements

static void fillTraceRowMajor(std::vector<uint64_t> &trace, uint64_t nRows, uint64_t nCols)
{
    // Values >= p (and p) so the GPU gather's canonicalization is exercised.
    uint64_t x = 0x243F6A8885A308D3ULL;
    for (uint64_t r = 0; r < nRows; ++r)
        for (uint64_t c = 0; c < nCols; ++c)
        {
            x = x * 6364136223846793005ULL + 1442695040888963407ULL;
            const uint64_t idx = r * nCols + c;
            if (idx % 7 == 6)      trace[idx] = sha256core::GL_P + (x % 0xFFFFFFFFULL);
            else if (idx % 11 == 5) trace[idx] = sha256core::GL_P;   // == 0 after reduction
            else                    trace[idx] = x % sha256core::GL_P;
        }
}

static void checkMerkletreeMatches(uint64_t nRows, uint64_t nCols)
{
    constexpr uint64_t ARITY = 2;
    const uint64_t treeElems = getTreeNumElements(nRows, ARITY);

    std::vector<uint64_t> trace(nRows * nCols);
    fillTraceRowMajor(trace, nRows, nCols);

    // --- CPU ---
    std::vector<Goldilocks::Element> cpuTree(treeElems);
    Sha256Goldilocks::merkletree(cpuTree.data(),
                                 reinterpret_cast<Goldilocks::Element *>(trace.data()),
                                 nCols, nRows, ARITY);

    // --- GPU (RowMajor, the layout the CPU class assumes) ---
    uint64_t *d_trace = nullptr, *d_tree = nullptr;
    CUDA_ASSERT(cudaMalloc(&d_trace, trace.size() * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMalloc(&d_tree, treeElems * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMemcpy(d_trace, trace.data(), trace.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemset(d_tree, 0, treeElems * sizeof(uint64_t)));

    Sha256GoldilocksGPU::merkletree((uint32_t)ARITY, d_tree, d_trace, nCols, nRows, Layout::RowMajor, 0);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    std::vector<uint64_t> gpuTree(treeElems);
    CUDA_ASSERT(cudaMemcpy(gpuTree.data(), d_tree, treeElems * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    int reported = 0;
    for (uint64_t i = 0; i < treeElems && reported < 8; ++i)
        if (gpuTree[i] != Goldilocks::toU64(cpuTree[i]))
        {
            EXPECT_EQ(gpuTree[i], Goldilocks::toU64(cpuTree[i]))
                << "tree element " << i << " differs (nRows " << nRows << ", nCols " << nCols
                << "); leaves occupy [0, " << nRows * 4 << ")";
            ++reported;
        }
    ASSERT_EQ(reported, 0) << "CPU and GPU merkle trees diverge";

    CUDA_ASSERT(cudaFree(d_trace));
    CUDA_ASSERT(cudaFree(d_tree));
}

/// Widths covering every padding case: 6 (fits the tail), 7 and 8/16 (extra
/// block), plus non-multiples and a wide row.
TEST(Sha256GoldilocksGPU, MerkletreeMatchesCpuAcrossLeafWidths)
{
    for (uint64_t nCols : {1u, 3u, 6u, 7u, 8u, 9u, 16u, 17u, 24u, 38u, 66u, 130u})
        checkMerkletreeMatches(1u << 10, nCols);
}

/// Non-power-of-arity row counts exercise the zero padding per level.
TEST(Sha256GoldilocksGPU, MerkletreeMatchesCpuAcrossRowCounts)
{
    for (uint64_t nRows : {2u, 3u, 5u, 7u, 8u, 100u, 1000u, 1024u, 4096u})
        checkMerkletreeMatches(nRows, 12);
}

/// The path verifyMerkleRoot takes, so verifier and prover roots must match.
TEST(Sha256GoldilocksGPU, MerkletreeReduceMatchesCpu)
{
    for (uint64_t n : {2u, 3u, 16u, 17u, 1000u})
    {
        std::vector<uint64_t> leaves(n * 4);
        uint64_t x = 0x13198A2E03707344ULL;
        for (auto &w : leaves) { x = x * 2862933555777941757ULL + 3037000493ULL; w = x % sha256core::GL_P; }

        Goldilocks::Element cpuRoot[4];
        Sha256Goldilocks::merkletreeReduce(cpuRoot,
                                           reinterpret_cast<Goldilocks::Element *>(leaves.data()), n, 2);

        uint64_t *d_in, *d_root;
        CUDA_ASSERT(cudaMalloc(&d_in, leaves.size() * sizeof(uint64_t)));
        CUDA_ASSERT(cudaMalloc(&d_root, 4 * sizeof(uint64_t)));
        CUDA_ASSERT(cudaMemcpy(d_in, leaves.data(), leaves.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        Sha256GoldilocksGPU::merkletreeReduce(d_root, d_in, n, 2, 0);
        CUDA_ASSERT(cudaDeviceSynchronize());
        uint64_t gpuRoot[4];
        CUDA_ASSERT(cudaMemcpy(gpuRoot, d_root, sizeof(gpuRoot), cudaMemcpyDeviceToHost));

        for (int i = 0; i < 4; ++i)
            EXPECT_EQ(gpuRoot[i], Goldilocks::toU64(cpuRoot[i]))
                << "merkletreeReduce root word " << i << " differs at n = " << n;

        CUDA_ASSERT(cudaFree(d_in));
        CUDA_ASSERT(cudaFree(d_root));
    }
}

/// getBufferOffset's contract: same digest whichever layout holds the trace.
TEST(Sha256GoldilocksGPU, LeafHashIsLayoutInvariant)
{
    const uint64_t nRows = 1u << 10, nCols = 24;
    std::vector<uint64_t> rowMajor(nRows * nCols);
    fillTraceRowMajor(rowMajor, nRows, nCols);

    std::vector<uint64_t> colMajor(nRows * nCols);
    for (uint64_t r = 0; r < nRows; ++r)
        for (uint64_t c = 0; c < nCols; ++c)
            colMajor[c * nRows + r] = rowMajor[r * nCols + c];

    uint64_t *d_rm, *d_cm, *d_out_rm, *d_out_cm;
    CUDA_ASSERT(cudaMalloc(&d_rm, rowMajor.size() * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMalloc(&d_cm, colMajor.size() * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMalloc(&d_out_rm, nRows * 4 * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMalloc(&d_out_cm, nRows * 4 * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMemcpy(d_rm, rowMajor.data(), rowMajor.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(d_cm, colMajor.data(), colMajor.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    Sha256GoldilocksGPU::linearHash(d_out_rm, d_rm, nCols, nRows, Layout::RowMajor, 0);
    Sha256GoldilocksGPU::linearHash(d_out_cm, d_cm, nCols, nRows, Layout::ColMajor, 0);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    std::vector<uint64_t> a(nRows * 4), b(nRows * 4);
    CUDA_ASSERT(cudaMemcpy(a.data(), d_out_rm, a.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(b.data(), d_out_cm, b.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    for (uint64_t i = 0; i < a.size(); ++i)
        ASSERT_EQ(a[i], b[i]) << "RowMajor and ColMajor leaf digests differ at word " << i;

    // ...and both must equal the CPU's contiguous hash of the same row.
    for (uint64_t r : std::vector<uint64_t>{0, 1, 513, nRows - 1})
    {
        uint64_t want[4];
        sha256core::hash_le64(&rowMajor[r * nCols], (uint32_t)nCols, want);
        for (int i = 0; i < 4; ++i)
            ASSERT_EQ(a[r * 4 + i], want[i]) << "row " << r << " word " << i << " differs from the core";
    }

    CUDA_ASSERT(cudaFree(d_rm));
    CUDA_ASSERT(cudaFree(d_cm));
    CUDA_ASSERT(cudaFree(d_out_rm));
    CUDA_ASSERT(cudaFree(d_out_cm));
}

