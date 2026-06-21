#include <gtest/gtest.h>
#include "../src/poseidon2_goldilocks.cuh"
#include "../src/poseidon2_goldilocks.hpp"
#include "../src/goldilocks_tooling.hpp"
#include "../src/ntt_goldilocks.hpp"
#include "../src/ntt_goldilocks.cuh"
#include "../src/goldilocks_tooling.cuh"

TEST(GOLDILOCKS_TEST, ntt_gpu_intt_roundtrip)
{
    constexpr uint64_t n_bits = 20;
    constexpr uint64_t domain_size = 1ULL << n_bits;
    constexpr uint64_t nCols = 1;

    // Build Fibonacci input
    std::vector<Goldilocks::Element> h_original(domain_size);
    h_original[0] = h_original[1] = Goldilocks::one();
    for (uint64_t i = 2; i < domain_size; i++)
        h_original[i] = h_original[i-1] + h_original[i-2];

    // CPU NTT: time domain → frequency domain
    std::vector<Goldilocks::Element> h_ntt(h_original);
    NTT_Goldilocks cpu_ntt(domain_size);
    cpu_ntt.NTT(h_ntt.data(), h_ntt.data(), domain_size);

    // Upload frequency-domain data to GPU
    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    NTTGoldilocksGPU gpu_ntt(20, 1, &gpu_id);

    gl64_t *d_data;
    CHECKCUDAERR(cudaMalloc((void**)&d_data, domain_size * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(d_data, h_ntt.data(), domain_size * sizeof(gl64_t), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    // GPU INTT: frequency domain → time domain
    gpu_ntt.INTT(d_data, n_bits, nCols, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Download result and compare with original
    std::vector<Goldilocks::Element> h_result(domain_size);
    CHECKCUDAERR(cudaMemcpy(h_result.data(), d_data, domain_size * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

    for (uint64_t i = 0; i < domain_size; i++)
        ASSERT_EQ(Goldilocks::toU64(h_result[i]), Goldilocks::toU64(h_original[i])) << "Mismatch at i=" << i;

    CHECKCUDAERR(cudaFree(d_data));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// Round-trip test: GPU NTT → GPU INTT must recover the original input.
TEST(GOLDILOCKS_TEST, ntt_gpu_ntt_intt_roundtrip)
{
    constexpr uint64_t n_bits = 20;
    constexpr uint64_t domain_size = 1ULL << n_bits;
    constexpr uint64_t nCols = 1;

    // Build Fibonacci input
    std::vector<Goldilocks::Element> h_original(domain_size);
    h_original[0] = h_original[1] = Goldilocks::one();
    for (uint64_t i = 2; i < domain_size; i++)
        h_original[i] = h_original[i-1] + h_original[i-2];

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    NTTGoldilocksGPU gpu_ntt(n_bits, 1, &gpu_id);

    gl64_t *d_data;
    CHECKCUDAERR(cudaMalloc((void**)&d_data, domain_size * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(d_data, h_original.data(), domain_size * sizeof(gl64_t), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));

    // GPU NTT then GPU INTT should recover original
    gpu_ntt.NTT(d_data, n_bits, nCols, stream);
    gpu_ntt.INTT(d_data, n_bits, nCols, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    std::vector<Goldilocks::Element> h_result(domain_size);
    CHECKCUDAERR(cudaMemcpy(h_result.data(), d_data, domain_size * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

    for (uint64_t i = 0; i < domain_size; i++)
        ASSERT_EQ(Goldilocks::toU64(h_result[i]), Goldilocks::toU64(h_original[i])) << "Mismatch at i=" << i;

    CHECKCUDAERR(cudaFree(d_data));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// Round-trip test: CPU NTT → GPU INTT must recover the original input, nCols=TILE_WIDTH.
TEST(GOLDILOCKS_TEST, ntt_gpu_intt_multicol)
{
    constexpr uint64_t n_bits  = 16;
    constexpr uint64_t nRows   = 1ULL << n_bits;
    constexpr uint64_t nCols   = TILE_WIDTH;
    constexpr uint64_t nElems  = nRows * nCols;

    // Build Fibonacci input in row-major order
    std::vector<Goldilocks::Element> h_original(nElems);
    h_original[0] = h_original[1] = Goldilocks::one();
    for (uint64_t i = 2; i < nElems; i++)
        h_original[i] = h_original[i-1] + h_original[i-2];

    // CPU NTT over all columns at once (row-major interleaved layout)
    std::vector<Goldilocks::Element> h_ntt(h_original);
    NTT_Goldilocks cpu_ntt(nRows);
    cpu_ntt.NTT(h_ntt.data(), h_ntt.data(), nRows, nCols);

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    NTTGoldilocksGPU gpu_ntt(16, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);

    // INTT(nBits=16, nCols=4) resolves to Layout::ColMajor (flat, sppark): nCols <= 500. Lay the input
    // out the SAME way (fromRowMajorToColMajor with ColMajor) so the in-place transform reads it
    // correctly, then read the result back with the matching flat formula (col*nRows + row).
    gl64_t *d_flat, *d_colmajor;
    CHECKCUDAERR(cudaMalloc((void**)&d_flat,     nElems * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_colmajor, nElems * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(d_flat, h_ntt.data(), nElems * sizeof(gl64_t), cudaMemcpyHostToDevice));
    fromRowMajorToColMajor(nRows, nCols, d_flat, d_colmajor, Layout::ColMajor, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // GPU INTT in-place on flat column-major data
    gpu_ntt.INTT(d_colmajor, n_bits, nCols, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // Download flat column-major result and reorder back to row-major (col*nRows + row).
    std::vector<Goldilocks::Element> h_colmajor(nElems);
    CHECKCUDAERR(cudaMemcpy(h_colmajor.data(), d_colmajor, nElems * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

    std::vector<Goldilocks::Element> h_result(nElems);
    for (uint64_t row = 0; row < nRows; row++)
        for (uint64_t col = 0; col < nCols; col++)
            h_result[row * nCols + col] = h_colmajor[col * nRows + row];

    for (uint64_t i = 0; i < nElems; i++)
        ASSERT_EQ(Goldilocks::toU64(h_result[i]), Goldilocks::toU64(h_original[i]))
            << "Mismatch at i=" << i;

    CHECKCUDAERR(cudaFree(d_flat));
    CHECKCUDAERR(cudaFree(d_colmajor));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// Correctness test: GPU LDE must match CPU LDE element-by-element.
TEST(GOLDILOCKS_TEST, ntt_gpu_lde)
{
    constexpr uint64_t n_bits     = 12;
    constexpr uint64_t n_bits_ext = 14; // 4× blowup
    constexpr uint64_t nRows      = 1ULL << n_bits;
    constexpr uint64_t nRows_ext  = 1ULL << n_bits_ext;
    constexpr uint64_t nCols      = 1;

    std::vector<Goldilocks::Element> h_input(nRows);
    for (uint64_t i = 0; i < nRows; i++)
        h_input[i] = Goldilocks::fromU64(i + 1);

    // CPU reference
    std::vector<Goldilocks::Element> h_cpu(nRows_ext);
    NTT_Goldilocks cpu_ntt(nRows_ext);
    cpu_ntt.LDE(h_cpu.data(), h_input.data(), nRows_ext, nRows, nCols);

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    NTTGoldilocksGPU gpu_ntt(24, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);

    gl64_t *d_src, *d_dst;
    CHECKCUDAERR(cudaMalloc((void**)&d_src, nRows     * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_dst, nRows_ext * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMemcpy(d_src, h_input.data(), nRows * sizeof(gl64_t), cudaMemcpyHostToDevice));

    gpu_ntt.LDE(d_dst, 0, d_src, 0, n_bits, n_bits_ext, nCols, timer, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    std::vector<Goldilocks::Element> h_gpu(nRows_ext);
    CHECKCUDAERR(cudaMemcpy(h_gpu.data(), d_dst, nRows_ext * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

    for (uint64_t i = 0; i < nRows_ext; i++)
        ASSERT_EQ(Goldilocks::toU64(h_gpu[i]), Goldilocks::toU64(h_cpu[i]))
            << "LDE mismatch at i=" << i;

    CHECKCUDAERR(cudaFree(d_src));
    CHECKCUDAERR(cudaFree(d_dst));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// LDE + buildMerkleTreeGPU(Layout::ColMajor) root must match CPU merkletree_seq built from the same extended polynomial.
TEST(GOLDILOCKS_TEST, ntt_gpu_lde_merkletree)
{
    constexpr uint64_t n_bits     = 12;
    constexpr uint64_t n_bits_ext = 14;
    constexpr uint64_t nRows      = 1ULL << n_bits;
    constexpr uint64_t nRows_ext  = 1ULL << n_bits_ext;
    constexpr uint64_t nCols      = 1;
    constexpr uint32_t arity      = 3;

    std::vector<Goldilocks::Element> h_input(nRows);
    for (uint64_t i = 0; i < nRows; i++)
        h_input[i] = Goldilocks::fromU64(i + 1);

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<12>::initConstants(&gpu_id, 1);
    NTTGoldilocksGPU gpu_ntt(14, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);

    uint64_t tree_size = getTreeNumElements(nRows_ext, arity);

    gl64_t *d_src, *d_lde_mt;
    Goldilocks::Element *d_tree_gpu;
    CHECKCUDAERR(cudaMalloc((void**)&d_src,      nRows     * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_lde_mt,   nRows_ext * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_tree_gpu, tree_size * sizeof(Goldilocks::Element)));

    // GPU: LDE then merkle tree
    CHECKCUDAERR(cudaMemcpy(d_src, h_input.data(), nRows * sizeof(gl64_t), cudaMemcpyHostToDevice));
    gpu_ntt.LDE(d_lde_mt, 0, d_src, 0, n_bits, n_bits_ext, nCols, timer, stream);
    Poseidon2GoldilocksGPU<12>::merkletree(arity, (uint64_t*)d_tree_gpu, (uint64_t*)d_lde_mt, nCols, nRows_ext, Layout::ColMajor, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // CPU reference: LDE then merkletree_seq
    std::vector<Goldilocks::Element> h_cpu_lde(nRows_ext);
    NTT_Goldilocks cpu_ntt(nRows_ext);
    cpu_ntt.LDE(h_cpu_lde.data(), h_input.data(), nRows_ext, nRows, nCols);

    std::vector<Goldilocks::Element> h_cpu_tree(tree_size);
    Poseidon2Goldilocks<12>::merkletree(h_cpu_tree.data(), h_cpu_lde.data(), nCols, nRows_ext, arity, Poseidon2Mode::Scalar);

    // Download GPU root and compare with CPU root element by element
    std::vector<Goldilocks::Element> h_gpu_root(HASH_SIZE);
    CHECKCUDAERR(cudaMemcpy(h_gpu_root.data(), d_tree_gpu + (tree_size - HASH_SIZE),
                             HASH_SIZE * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

    Goldilocks::Element *h_cpu_root = h_cpu_tree.data() + (tree_size - HASH_SIZE);
    for (int i = 0; i < HASH_SIZE; i++)
        ASSERT_EQ(Goldilocks::toU64(h_gpu_root[i]), Goldilocks::toU64(h_cpu_root[i]))
            << "Merkle root mismatch at element " << i;

    CHECKCUDAERR(cudaFree(d_src));
    CHECKCUDAERR(cudaFree(d_lde_mt));
    CHECKCUDAERR(cudaFree(d_tree_gpu));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// LDE + buildMerkleTreeGPU with multiple columns. LDE(nBits=12,nCols=12) resolves to Layout::ColMajor
// (flat); input is laid out flat column-major, and the tree is built over the flat LDE output.
// GPU root must match CPU LDE + merkletree_seq (both row-major).
TEST(GOLDILOCKS_TEST, ntt_gpu_lde_merkletree_multicol)
{
    constexpr uint64_t n_bits     = 12;
    constexpr uint64_t n_bits_ext = 14;
    constexpr uint64_t nRows      = 1ULL << n_bits;
    constexpr uint64_t nRows_ext  = 1ULL << n_bits_ext;
    constexpr uint64_t nCols      = 12;
    constexpr uint32_t arity      = 3;

    // Row-major input: h_input[row * nCols + col]
    std::vector<Goldilocks::Element> h_input(nRows * nCols);
    for (uint64_t i = 0; i < nRows * nCols; i++)
        h_input[i] = Goldilocks::fromU64(i + 1);

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<12>::initConstants(&gpu_id, 1);
    NTTGoldilocksGPU gpu_ntt(14, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);

    uint64_t tree_size = getTreeNumElements(nRows_ext, arity);

    gl64_t *d_flat, *d_colmajor, *d_lde;
    Goldilocks::Element *d_tree_gpu;
    CHECKCUDAERR(cudaMalloc((void**)&d_flat,     nRows     * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_colmajor, nRows     * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_lde,      nRows_ext * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_tree_gpu, tree_size * sizeof(Goldilocks::Element)));

    // GPU: row-major → flat column-major → LDE + merkle tree
    CHECKCUDAERR(cudaMemcpy(d_flat, h_input.data(), nRows * nCols * sizeof(gl64_t), cudaMemcpyHostToDevice));
    fromRowMajorToColMajor(nRows, nCols, d_flat, d_colmajor, Layout::ColMajor, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));
    gpu_ntt.LDE(d_lde, 0, d_colmajor, 0, n_bits, n_bits_ext, nCols, timer, stream);
    Poseidon2GoldilocksGPU<12>::merkletree(arity, (uint64_t*)d_tree_gpu, (uint64_t*)d_lde, nCols, nRows_ext, Layout::ColMajor, stream);
    CHECKCUDAERR(cudaStreamSynchronize(stream));

    // CPU reference: LDE (row-major) then merkletree_seq (row-major)
    std::vector<Goldilocks::Element> h_cpu_lde(nRows_ext * nCols);
    NTT_Goldilocks cpu_ntt(nRows_ext);
    cpu_ntt.LDE(h_cpu_lde.data(), h_input.data(), nRows_ext, nRows, nCols);

    std::vector<Goldilocks::Element> h_cpu_tree(tree_size);
    Poseidon2Goldilocks<12>::merkletree(h_cpu_tree.data(), h_cpu_lde.data(), nCols, nRows_ext, arity, Poseidon2Mode::Scalar);

    // Download GPU root and compare with CPU root
    std::vector<Goldilocks::Element> h_gpu_root(HASH_SIZE);
    CHECKCUDAERR(cudaMemcpy(h_gpu_root.data(), d_tree_gpu + (tree_size - HASH_SIZE),
                             HASH_SIZE * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));

    Goldilocks::Element *h_cpu_root = h_cpu_tree.data() + (tree_size - HASH_SIZE);
    for (int i = 0; i < HASH_SIZE; i++)
        ASSERT_EQ(Goldilocks::toU64(h_gpu_root[i]), Goldilocks::toU64(h_cpu_root[i]))
            << "Merkle root mismatch at element " << i << " (nCols=" << nCols << ")";

    CHECKCUDAERR(cudaFree(d_flat));
    CHECKCUDAERR(cudaFree(d_colmajor));
    CHECKCUDAERR(cudaFree(d_lde));
    CHECKCUDAERR(cudaFree(d_tree_gpu));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}

// Keccakf-like ColMajorTiled stage-1 flow: nBits=17, nCols>500 -> resolveLayout returns ColMajorTiled,
// so the trace is stored ColMajorTiled, LDE runs the NATIVE tiled path, and the Merkle reads tiled.
// (1) GPU root must match CPU LDE + merkletree_seq (row-major reference). (2) The GPU root must be
// DETERMINISTIC across two independent runs (catches uninitialised in-tile padding reads).
TEST(GOLDILOCKS_TEST, ntt_gpu_lde_merkletree_tiled_keccakf)
{
    constexpr uint64_t n_bits     = 17;
    constexpr uint64_t n_bits_ext = 18;
    constexpr uint64_t nRows      = 1ULL << n_bits;
    constexpr uint64_t nRows_ext  = 1ULL << n_bits_ext;
    constexpr uint64_t nCols      = 600;          // > 500 -> ColMajorTiled
    constexpr uint32_t arity      = 3;

    ASSERT_EQ(resolveLayout(n_bits, nCols), Layout::ColMajorTiled) << "test must exercise the tiled path";

    std::vector<Goldilocks::Element> h_input(nRows * nCols);
    for (uint64_t i = 0; i < nRows * nCols; i++)
        h_input[i] = Goldilocks::fromU64((i * 2654435761ULL + 1) & 0xffffffffULL);

    uint32_t gpu_id = 0;
    cudaGetDevice((int*)&gpu_id);
    Poseidon2GoldilocksGPU<12>::initConstants(&gpu_id, 1);
    NTTGoldilocksGPU gpu_ntt(n_bits_ext, 1, &gpu_id);

    cudaStream_t stream;
    CHECKCUDAERR(cudaStreamCreate(&stream));
    TimerGPU timer(stream);

    uint64_t tree_size = getTreeNumElements(nRows_ext, arity);

    gl64_t *d_flat, *d_tiled, *d_lde;
    Goldilocks::Element *d_tree_gpu;
    CHECKCUDAERR(cudaMalloc((void**)&d_flat,     nRows     * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_tiled,    nRows     * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_lde,      nRows_ext * nCols * sizeof(gl64_t)));
    CHECKCUDAERR(cudaMalloc((void**)&d_tree_gpu, tree_size * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(d_flat, h_input.data(), nRows * nCols * sizeof(gl64_t), cudaMemcpyHostToDevice));

    // CPU reference: LDE (row-major) then merkletree_seq (row-major)
    std::vector<Goldilocks::Element> h_cpu_lde(nRows_ext * nCols);
    NTT_Goldilocks cpu_ntt(nRows_ext);
    cpu_ntt.LDE(h_cpu_lde.data(), h_input.data(), nRows_ext, nRows, nCols);
    std::vector<Goldilocks::Element> h_cpu_tree(tree_size);
    Poseidon2Goldilocks<12>::merkletree(h_cpu_tree.data(), h_cpu_lde.data(), nCols, nRows_ext, arity, Poseidon2Mode::Scalar);
    const Goldilocks::Element *h_cpu_root = h_cpu_tree.data() + (tree_size - HASH_SIZE);

    std::vector<Goldilocks::Element> root_run0(HASH_SIZE), root_run1(HASH_SIZE);
    for (int run = 0; run < 2; run++) {
        // Wipe the LDE buffer between runs so a stale value can't masquerade as determinism.
        CHECKCUDAERR(cudaMemsetAsync(d_lde, 0xAB, nRows_ext * nCols * sizeof(gl64_t), stream));
        // row-major -> ColMajorTiled storage
        fromRowMajorToColMajor(nRows, nCols, d_flat, d_tiled, Layout::ColMajorTiled, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
        gpu_ntt.LDE(d_lde, 0, d_tiled, 0, n_bits, n_bits_ext, nCols, timer, stream);
        Poseidon2GoldilocksGPU<12>::merkletree(arity, (uint64_t*)d_tree_gpu, (uint64_t*)d_lde, nCols, nRows_ext, Layout::ColMajorTiled, stream);
        CHECKCUDAERR(cudaStreamSynchronize(stream));
        std::vector<Goldilocks::Element> &dst = (run == 0) ? root_run0 : root_run1;
        CHECKCUDAERR(cudaMemcpy(dst.data(), d_tree_gpu + (tree_size - HASH_SIZE), HASH_SIZE * sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost));
    }

    for (int i = 0; i < HASH_SIZE; i++)
        ASSERT_EQ(Goldilocks::toU64(root_run0[i]), Goldilocks::toU64(root_run1[i]))
            << "NON-DETERMINISTIC tiled root at element " << i << " (uninitialised in-tile padding?)";
    for (int i = 0; i < HASH_SIZE; i++)
        ASSERT_EQ(Goldilocks::toU64(root_run0[i]), Goldilocks::toU64(h_cpu_root[i]))
            << "tiled GPU root != CPU root at element " << i;

    CHECKCUDAERR(cudaFree(d_flat));
    CHECKCUDAERR(cudaFree(d_tiled));
    CHECKCUDAERR(cudaFree(d_lde));
    CHECKCUDAERR(cudaFree(d_tree_gpu));
    CHECKCUDAERR(cudaStreamDestroy(stream));
    NTTGoldilocksGPU::freeConstants();
}
