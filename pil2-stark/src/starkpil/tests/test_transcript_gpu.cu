// TranscriptGL_GPU against the SAME vectors as the CPU transcript.
//
// This is a stronger claim than characterization: the GPU prover and the CPU verifier must draw
// identical challenges, so any divergence here is a proof that cannot verify. The vectors are
// shared with test_transcript_cpu.cpp, and for blake3/sha256 they came from the Rust transcripts,
// so one file pins GPU == CPU == Rust.

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "transcriptGL.cuh"
#include "hash_family.hpp"
#include "transcript_vectors.hpp"

#define CUDA_ASSERT(call)                                                       \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        ASSERT_EQ(_e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(_e);  \
    } while (0)

static const uint64_t GL_P = 0xFFFFFFFF00000001ULL;

static uint64_t w(uint64_t i) { return (i * 0x9E3779B97F4A7C15ULL) % GL_P; }

// The script from test_transcript_cpu.cpp, driven through device buffers.
static std::vector<uint64_t> runScriptGPU(uint64_t arity, bool parallel)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    TranscriptGL_GPU t(arity, false, stream, parallel);

    uint64_t *d_in, *d_field, *d_state, *d_perms, *d_scratch;
    EXPECT_EQ(cudaMalloc(&d_in, 32 * sizeof(uint64_t)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_field, 3 * sizeof(uint64_t)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_state, 8 * sizeof(uint64_t)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_perms, 8 * sizeof(uint64_t)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_scratch, 64 * sizeof(uint64_t)), cudaSuccess);

    std::vector<uint64_t> host_in(32);
    for (uint64_t i = 0; i < 32; ++i) host_in[i] = w(i);
    EXPECT_EQ(cudaMemcpy(d_in, host_in.data(), 32 * sizeof(uint64_t), cudaMemcpyHostToDevice), cudaSuccess);

    std::vector<uint64_t> out;
    auto put = [&](uint64_t a, uint64_t b) {
        t.put((Goldilocks::Element *)(d_in + a), b - a, stream);
    };
    auto field = [&]() {
        t.getField(d_field, stream);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        uint64_t h[3];
        EXPECT_EQ(cudaMemcpy(h, d_field, sizeof(h), cudaMemcpyDeviceToHost), cudaSuccess);
        for (int i = 0; i < 3; ++i) out.push_back(h[i]);
    };

    put(0, 4);
    for (int i = 0; i < 3; ++i) field();
    put(4, 20);
    for (int i = 0; i < 2; ++i) field();
    put(20, 21);
    field();

    t.getState((Goldilocks::Element *)d_state, stream);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    uint64_t hs[4];
    EXPECT_EQ(cudaMemcpy(hs, d_state, sizeof(hs), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int i = 0; i < 4; ++i) out.push_back(hs[i]);

    t.getPermutations(d_perms, 8, 12, (Goldilocks::Element *)d_scratch, stream);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    uint64_t hp[8];
    EXPECT_EQ(cudaMemcpy(hp, d_perms, sizeof(hp), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int i = 0; i < 8; ++i) out.push_back(hp[i]);

    cudaFree(d_in);
    cudaFree(d_field);
    cudaFree(d_state);
    cudaFree(d_perms);
    cudaFree(d_scratch);
    cudaStreamDestroy(stream);
    return out;
}

// `define_hash_family` allows one family per process, which is right for production and wrong for
// a test binary covering four.
static void forceFamily(HashFamily f) { g_hash_family = f; }

static void expectSequence(HashFamily f, uint64_t arity, bool parallel, const uint64_t *want, const char *name)
{
    forceFamily(f);
    const std::vector<uint64_t> got = runScriptGPU(arity, parallel);
    ASSERT_EQ(got.size(), 30u) << name;
    for (size_t i = 0; i < 30; ++i)
        EXPECT_EQ(got[i], want[i]) << name << ": word " << i << " differs from the CPU/Rust sequence";
}

TEST(TranscriptGLGpu, Sha256MatchesTheCpuSequence)
{
    expectSequence(HashFamily::Sha256, 2, false, TRX_EXPECT_SHA256, "sha256");
}

TEST(TranscriptGLGpu, Blake3MatchesTheCpuSequence)
{
    expectSequence(HashFamily::Blake3, 2, false, TRX_EXPECT_BLAKE3, "blake3");
}

TEST(TranscriptGLGpu, Poseidon2MatchesTheCpuSequence)
{
    uint32_t gpu = 0;
    TranscriptGL_GPU::init_const(&gpu, 1, 4);
    expectSequence(HashFamily::Poseidon2, 4, false, TRX_EXPECT_POSEIDON2, "Poseidon2");
}

TEST(TranscriptGLGpu, Poseidon1MatchesTheCpuSequence)
{
    uint32_t gpu = 0;
    TranscriptGL_GPU::init_const(&gpu, 1, 4);
    expectSequence(HashFamily::Poseidon1, 4, false, TRX_EXPECT_POSEIDON1, "Poseidon1");
}

/// The warp-parallel sponge path must agree with the scalar one; only Poseidon has both.
TEST(TranscriptGLGpu, TheParallelSpongePathAgreesWithTheScalarOne)
{
    uint32_t gpu = 0;
    TranscriptGL_GPU::init_const(&gpu, 1, 4);
    forceFamily(HashFamily::Poseidon2);
    EXPECT_EQ(runScriptGPU(4, true), runScriptGPU(4, false));
}

// Own main: libstarksgpu.a carries deviceQuery.cu, whose `main` an archive would otherwise supply
// in place of gtest_main's.
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
