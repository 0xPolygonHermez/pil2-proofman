// The GPU expander is a third implementation of the band fill: same permutations and row
// layouts, written against gl64_t. GPU proofs run it in place of the host expander, so the two
// must produce byte-identical traces; drift would only show up as a proof that fails to verify.
//
// Builds a synthetic trace with realistic boundary cells, fills it once on the host and once on
// the device, and compares every cell.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#include "../src/goldilocks_base_field.hpp"
#include "../src/poseidon_goldilocks.hpp"
#include "../src/poseidon2_goldilocks.hpp"

#include "gate_bands.hpp"
#include "gate_bands_cpu.hpp"

// Pulled in directly so the test needs no extra link rule; the prover library compiles the
// same file separately.
#include "gate_bands_gpu.cu"

using GL = Goldilocks::Element;

namespace {

constexpr int WID = 16;

// One band per kind, laid out end to end, with a few spare rows after the last.
struct Layout {
    std::vector<uint64_t> bands;   // {row, kind} pairs
    uint64_t nRows = 0;
};

Layout buildLayout()
{
    Layout l;
    for (uint64_t kind = GB_POSEIDON1_COMPRESSOR_SPONGE; kind <= GB_POSEIDON2_AGGREGATION_COMPRESSION; kind++) {
        l.bands.push_back(l.nRows);
        l.bands.push_back(kind);
        l.nRows += (uint64_t)gate_bands::band_rows(kind);
    }
    l.nRows += 4;
    return l;
}

// The boundary cells the map would place: input at the band's first row, key bits where the
// setup puts them, and the correct output at the last row -- the host expander cross-checks
// the output, so a made-up one is rejected.
void placeBoundary(std::vector<GL> &trace, uint64_t nCols, uint64_t row, uint64_t kind,
                   std::mt19937_64 &rng)
{
    uint64_t in[WID];
    for (int i = 0; i < WID; i++) in[i] = rng() % GOLDILOCKS_PRIME;
    for (int i = 0; i < WID; i++) trace[row * nCols + i] = Goldilocks::fromU64(in[i]);

    const bool agg = gate_bands::is_aggregation(kind);
    uint64_t key = 0;
    if (gate_bands::is_compression(kind)) {
        key = rng() & 3;
        if (agg) {
            trace[(row + 1) * nCols + 15] = Goldilocks::fromU64(key & 1);
            trace[(row + 2) * nCols + 15] = Goldilocks::fromU64((key >> 1) & 1);
        } else {
            trace[row * nCols + 16] = Goldilocks::fromU64(key & 1);
            trace[row * nCols + 17] = Goldilocks::fromU64((key >> 1) & 1);
        }
    }

    uint64_t im[gate_bands::POS_IM_GROUPS * WID], out[WID];
    const bool compression = gate_bands::is_compression(kind);
    if (gate_bands::is_poseidon1(kind)) gate_bands::poseidon1::snapshots(im, out, in, key, compression);
    else                                gate_bands::poseidon2::snapshots(im, out, in, key, compression);

    const uint64_t outRow = row + (agg ? gate_bands::AGG_OUT_ROW : (uint64_t)(gate_bands::POS_ROWS - 1));
    for (int i = 0; i < WID; i++) trace[outRow * nCols + i] = Goldilocks::fromU64(out[i]);
}

}  // namespace

TEST(GateBandsGPU, MatchesTheHostExpander)
{
    const Layout l = buildLayout();
    const uint64_t nBands = l.bands.size() / 2;
    // Wide enough for the aggregation geometry's second chain slot (a[32..47]).
    const uint64_t nCols = 48;

    std::mt19937_64 rng(0x6BAD5EEDULL);
    std::vector<GL> seed(l.nRows * nCols, Goldilocks::zero());
    for (uint64_t i = 0; i < nBands; i++) placeBoundary(seed, nCols, l.bands[i * 2], l.bands[i * 2 + 1], rng);

    // An exec buffer carrying just this band section: an empty header -- no additions and no
    // map -- then the section's version and count.
    std::vector<uint64_t> exec{exec_layout::EXEC_MAGIC | exec_layout::EXEC_FORMAT_VERSION, 0, 0, 0,
                               gate_bands::GATE_BAND_FORMAT_VERSION, nBands};
    exec.insert(exec.end(), l.bands.begin(), l.bands.end());

    std::vector<GL> host = seed;
    const gate_bands::ExpandResult hostRes =
        gate_bands::expand_gate_bands(host.data(), exec.data(), nCols, exec.size(), l.nRows);
    ASSERT_EQ((int)hostRes.status, (int)gate_bands::ExpandStatus::Ok)
        << "host expander rejected band " << hostRes.badBand << " (kind " << hostRes.kind << ")";
    ASSERT_EQ(hostRes.nBands, nBands);

    uint64_t *d_trace = nullptr, *d_bands = nullptr;
    const size_t traceBytes = seed.size() * sizeof(uint64_t);
    ASSERT_EQ(cudaMalloc(&d_trace, traceBytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_bands, l.bands.size() * sizeof(uint64_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_trace, seed.data(), traceBytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_bands, l.bands.data(), l.bands.size() * sizeof(uint64_t),
                         cudaMemcpyHostToDevice), cudaSuccess);

    uploadGateBandConstantsGPU();
    expandGateBandsGPU(d_trace, nCols, d_bands, nBands, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint64_t> device(seed.size());
    ASSERT_EQ(cudaMemcpy(device.data(), d_trace, traceBytes, cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d_trace);
    cudaFree(d_bands);

    uint64_t diffs = 0;
    for (uint64_t r = 0; r < l.nRows; r++) {
        for (uint64_t c = 0; c < nCols; c++) {
            const uint64_t h = Goldilocks::toU64(host[r * nCols + c]);
            const uint64_t g = device[r * nCols + c];
            if (h != g && ++diffs <= 8) {
                ADD_FAILURE() << "row " << r << " col " << c << ": host " << h << " device " << g;
            }
        }
    }
    ASSERT_EQ(diffs, 0u) << diffs << " cells differ between the host and device expanders";

    // And the fill is not vacuous: the interiors really were written.
    uint64_t nonZero = 0;
    for (const uint64_t v : device) nonZero += (v != 0);
    ASSERT_GT(nonZero, nBands * WID * 4) << "device trace looks unfilled";
}
