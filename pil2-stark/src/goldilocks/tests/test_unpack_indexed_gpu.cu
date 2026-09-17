// The prover's indexed unpack (unpack_indexed, driven by unpack_trace) is a THIRD copy of the
// walk, independent of the slot kernel the stream-commit tests cover. It is what produces cm1
// for every GPU proof, so drift here reconstructs a different trace while the slot-root parity
// tests still pass. This drives the real kernel and compares every cell against the CPU walk.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#include "cuda_utils.cuh"
#include "unpack_indexed_device.cuh"
#include "unpack_indexed_row.hpp"

namespace {

void packBits(const std::vector<uint64_t> &vals, const std::vector<uint64_t> &widths,
              uint64_t words, uint64_t *out)
{
    for (uint64_t w = 0; w < words; w++) out[w] = 0;
    uint64_t idx = 0, off = 0;
    for (size_t i = 0; i < vals.size(); i++) {
        const uint64_t nbits = widths[i];
        const uint64_t mask = (nbits == 64) ? ~0ULL : ((1ULL << nbits) - 1ULL);
        const uint64_t v = vals[i] & mask;
        const uint64_t left = 64 - off;
        if (nbits <= left) {
            out[idx] |= v << off;
            off += nbits;
            if (off == 64 && idx + 1 < words) { idx++; off = 0; }
        } else {
            out[idx] |= (v & ((1ULL << left) - 1ULL)) << off;
            idx++;
            out[idx] |= v >> left;
            off = nbits - left;
        }
    }
}

uint64_t wordsFor(const std::vector<uint64_t> &w)
{
    uint64_t b = 0; for (uint64_t x : w) b += x; return (b + 63) / 64;
}

} // namespace

// Distinct entries per lane and per row: a single shared table cursor, or every lane reading
// lane 0's entry, both reproduce the right answer only when the lanes happen to coincide.
TEST(UNPACK_INDEXED_GPU, matches_the_cpu_walk_cell_for_cell)
{
    for (const uint64_t lanes : {1ull, 2ull, 4ull}) {
        const uint64_t INDEX_BITS = 32, nRows = 1024, nEntries = 13;

        // {width, sub-columns per lane, instruction-derived}, expanded lane-major. A C array,
        // not tuples + structured bindings: the goldilocks Makefile's NVCCFLAGS set no
        // -std=c++17, so .cu tests get whatever nvcc defaults to (C++14 on CUDA 11).
        const uint64_t FIELDS[][3] = {
            {32, 2, 0}, {1, 1, 0}, {32, 1, 1}, {64, 1, 1}, {8, 1, 1}, {38, 1, 0}, {1, 1, 1},
        };
        const uint64_t N_FIELDS = sizeof(FIELDS) / sizeof(FIELDS[0]);

        std::vector<uint64_t> widths, rowW, tabW, entryCol;
        std::vector<uint8_t> colSource, colLane;
        for (uint64_t l = 0; l < lanes; l++) rowW.push_back(INDEX_BITS);
        for (uint64_t f = 0; f < N_FIELDS; f++) {
            const uint64_t width = FIELDS[f][0], sub = FIELDS[f][1];
            const bool instr = FIELDS[f][2] != 0;
            const uint64_t entryBase = tabW.size();
            for (uint64_t l = 0; l < lanes; l++)
                for (uint64_t sc = 0; sc < sub; sc++) {
                    widths.push_back(width);
                    colSource.push_back(instr ? 1 : 0);
                    colLane.push_back(static_cast<uint8_t>(l));
                    entryCol.push_back(instr ? entryBase + sc : 0);
                    if (!instr) rowW.push_back(width);
                }
            if (instr) for (uint64_t sc = 0; sc < sub; sc++) tabW.push_back(width);
        }

        const uint64_t nCols = widths.size();
        const uint64_t rowWords = wordsFor(rowW), entWords = wordsFor(tabW);

        // Instruction table, then the compact rows that index into it.
        std::vector<uint64_t> table(nEntries * entWords);
        std::vector<std::vector<uint64_t>> entryVals(nEntries, std::vector<uint64_t>(tabW.size()));
        uint64_t x = 0x9E3779B97F4A7C15ull;
        auto next = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
        for (uint64_t e = 0; e < nEntries; e++) {
            for (size_t i = 0; i < tabW.size(); i++) entryVals[e][i] = next();
            packBits(entryVals[e], tabW, entWords, &table[e * entWords]);
        }

        std::vector<uint64_t> hRows(nRows * rowWords);
        for (uint64_t r = 0; r < nRows; r++) {
            std::vector<uint64_t> vals(rowW.size());
            for (uint64_t l = 0; l < lanes; l++) vals[l] = (r * 7 + l * 13 + 1) % nEntries;
            for (size_t i = lanes; i < vals.size(); i++) vals[i] = next();
            packBits(vals, rowW, rowWords, &hRows[r * rowWords]);
        }

        // Reference: the CPU walk, row by row.
        std::vector<uint64_t> ref(nRows * nCols);
        for (uint64_t r = 0; r < nRows; r++)
            ASSERT_TRUE(unpackIndexedRow(&hRows[r * rowWords], rowWords, table.data(), entWords,
                                         nEntries, INDEX_BITS, lanes, widths.data(),
                                         colSource.data(), colLane.data(), nCols,
                                         &ref[r * nCols]));

        // Under test: the prover kernel, exactly as unpack_trace launches it.
        const Layout layout = resolveLayout(10, nCols);
        uint64_t *dRows, *dTable, *dInfo, *dOut;
        uint8_t *dCS, *dCL;
        CHECKCUDAERR(cudaMalloc(&dRows, hRows.size() * 8));
        CHECKCUDAERR(cudaMalloc(&dTable, table.size() * 8));
        CHECKCUDAERR(cudaMalloc(&dInfo, nCols * 8));
        CHECKCUDAERR(cudaMalloc(&dOut, nRows * nCols * 8));
        CHECKCUDAERR(cudaMalloc(&dCS, nCols));
        CHECKCUDAERR(cudaMalloc(&dCL, nCols));
        CHECKCUDAERR(cudaMemcpy(dRows, hRows.data(), hRows.size() * 8, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(dTable, table.data(), table.size() * 8, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(dInfo, widths.data(), nCols * 8, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(dCS, colSource.data(), nCols, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(dCL, colLane.data(), nCols, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemset(dOut, 0xA5, nRows * nCols * 8));

        const uint32_t TPB = 256;
        unpack_indexed<<<(nRows + TPB - 1) / TPB, TPB, nCols * sizeof(uint64_t)>>>(
            dRows, dTable, dOut, nRows, nCols, rowWords, entWords, dInfo, dCS, dCL,
            INDEX_BITS, lanes, nEntries, layout);
        CHECKCUDAERR(cudaGetLastError());
        CHECKCUDAERR(cudaDeviceSynchronize());

        std::vector<uint64_t> got(nRows * nCols);
        CHECKCUDAERR(cudaMemcpy(got.data(), dOut, got.size() * 8, cudaMemcpyDeviceToHost));
        // getBufferOffset is __device__ only, so index the host copy directly. ColMajor is
        // (row,col) -> col*nRows + row; asserted rather than assumed, so a resolveLayout
        // change fails here instead of silently comparing the wrong cells.
        ASSERT_EQ(static_cast<int>(layout), static_cast<int>(Layout::ColMajor));
        for (uint64_t r = 0; r < nRows; r++)
            for (uint64_t c = 0; c < nCols; c++)
                ASSERT_EQ(got[c * nRows + r], ref[r * nCols + c])
                    << "lanes=" << lanes << " row " << r << " col " << c;

        CHECKCUDAERR(cudaFree(dRows)); CHECKCUDAERR(cudaFree(dTable));
        CHECKCUDAERR(cudaFree(dInfo)); CHECKCUDAERR(cudaFree(dOut));
        CHECKCUDAERR(cudaFree(dCS));   CHECKCUDAERR(cudaFree(dCL));
    }
}

// A single-lane air carries no lane map; the prover leaves d_col_lane null there.
TEST(UNPACK_INDEXED_GPU, a_null_lane_map_reads_lane_zero)
{
    const uint64_t INDEX_BITS = 32, nRows = 256, nEntries = 5, nCols = 4;
    const std::vector<uint64_t> widths{16, 32, 1, 64};
    const std::vector<uint8_t> colSource{0, 1, 0, 1}, colLane{0, 0, 0, 0};
    const std::vector<uint64_t> rowW{INDEX_BITS, 16, 1}, tabW{32, 64};
    const uint64_t rowWords = wordsFor(rowW), entWords = wordsFor(tabW);

    std::vector<uint64_t> table(nEntries * entWords);
    uint64_t x = 0x243F6A8885A308D3ull;
    auto next = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    for (uint64_t e = 0; e < nEntries; e++)
        packBits({next(), next()}, tabW, entWords, &table[e * entWords]);

    std::vector<uint64_t> hRows(nRows * rowWords);
    for (uint64_t r = 0; r < nRows; r++)
        packBits({r % nEntries, next(), next()}, rowW, rowWords, &hRows[r * rowWords]);

    std::vector<uint64_t> ref(nRows * nCols);
    for (uint64_t r = 0; r < nRows; r++)
        ASSERT_TRUE(unpackIndexedRow(&hRows[r * rowWords], rowWords, table.data(), entWords,
                                     nEntries, INDEX_BITS, 1, widths.data(), colSource.data(),
                                     colLane.data(), nCols, &ref[r * nCols]));

    const Layout layout = resolveLayout(8, nCols);
    uint64_t *dRows, *dTable, *dInfo, *dOut;
    uint8_t *dCS;
    CHECKCUDAERR(cudaMalloc(&dRows, hRows.size() * 8));
    CHECKCUDAERR(cudaMalloc(&dTable, table.size() * 8));
    CHECKCUDAERR(cudaMalloc(&dInfo, nCols * 8));
    CHECKCUDAERR(cudaMalloc(&dOut, nRows * nCols * 8));
    CHECKCUDAERR(cudaMalloc(&dCS, nCols));
    CHECKCUDAERR(cudaMemcpy(dRows, hRows.data(), hRows.size() * 8, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(dTable, table.data(), table.size() * 8, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(dInfo, widths.data(), nCols * 8, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(dCS, colSource.data(), nCols, cudaMemcpyHostToDevice));

    const uint32_t TPB = 128;
    // lanes = 0 AND a null map: the single-lane shape the prover actually registers.
    unpack_indexed<<<(nRows + TPB - 1) / TPB, TPB, nCols * sizeof(uint64_t)>>>(
        dRows, dTable, dOut, nRows, nCols, rowWords, entWords, dInfo, dCS, nullptr,
        INDEX_BITS, 0, nEntries, layout);
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaDeviceSynchronize());

    std::vector<uint64_t> got(nRows * nCols);
    CHECKCUDAERR(cudaMemcpy(got.data(), dOut, got.size() * 8, cudaMemcpyDeviceToHost));
    ASSERT_EQ(static_cast<int>(layout), static_cast<int>(Layout::ColMajor));
    for (uint64_t r = 0; r < nRows; r++)
        for (uint64_t c = 0; c < nCols; c++)
            ASSERT_EQ(got[c * nRows + r], ref[r * nCols + c]) << "row " << r << " col " << c;

    CHECKCUDAERR(cudaFree(dRows)); CHECKCUDAERR(cudaFree(dTable));
    CHECKCUDAERR(cudaFree(dInfo)); CHECKCUDAERR(cudaFree(dOut)); CHECKCUDAERR(cudaFree(dCS));
}
