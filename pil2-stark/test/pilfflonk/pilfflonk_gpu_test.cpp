// Tests of the GPU path's entry points (pilfflonk/docs/performance.md#selection-memory-and-errors),
// in every build: pilfflonk_gpu_available, and pilfflonk_ctx_new_on and ProvingKey::load, which
// refuse the GPU, saying why, before they read anything, in a library built without it
// (make pilfflonk_test) and on a machine without one (make pilfflonk_gpu_test there). Where there is
// one, the tests of each module compare its GPU path with its CPU one (gpuUnderTest):
// pilfflonk_lde_test.cpp the transforms, pilfflonk_commit_test.cpp the MSM,
// pilfflonk_prover_test.cpp whole proofs and a key on the GPU; and this file the kernels of the
// device path (pilfflonk_kernels.hpp, pilfflonk_lde_kernels.hpp), the LDE on the device and GpuKey's
// commitments, byte for byte against the CPU's code on seeded inputs, edge values (0, 1, r − 1, all
// equal) and sizes around the warp, the block of 256 threads and powers of two.
#include "pilfflonk_test.hpp"
#include "pilfflonk_test_ptau.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "pilfflonk_api.hpp"
#include "pilfflonk_gpu.hpp"
#include "pilfflonk_proving_key.hpp"
#ifdef __USE_CUDA__
#include "pilfflonk_commit.hpp"
#include "pilfflonk_kernels.hpp"
#include "pilfflonk_key_gpu.hpp"
#include "pilfflonk_lde.hpp"
#include "pilfflonk_lde_gpu.hpp"
#include "pilfflonk_lde_kernels.hpp"
#include "pilfflonk_opening_gpu.hpp"
#include "pilfflonk_shplonk_prover.hpp"
#include "pilfflonk_srs.hpp"
#include "pilfflonk_transcript.hpp"

// The PLONK GPU prover's helpers (rapidsnark/plonk_prover.cu).
extern "C" void gpu_plonk_memcpy_h2d(void *dst, const void *src, size_t bytes);
extern "C" void gpu_plonk_memcpy_d2h(void *dst, const void *src, size_t bytes);
extern "C" void gpu_plonk_precompute_omega_tables_async(void *dBases, void *dTid, const void *omega4xPtr,
                                                        uint32_t blockSize, uint32_t numBlocks, void *stream);
#endif

namespace PilFflonkTest {

namespace {

bool contains(const char *s, const char *part) { return std::strstr(s, part) != nullptr; }

// What a refused GPU says: there is none, and why.
#ifdef __USE_CUDA__
constexpr const char *NO_GPU = "no GPU: CUDA sees no device";
#else
constexpr const char *NO_GPU = "no GPU: this library was built without it";
#endif

void testAvailability() {
#ifdef __USE_CUDA__
    assert(pilfflonk_gpu_available() == (PilFflonk::Gpu::available() ? 1 : 0));
#else
    assert(pilfflonk_gpu_available() == 0);
#endif
    assert(pilfflonk_gpu_available() == (PilFflonk::gpuAvailable() ? 1 : 0));
    assert(pilfflonk_last_status() == PILFFLONK_OK && pilfflonk_last_error()[0] == '\0');
}

void testCtxNewOn() {
    TestDir dir;
    const std::string missing = dir.path() + "/nothing";

    assert(pilfflonk_ctx_new_on(missing.c_str(), 2) == nullptr);
    assert(pilfflonk_last_status() == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(contains(pilfflonk_last_error(), "pilfflonk_ctx_new_on: device 2 is neither the CPU (0) nor the GPU (1)"));
    assert(pilfflonk_ctx_new_on(nullptr, PILFFLONK_DEVICE_CPU) == nullptr);
    assert(pilfflonk_last_status() == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(contains(pilfflonk_last_error(), "proving_key_dir is NULL"));

    // The CPU reads the key, which is not there; so does pilfflonk_ctx_new, the same ctx.
    assert(pilfflonk_ctx_new_on(missing.c_str(), PILFFLONK_DEVICE_CPU) == nullptr);
    assert(pilfflonk_last_status() == PILFFLONK_ERR_IO && contains(pilfflonk_last_error(), "cannot open"));

    if (pilfflonk_gpu_available()) {
        // A GPU there: it reads the key too.
        assert(pilfflonk_ctx_new_on(missing.c_str(), PILFFLONK_DEVICE_GPU) == nullptr);
        assert(pilfflonk_last_status() == PILFFLONK_ERR_IO);
        return;
    }
    // No GPU: refused before anything is read, so not an IoError, and the reason given.
    assert(pilfflonk_ctx_new_on(missing.c_str(), PILFFLONK_DEVICE_GPU) == nullptr);
    assert(pilfflonk_last_status() == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(contains(pilfflonk_last_error(), "pilfflonk_ctx_new_on: ProvingKey::load: "));
    assert(contains(pilfflonk_last_error(), NO_GPU));
    try {
        PilFflonk::ProvingKey::load(missing, PilFflonk::Device::Gpu);
        assert(!"expected std::invalid_argument");
    } catch (const std::invalid_argument &e) {
        assert(contains(e.what(), NO_GPU));
    }
}

#ifdef __USE_CUDA__

using Engine = AltBn128::Engine;
using FrElement = Engine::FrElement;
using G1Point = Engine::G1Point;
using G1PointAffine = Engine::G1PointAffine;
using Column = std::vector<FrElement>;
using PilFflonk::DeviceBuffer;
using PilFflonk::Lde;
using PilFflonk::Poly;

Engine &E = Engine::engine;

// Around a warp, a block of 256 threads and powers of two, and one of 2^20 + 1.
const std::vector<uint64_t> SIZES = {1, 2, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 4095, 4097, 65535, 65537,
                                     (uint64_t(1) << 20) + 1};

// Elements from a fixed seed: 0, 1, r − 1 and three equal ones first, then below 2^253 < r as
// Montgomery limbs.
class Random {
public:
    explicit Random(uint64_t seed) : generator(seed) {}

    FrElement element() {
        FrElement e;
        for (uint64_t &limb : e.v) {
            limb = generator();
        }
        e.v[3] >>= 3;
        return e;
    }

    Column column(uint64_t n) {
        const FrElement same = element();
        const FrElement edges[] = {E.fr.zero(), E.fr.one(), E.fr.negOne(), same, same, same};
        Column c(n);
        for (uint64_t i = 0; i < n; ++i) {
            c[i] = i < 6 ? edges[i] : element();
        }
        return c;
    }

    std::mt19937_64 &engine() { return generator; }

private:
    std::mt19937_64 generator;
};

template <typename T>
DeviceBuffer upload(const std::vector<T> &host) {
    DeviceBuffer device(host.size() * sizeof(T));
    gpu_plonk_memcpy_h2d(device.data(), host.data(), device.size());
    return device;
}

template <typename T>
std::vector<T> download(const DeviceBuffer &device, uint64_t n) {
    std::vector<T> host(n);
    gpu_plonk_memcpy_d2h(host.data(), device.data(), n * sizeof(T));
    return host;
}

FrElement *elementsOf(const DeviceBuffer &device) { return reinterpret_cast<FrElement *>(device.data()); }

Column download(const FrElement *device, uint64_t n) {
    Column host(n);
    gpu_plonk_memcpy_d2h(host.data(), device, n * sizeof(FrElement));
    return host;
}

bool same(const Column &a, const Column &b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(FrElement)) == 0;
}

// A point in affine coordinates, which are unique, as bytes.
std::vector<uint8_t> affineBytes(G1Point p) {
    G1PointAffine a;
    E.g1.copy(a, p);
    return std::vector<uint8_t>(reinterpret_cast<uint8_t *>(&a), reinterpret_cast<uint8_t *>(&a) + sizeof(a));
}

// The h^(256·b) and h^t tables of pilfflonk_gpu_pack_shift, for n scalars.
struct ShiftTables {
    explicit ShiftTables(uint64_t n) : blocks(((n >> 8) + 1) * sizeof(FrElement)), powers(256 * sizeof(FrElement)) {
        gpu_plonk_precompute_omega_tables_async(blocks.data(), powers.data(), &PilFflonk::msmShiftRatio(), 256,
                                                static_cast<uint32_t>((n >> 8) + 1), nullptr);
    }
    DeviceBuffer blocks, powers;
};

// The witness of N rows of C canonical scalars into C of the W = C + 2 columns of stage 1, in a
// shuffled order, as the Instance transposes it on the CPU: the other columns untouched.
void testTransposeWitness(Random &random) {
    for (uint64_t N : SIZES) {
        for (uint64_t C : {uint64_t(1), uint64_t(3), uint64_t(9)}) {
            if (C == 9 && N > 65537) {
                continue;
            }
            const uint64_t W = C + 2;
            std::vector<uint64_t> positions(W);
            std::iota(positions.begin(), positions.end(), 0);
            std::shuffle(positions.begin(), positions.end(), random.engine());
            positions.resize(C);
            const Column values = random.column(N * C);
            std::vector<uint8_t> raw(N * C * PilFflonk::FR_BYTES);
            Column expected(W * N, E.fr.zero());
            for (uint64_t v = 0; v < N * C; ++v) {
                FrElement canonical;
                E.fr.fromMontgomery(canonical, values[v]);
                std::memcpy(raw.data() + v * PilFflonk::FR_BYTES, canonical.v, PilFflonk::FR_BYTES);
                const uint8_t *scalar = raw.data() + v * PilFflonk::FR_BYTES;
                expected[positions[v % C] * N + v / C] = PilFflonk::fromCanonicalFr(scalar);
            }
            const DeviceBuffer dRaw = upload(raw), dPositions = upload(positions);
            const DeviceBuffer dColumns = upload(Column(W * N, E.fr.zero()));
            pilfflonk_gpu_transpose_witness(dColumns.data(), dRaw.data(),
                                            reinterpret_cast<const uint64_t *>(dPositions.data()), N, C);
            assert(same(download<FrElement>(dColumns, W * N), expected));
        }
    }
}

// pack() of k polynomials of `length` coefficients, one of a lower degree and one zero, plus the
// shift h^(i+1) of Gpu::shift, for MSMs of k·length scalars and more; the polynomials on the device
// in the reverse order, through the offsets. And the shift alone, with no polynomials.
void testPackShift(Random &random) {
    const ShiftTables tables(3 * SIZES.back() + 300);
    for (uint64_t k : {uint64_t(1), uint64_t(2), uint64_t(3), uint64_t(12)}) {
        uint64_t extra = 0;
        for (uint64_t length : SIZES) {
            if (k == 12 && length > 65537) {
                continue;
            }
            // MSMs of the packing's length, and of 1 and 300 more scalars.
            extra = extra == 0 ? 1 : extra == 1 ? 300 : 0;
            std::vector<Column> columns;
            std::vector<std::unique_ptr<Poly>> polys;
            Column onDevice(k * length);
            std::vector<uint64_t> offsets(k);
            for (uint64_t j = 0; j < k; ++j) {
                columns.push_back(random.column(length));
                if (j == 1) {
                    std::fill(columns[j].begin() + (length + 1) / 2, columns[j].end(), E.fr.zero());
                }
                if (j == 2) {
                    std::fill(columns[j].begin(), columns[j].end(), E.fr.zero());
                }
                offsets[j] = (k - 1 - j) * length;
                std::copy(columns[j].begin(), columns[j].end(), onDevice.begin() + offsets[j]);
            }
            for (Column &c : columns) {
                polys.emplace_back(Poly::fromReservedBuffer(E, c.data(), length));
            }
            std::vector<Poly *> pointers;
            for (const auto &p : polys) {
                pointers.push_back(p.get());
            }
            const uint64_t n = k * length + extra;
            Column packed(std::max(PilFflonk::packedBufferLength(k, length), n), E.fr.zero());
            const uint64_t nCoefs = PilFflonk::pack(pointers.data(), k, packed.data(), packed.size());
            std::fill(packed.begin() + nCoefs, packed.end(), E.fr.zero());
            Column rho(n), expected(n);
            PilFflonk::Gpu::shift(rho.data(), n);
            for (uint64_t i = 0; i < n; ++i) {
                E.fr.add(expected[i], packed[i], rho[i]);
            }
            const DeviceBuffer dBase = upload(onDevice), dOffsets = upload(offsets), dOut(n * sizeof(FrElement));
            pilfflonk_gpu_pack_shift(dOut.data(), n, dBase.data(), reinterpret_cast<const uint64_t *>(dOffsets.data()),
                                     k, length, tables.blocks.data(), tables.powers.data());
            assert(same(download<FrElement>(dOut, n), expected));
            pilfflonk_gpu_pack_shift(dOut.data(), n, nullptr, nullptr, 0, 0, tables.blocks.data(),
                                     tables.powers.data());
            assert(same(download<FrElement>(dOut, n), rho));
        }
    }
}

// Poly::blindCoefficients of nPolys polynomials of n coefficients and room for b factors, b above
// n too, where the CPU's order of the factors matters.
void testBlind(Random &random) {
    for (uint64_t n : {uint64_t(1), uint64_t(2), uint64_t(3), uint64_t(256), uint64_t(1025), uint64_t(65537)}) {
        for (uint64_t b : {uint64_t(1), uint64_t(2), uint64_t(3), uint64_t(5)}) {
            for (uint64_t nPolys : {uint64_t(1), uint64_t(3)}) {
                const uint64_t length = n + b;
                Column coefs(nPolys * length, E.fr.zero());
                std::vector<uint64_t> offsets(nPolys);
                for (uint64_t t = 0; t < nPolys; ++t) {
                    offsets[t] = (nPolys - 1 - t) * length;
                    const Column c = random.column(n);
                    std::copy(c.begin(), c.end(), coefs.begin() + offsets[t]);
                }
                const Column factors = random.column(nPolys * b);
                Column expected = coefs, mutableFactors = factors;
                for (uint64_t t = 0; t < nPolys; ++t) {
                    std::unique_ptr<Poly> p(Poly::fromReservedBuffer(E, expected.data() + offsets[t], length));
                    p->blindCoefficients(mutableFactors.data() + t * b, static_cast<uint32_t>(b));
                }
                const DeviceBuffer dCoefs = upload(coefs), dOffsets = upload(offsets), dFactors = upload(factors);
                pilfflonk_gpu_blind(dCoefs.data(), reinterpret_cast<const uint64_t *>(dOffsets.data()), nPolys, n,
                                    dFactors.data(), b);
                assert(same(download<FrElement>(dCoefs, coefs.size()), expected));
            }
        }
    }
}

// 1 + the highest non-zero index, as fixDegree finds it: of zero polynomials, of one non-zero
// coefficient first, last or anywhere, and of random ones; then of 70000 polynomials of 3
// coefficients, more than a launch has rows.
void testCountCoefficients(Random &random) {
    auto reference = [](const FrElement *p, uint64_t length) {
        for (uint64_t i = length; i-- > 0;) {
            if (!E.fr.isZero(p[i])) {
                return i + 1;
            }
        }
        return uint64_t(0);
    };
    for (uint64_t length : SIZES) {
        const uint64_t nPolys = 6;
        Column coefs(nPolys * length, E.fr.zero());
        const uint64_t anywhere = random.engine()() % length;
        coefs[1 * length] = E.fr.negOne();
        coefs[2 * length + length - 1] = E.fr.one();
        coefs[3 * length + anywhere] = random.element();
        const Column r = random.column(length);
        std::copy(r.begin(), r.end(), coefs.begin() + 4 * length);
        std::copy(r.begin(), r.begin() + anywhere, coefs.begin() + 5 * length);
        std::vector<uint64_t> offsets(nPolys), expected(nPolys);
        for (uint64_t t = 0; t < nPolys; ++t) {
            offsets[t] = t * length;
            expected[t] = reference(coefs.data() + offsets[t], length);
        }
        std::unique_ptr<Poly> p(Poly::fromReservedBuffer(E, coefs.data() + 4 * length, length));
        assert(expected[4] == p->getDegree() + 1 || (expected[4] == 0 && p->getDegree() == 0));
        const DeviceBuffer dCoefs = upload(coefs), dOffsets = upload(offsets);
        const DeviceBuffer dCounts(nPolys * sizeof(uint64_t));
        pilfflonk_gpu_count_coefficients(reinterpret_cast<uint64_t *>(dCounts.data()), dCoefs.data(),
                                         reinterpret_cast<const uint64_t *>(dOffsets.data()), nPolys, length);
        assert(download<uint64_t>(dCounts, nPolys) == expected);
    }
    const uint64_t nPolys = 70000, length = 3;
    Column coefs(nPolys * length, E.fr.zero());
    std::vector<uint64_t> offsets(nPolys), expected(nPolys);
    for (uint64_t t = 0; t < nPolys; ++t) {
        offsets[t] = t * length;
        expected[t] = t % 4;
        if (expected[t] > 0) {
            coefs[t * length + expected[t] - 1] = E.fr.one();
        }
    }
    const DeviceBuffer dCoefs = upload(coefs), dOffsets = upload(offsets), dCounts(nPolys * sizeof(uint64_t));
    pilfflonk_gpu_count_coefficients(reinterpret_cast<uint64_t *>(dCounts.data()), dCoefs.data(),
                                     reinterpret_cast<const uint64_t *>(dOffsets.data()), nPolys, length);
    assert(download<uint64_t>(dCounts, nPolys) == expected);
}

// GpuKey::commit, the MSM from the device with the length of the layout, is commitPacked's point:
// for MSMs longer than the packing (zero scalars add nothing), of every packing, and of zero
// polynomials (the point at infinity on both); refused for a length with no shift sum.
void testTheDeviceCommitsAsTheCpu(Random &random) {
    constexpr uint64_t N_G1 = 1024;
    TestDir dir;
    const std::string ptau = dir.file("gpu_kernels.ptau");
    writeTestPtau(ptau, N_G1);
    const PilFflonk::Srs srs = PilFflonk::Srs::fromPtau(ptau, N_G1);
    PilFflonk::GpuKey key(srs);
    const DeviceBuffer work(N_G1 * sizeof(FrElement));
    for (uint64_t k : {uint64_t(1), uint64_t(2), uint64_t(3)}) {
        for (uint64_t length : {uint64_t(1), uint64_t(33), uint64_t(100), uint64_t(255)}) {
            for (bool zero : {false, true}) {
                std::vector<Column> columns;
                std::vector<std::unique_ptr<Poly>> polys;
                std::vector<Poly *> pointers;
                Column onDevice;
                std::vector<uint64_t> offsets;
                for (uint64_t j = 0; j < k; ++j) {
                    columns.push_back(zero ? Column(length, E.fr.zero()) : random.column(length));
                    offsets.push_back(onDevice.size());
                    onDevice.insert(onDevice.end(), columns[j].begin(), columns[j].end());
                }
                for (Column &c : columns) {
                    polys.emplace_back(Poly::fromReservedBuffer(E, c.data(), length));
                    pointers.push_back(polys.back().get());
                }
                const uint64_t n = k * length + 17;
                key.addShiftSum(n, work.data());
                const DeviceBuffer dBase = upload(onDevice), dOffsets = upload(offsets);
                const G1Point onGpu = key.commit(dBase.data(), reinterpret_cast<const uint64_t *>(dOffsets.data()), k,
                                                 length, n, work.data());
                assert(affineBytes(onGpu) == affineBytes(PilFflonk::commitPacked(srs, pointers.data(), k)));
            }
        }
    }
    bool refused = false;
    try {
        key.commit(work.data(), nullptr, 1, 1, 5, work.data());
    } catch (const std::logic_error &e) {
        refused = contains(e.what(), "no shift sum for MSMs of 5 scalars");
    }
    assert(refused);
}

// Lde::extendCosetPart on the device (extendCosetPartOnDevice) is the CPU's, byte for byte, at every
// part of the parts of N, 2N and N' points: for one column of each size of SIZES the coset holds,
// with fewer, as many and more coefficients than a part (so more than 2S too), and of the real
// ones, N for a fixed column and N + |O| + 1 for a committed one; each column from a buffer of its
// own, all in one call, and more of them than a launch of the fold takes.
void testExtendCosetPartOnDevice(Random &random) {
    struct Case {
        uint64_t nBits, nBitsExt;
    };
    for (const Case &test : std::vector<Case>{{0, 2}, {3, 5}, {4, 7}, {8, 11}, {18, 21}}) {
        const Lde lde(test.nBits, test.nBitsExt);
        const uint64_t N = lde.domainSize(), NExt = lde.extendedSize();
        std::vector<uint64_t> lengths;
        for (uint64_t n : SIZES) {
            if (n <= NExt) {
                lengths.push_back(n);
            }
        }
        for (uint64_t blind : {uint64_t(0), uint64_t(2), uint64_t(3), uint64_t(4)}) {
            if (N + blind <= NExt) {
                lengths.push_back(N + blind);
            }
        }
        while (NExt <= 2048 && lengths.size() <= 32) {
            const std::vector<uint64_t> again = lengths;
            lengths.insert(lengths.end(), again.begin(), again.end());
        }
        std::vector<Column> columns;
        std::vector<DeviceBuffer> buffers;
        std::vector<const FrElement *> sources;
        for (uint64_t n : lengths) {
            columns.push_back(random.column(n));
            buffers.push_back(upload(columns.back()));
            sources.push_back(elementsOf(buffers.back()));
        }
        const DeviceBuffer tables(PilFflonk::ldeTableElements(lde) * sizeof(FrElement));
        std::vector<uint64_t> partSizes = {test.nBits};
        for (uint64_t bits : {test.nBits + 1, test.nBitsExt}) {
            if (bits <= test.nBitsExt && bits != partSizes.back()) {
                partSizes.push_back(bits);
            }
        }
        for (uint64_t partBits : partSizes) {
            const uint64_t S = uint64_t(1) << partBits;
            const DeviceBuffer evals(lengths.size() * S * sizeof(FrElement));
            for (uint64_t part = 0; part < NExt / S; ++part) {
                PilFflonk::extendCosetPartOnDevice(lde, sources.data(), lengths.data(), lengths.size(), partBits, part,
                                                   elementsOf(evals), elementsOf(tables));
                const Column onGpu = download<FrElement>(evals, lengths.size() * S);
                for (uint64_t t = 0; t < lengths.size(); ++t) {
                    Column expected(S);
                    const FrElement *in = columns[t].data();
                    FrElement *out = expected.data();
                    lde.extendCosetPart(&in, &out, 1, lengths[t], partBits, part);
                    assert(std::memcmp(onGpu.data() + t * S, expected.data(), S * sizeof(FrElement)) == 0);
                }
            }
        }
    }
}

// Lde::interpolateCoset on the device (interpolateCosetOnDevice) is the CPU's, byte for byte, on
// cosets of 1 to 2^21 points; and the scaling by the powers of a base (pilfflonk_gpu_mul_by_powers)
// is the product by each power, one after another, on every size of SIZES.
void testInterpolateCosetOnDevice(Random &random) {
    for (uint64_t nBitsExt : {0, 1, 3, 8, 9, 12, 16, 21}) {
        const Lde lde(0, nBitsExt);
        const uint64_t NExt = lde.extendedSize();
        Column values = random.column(NExt);
        const DeviceBuffer onDevice = upload(values), tables(PilFflonk::ldeTableElements(lde) * sizeof(FrElement));
        PilFflonk::interpolateCosetOnDevice(lde, elementsOf(onDevice), elementsOf(tables));
        FrElement *inPlace = values.data();
        lde.interpolateCoset(&inPlace, &inPlace, 1);
        assert(same(download<FrElement>(onDevice, NExt), values));
    }

    const FrElement base = random.element();
    const uint64_t nBlocks = (SIZES.back() + 255) / 256;
    const DeviceBuffer blocks(nBlocks * sizeof(FrElement)), powers(256 * sizeof(FrElement));
    gpu_plonk_precompute_omega_tables_async(blocks.data(), powers.data(), &base, 256, static_cast<uint32_t>(nBlocks),
                                            nullptr);
    for (uint64_t n : SIZES) {
        const Column data = random.column(n);
        Column expected(n);
        FrElement factor = E.fr.one();
        for (uint64_t i = 0; i < n; ++i) {
            E.fr.mul(expected[i], data[i], factor);
            E.fr.mul(factor, factor, base);
        }
        const DeviceBuffer onDevice = upload(data);
        pilfflonk_gpu_mul_by_powers(onDevice.data(), n, blocks.data(), powers.data());
        assert(same(download<FrElement>(onDevice, n), expected));
    }
}

void testDeviceKernels() {
    if (!gpuUnderTest("the device path's kernels")) {
        return;
    }
    Random random(58);
    testTransposeWitness(random);
    testPackShift(random);
    testBlind(random);
    testCountCoefficients(random);
    testTheDeviceCommitsAsTheCpu(random);
    testExtendCosetPartOnDevice(random);
    testInterpolateCosetOnDevice(random);
}

// An f of a SHPLONK opening on the device: its k, its offsets, and its p_j's coefficients, `length`
// random ones each if not 0, and otherwise N + |O| + 1 (a blinded column's bound) of assorted degrees,
// as pilfflonk_shplonk_test.cpp's: full, half, full minus one, and none for j = 2 when k > 2.
struct ShplonkF {
    uint64_t k;
    std::vector<int64_t> offsets;
    uint64_t length = 0;
};

struct ShplonkCase {
    uint64_t nBits;
    std::vector<ShplonkF> fs;
};

std::vector<ShplonkF> everyK(const std::vector<int64_t> &offsets) {
    std::vector<ShplonkF> fs;
    for (uint64_t k : {uint64_t(1), uint64_t(2), uint64_t(3), uint64_t(4), uint64_t(6), uint64_t(12)}) {
        fs.push_back({k, offsets});
    }
    return fs;
}

// pilfflonk_shplonk_test.cpp's shapes, every k with the offsets of the wrap's layouts, and the degrees
// at the edges: fewer coefficients than roots (f = r, nothing in W), constants (a component of one
// coefficient, whose division is that of a constant), p_j of one coefficient above k·|O|; every f = r
// (W = 0, [W]₁ at infinity); N = 2, where ω_N = −1; and N = 2^11, whose divisions are of more than
// one block of the scan, L's of about 12 times as many coefficients.
std::vector<ShplonkCase> shplonkCases() {
    return {
        {4, everyK({0})},
        {4, everyK({0, 1})},
        {4, everyK({-1, 0, 1, 2})},
        {3, {{4, {0, 1}}, {4, {0}}, {3, {-1, 0, 1, 2}}, {1, {0}}, {12, {0, 1}}, {2, {-1, 2}}, {6, {1}}}},
        {4, {{3, {0, 1}, 1}, {2, {-1, 0, 1, 2}}, {4, {0}, 2}, {12, {1}, 2}, {1, {0, 1}, 1}, {1, {0}, 1}, {2, {0}}}},
        {4, {{3, {0}, 1}, {1, {0}, 1}, {2, {0, 1}, 1}}},
        {1, {{2, {-1, 0}}, {1, {0}}, {4, {-1}}}},
        {11, {{1, {0, 1}}, {2, {-1, 0}}, {12, {0}}}},
    };
}

std::unique_ptr<Poly> shplonkComponent(const ShplonkF &f, uint64_t j, uint64_t N, Random &random) {
    const uint64_t length = f.length != 0 ? f.length : N + f.offsets.size() + 1;
    const uint64_t degrees[] = {length - 1, length / 2, length - 2, length - 1};
    std::unique_ptr<Poly> p(new Poly(E, length));
    if (f.length != 0 || f.k <= 2 || j != 2) {
        for (uint64_t i = 0; i <= (f.length != 0 ? length - 1 : degrees[j % 4]); ++i) {
            p->coef[i] = random.element();
        }
    }
    p->fixDegree();
    return p;
}

// What a call throws, or "" if nothing.
template <typename Call> std::string thrownBy(Call call) {
    try {
        call();
    } catch (const std::exception &e) {
        return e.what();
    }
    return "";
}

// A whole opening from a transcript of one element: its proof's α, [W]₁, y and [W']₁, or what it
// throws.
template <typename Open> std::vector<uint8_t> openingOutcome(Open open) {
    PilFflonk::Transcript t;
    t.absorb(std::vector<FrElement>{E.fr.one()});
    std::vector<uint8_t> out;
    const std::string error = thrownBy([&] {
        const PilFflonk::ShplonkProof proof = open(t);
        for (const FrElement *e : {&proof.alpha, &proof.y}) {
            out.insert(out.end(), reinterpret_cast<const uint8_t *>(e), reinterpret_cast<const uint8_t *>(e + 1));
        }
        for (const G1Point &p : {proof.w, proof.wp}) {
            const std::vector<uint8_t> bytes = affineBytes(p);
            out.insert(out.end(), bytes.begin(), bytes.end());
        }
    });
    return error.empty() ? out : std::vector<uint8_t>(error.begin(), error.end());
}

// OpeningGpu against ShplonkProver on the host, byte for byte, on a case's p_j on the device: the
// evaluations; W and W' (quotientW, quotientWp) and their commitments; a whole opening; and, with an
// interpolant off by one (r_0 at X^0, the last r_i at its top coefficient), the same errors: f_i − r_i
// is not divisible, nor is L. Its MSMs are of workLength() scalars, more than W's and W''s.
void testShplonkCase(const ShplonkCase &c, Random &random, const PilFflonk::Srs &srs, PilFflonk::GpuKey &key,
                     const DeviceBuffer &work) {
    const uint64_t N = uint64_t(1) << c.nBits;
    std::vector<std::unique_ptr<Poly>> polys;
    std::vector<DeviceBuffer> onDevice;
    PilFflonk::ShplonkOpening opening;
    opening.nBits = c.nBits;
    opening.powerW = 1;
    opening.xiSeed = random.element();
    PilFflonk::OpeningGpu::Components components;
    PilFflonk::ShplonkBounds bounds;
    for (const ShplonkF &f : c.fs) {
        PilFflonk::ShplonkPolynomial p;
        p.offsets = f.offsets;
        components.emplace_back();
        for (uint64_t j = 0; j < f.k; ++j) {
            polys.push_back(shplonkComponent(f, j, N, random));
            const Poly &poly = *polys.back();
            p.components.push_back(polys.back().get());
            onDevice.push_back(upload(Column(poly.coef, poly.coef + poly.getLength())));
            components.back().push_back(reinterpret_cast<const FrElement *>(onDevice.back().data()));
            bounds.component = std::max(bounds.component, poly.getLength());
        }
        bounds.component = std::max<uint64_t>(bounds.component, f.offsets.size());
        bounds.nEvaluations += f.k * f.offsets.size();
        bounds.nPoints += f.offsets.size();
        opening.powerW = std::lcm(opening.powerW, f.k);
        opening.polynomials.push_back(std::move(p));
    }
    const PilFflonk::ShplonkProver host(opening);
    bounds.length = bounds.wMsm = bounds.wpMsm = host.workLength();
    key.addShiftSum(bounds.length, work.data());
    const DeviceBuffer workspace(PilFflonk::shplonkWorkspaceBytes(bounds));
    PilFflonk::OpeningGpu gpu(key, components, bounds, workspace.data());

    const PilFflonk::ShplonkProver device(opening,
                                          [&](const PilFflonk::ShplonkProver &prover) { return gpu.evaluate(prover); });
    assert(device.size() == host.size());
    for (uint64_t i = 0; i < host.size(); ++i) {
        assert(same(device.evaluations()[i], host.evaluations()[i]));
    }

    const PilFflonk::ShplonkProver::Interpolants r = host.interpolants();
    const FrElement alpha = random.element(), y = random.element();
    const std::unique_ptr<Poly> W = host.quotientW(r, alpha);
    gpu.computeW(device, r, alpha);
    assert(same(download(gpu.quotient(), W->getLength()), Column(W->coef, W->coef + W->getLength())));
    assert(affineBytes(gpu.commitW()) == affineBytes(srs.commit(W->coef, W->getDegree() + 1)));
    const std::unique_ptr<Poly> Wp = host.quotientWp(r, alpha, y, *W);
    gpu.computeWp(device, r, alpha, y);
    assert(same(download(gpu.quotient(), Wp->getLength()), Column(Wp->coef, Wp->coef + Wp->getLength())));
    assert(affineBytes(gpu.commitWp()) == affineBytes(srs.commit(Wp->coef, Wp->getDegree() + 1)));

    assert(openingOutcome([&](PilFflonk::Transcript &t) { return device.open(srs, t, gpu); }) ==
           openingOutcome([&](PilFflonk::Transcript &t) { return host.open(srs, t); }));

    for (uint64_t i : {uint64_t(0), host.size() - 1}) {
        PilFflonk::ShplonkProver::Interpolants off = host.interpolants();
        Poly &ri = *off[i];
        const uint64_t at = i == 0 ? 0 : ri.getLength() - 1;
        E.fr.add(ri.coef[at], ri.coef[at], E.fr.one());
        ri.fixDegree();
        const std::string remainder = thrownBy([&] { host.quotientW(off, alpha); });
        assert(remainder == "ShplonkProver: f_" + std::to_string(i) + " - r_i is not divisible");
        assert(thrownBy([&] { gpu.computeW(device, off, alpha); }) == remainder);
        gpu.computeW(device, r, alpha);
        const std::string l = thrownBy([&] { host.quotientWp(off, alpha, y, *W); });
        assert(l == "ShplonkProver: L is not divisible");
        assert(thrownBy([&] { gpu.computeWp(device, off, alpha, y); }) == l);
    }

    // Not its opening: refused.
    const PilFflonk::OpeningGpu other(key, PilFflonk::OpeningGpu::Components(components.begin(), components.end() - 1),
                                      bounds, workspace.data());
    assert(contains(thrownBy([&] { other.evaluate(host); }).c_str(),
                    "OpeningGpu: the opening is not of its components"));
}

void testShplonkOnTheDevice() {
    if (!gpuUnderTest("SHPLONK on the device")) {
        return;
    }
    // Enough powers for the longest f, 12·(2^11 + 2) coefficients.
    constexpr uint64_t N_G1 = uint64_t(1) << 15;
    TestDir dir;
    const std::string ptau = dir.file("gpu_shplonk.ptau");
    writeTestPtau(ptau, N_G1);
    const PilFflonk::Srs srs = PilFflonk::Srs::fromPtau(ptau, N_G1);
    PilFflonk::GpuKey key(srs);
    const DeviceBuffer work(N_G1 * sizeof(FrElement));
    Random random(62);
    for (const ShplonkCase &c : shplonkCases()) {
        testShplonkCase(c, random, srs, key, work);
    }
}

// What a key on the GPU budgets and refuses, which needs no GPU: sppark's MSM's own memory at 50 M
// points on 170 SMs (pippenger.cuh: windows of 18 bits, 15 of them, buckets of 128 bytes, and 4
// batches of 12.5 M digits and temporaries), and the refusal's message.
void testBudgets() {
    assert(PilFflonk::spparkMsmBytes(50000000, 170) == 259696640 + 950000000);
    assert(PilFflonk::spparkMsmBytes(0, 170) == 0);
    PilFflonk::requireDeviceMemory("the test", 10, 10);
    bool refused = false;
    try {
        PilFflonk::requireDeviceMemory("the test", 11, 10);
    } catch (const std::invalid_argument &e) {
        refused = contains(e.what(), "not enough GPU memory for the test: it needs 11 bytes of device memory, and the "
                                     "device has 10 free");
    }
    assert(refused);
}

#endif

} // namespace

bool gpuUnderTest(const char *what) {
    if (pilfflonk_gpu_available() == 1) {
        return true;
    }
    const char *required = std::getenv("PILFFLONK_GPU");
    if (required != nullptr && std::strcmp(required, "1") == 0) {
        std::fprintf(stderr, "pilfflonk_test: PILFFLONK_GPU=1, and there is no GPU for %s\n", what);
        assert(!"PILFFLONK_GPU=1 requires a GPU");
    }
    std::printf("pilfflonk_test: no GPU, %s skipped\n", what);
    return false;
}

void runGpuTests() {
    testAvailability();
    testCtxNewOn();
#ifdef __USE_CUDA__
    testBudgets();
    testDeviceKernels();
    testShplonkOnTheDevice();
#endif
}

} // namespace PilFflonkTest
