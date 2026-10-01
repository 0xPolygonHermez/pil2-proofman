// Tests of the GPU path's entry points (pilfflonk/docs/performance.md#selection-memory-and-errors),
// in every build: pilfflonk_gpu_available, and pilfflonk_ctx_new_on and ProvingKey::load, which
// refuse the GPU, saying why, before they read anything, in a library built without it
// (make pilfflonk_test) and on a machine without one (make pilfflonk_gpu_test there). Where there is
// one, the tests of each module compare its GPU path with its CPU one (gpuUnderTest):
// pilfflonk_lde_test.cpp the transforms, pilfflonk_commit_test.cpp the MSM,
// pilfflonk_prover_test.cpp whole proofs and a key on the GPU; and this file the kernels of the
// device path (pilfflonk_kernels.hpp) and GpuKey's commitments, byte for byte against the CPU's
// code on seeded inputs, edge values (0, 1, r − 1, all equal) and sizes around the warp, the block
// of 256 threads and powers of two.
#include "pilfflonk_test.hpp"
#include "pilfflonk_test_ptau.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "pilfflonk_api.hpp"
#include "pilfflonk_gpu.hpp"
#include "pilfflonk_proving_key.hpp"
#ifdef __USE_CUDA__
#include "pilfflonk_commit.hpp"
#include "pilfflonk_kernels.hpp"
#include "pilfflonk_key_gpu.hpp"
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

bool same(const Column &a, const Column &b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(FrElement)) == 0;
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
    auto affine = [](G1Point p) {
        G1PointAffine a;
        E.g1.copy(a, p);
        return std::vector<uint8_t>(reinterpret_cast<uint8_t *>(&a), reinterpret_cast<uint8_t *>(&a) + sizeof(a));
    };
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
                assert(affine(onGpu) == affine(PilFflonk::commitPacked(srs, pointers.data(), k)));
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
#endif
}

} // namespace PilFflonkTest
