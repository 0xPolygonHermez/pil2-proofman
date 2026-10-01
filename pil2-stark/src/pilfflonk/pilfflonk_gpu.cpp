// pilfflonk's GPU path (pilfflonk/docs/performance.md#gpu): compiled into the GPU library only
// (libstarksgpu.a, the Makefile's %_gpu.cpp rule), with g++, as final_snark_proof_gpu.cpp. It calls
// the GPU entry points of pil2-stark through their C linkage, as plonk_prover_gpu.c.cuh does, and
// has no kernel of its own.
#include "pilfflonk_gpu.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "pilfflonk_lde.hpp"

// The MSM and the NTTs (bn128/src/msm/msm_bn128.cu, bn128/src/ntt/ntt_bn128.cu) and the device
// memory helpers (rapidsnark/plonk_prover.cu), declared as plonk_prover_gpu.c.cuh declares them; and
// sppark's probe for a usable GPU (external/sppark/util/all_gpus.cpp).
extern "C" void msm_bn128_gpu_dev_ptr(void *out, const void *d_points, const void *d_scalars, size_t npoints,
                                      bool montgomery);
extern "C" void ntt_bn128_gpu_dev_ptr(void *d_data, uint32_t lg_n);
extern "C" void intt_bn128_gpu_dev_ptr(void *d_data, uint32_t lg_n);
extern "C" void gpu_plonk_memcpy_h2d(void *dst, const void *src, size_t bytes);
extern "C" void gpu_plonk_memcpy_d2h(void *dst, const void *src, size_t bytes);
extern "C" void gpu_plonk_cuda_malloc(void **dBuffer, uint64_t buffeSize);
extern "C" void gpu_plonk_cuda_free(void *dBuffer);
extern "C" void gpu_plonk_cuda_device_sync();
extern "C" void gpu_plonk_set_device(int gpuId);
extern "C" bool cuda_available();

namespace PilFflonk {

namespace {

using Engine = AltBn128::Engine;

// The PLONK GPU prover's default device.
constexpr int DEVICE = 0;

// The largest NTT: sppark's BN254 parameters have the roots of up to 2^28 points
// (ntt/parameters/alt_bn128.h, S = 28), the 2-adicity of r.
constexpr uint64_t MAX_NTT_BITS = 28;

std::invalid_argument invalid(const char *function, const std::string &message) {
    return std::invalid_argument(std::string("Gpu::") + function + ": " + message);
}

bool allZero(const FrElement *values, uint64_t n) {
    Engine::Fr &fr = Engine::engine.fr;
    bool zero = true;
#pragma omp parallel for reduction(&& : zero)
    for (uint64_t i = 0; i < n; ++i) {
        zero = zero && fr.isZero(values[i]);
    }
    return zero;
}

// h, the ratio of the shift ρ_i = h^(i+1) (pilfflonk_gpu.hpp). Any element of large order does; a
// fixed one makes every intermediate value the same from run to run.
const FrElement &shiftRatio() {
    static const FrElement ratio = [] {
        FrElement h;
        Engine::engine.fr.fromString(h, "6277101735386680763835789423207666416102355444464034512659");
        return h;
    }();
    return ratio;
}

// out[i] = scalars[i] + ρ_i, or ρ_i if scalars is null, for i < n. out may be scalars. In chunks,
// each of which starts from its first ρ (one exponentiation) and multiplies by h from there.
void shifted(FrElement *out, const FrElement *scalars, uint64_t n) {
    Engine::Fr &fr = Engine::engine.fr;
    constexpr uint64_t CHUNK = uint64_t(1) << 14;
    const FrElement &h = shiftRatio();
#pragma omp parallel for schedule(static)
    for (uint64_t start = 0; start < n; start += CHUNK) {
        FrElement rho = power(h, start + 1);
        const uint64_t end = std::min(n, start + CHUNK);
        for (uint64_t i = start; i < end; ++i) {
            if (scalars != nullptr) {
                fr.add(out[i], scalars[i], rho);
            } else {
                out[i] = rho;
            }
            fr.mul(rho, rho, h);
        }
    }
}

} // namespace

bool Gpu::available() { return cuda_available(); }

Gpu::Gpu(const G1PointAffine *points, uint64_t n) {
    if (!available()) {
        throw invalid("Gpu", "no GPU: CUDA sees no device of compute capability 7.0 or above (or no driver)");
    }
    if (points == nullptr) {
        throw invalid("Gpu", "points is null");
    }
    if (n == 0) {
        throw invalid("Gpu", "no points");
    }
    gpu_plonk_set_device(DEVICE);
    gpu_plonk_cuda_malloc(&devicePoints, n * sizeof(G1PointAffine));
    gpu_plonk_memcpy_h2d(devicePoints, points, n * sizeof(G1PointAffine));
    nDevicePoints = n;
}

Gpu::~Gpu() {
    gpu_plonk_cuda_free(deviceScratch);
    gpu_plonk_cuda_free(devicePoints);
}

void *Gpu::scratch(uint64_t n) const {
    if (n > scratchElements) {
        gpu_plonk_cuda_free(deviceScratch);
        deviceScratch = nullptr;
        scratchElements = 0;
        gpu_plonk_cuda_malloc(&deviceScratch, n * sizeof(FrElement));
        scratchElements = n;
    }
    return deviceScratch;
}

G1Point Gpu::msm(const FrElement *scalars, uint64_t n) const {
    Engine &E = Engine::engine;
    if (n > nDevicePoints) {
        throw invalid("msm", std::to_string(n) + " scalars for the " + std::to_string(nDevicePoints) +
                                 " points on the device");
    }
    G1Point result;
    if (n == 0) {
        E.g1.copy(result, E.g1.zero());
        return result;
    }
    if (scalars == nullptr) {
        throw invalid("msm", "scalars is null");
    }

    const std::lock_guard<std::mutex> guard(lock);
    gpu_plonk_set_device(DEVICE);
    FrElement *shiftedScalars = staging(n);
    auto shiftSum = shiftSums.find(n);
    if (shiftSum == shiftSums.end()) {
        shifted(shiftedScalars, nullptr, n);
        G1Point sum = deviceMsm(shiftedScalars, n);
        if (E.g1.isZero(sum)) {
            throw std::runtime_error("Gpu::msm: the GPU's MSM of the shift of " + std::to_string(n) +
                                     " points gave the point at infinity: msm_bn128_gpu_dev_ptr failed (or τ is a "
                                     "root of the shift's polynomial)");
        }
        shiftSum = shiftSums.emplace(n, sum).first;
    }
    shifted(shiftedScalars, scalars, n);
    G1Point shiftedSum = deviceMsm(shiftedScalars, n);
    if (E.g1.isZero(shiftedSum) && !allZero(shiftedScalars, n)) {
        throw std::runtime_error("Gpu::msm: the GPU's MSM of " + std::to_string(n) +
                                 " points gave the point at infinity for shifted scalars that are not all zero: "
                                 "msm_bn128_gpu_dev_ptr failed (or τ is a root of the shifted polynomial)");
    }
    E.g1.sub(result, shiftedSum, shiftSum->second);
    return result;
}

void Gpu::shift(FrElement *out, uint64_t n) {
    if (out == nullptr && n > 0) {
        throw invalid("shift", "out is null");
    }
    shifted(out, nullptr, n);
}

G1Point Gpu::deviceMsm(const FrElement *scalars, uint64_t n) const {
    Engine &E = Engine::engine;
    // sppark's jacobian_t<fp_t>: (X, Y, Z) in Montgomery form, the point (X/Z², Y/Z³).
    struct Jacobian {
        Engine::F1Element X, Y, Z;
    } jacobian;
    void *dScalars = scratch(n);
    gpu_plonk_memcpy_h2d(dScalars, scalars, n * sizeof(FrElement));
    // The copy is on the default stream, and sppark's MSM on streams of its own.
    gpu_plonk_cuda_device_sync();
    msm_bn128_gpu_dev_ptr(&jacobian, devicePoints, dScalars, n, true);
    // ffiasm's extended Jacobian point: (x, y, zz, zzz) is (x/zz, y/zzz), zz = Z² and zzz = Z³.
    G1Point result;
    result.x = jacobian.X;
    result.y = jacobian.Y;
    E.f1.square(result.zz, jacobian.Z);
    E.f1.mul(result.zzz, result.zz, jacobian.Z);
    return result;
}

FrElement *Gpu::staging(uint64_t n) const {
    if (n > stagingElements) {
        hostStaging.reset();
        stagingElements = 0;
        hostStaging.reset(new FrElement[n]);
        stagingElements = n;
    }
    return hostStaging.get();
}

void Gpu::transform(const FrElement *in, FrElement *out, uint64_t bits, bool inverse) const {
    const char *function = inverse ? "intt" : "ntt";
    if (bits > MAX_NTT_BITS) {
        throw invalid(function, "2^" + std::to_string(bits) + " points, and r has roots of unity of up to 2^" +
                                    std::to_string(MAX_NTT_BITS));
    }
    if (in == nullptr || out == nullptr) {
        throw invalid(function, in == nullptr ? "in is null" : "out is null");
    }
    const uint64_t bytes = (uint64_t(1) << bits) * sizeof(FrElement);
    if (bits == 0) {
        // One value is its own transform, either way; sppark's NTT starts at two points.
        if (out != in) {
            *out = *in;
        }
        return;
    }
    const std::lock_guard<std::mutex> guard(lock);
    gpu_plonk_set_device(DEVICE);
    void *data = scratch(uint64_t(1) << bits);
    gpu_plonk_memcpy_h2d(data, in, bytes);
    // As the PLONK GPU prover before each NTT: the copy is on the default stream, the NTT on sppark's.
    gpu_plonk_cuda_device_sync();
    if (inverse) {
        intt_bn128_gpu_dev_ptr(data, static_cast<uint32_t>(bits));
    } else {
        ntt_bn128_gpu_dev_ptr(data, static_cast<uint32_t>(bits));
    }
    // The NTT has synchronised its stream; the copy back waits for it.
    gpu_plonk_memcpy_d2h(out, data, bytes);
}

void Gpu::ntt(const FrElement *in, FrElement *out, uint64_t bits) const { transform(in, out, bits, false); }

void Gpu::intt(const FrElement *in, FrElement *out, uint64_t bits) const { transform(in, out, bits, true); }

} // namespace PilFflonk
