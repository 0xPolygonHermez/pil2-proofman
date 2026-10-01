// pilfflonk's GPU path (plan M43): compiled into the GPU library only (libstarksgpu.a, the Makefile's
// %_gpu.cpp rule), with g++, as final_snark_proof_gpu.cpp. It calls the GPU entry points of
// pil2-stark through their C linkage, as plonk_prover_gpu.c.cuh does, and has no kernel of its own.
#include "pilfflonk_gpu.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

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

    // sppark's jacobian_t<fp_t>: (X, Y, Z) in Montgomery form, the point (X/Z², Y/Z³).
    struct Jacobian {
        Engine::F1Element X, Y, Z;
    } jacobian;
    {
        const std::lock_guard<std::mutex> guard(lock);
        gpu_plonk_set_device(DEVICE);
        void *dScalars = scratch(n);
        gpu_plonk_memcpy_h2d(dScalars, scalars, n * sizeof(FrElement));
        // The copy is on the default stream, and sppark's MSM on streams of its own.
        gpu_plonk_cuda_device_sync();
        msm_bn128_gpu_dev_ptr(&jacobian, devicePoints, dScalars, n, true);
    }
    // ffiasm's extended Jacobian point: (x, y, zz, zzz) is (x/zz, y/zzz), zz = Z² and zzz = Z³.
    result.x = jacobian.X;
    result.y = jacobian.Y;
    E.f1.square(result.zz, jacobian.Z);
    E.f1.mul(result.zzz, result.zz, jacobian.Z);
    if (E.g1.isZero(result) && !allZero(scalars, n)) {
        throw std::runtime_error("Gpu::msm: the GPU's MSM of " + std::to_string(n) +
                                 " points gave the point at infinity for scalars that are not all zero: "
                                 "msm_bn128_gpu_dev_ptr failed (or τ is a root of the polynomial)");
    }
    return result;
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
