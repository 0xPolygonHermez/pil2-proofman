// pilfflonk's GPU and its MSMs and NTTs on device data (pilfflonk_gpu.hpp): compiled into the GPU
// library only (libstarksgpu.a, the Makefile's %_gpu.cpp rule), with g++, as
// final_snark_proof_gpu.cpp. It calls the GPU entry points of pil2-stark through their C linkage, as
// plonk_prover_gpu.c.cuh does, and has no kernel of its own.
#include "pilfflonk_gpu.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

#include "pilfflonk_error.hpp"
#include "pilfflonk_kernels.hpp"

namespace PilFflonk {

namespace {

using Engine = AltBn128::Engine;

// The PLONK GPU prover's default device.
constexpr int DEVICE = 0;

constexpr InvalidArgument invalid("Gpu::");

} // namespace

// Any element of large order does; a fixed one makes every intermediate value the same from run to
// run.
const FrElement &msmShiftRatio() {
    static const FrElement ratio = [] {
        FrElement h;
        Engine::engine.fr.fromString(h, "6277101735386680763835789423207666416102355444464034512659");
        return h;
    }();
    return ratio;
}

G1Point msmOnDevice(const void *points, const void *scalars, uint64_t n) {
    Engine &E = Engine::engine;
    // sppark's jacobian_t<fp_t>: (X, Y, Z) in Montgomery form, the point (X/Z², Y/Z³).
    struct Jacobian {
        Engine::F1Element X, Y, Z;
    } jacobian;
    const DeviceScope device;
    gpu_plonk_cuda_device_sync();
    msm_bn128_gpu_dev_ptr(&jacobian, points, scalars, n, true);
    // ffiasm's extended Jacobian point: (x, y, zz, zzz) is (x/zz, y/zzz), zz = Z² and zzz = Z³.
    G1Point result;
    result.x = jacobian.X;
    result.y = jacobian.Y;
    E.f1.square(result.zz, jacobian.Z);
    E.f1.mul(result.zzz, result.zz, jacobian.Z);
    return result;
}

void transformOnDevice(void *data, uint64_t bits, bool inverse) {
    // sppark's NTT starts at two points.
    if (bits == 0) {
        return;
    }
    // As the PLONK GPU prover before each NTT: the data may come from the default stream, and the
    // NTT runs on sppark's, which it synchronises before it returns.
    const DeviceScope device;
    gpu_plonk_cuda_device_sync();
    if (inverse) {
        intt_bn128_gpu_dev_ptr(data, static_cast<uint32_t>(bits));
    } else {
        ntt_bn128_gpu_dev_ptr(data, static_cast<uint32_t>(bits));
    }
}

DeviceScope::DeviceScope() : previous(pilfflonk_gpu_current_device()) {
    if (previous != DEVICE) {
        gpu_plonk_set_device(DEVICE);
    }
}

DeviceScope::~DeviceScope() {
    if (previous != DEVICE) {
        gpu_plonk_set_device(previous);
    }
}

bool Gpu::available() { return cuda_available(); }

Gpu::Gpu(const G1PointAffine *points, uint64_t n, const Upload &upload) {
    if (!available()) {
        throw invalid("Gpu", "no GPU: CUDA sees no device of compute capability 7.0 or above (or no driver)");
    }
    if (points == nullptr) {
        throw invalid("Gpu", "points is null");
    }
    if (n == 0) {
        throw invalid("Gpu", "no points");
    }
    const DeviceScope device;
    gpu_plonk_cuda_malloc(&devicePoints, n * sizeof(G1PointAffine));
    nDevicePoints = n;
    upload(devicePoints, points, n * sizeof(G1PointAffine));
}

Gpu::~Gpu() {
    const DeviceScope device;
    gpu_plonk_cuda_free(devicePoints);
}

} // namespace PilFflonk
