// pilfflonk's GPU and its MSMs and NTTs on device data (pilfflonk_gpu.hpp): compiled into the GPU
// library only (libstarksgpu.a, the Makefile's %_gpu.cpp rule), with g++, as
// final_snark_proof_gpu.cpp. It calls the GPU entry points of pil2-stark through their C linkage, as
// plonk_prover_gpu.c.cuh does, and has no kernel of its own.
#include "pilfflonk_gpu.hpp"

#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "pilfflonk_error.hpp"
#include "pilfflonk_kernels.hpp"

namespace PilFflonk {

namespace {

using Engine = AltBn128::Engine;

// The device of the GPU path (setGpuDevice): device 0, the PLONK GPU prover's default, until set.
std::atomic<int> gpuDevice{0};
// The GpuKeys alive (holdGpuDevice): their memory is on gpuDevice, which then stays.
std::atomic<int> gpuKeys{0};

constexpr InvalidArgument invalid("Gpu::");

} // namespace

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

void setGpuDevice(int device) {
    const int count = pilfflonk_gpu_device_count();
    if (device < 0 || device >= count) {
        throw std::invalid_argument("setGpuDevice: CUDA has no device " + std::to_string(device) + " (it sees " +
                                    std::to_string(count) + ")");
    }
    if (device != gpuDevice.load() && gpuKeys.load() > 0) {
        throw std::invalid_argument("setGpuDevice: a key is loaded on device " + std::to_string(gpuDevice.load()) +
                                    ", and its memory cannot follow to device " + std::to_string(device));
    }
    gpuDevice.store(device);
}

void holdGpuDevice() { gpuKeys.fetch_add(1); }

void releaseGpuDevice() { gpuKeys.fetch_sub(1); }

DeviceScope::DeviceScope() : previous(pilfflonk_gpu_current_device()), device(gpuDevice.load()) {
    if (previous != device) {
        gpu_plonk_set_device(device);
    }
}

DeviceScope::~DeviceScope() {
    if (previous != device) {
        gpu_plonk_set_device(previous);
    }
}

bool Gpu::available() { return cuda_available(); }

Gpu::Gpu(uint64_t n, const Fill &fill, void *at) {
    if (!available()) {
        throw invalid("Gpu", "no GPU: CUDA sees no device of compute capability 7.0 or above (or no driver)");
    }
    if (n == 0) {
        throw invalid("Gpu", "no points");
    }
    const DeviceScope device;
    owned = at == nullptr;
    if (owned) {
        gpu_plonk_cuda_malloc(&devicePoints, n * sizeof(G1PointAffine));
    } else {
        devicePoints = at;
    }
    nDevicePoints = n;
    try {
        fill(devicePoints, n * sizeof(G1PointAffine));
    } catch (...) {
        // A constructor that throws runs no destructor.
        if (owned) {
            gpu_plonk_cuda_free(devicePoints);
        }
        throw;
    }
}

Gpu::~Gpu() {
    if (!owned) {
        return;
    }
    const DeviceScope device;
    gpu_plonk_cuda_free(devicePoints);
}

} // namespace PilFflonk
