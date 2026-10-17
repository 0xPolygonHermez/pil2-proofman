#ifndef PILFFLONK_GPU_HPP
#define PILFFLONK_GPU_HPP

#include <atomic>
#include <cstdint>
#include <functional>

#include "alt_bn128.hpp"

namespace PilFflonk {

using FrElement = AltBn128::Engine::FrElement;
using G1Point = AltBn128::Engine::G1Point;
using G1PointAffine = AltBn128::Engine::G1PointAffine;

// Where a ProvingKey runs its proofs (pilfflonk/docs/performance.md#gpu). The proof is the same, bit
// for bit, on either.
enum class Device {
    Cpu, // ffiasm's MSM and FFT, and the host's code
    Gpu, // the device's (GpuKey, pilfflonk_key_gpu.hpp)
};

// The bytes the GPU path copies between the host and the device, each way, for the -vv log of each
// phase (CopyLog, pilfflonk_key_gpu.hpp). Safe to use from several threads.
class CopyVolume {
public:
    struct Totals {
        uint64_t toDevice = 0;
        uint64_t toHost = 0;
    };

    void addToDevice(uint64_t bytes) { h2d += bytes; }
    void addToHost(uint64_t bytes) { d2h += bytes; }
    Totals totals() const { return Totals{h2d.load(), d2h.load()}; }

private:
    std::atomic<uint64_t> h2d{0};
    std::atomic<uint64_t> d2h{0};
};

// How a key on the GPU (GpuKey, pilfflonk_key_gpu.hpp) uses the device's memory.
struct GpuKeyOptions {
    // The most device memory the key may hold and leave for a proof, in bytes, or 0 for what the
    // device has free.
    uint64_t memoryLimit = 0;
    // A device buffer of arenaBytes bytes on device 0 for the arena of its proofs, in place of one the
    // key allocates: a wrap's pre-reserved memory, which others use between the proofs (proofman's
    // unified buffer). The key writes it only while a proof holds it (GpuKey::Lease), never while it
    // loads, and never frees it: it must outlive the key.
    void *arena = nullptr;
    uint64_t arenaBytes = 0;
    // The arena's buffer is the key's alone while it lives: it holds all its device memory, its own
    // from the start and its proofs' arena from the end (DevicePool), and the key writes it as its own.
    bool exclusive = false;
    // With exclusive, the buffer is others' between the proofs (a preloaded wrap's): once loaded the
    // key copies what it holds there to pinned host memory (GpuKey::snapshot), which restore() writes
    // back before each proof.
    bool restorable = false;
};

// The device memory a key on the GPU needs (ProvingKey::requiredDeviceBytes), in bytes: the arena of
// its proofs, which a buffer given to it (GpuKeyOptions::arena) must hold, and the most it holds and
// allocates beside that buffer: the SRS's powers and the tables of the MSMs' shift, each AIR's fixed
// columns' coefficients, bytecode and interpreter, the scratch of an AIR's loading (which it
// allocates while the AIR loads, as it never writes a given arena then) and what a proof allocates
// besides (sppark's MSM). A key loads on a given arena with a memoryLimit (or free memory) of
// `beside`, and not of one byte less; on its own arena, which holds its loading's scratch, with
// arena + beside at most.
struct DeviceBytes {
    uint64_t arena = 0;
    uint64_t beside = 0;
};

// The GPU of a key on it (pilfflonk/docs/performance.md#what-runs-on-the-gpu): whether there is one,
// and the copy on the device of the SRS's powers [τ^i]₁, which the MSMs of a key on the GPU read
// (GpuKey, pilfflonk_key_gpu.hpp, which keeps every other datum of the key and its proofs on the
// device too). The MSMs and NTTs below are the GPU entry points that pil2-stark already has, called
// as the PLONK GPU prover calls them (rapidsnark/plonk_prover_gpu.c.cuh):
// - msm_bn128_gpu_dev_ptr (bn128/src/msm/msm_bn128.cu, sppark's Pippenger), with montgomery = true:
//   the scalars go as ffiasm keeps them, in Montgomery form, and the result is sppark's Jacobian
//   point, whose X/Z² and Y/Z³ are ffiasm's x/zz and y/zzz (multiExponentiationGPU_devptr);
// - ntt_bn128_gpu_dev_ptr and intt_bn128_gpu_dev_ptr (bn128/src/ntt/ntt_bn128.cu, sppark's NTT), in
//   natural order in and out. sppark's roots are ffiasm's, ω_{2^k} = 5^((r-1)/2^k), in the same
//   Montgomery form (ntt/parameters/alt_bn128.h), and its inverse scales by 1/2^k, as ffiasm's ifft;
// - gpu_plonk_cuda_malloc and _free, gpu_plonk_memcpy_h2d, gpu_plonk_cuda_device_sync and
//   gpu_plonk_set_device (rapidsnark/plonk_prover.cu), for the device's memory.
// Both sides compute in exact field arithmetic and keep every element in its canonical Montgomery
// form (limbs below r), and a commitment leaves the prover in affine coordinates: the results are
// those of ffiasm's MSM and FFT bit for bit.
//
// Device 0, the PLONK GPU prover's default, whatever the calling thread's current device: every
// function of the GPU path a caller reaches makes it current while it runs (DeviceScope). Only a
// library built with the GPU
// (provers/starks-lib-c/build.rs found nvcc: libstarksgpu.a, its sources compiled with __USE_CUDA__)
// defines it, in pilfflonk_gpu.cpp; the code that uses it is under __USE_CUDA__, and elsewhere
// gpuAvailable() is false. A CUDA failure inside the helpers above (out of device memory, a lost
// device) aborts the process, as it does in the PLONK GPU prover.
class Gpu {
public:
    // Whether this library has the GPU path and sees a GPU it can use: sppark's cuda_available()
    // (external/sppark/util/all_gpus.cpp), a device of compute capability 7.0 or above with
    // cooperative launch. False, without failing, where CUDA finds no device or no driver.
    static bool available();

    // A copy of `bytes` bytes from host `src` to device `dst`, which returns once src may change.
    using Upload = std::function<void(void *dst, const void *src, uint64_t bytes)>;
    // A copy on the device of the n points (n >= 1) at `points`, the powers [τ^i]₁ of an SRS, copied
    // by `upload` (a GpuKey's pinned Staging, which counts them). Throws std::invalid_argument if
    // !available(), points is null or n is 0.
    Gpu(const G1PointAffine *points, uint64_t n, const Upload &upload, void *at = nullptr);
    ~Gpu();
    Gpu(const Gpu &) = delete;
    Gpu &operator=(const Gpu &) = delete;

    uint64_t nPoints() const { return nDevicePoints; }
    // The points on the device, which a GpuKey's MSMs read.
    const void *devicePowers() const { return devicePoints; }

private:
    void *devicePoints = nullptr;
    uint64_t nDevicePoints = 0;
    bool owned = true; // not given at construction
};

// While it lives, the GPU's device (device 0, Gpu's) is the calling thread's current CUDA device;
// then the one the thread had before is again. Every function of the GPU path that a caller reaches
// holds one while it runs (a key's load, an instance's and an opening's calls, their device memory's
// and streams' release, ProvingKey::requiredDeviceBytes), as the repo's other GPU entry points set
// their device (gen_recursive_proof_final_gpu): a thread whose current device is another, as one of
// a multi-GPU process may be, proves on device 0 and keeps its own. Only a library built with the
// GPU defines it, in pilfflonk_gpu.cpp.
class DeviceScope {
public:
    DeviceScope();
    ~DeviceScope();
    DeviceScope(const DeviceScope &) = delete;
    DeviceScope &operator=(const DeviceScope &) = delete;

private:
    int previous;
};

// Σ_{i<n} scalars[i]·points[i] for n >= 1 scalars in Montgomery form and n points [τ^i]₁, all on the
// device: msm_bn128_gpu_dev_ptr, after the device is synchronised (the scalars may come from the
// default stream, and sppark's MSM runs on its own), as ffiasm's point. The point at infinity is
// also sppark's report of a failure.
G1Point msmOnDevice(const void *points, const void *scalars, uint64_t n);

// The NTT (ffiasm's fft) or the INTT (its ifft) of the 2^bits values at `data`, on the device, in
// place and in natural order: ntt_bn128_gpu_dev_ptr or intt_bn128_gpu_dev_ptr after the device is
// synchronised. Nothing for bits = 0, a value being its own transform. bits <= 28.
void transformOnDevice(void *data, uint64_t bits, bool inverse);

// Gpu::available() in a library built with the GPU, and false in one built without.
inline bool gpuAvailable() {
#ifdef __USE_CUDA__
    return Gpu::available();
#else
    return false;
#endif
}

} // namespace PilFflonk

#endif
