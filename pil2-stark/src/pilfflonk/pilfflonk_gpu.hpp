#ifndef PILFFLONK_GPU_HPP
#define PILFFLONK_GPU_HPP

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>

#include "alt_bn128.hpp"

namespace PilFflonk {

using FrElement = AltBn128::Engine::FrElement;
using G1Point = AltBn128::Engine::G1Point;
using G1PointAffine = AltBn128::Engine::G1PointAffine;

// Where a ProvingKey runs the MSMs and the NTTs of its proofs (pilfflonk/docs/performance.md#gpu).
// The proof is the same, bit for bit, on either.
enum class Device {
    Cpu, // ffiasm's MSM and FFT
    Gpu, // Gpu's, below
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

// The MSMs and the NTTs of the prover on the GPU from host memory
// (pilfflonk/docs/performance.md#what-runs-on-the-gpu): each copies its data to the device and its
// result back. A key on the GPU (GpuKey, pilfflonk_key_gpu.hpp) keeps the data of its stage and
// fixed commitments and of the opening on the device instead, and Q's LDE runs there (LdeGpu): it
// uses this one only for the MSMs of Q, which still run from the host. It has no kernel: it calls
// the GPU entry points that pil2-stark already has, as the PLONK GPU prover calls them
// (rapidsnark/plonk_prover_gpu.c.cuh):
// - msm_bn128_gpu_dev_ptr (bn128/src/msm/msm_bn128.cu, sppark's Pippenger), with montgomery = true,
//   on a copy of the SRS's powers [τ^i]₁ kept on the device: the scalars go as ffiasm keeps them,
//   in Montgomery form, and the result is sppark's Jacobian point, whose X/Z² and Y/Z³ are ffiasm's
//   x/zz and y/zzz (multiExponentiationGPU_devptr);
// - ntt_bn128_gpu_dev_ptr and intt_bn128_gpu_dev_ptr (bn128/src/ntt/ntt_bn128.cu, sppark's NTT), in
//   natural order in and out. sppark's roots are ffiasm's, ω_{2^k} = 5^((r-1)/2^k), in the same
//   Montgomery form (ntt/parameters/alt_bn128.h), and its inverse scales by 1/2^k, as ffiasm's ifft;
// - gpu_plonk_cuda_malloc and _free, gpu_plonk_memcpy_h2d and _d2h, gpu_plonk_cuda_device_sync and
//   gpu_plonk_set_device (rapidsnark/plonk_prover.cu), for the device's memory.
// Both sides compute in exact field arithmetic and keep every element in its canonical Montgomery
// form (limbs below r), and a commitment leaves the prover in affine coordinates: the results are
// those of ffiasm's MSM and FFT bit for bit. The elementwise work around them (the coset shifts,
// the folding of Lde::extendCosetPart, the packing) stays on the CPU, where Lde and commitPacked do
// it: the PLONK GPU prover has no helper for a coset on the device.
//
// The MSM shifts its scalars: it computes Σ (s_i + ρ_i)·[τ^i]₁ − Σ ρ_i·[τ^i]₁, with
// ρ_i = h^(i+1) for a fixed h (shift). sppark's Pippenger is fast when the scalars look random and
// slow when many of them repeat, as when every coefficient of L1 is 1/N: in each window, every
// point whose scalar has the same digit goes to one bucket, and few threads sum it. On an RTX 5090,
// 2^23 points take 0.05 s with random scalars and 26 s with equal ones. s + ρ looks random
// whatever s is, and Σ ρ_i·[τ^i]₁ depends only on n, so it is computed once for each n. The
// difference is the same point, and the commitment, in affine coordinates, is ffiasm's bit for bit.
//
// Device 0, the PLONK GPU prover's default. One device buffer holds the scalars of an MSM or the
// data of an NTT, and one host buffer the shifted scalars, each grown to the largest asked for, and
// every call holds a lock on them: a ProvingKey, and the Gpu it holds, may be shared by several
// threads, whose calls take turns.
//
// Only a library built with the GPU (provers/starks-lib-c/build.rs found nvcc: libstarksgpu.a, its
// sources compiled with __USE_CUDA__) defines it, in pilfflonk_gpu.cpp; the code that uses it is
// under __USE_CUDA__, and elsewhere gpuAvailable() is false. A CUDA failure inside the helpers above
// (out of device memory, a lost device) aborts the process, as it does in the PLONK GPU prover; the
// MSM's own failure, which it reports as the point at infinity, is thrown (msm).
class Gpu {
public:
    // Whether this library has the GPU path and sees a GPU it can use: sppark's cuda_available()
    // (external/sppark/util/all_gpus.cpp), a device of compute capability 7.0 or above with
    // cooperative launch. False, without failing, where CUDA finds no device or no driver.
    static bool available();

    // A copy on the device of the n points (n >= 1) at `points`, the powers [τ^i]₁ of an SRS, for
    // msm. Its copies to and from the device are counted in `copies`, if not null, which must outlive
    // it. Throws std::invalid_argument if !available(), points is null or n is 0.
    Gpu(const G1PointAffine *points, uint64_t n, CopyVolume *copies = nullptr);
    ~Gpu();
    Gpu(const Gpu &) = delete;
    Gpu &operator=(const Gpu &) = delete;

    uint64_t nPoints() const { return nDevicePoints; }
    // The points on the device, which a GpuKey's MSMs share.
    const void *devicePowers() const { return devicePoints; }

    // Σ_{i<n} scalars[i]·points[i], the scalars in Montgomery form: what Srs::commit computes with
    // ffiasm's MSM, through the shift above. With n = 0, the point at infinity. Throws
    // std::invalid_argument if n > nPoints() or scalars is null with n > 0, and
    // std::runtime_error if the GPU gives the point at infinity for shifted scalars that are not
    // all zero, or for the shift: msm_bn128_gpu_dev_ptr's report of a failure (or τ is a root of
    // that polynomial, which a secret τ makes as likely as guessing it).
    G1Point msm(const FrElement *scalars, uint64_t n) const;

    // The shift msm adds to its first n scalars: out[i] = ρ_i = h^(i+1), in Montgomery form. For
    // the tests. Throws std::invalid_argument if out is null with n > 0.
    static void shift(FrElement *out, uint64_t n);

    // The NTT (ffiasm's fft) or the INTT (its ifft) of the 2^bits values at `in`, in natural order,
    // written to `out`, which may be `in` (in place) or else must not overlap it. Throws
    // std::invalid_argument if bits > 28, the 2-adicity of r, or a pointer is null.
    void ntt(const FrElement *in, FrElement *out, uint64_t bits) const;
    void intt(const FrElement *in, FrElement *out, uint64_t bits) const;

private:
    void transform(const FrElement *in, FrElement *out, uint64_t bits, bool inverse) const;
    // msm_bn128_gpu_dev_ptr of the n scalars at `scalars` (host memory), as ffiasm's point; with
    // the lock held.
    G1Point deviceMsm(const FrElement *scalars, uint64_t n) const;
    // The device buffer, of n elements at least; with the lock held.
    void *scratch(uint64_t n) const;
    // The host buffer of the shifted scalars, of n elements at least; with the lock held.
    FrElement *staging(uint64_t n) const;

    void *devicePoints = nullptr;
    uint64_t nDevicePoints = 0;
    CopyVolume *volume = nullptr;
    mutable std::mutex lock;
    mutable void *deviceScratch = nullptr;
    mutable uint64_t scratchElements = 0;
    mutable std::unique_ptr<FrElement[]> hostStaging;
    mutable uint64_t stagingElements = 0;
    // Σ_{i<n} ρ_i·points[i] for each n msm has seen.
    mutable std::map<uint64_t, G1Point> shiftSums;
};

// The ratio h of the shift ρ_i = h^(i+1) of the MSMs' scalars (Gpu::msm), in Montgomery form.
const FrElement &msmShiftRatio();

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
