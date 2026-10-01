#ifndef PILFFLONK_OPENING_GPU_HPP
#define PILFFLONK_OPENING_GPU_HPP

// The device side of an Opening on a key on the GPU (pilfflonk_prover.hpp, pilfflonk_key_gpu.hpp):
// SHPLONK's evaluations, W, W' and their commitments, computed on the device from the committed
// polynomials there. Only a library built with the GPU (__USE_CUDA__) has it, in
// pilfflonk_opening_gpu.cpp.

#include <cstdint>
#include <memory>
#include <vector>

#include "pilfflonk_key_gpu.hpp"
#include "pilfflonk_shplonk_prover.hpp"

namespace PilFflonk {

class AirKey;   // pilfflonk_proving_key.hpp
class Instance; // pilfflonk_prover.hpp

// The most an OpeningGpu holds of an opening, in elements, from the shapes of the f_i it opens.
struct ShplonkBounds {
    uint64_t nEvaluations = 0; // Σ_i k_i·|O_i| = Σ_i |T_i|: the evaluations, and r_i's coefficients
    uint64_t nPoints = 0;      // Σ_i |O_i|
    uint64_t component = 0;    // the coefficients of a p_j, and |O_i|
    uint64_t length = 0;       // W's, L's and W''s: ShplonkProver::workLength()
    // The MSMs of W and W', lengths with a shift sum (GpuKey::addShiftSum): at least the bounds of
    // ShplonkProver::checkW and checkWp.
    uint64_t wMsm = 0;
    uint64_t wpMsm = 0;
};

// The bounds of an opening of an instance of `air`, the f of its layout, from the layout: f's degree
// bound and |T_f| = k·|O_f|, and its p_j's coefficients (N of a fixed column, N + blindLength(f) of a
// committed one, a piece's bound of Q's). The MSMs are of max(1, max_f (degree_f − |T_f|)) and
// max(1, max_f degree_f − 1) scalars: GpuAirKey computes their shift sums when it loads.
ShplonkBounds shplonkBounds(const AirKey &air);

// The device memory, in bytes, of an OpeningGpu's workspace for `bounds`.
uint64_t shplonkWorkspaceBytes(const ShplonkBounds &bounds);

// SHPLONK on the device (pilfflonk/docs/protocol.md#shplonk-opening), for an opening whose p_j are on
// it: ShplonkProver's evaluations (its Evaluator) and its quotients (ShplonkQuotients), the same bit
// for bit, while the host keeps the transcript, the interpolants r_i, the roots and points, and L's
// scalars (ShplonkProver::lScalars).
//
// The packing f(X) = Σ_{j<k} p_j(X^k)·X^j makes the division of f − r by Z_{T_i}(X) =
// Π_{s in O_i} (X^k − ξ·ω_N^s) one of each component by Π_s (Y − ξ·ω_N^s), Y = X^k:
// (f − r)/Z_{T_i} = Σ_j q_j(X^k)·X^j, q_j = (p_j − r^(j))/Π_s (Y − ξ·ω_N^s), with r^(j)[c] = r[c·k + j];
// it is exact if and only if each one is. So W is built a component at a time: p_j − r^(j) into a
// buffer, divided by one Y − β after another (gpu_plonk_compute_div_zerofier, with q_0 = −a_0/β from
// a_0 on the host), and α^i·q_j added to W at c·k + j. L = w·W + Σ_i f[i]·(f_i − r_i(y)) is built in
// W's buffer and divided by X − y. A division by X − β of n coefficients writes the series of
// a/(X − β) up to X^(n−1): its coefficient n − 1 is zero if and only if it is exact, and below it is
// the quotient. A buffer of one coefficient divides only if it is zero, as divideExactly.
//
// The evaluations: each p_j at each point, Σ_i coefs[i]·x^i with x^i from tables of powers of each
// distinct point (gpu_plonk_precompute_omega_tables_async), all in one launch.
//
// It reads the degree of each p_j from the prover's components (ShplonkComponent: a copy on the host,
// or elsewhere, as Instance::component gives Q's pieces), and copies to the device the r_i and to the
// host a few elements per division (counted in the key's CopyVolume).
class OpeningGpu final : public ShplonkQuotients {
public:
    // By f_i, then j: the p_j of f_i on the device, f_i in the order of the ShplonkProver the opening
    // is of, which must have as many and as long (ShplonkProver::components).
    using Components = std::vector<std::vector<const FrElement *>>;

    // Over `components` and `workspace`, shplonkWorkspaceBytes(bounds) bytes of device memory it uses
    // while it lives; it commits with key's MSMs of bounds.wMsm and bounds.wpMsm scalars, which must
    // have their shift sums.
    OpeningGpu(const GpuKey &key, Components components, const ShplonkBounds &bounds, uint8_t *workspace);
    OpeningGpu(const OpeningGpu &) = delete;
    OpeningGpu &operator=(const OpeningGpu &) = delete;

    // The device side of an Opening of `instance` alone, which holds its key's arena (GpuKey::Lease):
    // its f in the order of its AIR's layout, the global order for one instance, with its committed
    // polynomials and Q's pieces where InstanceGpu left them, and its fixed ones in the GpuAirKey; its
    // workspace at ArenaLayout::shplonk. Nothing is copied.
    static std::unique_ptr<OpeningGpu> ofInstance(const Instance &instance);

    // ShplonkProver's evaluations for `prover` (an Evaluator). Throws std::logic_error if the prover
    // is not of these components and bounds.
    ShplonkProver::Evaluations evaluate(const ShplonkProver &prover) const;

    // ShplonkQuotients, the same as ShplonkProver::quotientW and quotientWp, and their commitments,
    // throwing what they throw (but std::invalid_argument for arguments open() never gives:
    // std::logic_error here).
    void computeW(const ShplonkProver &prover, const ShplonkProver::Interpolants &r, const FrElement &alpha) override;
    G1Point commitW() override;
    void computeWp(const ShplonkProver &prover, const ShplonkProver::Interpolants &r, const FrElement &alpha,
                   const FrElement &y) override;
    G1Point commitWp() override;

    // W after computeW, and W' after computeWp, on the device: the ShplonkProver::workLength()
    // coefficients of quotientW's and quotientWp's Poly. For tests.
    const FrElement *quotient() const;

private:
    // Throws std::logic_error unless `prover` is of these components and fits the bounds.
    void requireShape(const ShplonkProver &prover) const;
    // The element at src on the device.
    FrElement read(const FrElement *src) const;
    // 1 + the index of the highest coefficient not zero of the n at data (0 if all are).
    uint64_t count(const FrElement *data, uint64_t n) const;
    // The n coefficients at data divided by X − β in place, the quotient in the first max(n − 1, 1),
    // with `pairs` for the scan's (gpu_plonk_compute_div_zerofier's work); whether it was exact.
    bool divide(FrElement *data, uint64_t n, const FrElement &beta, void *pairs) const;
    // r's coefficients to the device, r_i at the first of its |T_i|, zero above its own.
    void uploadInterpolants(const ShplonkProver &prover, const ShplonkProver::Interpolants &r) const;
    // The quotient of `count` coefficients committed with an MSM of n scalars.
    G1Point commit(uint64_t count, uint64_t n) const;

    template <typename T> T *at(uint64_t offset) const { return reinterpret_cast<T *>(workspace + offset); }

    const GpuKey &key;
    Components components;
    ShplonkBounds bounds;
    uint8_t *workspace;
    uint64_t wCount = 0;  // W's coefficients, as count() finds them
    uint64_t wpCount = 0; // W''s
};

} // namespace PilFflonk

#endif
