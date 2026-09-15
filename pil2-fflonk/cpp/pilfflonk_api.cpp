#include "pilfflonk_api.hpp"

#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include <alt_bn128.hpp>

namespace {

using Engine = AltBn128::Engine;
using FrEl = Engine::FrElement;
using G1Point = Engine::G1Point;
using G1PointAffine = Engine::G1PointAffine;

// Per-thread so concurrent callers do not overwrite each other's message.
thread_local std::string lastError;

// Every entry point funnels through this: C++ exceptions must not unwind into
// Rust, so they are caught here and turned into a return code.
template <typename Body>
int guard(Body body) {
    try {
        lastError.clear();
        body();
        return 0;
    } catch (const std::exception &e) {
        lastError = e.what();
        return 1;
    } catch (...) {
        lastError = "unknown error";
        return 2;
    }
}

} // namespace

extern "C" const char *pilfflonk_last_error(void) {
    return lastError.empty() ? nullptr : lastError.c_str();
}

extern "C" int pilfflonk_msm(const uint8_t *ptau, const uint8_t *coeffs, uint64_t n, uint8_t *out) {
    return guard([&] {
        if (out == nullptr) throw std::runtime_error("pilfflonk_msm: out is null");
        if (n == 0) {
            // The empty sum is the point at infinity, which this
            // representation writes as all zeroes.
            std::memset(out, 0, PILFFLONK_G1_AFFINE_BYTES);
            return;
        }
        if (ptau == nullptr || coeffs == nullptr) throw std::runtime_error("pilfflonk_msm: null input buffer");

        auto &E = Engine::engine;

        // multiMulByScalar consumes scalars in canonical form, while the
        // proving key stores them in Montgomery form -- committing the stored
        // values directly would silently compute the wrong point. The copy is
        // also what keeps the caller's buffer const.
        std::vector<FrEl> scalars(n);
        std::memcpy(scalars.data(), coeffs, n * sizeof(FrEl));
        for (uint64_t i = 0; i < n; i++) {
            E.fr.fromMontgomery(scalars[i], scalars[i]);
        }

        // nx/x describe how ffiasm splits the work across threads; one span of
        // the whole range asks it to choose for itself.
        uint64_t lengths[1] = {n};
        G1Point result;
        E.g1.multiMulByScalar(
            result, (G1PointAffine *)ptau, (uint8_t *)scalars.data(), sizeof(FrEl), n, 1, lengths);

        G1PointAffine affine;
        E.g1.copy(affine, result);
        std::memcpy(out, &affine, PILFFLONK_G1_AFFINE_BYTES);
    });
}
