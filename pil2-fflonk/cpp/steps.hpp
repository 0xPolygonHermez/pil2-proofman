#ifndef PIL2_FFLONK_STEPS_HPP
#define PIL2_FFLONK_STEPS_HPP

#include <cstdint>
#include <memory>
#include <string>

#include <alt_bn128.hpp>
#include <nlohmann/json.hpp>

#include "chelpers/pilfflonk_steps.hpp"
#include "fflonk_info.hpp"
#include "json_steps.hpp"

// The seam between the prover and the AIR's constraint code.
//
// pil-fflonk's prover calls a concrete `PilFflonkSteps`, whose bodies are
// generated C++ for one specific AIR. That makes the prover unable to prove
// anything the pil1 toolchain has not been run over -- and pil2 does not
// generate C++ at all.
//
// `Steps` is that call surface as an interface, with two implementations:
//
//   * `GeneratedSteps` forwards to the generated functions, so the reference
//     AIR keeps working exactly as before. It holds a `PilFflonkSteps` by value
//     rather than deriving from it, which is what lets `chelpers/` stay
//     byte-identical to what pil1 emitted.
//   * `JsonSteps`-backed `DataSteps` runs the same expressions from the AIR
//     description instead. A pil2 implementation goes here too, over the
//     bytecode in `fr_expressions.hpp`.
//
// The two are checked against each other in `tests/json_steps.cpp` and
// `tests/steps_agree.cpp`.

namespace PilFflonk {

class Steps {
public:
    virtual ~Steps() = default;

    /// Constant values the expressions hoist out. Generated code precomputes
    /// them into a buffer; an evaluator reading the AIR directly has its
    /// literals inline and needs none.
    virtual uint64_t getNumConstValues() = 0;
    virtual void setConstValues(AltBn128::Engine &E, PilFflonkStepsParams &params) = 0;

    virtual void publics(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i, uint64_t pub) = 0;

    virtual void step2prev(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i) = 0;
    virtual void step3prev(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i) = 0;
    virtual void step3(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i) = 0;
    virtual void step42ns(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i) = 0;
};

/// The generated C++, unchanged.
///
/// Only the `_first` variants are forwarded. pil1 emits `_first`, `_i` and
/// `_last` so a caller can specialise the domain's ends, but the prover calls
/// `_first` for every row -- the three differ only in how they were generated,
/// not in what they compute for an interior row, and `_first` handles the
/// boundary by wrapping like the others.
class GeneratedSteps : public Steps {
    PilFflonkSteps generated;

public:
    uint64_t getNumConstValues() override { return generated.getNumConstValues(); }

    void setConstValues(AltBn128::Engine &E, PilFflonkStepsParams &params) override {
        generated.setConstValues(E, params);
    }

    void publics(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i, uint64_t pub) override {
        generated.publics_first(E, params, i, pub);
    }

    void step2prev(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i) override {
        generated.step2prev_first(E, params, i);
    }

    void step3prev(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i) override {
        generated.step3prev_first(E, params, i);
    }

    void step3(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i) override {
        generated.step3_first(E, params, i);
    }

    void step42ns(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i) override {
        generated.step42ns_first(E, params, i);
    }
};

/// The same expressions, run from the AIR description.
///
/// Needs the domain size, since a primed access on the last row wraps to the
/// first and nothing in the code itself says how long the trace is.
class DataSteps : public Steps {
    JsonSteps publicsProgram;
    JsonSteps step2prevProgram;
    JsonSteps step3prevProgram;
    JsonSteps step3Program;
    JsonSteps step42nsProgram;
    uint64_t n;
    uint64_t nExtended;

public:
    DataSteps(FflonkInfo::FflonkInfo &info, const nlohmann::json &document, uint64_t n, uint64_t nExtended);

    // Literals are read from the description as they are met, so there is
    // nothing to precompute.
    uint64_t getNumConstValues() override { return 0; }
    void setConstValues(AltBn128::Engine &, PilFflonkStepsParams &) override {}

    void publics(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i, uint64_t pub) override;
    void step2prev(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i) override;
    void step3prev(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i) override;
    void step3(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i) override;

    /// Evaluated over the extended domain, so it wraps at `nExtended`.
    void step42ns(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i) override;
};

} // namespace PilFflonk

#endif // PIL2_FFLONK_STEPS_HPP
