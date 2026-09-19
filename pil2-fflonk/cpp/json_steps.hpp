#ifndef PIL2_FFLONK_JSON_STEPS_HPP
#define PIL2_FFLONK_JSON_STEPS_HPP

#include <cstdint>
#include <vector>

#include <alt_bn128.hpp>

#include "chelpers/pilfflonk_steps.hpp"
#include "fflonk_info.hpp"

// Evaluating a stage's constraint code from the AIR description, instead of
// from generated C++.
//
// pil-fflonk compiles each stage's expressions into `chelpers/` -- one C++
// function per stage, generated for one specific AIR. The same expressions are
// also in the `.fflonkinfo`, which `FflonkInfo` already parses into `Step`.
// This runs those.
//
// Two reasons it matters. The generated code only exists for an AIR someone has
// run the pil1 toolchain over, so a prover that depends on it cannot prove
// anything else. And pil2 does not emit C++ at all -- it emits the same
// expression model as uint16 bytecode, evaluated by `fr_expressions.hpp` and
// `fr_evaluator.hpp`. Driving the prover from data is what lets the second be
// substituted for the first.
//
// Reading the pil1 encoding makes this checkable against the generated code on
// the reference AIR, which is the point: it establishes that a data-driven
// evaluator reproduces the compiled one exactly before anything depends on it.

namespace PilFflonk {

using FrEl = typename AltBn128::Engine::FrElement;

/// Which domain a program runs over.
///
/// The trace steps evaluate over the base domain; the quotient step evaluates
/// over the extended one, reading the `_2ns` buffers instead. `varPolMap`
/// records only the base section, so the domain decides which of the pair a
/// polynomial resolves to.
enum class Domain { Base, Extended };

/// One stage's program, resolved against the AIR.
class JsonSteps {
public:
    /// `section` picks `first`, `i` or `last` from the step.
    enum class Section { First, Interior, Last };

    JsonSteps(FflonkInfo::FflonkInfo &info, const FflonkInfo::Step &code, Section section, Domain domain);

    /// Evaluate one row.
    ///
    /// `n` is the domain being walked. `primeStride` is how far a primed access
    /// reaches: one row on the base domain, but the extension factor on the
    /// extended one, where the next *trace* row is that many rows away. Using
    /// one there reads a neighbouring evaluation of the same row instead, which
    /// is wrong without being obviously so.
    void run(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i, uint64_t n,
             uint64_t primeStride = 1) const;

    size_t size() const { return program.size(); }

private:
    /// Where one operand reads from, or one destination writes to.
    struct Operand {
        FflonkInfo::StepType::eType kind;
        bool prime = false;
        /// Column within its section, and the section's width.
        uint64_t pos = 0;
        uint64_t width = 0;
        FflonkInfo::eSection section = FflonkInfo::eSection::cm1_n;
        FrEl number{};
    };

    struct Instruction {
        FflonkInfo::StepOperation::eOperation op;
        Operand dest;
        std::vector<Operand> src;
    };

    std::vector<Instruction> program;
    uint64_t tmps = 0;
    Domain domain;

    Operand resolve(FflonkInfo::FflonkInfo &info, const FflonkInfo::StepType &operand) const;
};

} // namespace PilFflonk

#endif // PIL2_FFLONK_JSON_STEPS_HPP
