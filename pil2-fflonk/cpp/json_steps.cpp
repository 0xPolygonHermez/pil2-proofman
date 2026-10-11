#include "json_steps.hpp"

#include <stdexcept>

namespace PilFflonk {

namespace {

/// The extended counterpart of a committed section.
///
/// The enumerators are laid out in base/extended pairs -- cm1_n, cm1_2ns,
/// cm2_n, cm2_2ns, ... -- so the extended one is the next value. Anything
/// without a pair is returned unchanged.
FflonkInfo::eSection extended(FflonkInfo::eSection section) {
    switch (section) {
        case FflonkInfo::eSection::cm1_n: return FflonkInfo::eSection::cm1_2ns;
        case FflonkInfo::eSection::cm2_n: return FflonkInfo::eSection::cm2_2ns;
        case FflonkInfo::eSection::cm3_n: return FflonkInfo::eSection::cm3_2ns;
        default: return section;
    }
}

FrEl *bufferFor(FflonkInfo::eSection section, PilFflonkStepsParams &params) {
    switch (section) {
        case FflonkInfo::eSection::cm1_n: return params.cm1_n;
        case FflonkInfo::eSection::cm2_n: return params.cm2_n;
        case FflonkInfo::eSection::cm3_n: return params.cm3_n;
        case FflonkInfo::eSection::tmpExp_n: return params.tmpExp_n;
        case FflonkInfo::eSection::cm1_2ns: return params.cm1_2ns;
        case FflonkInfo::eSection::cm2_2ns: return params.cm2_2ns;
        case FflonkInfo::eSection::cm3_2ns: return params.cm3_2ns;
        case FflonkInfo::eSection::q_2ns: return params.q_2ns;
        default: throw std::runtime_error("json_steps: no buffer for section " + std::to_string((int)section));
    }
}

} // namespace

JsonSteps::Operand JsonSteps::resolve(FflonkInfo::FflonkInfo &info, const FflonkInfo::StepType &operand) const {
    Operand out;
    out.kind = operand.type;
    out.prime = operand.prime;

    switch (operand.type) {
        case FflonkInfo::StepType::cm: {
            // `p` indexes the polynomial map, which says which buffer the value
            // lives in and at which column. The column is that sectionPos, not
            // the operand's `id` -- `id` is the polynomial's global index, and
            // the two coincide only for cm1_n, where the orderings agree.
            if (operand.p >= info.varPolMap.size()) {
                throw std::runtime_error("json_steps: polynomial " + std::to_string(operand.p) +
                                         " is outside varPolMap");
            }
            auto section = info.varPolMap[operand.p].section;
            out.section = (domain == Domain::Extended) ? extended(section) : section;
            out.pos = info.varPolMap[operand.p].sectionPos;
            out.width = info.mapSectionsN.section[out.section];
            break;
        }

        case FflonkInfo::StepType::tmpExp:
            // Intermediate expressions have a section of their own, so they
            // carry no polynomial-map index.
            out.section = FflonkInfo::eSection::tmpExp_n;
            out.pos = operand.id;
            out.width = info.mapSectionsN.section[FflonkInfo::eSection::tmpExp_n];
            break;

        case FflonkInfo::StepType::q:
            out.section = FflonkInfo::eSection::q_2ns;
            out.pos = 0;
            out.width = info.mapSectionsN.section[FflonkInfo::eSection::q_2ns];
            break;

        case FflonkInfo::StepType::_const:
            out.pos = operand.id;
            out.width = info.nConstants;
            break;

        case FflonkInfo::StepType::tmp:
        case FflonkInfo::StepType::challenge:
        case FflonkInfo::StepType::_public:
            out.pos = operand.id;
            break;

        case FflonkInfo::StepType::number:
            AltBn128::Engine::engine.fr.fromString(out.number, operand.value);
            break;

        case FflonkInfo::StepType::x:
            break;

        default:
            throw std::runtime_error("json_steps: unsupported operand type " + std::to_string((int)operand.type));
    }

    return out;
}

JsonSteps::JsonSteps(FflonkInfo::FflonkInfo &info, const FflonkInfo::Step &code, Section section, Domain domain)
    : tmps(code.tmpUsed), domain(domain) {
    const std::vector<FflonkInfo::StepOperation> *ops = nullptr;
    switch (section) {
        case Section::First: ops = &code.first; break;
        case Section::Interior: ops = &code.i; break;
        case Section::Last: ops = &code.last; break;
    }

    for (const auto &raw : *ops) {
        Instruction ins;
        ins.op = raw.op;
        ins.dest = resolve(info, raw.dest);
        for (const auto &s : raw.src) {
            ins.src.push_back(resolve(info, s));
        }

        const size_t arity = (raw.op == FflonkInfo::StepOperation::copy) ? 1 : 2;
        if (ins.src.size() != arity) {
            throw std::runtime_error("json_steps: operation has " + std::to_string(ins.src.size()) +
                                     " operands, want " + std::to_string(arity));
        }

        program.push_back(std::move(ins));
    }
}

void JsonSteps::run(AltBn128::Engine &E, PilFflonkStepsParams &params, uint64_t i, uint64_t n,
                    uint64_t primeStride) const {
    std::vector<FrEl> tmp(tmps);
    const bool ext = domain == Domain::Extended;

    auto index = [&](const Operand &o) -> uint64_t {
        // A primed access reads the next row, wrapping so the last row reads
        // the first -- the trace is cyclic.
        uint64_t row = o.prime ? (i + primeStride) % n : i;
        return o.pos + row * o.width;
    };

    auto read = [&](const Operand &o) -> FrEl {
        switch (o.kind) {
            case FflonkInfo::StepType::cm:
            case FflonkInfo::StepType::tmpExp:
            case FflonkInfo::StepType::q: return bufferFor(o.section, params)[index(o)];
            case FflonkInfo::StepType::_const: return (ext ? params.const_2ns : params.const_n)[index(o)];
            case FflonkInfo::StepType::tmp: return tmp[o.pos];
            case FflonkInfo::StepType::challenge: return params.challenges[o.pos];
            case FflonkInfo::StepType::_public: return params.publicInputs[o.pos];
            case FflonkInfo::StepType::number: return o.number;
            case FflonkInfo::StepType::x: return (ext ? params.x_2ns : params.x_n)[i];
            default: throw std::runtime_error("json_steps: unreadable operand");
        }
    };

    auto write = [&](const Operand &o, const FrEl &value) {
        switch (o.kind) {
            case FflonkInfo::StepType::cm:
            case FflonkInfo::StepType::tmpExp:
            case FflonkInfo::StepType::q: bufferFor(o.section, params)[index(o)] = value; return;
            case FflonkInfo::StepType::tmp: tmp[o.pos] = value; return;
            default: throw std::runtime_error("json_steps: operand is not a destination");
        }
    };

    for (const auto &ins : program) {
        FrEl value;
        switch (ins.op) {
            case FflonkInfo::StepOperation::add: E.fr.add(value, read(ins.src[0]), read(ins.src[1])); break;
            case FflonkInfo::StepOperation::sub: E.fr.sub(value, read(ins.src[0]), read(ins.src[1])); break;
            case FflonkInfo::StepOperation::mul: E.fr.mul(value, read(ins.src[0]), read(ins.src[1])); break;
            case FflonkInfo::StepOperation::copy: value = read(ins.src[0]); break;
        }
        write(ins.dest, value);
    }
}

} // namespace PilFflonk
