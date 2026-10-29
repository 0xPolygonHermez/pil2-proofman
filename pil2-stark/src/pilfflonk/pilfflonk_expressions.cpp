#include "pilfflonk_expressions.hpp"

#include <omp.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

#include "pilfflonk_error.hpp"
#include "pilfflonk_fr.hpp"
#include "pilfflonk_lde.hpp"
#include "pilfflonk_ntt.hpp"

namespace PilFflonk {

namespace {

using Engine = AltBn128::Engine;

constexpr InvalidArgument invalid("Expressions::");

FrElement fromUI(uint64_t value) {
    FrElement e;
    Engine::engine.fr.fromUI(e, value);
    return e;
}

// The values of a scalar operand of `type` in the prover mode, or null if it is no scalar the
// prover mode has.
const std::vector<FrElement> *scalarValues(const OperandTypes &types, uint32_t type, const ParserArgs &args,
                                           const ProverValues &values) {
    return type == types.numbers() ? &args.numbers : proverScalars(types, type, values);
}

std::string scalarName(const OperandTypes &types, uint32_t type) {
    if (type == types.publics()) return "public";
    if (type == types.airValues()) return "air value";
    if (type == types.proofValues()) return "proof value";
    if (type == types.airgroupValues()) return "airgroup value";
    if (type == types.challenges()) return "challenge";
    return "number";
}

// A source of an op on a block: its B values, or one for all of them.
struct Source {
    const FrElement *values;
    bool scalar;
};

// res[j] = op(a[j], b[j]) for j < n, a scalar source the same value on every row.
template <typename Op> inline void applyOp(FrElement *res, const Source &a, const Source &b, uint64_t n, Op op) {
    FrElement t;
    if (!a.scalar && !b.scalar) {
        for (uint64_t j = 0; j < n; ++j) {
            op(t, a.values[j], b.values[j]);
            res[j] = t;
        }
    } else if (a.scalar && !b.scalar) {
        const FrElement x = a.values[0];
        for (uint64_t j = 0; j < n; ++j) {
            op(t, x, b.values[j]);
            res[j] = t;
        }
    } else if (!a.scalar) {
        const FrElement y = b.values[0];
        for (uint64_t j = 0; j < n; ++j) {
            op(t, a.values[j], y);
            res[j] = t;
        }
    } else {
        op(t, a.values[0], b.values[0]);
        std::fill(res, res + n, t);
    }
}

} // namespace

const std::vector<FrElement> *proverScalars(const OperandTypes &types, uint32_t type, const ProverValues &values) {
    if (type == types.publics()) return &values.publics;
    if (type == types.airValues()) return &values.airValues;
    if (type == types.proofValues()) return &values.proofValues;
    if (type == types.airgroupValues()) return &values.airgroupValues;
    if (type == types.challenges()) return &values.challenges;
    return nullptr;
}

FrElement zhInvAt(uint64_t nBits, const FrElement &x) {
    Engine::Fr &fr = Engine::engine.fr;
    if (nBits > MAX_NBITS_EXT) {
        throw std::invalid_argument("zhInvAt: nBits = " + std::to_string(nBits) + " exceeds 28");
    }
    FrElement zh;
    fr.sub(zh, power(x, uint64_t(1) << nBits), fr.one());
    if (fr.isZero(zh)) {
        throw std::invalid_argument("zhInvAt: the point is in H, where Z_H vanishes");
    }
    FrElement zi;
    fr.inv(zi, zh);
    return zi;
}

ExpressionsDomain ExpressionsDomain::trace(uint64_t nBits) {
    if (nBits > MAX_NBITS_EXT) {
        throw std::invalid_argument("ExpressionsDomain::trace: nBits = " + std::to_string(nBits) + " exceeds 28");
    }
    return ExpressionsDomain(nBits, 0);
}

ExpressionsDomain ExpressionsDomain::coset(uint64_t nBits, uint64_t nBitsExt) {
    if (nBitsExt > MAX_NBITS_EXT || nBits > nBitsExt) {
        throw std::invalid_argument("ExpressionsDomain::coset: needs nBits <= nBitsExt <= 28, and they are " +
                                    std::to_string(nBits) + " and " + std::to_string(nBitsExt));
    }
    return cosetPart(nBits, nBitsExt, nBitsExt, 0);
}

CosetPart cosetPartPoints(uint64_t nBits, uint64_t nBitsExt, uint64_t partBits, uint64_t part) {
    if (nBitsExt > MAX_NBITS_EXT || nBits > partBits || partBits > nBitsExt) {
        throw std::invalid_argument("ExpressionsDomain::cosetPart: needs nBits <= partBits <= nBitsExt <= 28, and "
                                    "they are " +
                                    std::to_string(nBits) + ", " + std::to_string(partBits) + " and " +
                                    std::to_string(nBitsExt));
    }
    if (part >= (uint64_t(1) << (nBitsExt - partBits))) {
        throw std::invalid_argument("ExpressionsDomain::cosetPart: part " + std::to_string(part) + " of the " +
                                    std::to_string(uint64_t(1) << (nBitsExt - partBits)) + " of 2^" +
                                    std::to_string(partBits) + " points");
    }
    Engine::Fr &fr = Engine::engine.fr;
    const uint64_t n = uint64_t(1) << nBits;
    const uint64_t e = (uint64_t(1) << partBits) / n;
    // The part's shift c = g·ω_{N'}^part (g itself for part 0), and ω_m.
    CosetPart points;
    points.shift = fromUI(COSET_SHIFT);
    if (part > 0) {
        fr.mul(points.shift, points.shift, power(rootOfUnity(nBitsExt), part));
    }
    points.root = rootOfUnity(partBits);

    // Z_H(c·ω_m^i) = c^N·ω_e^i − 1, with ω_e = ω_m^N of order e: e values, repeated.
    points.zh.resize(e);
    const FrElement cN = power(points.shift, n);
    const FrElement we = power(points.root, n);
    FrElement factor = cN;
    for (uint64_t k = 0; k < e; ++k) {
        fr.sub(points.zh[k], factor, fr.one());
        fr.mul(factor, factor, we);
    }
    points.zhInv.resize(e);
    if (!batchInverse(points.zhInv.data(), points.zh.data(), e)) {
        throw std::logic_error("ExpressionsDomain::cosetPart: Z_H vanishes on the coset: g lies in no subgroup of "
                               "order 2^k");
    }
    return points;
}

ExpressionsDomain ExpressionsDomain::cosetPart(uint64_t nBits, uint64_t nBitsExt, uint64_t partBits, uint64_t part) {
    CosetPart points = cosetPartPoints(nBits, nBitsExt, partBits, part);
    ExpressionsDomain domain(nBits, partBits - nBits);
    domain.zhInv_ = std::move(points.zhInv);
    return domain;
}

Expressions::Expressions(const ExpressionsBin &_bin, const PilfflonkInfo &info)
    : bin(_bin), openingPoints(info.openingPoints) {
    if (bin.nStages != info.nStages) {
        throw FormatError("<air>.bin is of an AIR of " + std::to_string(bin.nStages) +
                          " stages, and the pilfflonkinfo of one of " + std::to_string(info.nStages));
    }
    const OperandTypes types = bin.types();
    auto check = [&](const ParserParams &p, const ParserArgs &args, const std::string &what) {
        for (uint64_t k = 0; k < uint64_t(p.nOps) * ARGS_PER_OP; k += ARGS_PER_OP) {
            const uint32_t *op = &args.args[p.argsOffset + k];
            for (int s = 0; s < 2; ++s) {
                const uint32_t type = op[2 + 3 * s], arg1 = op[3 + 3 * s], arg2 = op[4 + 3 * s];
                if (types.isColumn(type) && arg2 >= openingPoints.size()) {
                    throw FormatError("<air>.bin: " + what + " reads a column at opening point " +
                                      std::to_string(arg2) + ", and the AIR has " +
                                      std::to_string(openingPoints.size()));
                }
                if (type == types.zi() && arg1 != 1) {
                    throw FormatError("<air>.bin: " + what + " reads Zi of boundary " + std::to_string(arg1 - 1) +
                                      ", and pilfflonk has everyRow only");
                }
            }
        }
    };
    for (const auto &entry : bin.expressionsInfo) {
        check(entry.second, bin.expressionsBinArgsExpressions, "expression " + std::to_string(entry.first));
    }
    for (size_t i = 0; i < bin.constraintsInfoDebug.size(); ++i) {
        check(bin.constraintsInfoDebug[i], bin.expressionsBinArgsConstraints, "constraint " + std::to_string(i));
    }
}

void Expressions::calculateExpression(uint64_t expId, const ExpressionsDomain &domain, const ProverValues &values,
                                      FrElement *dest) const {
    calculate(checkedExpression(expId, domain.hasZi(), values, dest), bin.expressionsBinArgsExpressions, domain,
              values, dest);
}

void Expressions::calculateConstraint(uint64_t index, const ExpressionsDomain &domain, const ProverValues &values,
                                      FrElement *dest) const {
    calculate(checkedConstraint(index, domain.hasZi(), values, dest), bin.expressionsBinArgsConstraints, domain,
              values, dest);
}

const ParserParams &Expressions::checkedExpression(uint64_t expId, bool hasZi, const ProverValues &values,
                                                   const void *dest) const {
    const ParserParams &params = bin.expression(expId);
    checkOperands(params, bin.expressionsBinArgsExpressions, hasZi, values, dest, "calculateExpression");
    return params;
}

const ParserParams &Expressions::checkedConstraint(uint64_t index, bool hasZi, const ProverValues &values,
                                                   const void *dest) const {
    if (index >= bin.constraintsInfoDebug.size()) {
        throw invalid("calculateConstraint", "no constraint " + std::to_string(index) + " of " +
                                                 std::to_string(bin.constraintsInfoDebug.size()));
    }
    const ParserParams &params = bin.constraintsInfoDebug[index];
    checkOperands(params, bin.expressionsBinArgsConstraints, hasZi, values, dest, "calculateConstraint");
    return params;
}

std::vector<uint64_t> Expressions::shifts(uint64_t size, uint64_t extendBits) const {
    // (2^e·o) mod M.
    std::vector<uint64_t> out(openingPoints.size());
    for (size_t i = 0; i < openingPoints.size(); ++i) {
        const int64_t mm = static_cast<int64_t>(size);
        const uint64_t o = static_cast<uint64_t>(((openingPoints[i] % mm) + mm) % mm);
        out[i] = (o << extendBits) & (size - 1);
    }
    return out;
}

// Every operand, against what the values and the domain have: nothing is computed before.
void Expressions::checkOperands(const ParserParams &params, const ParserArgs &args, bool hasZi,
                                const ProverValues &values, const void *dest, const char *what) const {
    if (dest == nullptr) {
        throw invalid(what, "dest is null");
    }
    const OperandTypes types = bin.types();
    const uint32_t *code = &args.args[params.argsOffset];
    for (uint64_t k = 0; k < params.nOps; ++k) {
        const uint32_t *op = code + k * ARGS_PER_OP;
        for (int s = 0; s < 2; ++s) {
            const uint32_t type = op[2 + 3 * s], arg1 = op[3 + 3 * s];
            if (types.isColumn(type)) {
                if (type >= values.columns.size() || arg1 >= values.columns[type].size() ||
                    values.columns[type][arg1] == nullptr) {
                    throw invalid(what, "the code reads column " + std::to_string(arg1) +
                                            (type == 0 ? std::string(" of the fixed ones")
                                                       : " of stage " + std::to_string(type)) +
                                            ", which the values do not have");
                }
            } else if (type == types.zi()) {
                if (!hasZi) {
                    throw invalid(what, "the code reads Zi, which the domain does not have (on H, none)");
                }
            } else if (type == types.evals()) {
                throw invalid(what, "the code reads evaluations, which only the verifier mode has");
            } else if (type != types.tmp()) {
                const std::vector<FrElement> *vector = scalarValues(types, type, args, values);
                if (arg1 >= vector->size()) {
                    throw invalid(what, "the code reads " + scalarName(types, type) + " " + std::to_string(arg1) +
                                            ", of " + std::to_string(vector->size()));
                }
            }
        }
    }
}

void Expressions::calculate(const ParserParams &params, const ParserArgs &args, const ExpressionsDomain &domain,
                            const ProverValues &values, FrElement *dest) const {
    const OperandTypes types = bin.types();
    const uint64_t m = domain.size();
    const uint32_t *code = &args.args[params.argsOffset];
    const std::vector<uint64_t> openingShifts = shifts(m, domain.extendBits());

    const uint64_t blockRows = std::min(BLOCK_ROWS, m);
    const uint64_t nBlocks = m / blockRows;
    const uint64_t nTemp = params.nTemp;
    // Per thread: the temporaries of a block, and a buffer for each source that wraps around. Nothing
    // in the parallel region allocates or throws. Not cleared: a block writes each element before it
    // reads it, a temporary in an op before the ops after it (ExpressionsBin refuses code that reads
    // one first, and its destId is the last op's) and a buffer in load().
    const uint64_t perThread = (nTemp + 2) * blockRows;
    const std::unique_ptr<FrElement[]> scratch(new FrElement[perThread * omp_get_max_threads()]);
#pragma omp parallel for schedule(dynamic, 8)
    for (uint64_t block = 0; block < nBlocks; ++block) {
        FrElement *tmp = scratch.get() + perThread * omp_get_thread_num();
        FrElement *buffers[2] = {tmp + nTemp * blockRows, tmp + (nTemp + 1) * blockRows};
        const uint64_t row = block * blockRows;

        auto load = [&](const uint32_t *source, FrElement *buffer) -> Source {
            const uint32_t type = source[0], arg1 = source[1], arg2 = source[2];
            if (types.isColumn(type)) {
                const FrElement *column = values.columns[type][arg1];
                const uint64_t start = (row + openingShifts[arg2]) & (m - 1);
                if (start + blockRows <= m) {
                    return {column + start, false};
                }
                for (uint64_t j = 0; j < blockRows; ++j) {
                    buffer[j] = column[(start + j) & (m - 1)];
                }
                return {buffer, false};
            }
            if (type == types.zi()) {
                // Zi at point i is zi[i & mask]: one value for every point, or a block of its own
                // unless it repeats within the block.
                const FrElement *zi = domain.zhInv().data();
                const uint64_t mask = domain.zhInv().size() - 1;
                if (mask == 0) {
                    return {zi, true};
                }
                const uint64_t start = row & mask;
                if (start + blockRows <= mask + 1) {
                    return {zi + start, false};
                }
                for (uint64_t j = 0; j < blockRows; ++j) {
                    buffer[j] = zi[(row + j) & mask];
                }
                return {buffer, false};
            }
            if (type == types.tmp()) {
                return {tmp + arg1 * blockRows, false};
            }
            return {&(*scalarValues(types, type, args, values))[arg1], true};
        };

        for (uint64_t k = 0; k < params.nOps; ++k) {
            const uint32_t *op = code + k * ARGS_PER_OP;
            const Source a = load(op + 2, buffers[0]);
            const Source b = load(op + 5, buffers[1]);
            FrElement *res = tmp + op[1] * blockRows;
            // The op hoisted out of the rows; each row read before res is written (res may be a or b).
            switch (op[0]) {
            case OP_ADD:
                applyOp(res, a, b, blockRows, [](FrElement &r, const FrElement &x, const FrElement &y) { frAdd(r, x, y); });
                break;
            case OP_SUB:
                applyOp(res, a, b, blockRows, [](FrElement &r, const FrElement &x, const FrElement &y) { frSub(r, x, y); });
                break;
            case OP_MUL:
                applyOp(res, a, b, blockRows,
                        [](FrElement &r, const FrElement &x, const FrElement &y) { Fr_rawMMul(r.v, x.v, y.v); });
                break;
            default: // OP_SUB_SWAP: the reader refuses any other
                applyOp(res, a, b, blockRows, [](FrElement &r, const FrElement &x, const FrElement &y) { frSub(r, y, x); });
                break;
            }
        }
        std::copy(tmp + params.destId * blockRows, tmp + (params.destId + 1) * blockRows, dest + row);
    }
}

FrElement Expressions::evaluateExpressionAt(uint64_t expId, const PointValues &values) const {
    const ParserParams &params = bin.expression(expId);
    const ParserArgs &args = bin.expressionsBinArgsExpressions;
    const OperandTypes types = bin.types();
    Engine::Fr &fr = Engine::engine.fr;
    std::vector<FrElement> tmp(params.nTemp);

    auto value = [&](const uint32_t *source) -> const FrElement & {
        const uint32_t type = source[0], arg1 = source[1];
        auto at = [&](const std::vector<FrElement> &vector, const char *what) -> const FrElement & {
            if (arg1 >= vector.size()) {
                throw invalid("evaluateExpressionAt", std::string("the code reads ") + what + " " +
                                                          std::to_string(arg1) + ", of " +
                                                          std::to_string(vector.size()));
            }
            return vector[arg1];
        };
        if (types.isColumn(type)) {
            throw invalid("evaluateExpressionAt", "the code reads a column: at a point, it reads evaluations");
        }
        if (type == types.zi()) return values.zhInv;
        if (type == types.tmp()) return tmp[arg1];
        if (type == types.publics()) return at(values.publics, "public");
        if (type == types.numbers()) return at(args.numbers, "number");
        if (type == types.airValues()) return at(values.airValues, "air value");
        if (type == types.proofValues()) return at(values.proofValues, "proof value");
        if (type == types.airgroupValues()) return at(values.airgroupValues, "airgroup value");
        if (type == types.challenges()) return at(values.challenges, "challenge");
        return at(values.evals, "evaluation");
    };

    for (uint64_t k = 0; k < params.nOps; ++k) {
        const uint32_t *op = &args.args[params.argsOffset + k * ARGS_PER_OP];
        const FrElement x = value(op + 2);
        const FrElement y = value(op + 5);
        FrElement &res = tmp[op[1]];
        switch (op[0]) {
        case OP_ADD:
            fr.add(res, x, y);
            break;
        case OP_SUB:
            fr.sub(res, x, y);
            break;
        case OP_MUL:
            fr.mul(res, x, y);
            break;
        default:
            fr.sub(res, y, x);
            break;
        }
    }
    return tmp[params.destId];
}

} // namespace PilFflonk
