#include "pilfflonk_expressions.hpp"

#include <omp.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "pilfflonk_error.hpp"
#include "pilfflonk_fr.hpp"
#include "pilfflonk_lde.hpp"

namespace PilFflonk {

namespace {

using Engine = AltBn128::Engine;

std::invalid_argument invalid(const char *function, const std::string &message) {
    return std::invalid_argument(std::string("Expressions::") + function + ": " + message);
}

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

} // namespace

const std::vector<FrElement> *proverScalars(const OperandTypes &types, uint32_t type, const ProverValues &values) {
    if (type == types.publics()) return &values.publics;
    if (type == types.airValues()) return &values.airValues;
    if (type == types.proofValues()) return &values.proofValues;
    if (type == types.airgroupValues()) return &values.airgroupValues;
    if (type == types.challenges()) return &values.challenges;
    return nullptr;
}

std::vector<FrElement> excludedRoots(uint64_t nBits, const Boundary &b) {
    const uint64_t n = uint64_t(1) << nBits;
    if (b.offsetMin > n || b.offsetMax > n - b.offsetMin) {
        throw std::invalid_argument("everyFrame excludes " + std::to_string(b.offsetMin) + " + " +
                                    std::to_string(b.offsetMax) + " rows of " + std::to_string(n));
    }
    const FrElement w = rootOfUnity(nBits);
    std::vector<FrElement> roots;
    for (uint64_t j = 0; j < b.offsetMin; ++j) {
        roots.push_back(power(w, j));
    }
    for (uint64_t k = 0; k < b.offsetMax; ++k) {
        roots.push_back(power(w, n - k - 1));
    }
    return roots;
}

FrElement oneRowRoot(uint64_t nBits, BoundaryType type) {
    const uint64_t n = uint64_t(1) << nBits;
    return type == BoundaryType::FirstRow ? Engine::engine.fr.one() : power(rootOfUnity(nBits), n - 1);
}

FrElement rootOfUnity(uint64_t nBits) {
    if (nBits > MAX_NBITS_EXT) {
        throw std::invalid_argument("rootOfUnity: there is no root of unity of order 2^" + std::to_string(nBits) +
                                    ": r - 1 = 2^28 · odd");
    }
    // (r − 1) >> nBits, little-endian: r − 1 is FR_MODULUS_LE with its lowest byte 1 less (it is 1).
    uint8_t exponent[FR_BYTES];
    std::copy(FR_MODULUS_LE, FR_MODULUS_LE + FR_BYTES, exponent);
    exponent[0] -= 1;
    for (uint64_t s = 0; s < nBits; ++s) {
        for (size_t i = 0; i < FR_BYTES; ++i) {
            const uint8_t carry = i + 1 < FR_BYTES ? static_cast<uint8_t>(exponent[i + 1] << 7) : 0;
            exponent[i] = static_cast<uint8_t>(exponent[i] >> 1) | carry;
        }
    }
    FrElement result;
    Engine::engine.fr.exp(result, fromUI(COSET_SHIFT), exponent, FR_BYTES);
    return result;
}

std::vector<FrElement> zerofiersAt(uint64_t nBits, const std::vector<Boundary> &boundaries, const FrElement &x) {
    Engine::Fr &fr = Engine::engine.fr;
    if (nBits > MAX_NBITS_EXT) {
        throw std::invalid_argument("zerofiersAt: nBits = " + std::to_string(nBits) + " exceeds 28");
    }
    FrElement zh;
    fr.sub(zh, power(x, uint64_t(1) << nBits), fr.one());
    if (fr.isZero(zh)) {
        throw std::invalid_argument("zerofiersAt: the point is in H, where Z_H vanishes");
    }
    std::vector<FrElement> out;
    for (const Boundary &b : boundaries) {
        FrElement zi;
        switch (b.type) {
        case BoundaryType::EveryRow:
            fr.inv(zi, zh);
            break;
        case BoundaryType::FirstRow:
        case BoundaryType::LastRow: {
            FrElement den, denInv;
            fr.sub(den, x, oneRowRoot(nBits, b.type));
            fr.inv(denInv, den);
            fr.mul(zi, zh, denInv);
            break;
        }
        case BoundaryType::EveryFrame:
            zi = fr.one();
            for (const FrElement &root : excludedRoots(nBits, b)) {
                FrElement factor;
                fr.sub(factor, x, root);
                fr.mul(zi, zi, factor);
            }
            break;
        }
        out.push_back(zi);
    }
    return out;
}

ExpressionsDomain ExpressionsDomain::trace(uint64_t nBits) {
    if (nBits > MAX_NBITS_EXT) {
        throw std::invalid_argument("ExpressionsDomain::trace: nBits = " + std::to_string(nBits) + " exceeds 28");
    }
    return ExpressionsDomain(nBits, 0);
}

ExpressionsDomain ExpressionsDomain::coset(uint64_t nBits, uint64_t nBitsExt, const std::vector<Boundary> &boundaries) {
    if (nBitsExt > MAX_NBITS_EXT || nBits > nBitsExt) {
        throw std::invalid_argument("ExpressionsDomain::coset: needs nBits <= nBitsExt <= 28, and they are " +
                                    std::to_string(nBits) + " and " + std::to_string(nBitsExt));
    }
    return cosetPart(nBits, nBitsExt, nBitsExt, 0, boundaries);
}

CosetPart cosetPartPoints(uint64_t nBits, uint64_t nBitsExt, uint64_t partBits, uint64_t part,
                          const std::vector<Boundary> &boundaries) {
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
    for (const Boundary &b : boundaries) {
        if (b.type == BoundaryType::EveryFrame) {
            excludedRoots(nBits, b); // throws if it excludes too many rows
        }
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

ExpressionsDomain ExpressionsDomain::cosetPart(uint64_t nBits, uint64_t nBitsExt, uint64_t partBits, uint64_t part,
                                               const std::vector<Boundary> &boundaries) {
    const CosetPart points = cosetPartPoints(nBits, nBitsExt, partBits, part, boundaries);
    Engine::Fr &fr = Engine::engine.fr;
    ExpressionsDomain domain(nBits, partBits - nBits);
    const uint64_t m = uint64_t(1) << partBits;
    const uint64_t e = points.zh.size();

    // The part's points c·ω_m^i, each thread's chunk from c·ω_m^begin.
    const FrElement &c = points.shift;
    const FrElement &w = points.root;
    std::vector<FrElement> x(m);
#pragma omp parallel
    {
        const uint64_t nThreads = omp_get_num_threads();
        const uint64_t chunk = (m + nThreads - 1) / nThreads;
        const uint64_t begin = std::min(m, omp_get_thread_num() * chunk);
        const uint64_t end = std::min(m, begin + chunk);
        if (begin < end) {
            FrElement point;
            fr.mul(point, c, power(w, begin));
            for (uint64_t i = begin; i < end; ++i) {
                x[i] = point;
                fr.mul(point, point, w);
            }
        }
    }

    std::vector<FrElement> den;
    for (const Boundary &b : boundaries) {
        std::vector<FrElement> zi(m);
        switch (b.type) {
        case BoundaryType::EveryRow:
#pragma omp parallel for schedule(static)
            for (uint64_t i = 0; i < m; ++i) {
                zi[i] = points.zhInv[i & (e - 1)];
            }
            break;
        case BoundaryType::FirstRow:
        case BoundaryType::LastRow: {
            const FrElement root = oneRowRoot(nBits, b.type);
            den.resize(m);
#pragma omp parallel for schedule(static)
            for (uint64_t i = 0; i < m; ++i) {
                fr.sub(den[i], x[i], root);
            }
            if (!batchInverse(zi.data(), den.data(), m)) {
                throw std::logic_error("ExpressionsDomain::cosetPart: a point of the coset is a row of H");
            }
#pragma omp parallel for schedule(static)
            for (uint64_t i = 0; i < m; ++i) {
                fr.mul(zi[i], points.zh[i & (e - 1)], zi[i]);
            }
            break;
        }
        case BoundaryType::EveryFrame: {
            const std::vector<FrElement> roots = excludedRoots(nBits, b);
#pragma omp parallel for schedule(static)
            for (uint64_t i = 0; i < m; ++i) {
                FrElement acc = fr.one();
                for (const FrElement &root : roots) {
                    FrElement f;
                    fr.sub(f, x[i], root);
                    fr.mul(acc, acc, f);
                }
                zi[i] = acc;
            }
            break;
        }
        }
        domain.zerofiers_.push_back(std::move(zi));
    }
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
                if (type == types.zi() && arg1 - 1 >= info.boundaries.size()) {
                    throw FormatError("<air>.bin: " + what + " reads Zi of boundary " + std::to_string(arg1 - 1) +
                                      ", and the AIR has " + std::to_string(info.boundaries.size()));
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
    calculate(checkedExpression(expId, domain.nZerofiers(), values, dest), bin.expressionsBinArgsExpressions, domain,
              values, dest);
}

void Expressions::calculateConstraint(uint64_t index, const ExpressionsDomain &domain, const ProverValues &values,
                                      FrElement *dest) const {
    calculate(checkedConstraint(index, domain.nZerofiers(), values, dest), bin.expressionsBinArgsConstraints, domain,
              values, dest);
}

const ParserParams &Expressions::checkedExpression(uint64_t expId, uint64_t nZerofiers, const ProverValues &values,
                                                   const void *dest) const {
    const ParserParams &params = bin.expression(expId);
    checkOperands(params, bin.expressionsBinArgsExpressions, nZerofiers, values, dest, "calculateExpression");
    return params;
}

const ParserParams &Expressions::checkedConstraint(uint64_t index, uint64_t nZerofiers, const ProverValues &values,
                                                   const void *dest) const {
    if (index >= bin.constraintsInfoDebug.size()) {
        throw invalid("calculateConstraint", "no constraint " + std::to_string(index) + " of " +
                                                 std::to_string(bin.constraintsInfoDebug.size()));
    }
    const ParserParams &params = bin.constraintsInfoDebug[index];
    checkOperands(params, bin.expressionsBinArgsConstraints, nZerofiers, values, dest, "calculateConstraint");
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
void Expressions::checkOperands(const ParserParams &params, const ParserArgs &args, uint64_t nZerofiers,
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
                if (arg1 - 1 >= nZerofiers) {
                    throw invalid(what, "the code reads Zi of boundary " + std::to_string(arg1 - 1) +
                                            ", which the domain does not have (on H, none)");
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

    std::vector<const FrElement *> zerofiers(domain.nZerofiers());
    for (uint64_t b = 0; b < zerofiers.size(); ++b) {
        zerofiers[b] = domain.zerofier(b).data();
    }

    const uint64_t blockRows = std::min(BLOCK_ROWS, m);
    const uint64_t nBlocks = m / blockRows;
    const uint64_t nTemp = params.nTemp;
    // Per thread: the temporaries of a block, and a buffer for each source that wraps around. Nothing
    // in the parallel region allocates or throws.
    const uint64_t perThread = (nTemp + 2) * blockRows;
    std::vector<FrElement> scratch(perThread * omp_get_max_threads());
    Engine::Fr &fr = Engine::engine.fr;

#pragma omp parallel for schedule(static)
    for (uint64_t block = 0; block < nBlocks; ++block) {
        FrElement *tmp = scratch.data() + perThread * omp_get_thread_num();
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
                return {zerofiers[arg1 - 1] + row, false};
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
            for (uint64_t j = 0; j < blockRows; ++j) {
                const FrElement &x = a.values[a.scalar ? 0 : j];
                const FrElement &y = b.values[b.scalar ? 0 : j];
                // Into t first: res may be the temporary of a or b.
                FrElement t;
                switch (op[0]) {
                case OP_ADD:
                    fr.add(t, x, y);
                    break;
                case OP_SUB:
                    fr.sub(t, x, y);
                    break;
                case OP_MUL:
                    fr.mul(t, x, y);
                    break;
                default: // OP_SUB_SWAP: the reader refuses any other
                    fr.sub(t, y, x);
                    break;
                }
                res[j] = t;
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
        if (type == types.zi()) return at(values.zerofiers, "Zi of boundary");
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
        // Zi's arg1 is 1 + its boundary.
        uint32_t sources[2][3] = {{op[2], op[3], op[4]}, {op[5], op[6], op[7]}};
        for (auto &source : sources) {
            if (source[0] == types.zi()) {
                source[1] -= 1;
            }
        }
        const FrElement x = value(sources[0]);
        const FrElement y = value(sources[1]);
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
