#include "pilfflonk_prover.hpp"

#include <omp.h>

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "pilfflonk_error.hpp"
#include "pilfflonk_fr.hpp"
#include "thread_utils.hpp"

namespace PilFflonk {

namespace {

using Engine = AltBn128::Engine;

std::invalid_argument invalid(const char *function, const std::string &message) {
    return std::invalid_argument(std::string(function) + ": " + message);
}

// The values of stage 1 among all of a map's, by the stages of its entries: given[i] at the i-th entry
// of stage 1, zero at the others (of later stages, which the prover computes).
std::vector<FrElement> stageOneValues(const std::vector<uint64_t> &stages, std::vector<FrElement> given,
                                      const char *function, const char *what) {
    const uint64_t n = std::count(stages.begin(), stages.end(), uint64_t(1));
    if (given.size() != n) {
        throw invalid(function, std::to_string(given.size()) + " " + what + " for the " + std::to_string(n) +
                                    " of stage 1");
    }
    std::vector<FrElement> values(stages.size(), Engine::engine.fr.zero());
    uint64_t next = 0;
    for (uint64_t i = 0; i < stages.size(); ++i) {
        if (stages[i] == 1) {
            values[i] = given[next++];
        }
    }
    return values;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Instance
// ---------------------------------------------------------------------------------------------

Instance::Instance(const ProvingKey &_pk, uint64_t airgroupId, uint64_t airId, const uint8_t *stage1,
                   uint64_t stage1Bytes, std::vector<FrElement> airValues, std::vector<FrElement> publics,
                   std::vector<FrElement> proofValues, std::unique_ptr<BlindingSource> _blinding)
    : pk(_pk), key(_pk.air(airgroupId, airId)), blinding(std::move(_blinding)) {
    const char *function = "Instance";
    const PilfflonkInfo &info = key.info();
    const GlobalInfo &global = pk.globalInfo();
    if (!blinding) {
        throw invalid(function, "no blinding source");
    }
    if (publics.size() != global.nPublics) {
        throw invalid(function,
                      std::to_string(publics.size()) + " publics, and the proof has " + std::to_string(global.nPublics));
    }
    std::vector<uint64_t> airValueStages;
    for (const NameStageEntry &e : info.airValuesMap) {
        airValueStages.push_back(e.stage);
    }
    airValueValues = stageOneValues(airValueStages, std::move(airValues), function, "air values");
    proofValueValues = stageOneValues(global.proofValueStages, std::move(proofValues), function, "proof values");
    publicValues = std::move(publics);

    const uint64_t N = key.n();
    const uint64_t C = key.witnessColumns().size();
    if (C > std::numeric_limits<uint64_t>::max() / FR_BYTES / N || stage1Bytes != N * C * FR_BYTES) {
        throw invalid(function, "the stage-1 witness has " + std::to_string(stage1Bytes) + " bytes, and " +
                                    std::to_string(N) + " rows of " + std::to_string(C) + " columns have " +
                                    std::to_string(N * C * FR_BYTES));
    }
    if (stage1 == nullptr && stage1Bytes != 0) {
        throw invalid(function, "the stage-1 witness is null");
    }
    const uint64_t first = firstNonCanonicalFr(stage1, N * C);
    if (first < N * C) {
        throw invalid(function, "the stage-1 witness at row " + std::to_string(first / C) + ", column " +
                                    std::to_string(first % C) + " is not below r");
    }

    columns.resize(info.nStages + 1);
    for (uint64_t s = 1; s <= info.nStages; ++s) {
        columns[s].assign(key.cmIds()[s].size() * N, Engine::engine.fr.zero());
    }
    const std::vector<uint64_t> &positions = key.witnessColumns();
    FrElement *stageOne = columns[1].data();
#pragma omp parallel for
    for (uint64_t v = 0; v < N * C; ++v) {
        stageOne[positions[v % C] * N + v / C] = fromCanonicalFr(stage1 + v * FR_BYTES);
    }

    challengeValues.assign(info.challengesMap.size(), Engine::engine.fr.zero());
    coefBuffers.resize(info.cmPolsMap.size());
    polys.resize(info.cmPolsMap.size());
}

uint64_t Instance::nCommitments(uint64_t stage) const {
    const std::vector<LayoutEntry> &layout = key.info().layout;
    return std::count_if(layout.begin(), layout.end(), [&](const LayoutEntry &f) { return f.stage == stage; });
}

uint64_t Instance::nChallenges(uint64_t stage) const {
    const std::vector<ChallengeMapEntry> &map = key.info().challengesMap;
    return std::count_if(map.begin(), map.end(), [&](const ChallengeMapEntry &c) { return c.stage == stage; });
}

void Instance::placeChallenges(uint64_t stage, const std::vector<FrElement> &given,
                               std::vector<FrElement> &values) const {
    const std::vector<ChallengeMapEntry> &map = key.info().challengesMap;
    if (given.size() != nChallenges(stage)) {
        throw invalid("Instance", std::to_string(given.size()) + " challenges for the " +
                                      std::to_string(nChallenges(stage)) + " of stage " + std::to_string(stage));
    }
    for (const ChallengeMapEntry &c : map) {
        if (c.stage == stage && c.stageId >= given.size()) {
            throw FormatError(key.name() + ": challenge " + c.name + " has stageId " + std::to_string(c.stageId) +
                              ", and its stage has " + std::to_string(given.size()) + " challenges");
        }
    }
    for (uint64_t i = 0; i < map.size(); ++i) {
        if (map[i].stage == stage) {
            values[i] = given[map[i].stageId];
        }
    }
}

void Instance::setChallenges(uint64_t stage, const std::vector<FrElement> &given) {
    placeChallenges(stage, given, challengeValues);
}

ProverValues Instance::valuesOn(const std::vector<std::vector<FrElement>> &cols,
                                const std::vector<FrElement> &challenges) const {
    const PilfflonkInfo &info = key.info();
    const uint64_t N = key.n();
    ProverValues v;
    v.columns.resize(info.nStages + 1);
    for (uint64_t c = 0; c < info.nConstants; ++c) {
        v.columns[0].push_back(key.fixedEvaluations(c));
    }
    for (uint64_t s = 1; s <= info.nStages; ++s) {
        for (uint64_t p = 0; p < key.cmIds()[s].size(); ++p) {
            v.columns[s].push_back(cols[s].data() + p * N);
        }
    }
    v.publics = publicValues;
    v.challenges = challenges;
    v.airValues = airValueValues;
    v.proofValues = proofValueValues;
    v.airgroupValues.assign(info.airgroupValuesMap.size(), Engine::engine.fr.zero());
    return v;
}

void Instance::computeHintColumns(uint64_t stage, std::vector<std::vector<FrElement>> &cols,
                                  const std::vector<FrElement> &challenges) const {
    const std::vector<StdHint> &hints = key.stdHints();
    if (std::none_of(hints.begin(), hints.end(), [&](const StdHint &h) { return h.stage == stage; })) {
        return;
    }
    const PilfflonkInfo &info = key.info();
    const uint64_t N = key.n();
    Engine::Fr &fr = Engine::engine.fr;
    // The hints read the stages before this one (AirKey checked it), and the challenges of this one.
    const ProverValues values = valuesOn(cols, challenges);
    const ExpressionsDomain trace = ExpressionsDomain::trace(info.nBits);
    // An operand on every row of H, as addHintField reads it: a column at row i + offset, cyclically.
    auto evaluate = [&](const HintInput &in, FrElement *dest) {
        switch (in.kind) {
        case HintInput::Kind::Expression:
            key.expressions().calculateExpression(in.expId, trace, values, dest);
            break;
        case HintInput::Kind::Column: {
            const FrElement *source = in.column.type == 0 ? key.fixedEvaluations(in.column.index)
                                                          : cols[in.column.type].data() + in.column.index * N;
            const uint64_t shift = static_cast<uint64_t>(((in.offset % int64_t(N)) + int64_t(N)) % int64_t(N));
#pragma omp parallel for
            for (uint64_t i = 0; i < N; ++i) {
                dest[i] = source[(i + shift) % N];
            }
            break;
        }
        case HintInput::Kind::Number:
            std::fill(dest, dest + N, in.number);
            break;
        }
    };
    std::vector<FrElement> numerator(N), denominator(N), inverse(N);
    // calculateWitnessSTD's order: the products, then the sums (gen_proof.hpp).
    for (const bool prod : {true, false}) {
        for (const StdHint &hint : hints) {
            if (hint.stage != stage || hint.prod != prod) {
                continue;
            }
            evaluate(hint.numerator, numerator.data());
            evaluate(hint.denominator, denominator.data());
            if (!batchInverse(inverse.data(), denominator.data(), N)) {
                const uint64_t row = std::find_if(denominator.begin(), denominator.end(),
                                                  [&](const FrElement &d) { return fr.isZero(d); }) -
                                     denominator.begin();
                throw UnsatisfiedError(key.name() + ": the denominator of hint " + std::to_string(hint.hint) + " (" +
                                       hint.name + ", column " + info.cmPolsMap[hint.cmId].name + ") is 0 at row " +
                                       std::to_string(row) + ": the column has no value there");
            }
            // accMulHintFields: vals[i] = numerator[i]/denominator[i], then vals[i] = vals[i] ∘ vals[i − 1].
            FrElement *dest = cols[stage].data() + hint.stagePos * N;
#pragma omp parallel for
            for (uint64_t i = 0; i < N; ++i) {
                fr.mul(dest[i], numerator[i], inverse[i]);
            }
            for (uint64_t i = 1; i < N; ++i) {
                if (prod) {
                    fr.mul(dest[i], dest[i], dest[i - 1]);
                } else {
                    fr.add(dest[i], dest[i], dest[i - 1]);
                }
            }
        }
    }
}

void Instance::computeImPols(uint64_t stage) {
    if (stage <= imPolsComputed) {
        return;
    }
    computeImPols(stage, columns, challengeValues);
    imPolsComputed = stage;
}

void Instance::computeImPols(uint64_t stage, std::vector<std::vector<FrElement>> &cols,
                             const std::vector<FrElement> &challenges) const {
    const PilfflonkInfo &info = key.info();
    const uint64_t N = key.n();
    std::vector<uint64_t> pending;
    std::vector<bool> ready(key.cmIds()[stage].size(), true);
    for (uint64_t id = 0; id < info.cmPolsMap.size(); ++id) {
        const PolMapEntry &p = info.cmPolsMap[id];
        if (p.imPol && p.stage == stage) {
            pending.push_back(id);
            ready[p.stagePos] = false;
        }
    }
    if (pending.empty()) {
        return;
    }
    const ProverValues values = valuesOn(cols, challenges);
    const ExpressionsDomain trace = ExpressionsDomain::trace(info.nBits);
    // Each im pol once every column its code reads is: those of earlier stages are, and of this
    // stage the witness columns and the im pols computed so far (AirKey checked there is nothing else).
    while (!pending.empty()) {
        std::vector<uint64_t> waiting;
        for (uint64_t id : pending) {
            const PolMapEntry &p = info.cmPolsMap[id];
            const std::vector<ColumnRead> reads = columnsRead(key.bin(), p.expId);
            const bool computable = std::all_of(reads.begin(), reads.end(), [&](const ColumnRead &c) {
                return c.type != stage || ready[c.index];
            });
            if (!computable) {
                waiting.push_back(id);
                continue;
            }
            key.expressions().calculateExpression(p.expId, trace, values, cols[stage].data() + p.stagePos * N);
            ready[p.stagePos] = true;
        }
        if (waiting.size() == pending.size()) {
            throw FormatError(key.name() + ": the im pols of stage " + std::to_string(stage) +
                              " read each other in a cycle, starting with " + info.cmPolsMap[waiting[0]].name);
        }
        pending = std::move(waiting);
    }
}

std::vector<G1Point> Instance::commitF(uint64_t stage) {
    const PilfflonkInfo &info = key.info();
    const uint64_t N = key.n();
    std::vector<G1Point> commitments;
    for (uint64_t f = 0; f < info.layout.size(); ++f) {
        const LayoutEntry &entry = info.layout[f];
        if (entry.stage != stage) {
            continue;
        }
        const uint64_t k = entry.k;
        const uint64_t b = key.blindLength(f);
        std::vector<FrElement *> evals(k), coefs(k);
        for (uint64_t j = 0; j < k; ++j) {
            const uint64_t id = entry.pols[j].id;
            evals[j] = columns[stage].data() + info.cmPolsMap[id].stagePos * N;
            coefBuffers[id].reset(new FrElement[N + b]);
            coefs[j] = coefBuffers[id].get();
        }
        std::vector<std::unique_ptr<Poly>> interpolants = key.lde().intt(evals.data(), coefs.data(), k, b);
        // p'(X) = p(X) + (X^N − 1)·b(X): blindCoefficients adds b_i at N + i and subtracts it at i
        // (spec A.3). The factors are drawn f by f, and column by column within an f.
        std::vector<FrElement> factors(b);
        std::vector<Poly *> components(k);
        for (uint64_t j = 0; j < k; ++j) {
            blinding->fill(factors.data(), b);
            interpolants[j]->blindCoefficients(factors.data(), static_cast<uint32_t>(b));
            components[j] = interpolants[j].get();
            polys[entry.pols[j].id] = std::move(interpolants[j]);
        }
        commitments.push_back(commitPacked(pk.srs(), components.data(), k));
    }
    return commitments;
}

std::vector<G1Point> Instance::commitStage(uint64_t stage, const std::vector<FrElement> &challenges) {
    const char *function = "Instance::commitStage";
    const PilfflonkInfo &info = key.info();
    if (stage == 0 || stage > info.nStages) {
        throw invalid(function, "stage " + std::to_string(stage) + " is not one of the " +
                                    std::to_string(info.nStages) + " committed stages");
    }
    if (stage != next) {
        throw invalid(function, "stage " + std::to_string(stage) + " out of order: the next stage is " +
                                    std::to_string(next));
    }
    setChallenges(stage, challenges);
    computeHintColumns(stage, columns, challengeValues);
    computeImPols(stage);
    std::vector<G1Point> commitments = commitF(stage);
    ++next;
    return commitments;
}

std::vector<G1Point> Instance::commitQ(const std::vector<FrElement> &challenges) {
    const char *function = "Instance::commitQ";
    const PilfflonkInfo &info = key.info();
    if (next != info.qStage()) {
        throw invalid(function, next < info.qStage() ? "the stages are not all committed yet"
                                                     : "Q is committed already");
    }
    setChallenges(info.qStage(), challenges);

    // Every column Q's code reads, from its committed polynomial to the coset (Lde::extendCoset),
    // those with as many coefficients in one call.
    const Lde &lde = key.lde();
    const uint64_t M = lde.extendedSize();
    const std::vector<ColumnRead> &reads = key.qReads();
    std::vector<std::vector<FrElement>> coset(reads.size(), std::vector<FrElement>(M));
    ProverValues values;
    values.columns.resize(info.nStages + 1);
    values.columns[0].assign(info.nConstants, nullptr);
    for (uint64_t s = 1; s <= info.nStages; ++s) {
        values.columns[s].assign(key.cmIds()[s].size(), nullptr);
    }
    std::map<uint64_t, std::pair<std::vector<const FrElement *>, std::vector<FrElement *>>> byLength;
    for (uint64_t r = 0; r < reads.size(); ++r) {
        const ColumnRead &c = reads[r];
        const Poly *p = c.type == 0 ? key.fixedPolynomial(c.index) : polys[key.cmIds()[c.type][c.index]].get();
        auto &group = byLength[p->getLength()];
        group.first.push_back(p->coef);
        group.second.push_back(coset[r].data());
        values.columns[c.type][c.index] = coset[r].data();
    }
    for (auto &group : byLength) {
        lde.extendCoset(group.second.first.data(), group.second.second.data(), group.second.first.size(),
                        group.first);
    }
    values.publics = publicValues;
    values.challenges = challengeValues;
    values.airValues = airValueValues;
    values.proofValues = proofValueValues;
    values.airgroupValues.assign(info.airgroupValuesMap.size(), Engine::engine.fr.zero());

    const ExpressionsDomain domain = ExpressionsDomain::coset(info.nBits, key.degrees().nBitsExt, info.boundaries);
    std::vector<FrElement> qValues(M);
    key.expressions().calculateExpression(info.cExpId, domain, values, qValues.data());
    coset.clear();
    FrElement *qBuffer = qValues.data();
    lde.interpolateCoset(&qBuffer, &qBuffer, 1);

    // Q is a polynomial of its bound (spec A.1) if and only if every constraint holds on its rows.
    const uint64_t bound = key.degrees().qCoefficients;
    for (uint64_t j = M; j-- > bound;) {
        if (!Engine::engine.fr.isZero(qValues[j])) {
            throw UnsatisfiedError(
                "the witness does not satisfy the constraints of " + key.name() +
                ": Q has a coefficient of degree " + std::to_string(j) + " not zero, and a polynomial of at most " +
                std::to_string(bound) + " coefficients if it does (spec A.1)");
        }
    }

    // Its pieces (spec A.1, A.3): each its S coefficients of Q (the last one the rest up to the bound),
    // and each boundary b0·X^S + b1·X^(S+1) in the piece below it, b0 + b1·X out of the one above.
    // Unsplit, the one piece is Q, unblinded. AirDegrees gives every piece 2 coefficients at least.
    Engine::Fr &fr = Engine::engine.fr;
    const AirDegrees &d = key.degrees();
    const uint64_t m = d.qPieceCoefficients.size();
    std::vector<std::unique_ptr<Poly>> pieces(m);
    for (uint64_t i = 0; i < m; ++i) {
        const uint64_t start = i * d.qStride;
        const uint64_t length = i + 1 < m ? d.qStride : bound - start;
        pieces[i].reset(new Poly(Engine::engine, d.qPieceCoefficients[i]));
        ThreadUtils::parcpy(pieces[i]->coef, qValues.data() + start, length * sizeof(FrElement),
                            omp_get_max_threads());
    }
    FrElement factors[2];
    for (uint64_t i = 0; i + 1 < m; ++i) {
        blinding->fill(factors, 2);
        FrElement *below = pieces[i]->coef + d.qStride, *above = pieces[i + 1]->coef;
        below[0] = factors[0];
        below[1] = factors[1];
        fr.sub(above[0], above[0], factors[0]);
        fr.sub(above[1], above[1], factors[1]);
    }
    for (const std::unique_ptr<Poly> &piece : pieces) {
        piece->fixDegree();
    }

    // Each f of Q's stage, its pieces packed.
    std::vector<G1Point> commitments;
    for (const LayoutEntry &entry : info.layout) {
        if (entry.stage != info.qStage()) {
            continue;
        }
        std::vector<Poly *> components(entry.k);
        for (uint64_t j = 0; j < entry.k; ++j) {
            components[j] = pieces[info.cmPolsMap[entry.pols[j].id].stagePos].get();
        }
        commitments.push_back(commitPacked(pk.srs(), components.data(), entry.k));
    }
    qPieces = std::move(pieces);
    ++next;
    return commitments;
}

Instance::CheckTrace Instance::checkTrace(const std::vector<FrElement> &challenges) {
    const PilfflonkInfo &info = key.info();
    uint64_t expected = 0;
    for (uint64_t s = 2; s <= info.nStages; ++s) {
        expected += nChallenges(s);
    }
    if (challenges.size() != expected) {
        throw invalid("Instance::check", std::to_string(challenges.size()) +
                                             " challenges for the stages after the first, which have " +
                                             std::to_string(expected));
    }
    // Stage 1's im pols as commitStage(1) computes them, in the instance: they read no challenge.
    computeImPols(1);
    CheckTrace t;
    if (info.nStages == 1) {
        return t;
    }
    t.columns = columns;
    t.challenges = challengeValues;
    uint64_t at = 0;
    for (uint64_t s = 2; s <= info.nStages; ++s) {
        const uint64_t n = nChallenges(s);
        placeChallenges(s, std::vector<FrElement>(challenges.begin() + at, challenges.begin() + at + n), t.challenges);
        at += n;
    }
    for (uint64_t s = 2; s <= info.nStages; ++s) {
        computeHintColumns(s, t.columns, t.challenges);
        computeImPols(s, t.columns, t.challenges);
    }
    return t;
}

std::vector<std::vector<FrElement>> Instance::checkColumns(const std::vector<FrElement> &challenges) {
    CheckTrace t = checkTrace(challenges);
    return key.info().nStages == 1 ? columns : std::move(t.columns);
}

std::vector<ConstraintCheck> Instance::check(uint64_t maxRows, const std::vector<FrElement> &challenges) {
    const PilfflonkInfo &info = key.info();
    const std::vector<ParserParams> &constraints = key.bin().constraintsInfoDebug;
    for (uint64_t c = 0; c < constraints.size(); ++c) {
        if (constraints[c].stage > info.nStages) {
            throw invalid("Instance::check", "constraint " + std::to_string(c) + " (" + constraints[c].line +
                                                 ") is of stage " + std::to_string(constraints[c].stage) + ", and " +
                                                 key.name() + " has " + std::to_string(info.nStages) + " stages");
        }
    }
    const CheckTrace t = checkTrace(challenges);
    const ProverValues values =
        info.nStages == 1 ? valuesOn(columns, challengeValues) : valuesOn(t.columns, t.challenges);
    const ExpressionsDomain trace = ExpressionsDomain::trace(info.nBits);
    Engine::Fr &fr = Engine::engine.fr;
    std::vector<FrElement> numerator(key.n());
    std::vector<ConstraintCheck> checks(constraints.size());
    for (uint64_t c = 0; c < constraints.size(); ++c) {
        // AirKey checked that its rows lie in H: lastRow <= N.
        const uint64_t firstRow = constraints[c].firstRow, lastRow = constraints[c].lastRow;
        key.expressions().calculateConstraint(c, trace, values, numerator.data());
        uint64_t failed = 0;
#pragma omp parallel for reduction(+ : failed)
        for (uint64_t i = firstRow; i < lastRow; ++i) {
            if (!fr.isZero(numerator[i])) {
                ++failed;
            }
        }
        ConstraintCheck &result = checks[c];
        result.nFailed = failed;
        const uint64_t kept = std::min(failed, maxRows);
        result.rows.reserve(kept);
        for (uint64_t i = firstRow; result.rows.size() < kept; ++i) {
            if (!fr.isZero(numerator[i])) {
                result.rows.push_back(FailedRow{i, numerator[i]});
            }
        }
    }
    return checks;
}

const FrElement *Instance::column(uint64_t stage, uint64_t stagePos) const {
    if (stage == 0 || stage >= next || stage > key.info().nStages) {
        throw invalid("Instance::column", "stage " + std::to_string(stage) + " is not committed");
    }
    if (stagePos >= key.cmIds()[stage].size()) {
        throw invalid("Instance::column", "stage " + std::to_string(stage) + " has " +
                                              std::to_string(key.cmIds()[stage].size()) + " columns, and no stagePos " +
                                              std::to_string(stagePos));
    }
    return columns[stage].data() + stagePos * key.n();
}

Poly *Instance::polynomial(uint64_t f, uint64_t j) const {
    const PilfflonkInfo &info = key.info();
    const LayoutEntry &entry = info.layout.at(f);
    if (entry.stage == 0 || j >= entry.k) {
        return nullptr;
    }
    if (entry.stage == info.qStage()) {
        return qPiece(info.cmPolsMap[entry.pols[j].id].stagePos);
    }
    return polys[entry.pols[j].id].get();
}

Poly *Instance::qPiece(uint64_t i) const { return i < qPieces.size() ? qPieces[i].get() : nullptr; }

// ---------------------------------------------------------------------------------------------
// Opening
// ---------------------------------------------------------------------------------------------

Opening::Opening(const std::vector<const Instance *> &instances, const FrElement &xiSeed) {
    const char *function = "Opening";
    if (instances.empty()) {
        throw invalid(function, "no instances");
    }
    for (uint64_t i = 0; i < instances.size(); ++i) {
        if (instances[i] == nullptr) {
            throw invalid(function, "instance " + std::to_string(i) + " is null");
        }
    }
    pk = &instances[0]->provingKey();
    nBits = instances[0]->air().info().nBits;
    for (uint64_t i = 0; i < instances.size(); ++i) {
        const Instance &inst = *instances[i];
        const std::string which = "instance " + std::to_string(i);
        if (&inst.provingKey() != pk) {
            throw invalid(function, which + " is of another provingKey");
        }
        if (!inst.qCommitted()) {
            throw invalid(function, which + " has not committed Q yet");
        }
        if (inst.air().info().nBits != nBits) {
            throw invalid(function, which + " has 2^" + std::to_string(inst.air().info().nBits) +
                                        " rows, and the first one 2^" + std::to_string(nBits));
        }
        if (i > 0) {
            const Instance &prev = *instances[i - 1];
            if (std::make_pair(prev.airgroupId(), prev.airId()) > std::make_pair(inst.airgroupId(), inst.airId())) {
                throw invalid(function, "the instances are not in canonical order: " + which +
                                            " is of an AIR before the previous one's");
            }
        }
    }

    // The global order of A.5, and where each AIR's fixed f and each instance's other f start in it.
    ShplonkOpening opening;
    opening.nBits = nBits;
    opening.xiSeed = xiSeed;
    std::vector<uint64_t> fixedStart(instances.size());
    for (uint64_t i = 0; i < instances.size(); ++i) {
        const AirKey &air = instances[i]->air();
        if (i > 0 && &instances[i - 1]->air() == &air) {
            fixedStart[i] = fixedStart[i - 1];
            continue;
        }
        fixedStart[i] = opening.polynomials.size();
        for (uint64_t f = 0; f < air.nFixedF(); ++f) {
            const LayoutEntry &entry = air.info().layout[f];
            ShplonkPolynomial p;
            for (const LayoutPol &pol : entry.pols) {
                p.components.push_back(air.fixedPolynomial(pol.id));
            }
            p.offsets = entry.offsets;
            opening.polynomials.push_back(std::move(p));
        }
    }
    std::vector<uint64_t> instanceStart(instances.size());
    for (uint64_t i = 0; i < instances.size(); ++i) {
        const Instance &inst = *instances[i];
        const AirKey &air = inst.air();
        instanceStart[i] = opening.polynomials.size() - air.nFixedF();
        for (uint64_t f = air.nFixedF(); f < air.info().layout.size(); ++f) {
            const LayoutEntry &entry = air.info().layout[f];
            ShplonkPolynomial p;
            for (uint64_t j = 0; j < entry.k; ++j) {
                p.components.push_back(inst.polynomial(f, j));
            }
            p.offsets = entry.offsets;
            opening.polynomials.push_back(std::move(p));
        }
    }
    uint64_t powerW = 1;
    for (const ShplonkPolynomial &p : opening.polynomials) {
        const uint64_t k = p.components.size();
        const uint64_t gcd = std::gcd(powerW, k);
        if (k == 0 || powerW / gcd > std::numeric_limits<uint64_t>::max() / k) {
            throw FormatError("the lcm of the layouts' k does not fit in 64 bits");
        }
        powerW = powerW / gcd * k;
    }
    opening.powerW = powerW;
    shplonk = std::make_unique<ShplonkProver>(std::move(opening));

    Engine::Fr &fr = Engine::engine.fr;
    FrElement zh;
    fr.sub(zh, power(shplonk->xi(), uint64_t(1) << nBits), fr.one());
    if (fr.isZero(zh)) {
        throw std::runtime_error("Opening: ξ is in H, where Z_H vanishes");
    }

    // The evaluations, in the order of A.4 step 4: each at p_j(ξ·ω^prime) of its f.
    const ShplonkProver::Evaluations &evals = shplonk->evaluations();
    auto evaluation = [&](const AirKey &air, const EvMapEntry &e, uint64_t start) {
        const LayoutPosition &pos = e.type == PolType::Const ? air.constPosition(e.id) : air.cmPosition(e.id);
        const std::string column = std::string(e.type == PolType::Const ? "const " : "cm ") + std::to_string(e.id);
        if (pos.f == AirKey::NOT_COMMITTED) {
            throw FormatError(air.name() + ": the evMap opens " + column + ", which the layout does not commit");
        }
        const LayoutEntry &entry = air.info().layout[pos.f];
        const auto offset = std::find(entry.offsets.begin(), entry.offsets.end(), e.prime);
        if (offset == entry.offsets.end()) {
            throw FormatError(air.name() + ": the evMap opens " + column + " at " + std::to_string(e.prime) +
                              ", which its f does not open");
        }
        const uint64_t m = offset - entry.offsets.begin();
        proofEvaluations.push_back(evals[start + pos.f][m * entry.k + pos.j]);
    };
    for (uint64_t i = 0; i < instances.size(); ++i) {
        const AirKey &air = instances[i]->air();
        if (i > 0 && &instances[i - 1]->air() == &air) {
            continue;
        }
        for (const EvMapEntry &e : air.info().evMap) {
            if (e.type == PolType::Const) {
                evaluation(air, e, fixedStart[i]);
            }
        }
    }
    for (uint64_t i = 0; i < instances.size(); ++i) {
        const AirKey &air = instances[i]->air();
        for (const EvMapEntry &e : air.info().evMap) {
            if (e.type == PolType::Cm) {
                evaluation(air, e, instanceStart[i]);
            }
        }
    }

    // A.4 step 4.3: the Q_i(ξ) of each instance whose Q is split, in the order of its layout; and Q(ξ)
    // of every instance, Σ_i ξ^(i·S)·Q_i(ξ), by Horner from the last piece (Q_0(ξ) = Q(ξ) unsplit).
    for (uint64_t i = 0; i < instances.size(); ++i) {
        const AirKey &air = instances[i]->air();
        const std::vector<LayoutEntry> &layout = air.info().layout;
        const uint64_t m = air.nQPieces();
        if (m > 1) {
            for (uint64_t f = 0; f < layout.size(); ++f) {
                if (layout[f].stage == air.info().qStage()) {
                    for (uint64_t j = 0; j < layout[f].k; ++j) {
                        proofEvaluations.push_back(evals[instanceStart[i] + f][j]);
                    }
                }
            }
        }
        const FrElement shift = power(shplonk->xi(), air.degrees().qStride);
        FrElement q = fr.zero();
        for (uint64_t piece = m; piece-- > 0;) {
            const LayoutPosition &at = air.qPosition(piece);
            fr.mul(q, q, shift);
            fr.add(q, q, evals[instanceStart[i] + at.f][at.j]);
        }
        qValues.push_back(q);
    }
}

FrElement Opening::q(uint64_t instance) const {
    if (instance >= qValues.size()) {
        throw invalid("Opening::q", "there is no instance " + std::to_string(instance));
    }
    return qValues[instance];
}

Opening::Proof Opening::open(Transcript &transcript) const {
    Engine::Fr &fr = Engine::engine.fr;
    Proof proof;
    proof.shplonk = shplonk->open(pk->srs(), transcript);
    proof.inv = verifierInverse(*shplonk, proof.shplonk.y);
    FrElement zh;
    fr.sub(zh, power(shplonk->xi(), uint64_t(1) << nBits), fr.one());
    fr.inv(proof.invZh, zh);
    return proof;
}

} // namespace PilFflonk
