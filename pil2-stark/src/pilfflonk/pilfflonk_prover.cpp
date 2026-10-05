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
#include "timer.hpp"
#ifdef __USE_CUDA__
#include "pilfflonk_hints_gpu.hpp"
#include "pilfflonk_instance_gpu.hpp"
#include "pilfflonk_opening_gpu.hpp"
#endif

namespace PilFflonk {

namespace {

using Engine = AltBn128::Engine;

constexpr InvalidArgument invalid("");

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

// The rows of a chunk of accumulate() at least, below which it scans on one thread.
constexpr uint64_t MIN_SCAN_CHUNK = uint64_t(1) << 12;

// values[i] = values[0] ∘ … ∘ values[i] for i < n, ∘ the product (or the sum), as accMulHintFields
// accumulates: a blocked prefix, each chunk scanned on a thread of its own, then each chunk's carry
// (∘ of the chunks before it) applied to it. The field's arithmetic is exact, so it is the serial
// scan's, bit for bit.
void accumulate(FrElement *values, uint64_t n, bool product) {
    Engine::Fr &fr = Engine::engine.fr;
    auto combine = [&fr, product](FrElement &out, const FrElement &a, const FrElement &b) {
        if (product) {
            fr.mul(out, a, b);
        } else {
            fr.add(out, a, b);
        }
    };
    const uint64_t threads = std::max<uint64_t>(1, std::min<uint64_t>(omp_get_max_threads(), n / MIN_SCAN_CHUNK));
    const uint64_t chunk = (n + threads - 1) / threads;
    const uint64_t nChunks = chunk == 0 ? 0 : (n + chunk - 1) / chunk;
    // totals[c]: ∘ of chunk c's values, then of every chunk up to c.
    std::vector<FrElement> totals(nChunks);
#pragma omp parallel for schedule(static) num_threads(threads)
    for (uint64_t c = 0; c < nChunks; ++c) {
        const uint64_t begin = c * chunk, end = std::min(n, begin + chunk);
        for (uint64_t i = begin + 1; i < end; ++i) {
            combine(values[i], values[i], values[i - 1]);
        }
        totals[c] = values[end - 1];
    }
    for (uint64_t c = 1; c < nChunks; ++c) {
        combine(totals[c], totals[c], totals[c - 1]);
    }
#pragma omp parallel for schedule(static) num_threads(threads)
    for (uint64_t c = 1; c < nChunks; ++c) {
        const uint64_t begin = c * chunk, end = std::min(n, begin + chunk);
        for (uint64_t i = begin; i < end; ++i) {
            combine(values[i], values[i], totals[c - 1]);
        }
    }
}

// Every column Q's code reads (AirKey::qReads), column r of them from its polynomial (polys, by
// cmPolsMap index, or the key's fixed one) extended to part `part` of 2^partBits points
// (Lde::extendCosetPart) into partColumns + r·2^partBits, those of as many coefficients in one call.
void extendQPart(const AirKey &key, const std::vector<std::unique_ptr<Poly>> &polys, uint64_t partBits,
                 uint64_t part, FrElement *partColumns) {
    const std::vector<ColumnRead> &reads = key.qReads();
    const uint64_t S = uint64_t(1) << partBits;
    std::map<uint64_t, std::pair<std::vector<const FrElement *>, std::vector<FrElement *>>> byLength;
    for (uint64_t r = 0; r < reads.size(); ++r) {
        const ColumnRead &c = reads[r];
        const Poly *p = c.type == 0 ? key.fixedPolynomial(c.index) : polys[key.cmIds()[c.type][c.index]].get();
        auto &group = byLength[p->getLength()];
        group.first.push_back(p->coef);
        group.second.push_back(partColumns + r * S);
    }
    for (auto &group : byLength) {
        key.lde().extendCosetPart(group.second.first.data(), group.second.second.data(), group.second.first.size(),
                                  group.first, partBits, part);
    }
}

} // namespace

uint64_t hintRowShift(const HintInput &in, uint64_t N) {
    return static_cast<uint64_t>(((in.offset % int64_t(N)) + int64_t(N)) % int64_t(N));
}

std::vector<uint64_t> imPolOrder(const AirKey &air, uint64_t stage) {
    const PilfflonkInfo &info = air.info();
    std::vector<uint64_t> pending, order;
    std::vector<bool> ready(air.cmIds()[stage].size(), true);
    for (uint64_t id = 0; id < info.cmPolsMap.size(); ++id) {
        const PolMapEntry &p = info.cmPolsMap[id];
        if (p.imPol && p.stage == stage) {
            pending.push_back(id);
            ready[p.stagePos] = false;
        }
    }
    // Each im pol once every column its code reads is: those of earlier stages are, and of this
    // stage the witness columns and the im pols computed so far (AirKey checked there is nothing else).
    while (!pending.empty()) {
        std::vector<uint64_t> waiting;
        for (uint64_t id : pending) {
            const PolMapEntry &p = info.cmPolsMap[id];
            const std::vector<ColumnRead> reads = columnsRead(air.bin(), p.expId);
            const bool computable = std::all_of(reads.begin(), reads.end(), [&](const ColumnRead &c) {
                return c.type != stage || ready[c.index];
            });
            if (!computable) {
                waiting.push_back(id);
                continue;
            }
            order.push_back(id);
            ready[p.stagePos] = true;
        }
        if (waiting.size() == pending.size()) {
            throw FormatError(air.name() + ": the im pols of stage " + std::to_string(stage) +
                              " read each other in a cycle, starting with " + info.cmPolsMap[waiting[0]].name);
        }
        pending = std::move(waiting);
    }
    return order;
}

UnsatisfiedError zeroDenominatorError(const AirKey &air, const StdHint &hint, uint64_t row) {
    return UnsatisfiedError(air.name() + ": the denominator of hint " + std::to_string(hint.hint) + " (" + hint.name +
                            ", column " + air.info().cmPolsMap[hint.cmId].name + ") is 0 at row " +
                            std::to_string(row) + ": the column has no value there");
}

// ---------------------------------------------------------------------------------------------
// Instance
// ---------------------------------------------------------------------------------------------

Instance::Instance(const ProvingKey &_pk, uint64_t airgroupId, uint64_t airId, const uint8_t *stage1,
                   uint64_t stage1Bytes, std::vector<FrElement> airValues, std::vector<FrElement> publics,
                   std::vector<FrElement> proofValues, std::unique_ptr<BlindingSource> _blinding)
    : pk(_pk), key(_pk.air(airgroupId, airId)), blinding(std::move(_blinding)) {
    const char *function = "Instance";
    TimerStart(PILFFLONK_INSTANCE);
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
#ifdef __USE_CUDA__
    // On a key on the GPU, the columns are on the device only.
    if (key.device() != nullptr) {
        device = std::make_unique<InstanceGpu>(*key.device(), stage1);
        deviceCopies.resize(info.nStages + 1);
        copied.resize(info.nStages + 1);
    }
    if (device == nullptr)
#endif
    {
        for (uint64_t s = 1; s <= info.nStages; ++s) {
            columns[s].assign(key.cmIds()[s].size() * N, Engine::engine.fr.zero());
        }
        const std::vector<uint64_t> &positions = key.witnessColumns();
        FrElement *stageOne = columns[1].data();
#pragma omp parallel for
        for (uint64_t v = 0; v < N * C; ++v) {
            stageOne[positions[v % C] * N + v / C] = fromCanonicalFr(stage1 + v * FR_BYTES);
        }
    }

    challengeValues.assign(info.challengesMap.size(), Engine::engine.fr.zero());
    coefBuffers.resize(info.cmPolsMap.size());
    polys.resize(info.cmPolsMap.size());
    TimerStopAndLog(PILFFLONK_INSTANCE);
}

Instance::~Instance() = default;

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
    ProverValues v = scalarValues(challenges);
    v.columns.resize(info.nStages + 1);
    for (uint64_t c = 0; c < info.nConstants; ++c) {
        v.columns[0].push_back(key.fixedEvaluations(c));
    }
    for (uint64_t s = 1; s <= info.nStages; ++s) {
        for (uint64_t p = 0; p < key.cmIds()[s].size(); ++p) {
            v.columns[s].push_back(cols[s].data() + p * N);
        }
    }
    return v;
}

ProverValues Instance::scalarValues(const std::vector<FrElement> &challenges) const {
    ProverValues v;
    v.publics = publicValues;
    v.challenges = challenges;
    v.airValues = airValueValues;
    v.proofValues = proofValueValues;
    v.airgroupValues.assign(key.info().airgroupValuesMap.size(), Engine::engine.fr.zero());
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
    // The hints read the stages before this one and, of this one, the columns of the hints before
    // them (AirKey checked it), which are in cols as they are computed; and the challenges of this one.
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
            const uint64_t shift = hintRowShift(in, N);
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
    // In the STARK's order, which stdHints() has (gen_proof.hpp): calculateImHints, then
    // calculateWitnessSTD for the products and for the sums.
    for (const StdHint &hint : hints) {
        if (hint.stage != stage) {
            continue;
        }
        evaluate(hint.numerator, numerator.data());
        evaluate(hint.denominator, denominator.data());
        if (!batchInverse(inverse.data(), denominator.data(), N)) {
            const uint64_t row = std::find_if(denominator.begin(), denominator.end(),
                                              [&](const FrElement &d) { return fr.isZero(d); }) -
                                 denominator.begin();
            throw zeroDenominatorError(key, hint, row);
        }
        // multiplyHintFields (im_col) and accMulHintFields: vals[i] = numerator[i]/denominator[i]; the
        // latter then accumulates, vals[i] = vals[i] ∘ vals[i − 1].
        FrElement *dest = cols[stage].data() + hint.stagePos * N;
#pragma omp parallel for
        for (uint64_t i = 0; i < N; ++i) {
            fr.mul(dest[i], numerator[i], inverse[i]);
        }
        if (hint.kind != StdHint::Kind::ImCol) {
            accumulate(dest, N, hint.kind == StdHint::Kind::Prod);
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
    const std::vector<uint64_t> order = imPolOrder(key, stage);
    if (order.empty()) {
        return;
    }
    const ProverValues values = valuesOn(cols, challenges);
    const ExpressionsDomain trace = ExpressionsDomain::trace(info.nBits);
    for (uint64_t id : order) {
        const PolMapEntry &p = info.cmPolsMap[id];
        key.expressions().calculateExpression(p.expId, trace, values, cols[stage].data() + p.stagePos * N);
    }
}

std::vector<FrElement> Instance::drawBlinding(uint64_t stage) {
    const std::vector<LayoutEntry> &layout = key.info().layout;
    uint64_t n = 0;
    for (uint64_t f = 0; f < layout.size(); ++f) {
        if (layout[f].stage == stage) {
            n += layout[f].k * key.blindLength(f);
        }
    }
    std::vector<FrElement> factors(n);
    FrElement *next = factors.data();
    for (uint64_t f = 0; f < layout.size(); ++f) {
        if (layout[f].stage != stage) {
            continue;
        }
        const uint64_t b = key.blindLength(f);
        for (uint64_t j = 0; j < layout[f].k; ++j, next += b) {
            blinding->fill(next, b);
        }
    }
    return factors;
}

std::vector<G1Point> Instance::commitF(uint64_t stage) {
    const PilfflonkInfo &info = key.info();
    const uint64_t N = key.n();
    // p'(X) = p(X) + (X^N − 1)·b(X): blindCoefficients adds b_i at N + i and subtracts it at i
    // (pilfflonk/docs/protocol.md#blinding).
    std::vector<FrElement> factors = drawBlinding(stage);
#ifdef __USE_CUDA__
    if (device != nullptr) {
        return device->commitStage(stage, factors.data());
    }
#endif
    FrElement *next = factors.data();
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
        TimerStartExpr(PILFFLONK_INTT, f);
        std::vector<std::unique_ptr<Poly>> interpolants = key.lde().intt(evals.data(), coefs.data(), k, b);
        TimerStopAndLogExpr(PILFFLONK_INTT, f);
        std::vector<Poly *> components(k);
        for (uint64_t j = 0; j < k; ++j, next += b) {
            interpolants[j]->blindCoefficients(next, static_cast<uint32_t>(b));
            components[j] = interpolants[j].get();
            polys[entry.pols[j].id] = std::move(interpolants[j]);
        }
        TimerStartExpr(PILFFLONK_COMMIT, f);
        commitments.push_back(commitPacked(pk.srs(), components.data(), k));
        TimerStopAndLogExpr(PILFFLONK_COMMIT, f);
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
    TimerStartExpr(PILFFLONK_STAGE, stage);
#ifdef __USE_CUDA__
    // On a key on the GPU, the columns of the stage are computed where its commit reads them.
    if (device != nullptr) {
        computeStageColumns(*key.device(), stage, scalarValues(challengeValues));
    } else
#endif
    {
        TimerStartExpr(PILFFLONK_HINT_COLUMNS, stage);
        computeHintColumns(stage, columns, challengeValues);
        TimerStopAndLogExpr(PILFFLONK_HINT_COLUMNS, stage);
        TimerStartExpr(PILFFLONK_IM_POLS, stage);
        computeImPols(stage);
        TimerStopAndLogExpr(PILFFLONK_IM_POLS, stage);
    }
    std::vector<G1Point> commitments = commitF(stage);
    TimerStopAndLogExpr(PILFFLONK_STAGE, stage);
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
#ifdef __USE_CUDA__
    const CopyLog copies(pk.gpuKey(), "Q");
    qBegun = true;
#endif
    TimerStart(PILFFLONK_Q);
    const uint64_t partBits = qPartBits == 0 ? info.nBits : qPartBits;
    std::vector<G1Point> commitments;
#ifdef __USE_CUDA__
    if (device != nullptr) {
        commitments = commitQOnDevice(partBits);
    }
    if (device == nullptr)
#endif
    {
        commitments = commitQOnHost(partBits);
    }
    TimerStopAndLog(PILFFLONK_Q);
    ++next;
    return commitments;
}

ProverValues Instance::qValuesOn(const FrElement *columns, uint64_t S) const {
    const PilfflonkInfo &info = key.info();
    const std::vector<ColumnRead> &reads = key.qReads();
    ProverValues values = scalarValues(challengeValues);
    values.columns.resize(info.nStages + 1);
    values.columns[0].assign(info.nConstants, nullptr);
    for (uint64_t s = 1; s <= info.nStages; ++s) {
        values.columns[s].assign(key.cmIds()[s].size(), nullptr);
    }
    for (uint64_t r = 0; r < reads.size(); ++r) {
        values.columns[reads[r].type][reads[r].index] = columns + r * S;
    }
    return values;
}

UnsatisfiedError Instance::qAboveItsBound(uint64_t degree) const {
    // Q is a polynomial of its bound if and only if every constraint holds on its rows
    // (pilfflonk/docs/protocol.md#proof-sequence).
    return UnsatisfiedError("the witness does not satisfy the constraints of " + key.name() +
                            ": Q has a coefficient of degree " + std::to_string(degree) +
                            " not zero, and a polynomial of at most " + std::to_string(key.degrees().qCoefficients) +
                            " coefficients if it does (pilfflonk/docs/protocol.md#degrees)");
}

std::vector<FrElement> Instance::drawQBlinding() {
    const uint64_t boundaries = key.nQPieces() - 1;
    std::vector<FrElement> factors(2 * boundaries);
    for (uint64_t i = 0; i < boundaries; ++i) {
        blinding->fill(factors.data() + 2 * i, 2);
    }
    return factors;
}

std::vector<G1Point> Instance::commitQOnHost(uint64_t partBits) {
    // Q on the extended coset g·H' part by part (pilfflonk/docs/protocol.md#q-in-parts): each part
    // the 2^partBits points of Lde::extendCosetPart and ExpressionsDomain::cosetPart, and every
    // column Q's code reads extended to it from its committed polynomial (extendQPart). Point i of
    // part p is point p + nParts·i of the coset, where Q's value goes.
    const PilfflonkInfo &info = key.info();
    const Lde &lde = key.lde();
    const uint64_t M = lde.extendedSize();
    const uint64_t nBitsExt = key.degrees().nBitsExt;
    const uint64_t S = uint64_t(1) << partBits;
    const uint64_t nParts = M / S;
    // Q's N' values, and the S values on a part of column r of qReads at partColumns + r·S.
    std::unique_ptr<FrElement[]> qValues(new FrElement[M]());
    std::vector<FrElement> partColumns(key.qReads().size() * S);
    const ProverValues values = qValuesOn(partColumns.data(), S);

    // One part is the whole coset, in its order: Q goes straight into qValues.
    std::vector<FrElement> qPart(nParts > 1 ? S : 0);
    for (uint64_t part = 0; part < nParts; ++part) {
        TimerStartExpr(PILFFLONK_Q_EXTEND, part);
        extendQPart(key, polys, partBits, part, partColumns.data());
        TimerStopAndLogExpr(PILFFLONK_Q_EXTEND, part);
        TimerStartExpr(PILFFLONK_Q_DOMAIN, part);
        const ExpressionsDomain domain =
            ExpressionsDomain::cosetPart(info.nBits, nBitsExt, partBits, part, info.boundaries);
        TimerStopAndLogExpr(PILFFLONK_Q_DOMAIN, part);
        TimerStartExpr(PILFFLONK_Q_EVALUATE, part);
        FrElement *dest = nParts > 1 ? qPart.data() : qValues.get();
        key.expressions().calculateExpression(info.cExpId, domain, values, dest);
        if (nParts > 1) {
#pragma omp parallel for schedule(static)
            for (uint64_t i = 0; i < S; ++i) {
                qValues[part + nParts * i] = qPart[i];
            }
        }
        TimerStopAndLogExpr(PILFFLONK_Q_EVALUATE, part);
    }
    // Their memory back before the interpolation.
    std::vector<FrElement>().swap(partColumns);
    std::vector<FrElement>().swap(qPart);
    TimerStart(PILFFLONK_Q_INTERPOLATE);
    FrElement *coefs = qValues.get();
    lde.interpolateCoset(&coefs, &coefs, 1);
    TimerStopAndLog(PILFFLONK_Q_INTERPOLATE);

    const uint64_t bound = key.degrees().qCoefficients;
    for (uint64_t j = M; j-- > bound;) {
        if (!Engine::engine.fr.isZero(qValues[j])) {
            throw qAboveItsBound(j);
        }
    }

    // Its pieces (pilfflonk/docs/protocol.md#q-pieces): each its S coefficients of Q (the last one
    // the rest up to the bound), and each boundary b0·X^S + b1·X^(S+1) in the piece below it,
    // b0 + b1·X out of the one above. Unsplit, the one piece is Q, unblinded: its first bound
    // coefficients, where they are, in the buffer the instance keeps (qPieceCopy). AirDegrees gives
    // every piece 2 coefficients at least.
    Engine::Fr &fr = Engine::engine.fr;
    const AirDegrees &d = key.degrees();
    const uint64_t m = d.qPieceCoefficients.size();
    std::vector<std::unique_ptr<Poly>> pieces(m);
    if (m == 1) {
        qPieceCopy = std::move(qValues);
        pieces[0].reset(Poly::fromReservedBuffer(Engine::engine, qPieceCopy.get(), d.qPieceCoefficients[0]));
    } else {
        for (uint64_t i = 0; i < m; ++i) {
            const QPieceRange range = qPieceRange(d, i);
            pieces[i].reset(new Poly(Engine::engine, d.qPieceCoefficients[i]));
            ThreadUtils::parcpy(pieces[i]->coef, qValues.get() + range.start, range.length * sizeof(FrElement),
                                omp_get_max_threads());
        }
        // Q is in its pieces: its memory back before their commitments.
        qValues.reset();
    }
    const std::vector<FrElement> factors = drawQBlinding();
    for (uint64_t i = 0; i + 1 < m; ++i) {
        FrElement *below = pieces[i]->coef + d.qStride, *above = pieces[i + 1]->coef;
        const FrElement *b = factors.data() + 2 * i;
        below[0] = b[0];
        below[1] = b[1];
        fr.sub(above[0], above[0], b[0]);
        fr.sub(above[1], above[1], b[1]);
    }
    for (const std::unique_ptr<Poly> &piece : pieces) {
        piece->fixDegree();
    }

    // Each f of Q's stage, its pieces packed.
    TimerStart(PILFFLONK_Q_COMMIT);
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
    TimerStopAndLog(PILFFLONK_Q_COMMIT);
    qPieces = std::move(pieces);
    return commitments;
}

#ifdef __USE_CUDA__
std::vector<G1Point> Instance::commitQOnDevice(uint64_t partBits) {
    // As commitQOnHost, on the device (InstanceGpu::computeQ and commitQ), with the same check and
    // blinding factors.
    const uint64_t count =
        device->computeQ(partBits, [this](const FrElement *columns, uint64_t S) { return qValuesOn(columns, S); });
    if (count > key.degrees().qCoefficients) {
        throw qAboveItsBound(count - 1);
    }
    const std::vector<FrElement> factors = drawQBlinding();
    TimerStart(PILFFLONK_Q_COMMIT);
    std::vector<G1Point> commitments = device->commitQ(factors.data());
    TimerStopAndLog(PILFFLONK_Q_COMMIT);
    return commitments;
}
#endif

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
    witnessColumnsToHost("Instance::check");
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
        // Every column of the stage is computed below; on a key on the GPU, the instance has none here.
        t.columns[s].resize(key.cmIds()[s].size() * key.n(), Engine::engine.fr.zero());
        computeHintColumns(s, t.columns, t.challenges);
        computeImPols(s, t.columns, t.challenges);
    }
    return t;
}

void Instance::witnessColumnsToHost(const char *function) {
#ifdef __USE_CUDA__
    if (device == nullptr || !columns[1].empty()) {
        return;
    }
    if (qBegun) {
        throw invalid(function, "on a key on the GPU, the witness columns are on the device until Q is committed, and "
                                "Q's commitment has begun: check before commitQ");
    }
    const uint64_t N = key.n();
    columns[1].assign(key.cmIds()[1].size() * N, Engine::engine.fr.zero());
    for (uint64_t p : key.witnessColumns()) {
        stageColumnToHost(*key.device(), 1, p, columns[1].data() + p * N);
    }
#else
    (void)function;
#endif
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
    const uint64_t N = key.n();
#ifdef __USE_CUDA__
    if (device != nullptr) {
        std::vector<FrElement> &copy = deviceCopies[stage];
        std::vector<bool> &done = copied[stage];
        if (done.empty()) {
            copy.resize(key.cmIds()[stage].size() * N);
            done.assign(key.cmIds()[stage].size(), false);
        }
        if (!done[stagePos]) {
            if (qBegun) {
                throw invalid("Instance::column", "on a key on the GPU, the columns of stage " + std::to_string(stage) +
                                                      " are on the device until Q is committed, and Q's commitment "
                                                      "has begun: ask for them before commitQ");
            }
            stageColumnToHost(*key.device(), stage, stagePos, copy.data() + stagePos * N);
            done[stagePos] = true;
        }
        return copy.data() + stagePos * N;
    }
#endif
    return columns[stage].data() + stagePos * N;
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
    const uint64_t id = entry.pols[j].id;
#ifdef __USE_CUDA__
    // On a key on the GPU, a copy of the device's, once its stage is committed.
    if (device != nullptr && entry.stage < next && polys[id] == nullptr) {
        coefBuffers[id].reset(new FrElement[key.n() + key.blindLength(f)]);
        polys[id] = device->polynomialToHost(id, coefBuffers[id].get());
    }
#endif
    return polys[id].get();
}

ShplonkComponent Instance::component(uint64_t f, uint64_t j) const {
#ifdef __USE_CUDA__
    const PilfflonkInfo &info = key.info();
    const LayoutEntry &entry = info.layout.at(f);
    // On a key on the GPU, where the device keeps it, once its stage is committed.
    if (device != nullptr && entry.stage != 0 && j < entry.k && entry.stage < next) {
        if (entry.stage == info.qStage()) {
            const uint64_t piece = info.cmPolsMap[entry.pols[j].id].stagePos;
            return ShplonkComponent::elsewhere(key.degrees().qPieceCoefficients[piece], device->qPieceDegree(piece));
        }
        const uint64_t count = device->polynomialCount(entry.pols[j].id);
        return ShplonkComponent::elsewhere(key.n() + key.blindLength(f), count == 0 ? 0 : count - 1);
    }
#endif
    return polynomial(f, j);
}

Poly *Instance::qPiece(uint64_t i) const {
#ifdef __USE_CUDA__
    if (device != nullptr && qCommitted() && qPieces.empty()) {
        const std::vector<uint64_t> &bounds = key.degrees().qPieceCoefficients;
        qPieceCopy.reset(new FrElement[std::accumulate(bounds.begin(), bounds.end(), uint64_t(0))]);
        qPieces = device->qPiecesToHost(qPieceCopy.get());
    }
#endif
    return i < qPieces.size() ? qPieces[i].get() : nullptr;
}

void Instance::setQPartBits(uint64_t partBits) {
    const uint64_t nBits = key.info().nBits, nBitsExt = key.degrees().nBitsExt;
    if (partBits < nBits || partBits > nBitsExt) {
        throw invalid("Instance::setQPartBits", "parts of 2^" + std::to_string(partBits) +
                                                    " points, and Q's are of 2^" + std::to_string(nBits) + " to 2^" +
                                                    std::to_string(nBitsExt));
    }
#ifdef __USE_CUDA__
    if (device != nullptr) {
        device->requireQParts(partBits, "Instance::setQPartBits");
    }
#endif
    qPartBits = partBits;
}

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

    // The global order (pilfflonk/docs/protocol.md#global-order), and where each AIR's fixed f and
    // each instance's other f start in it.
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
                p.components.push_back(air.fixedComponent(pol.id));
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
                p.components.push_back(inst.component(f, j));
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
#ifdef __USE_CUDA__
    const CopyLog copies(pk->gpuKey(), "EVALUATIONS");
#endif
    TimerStart(PILFFLONK_EVALUATIONS);
#ifdef __USE_CUDA__
    if (pk->gpuKey() != nullptr) {
        // The instance holds the key's device memory, which no other instance does meanwhile: two
        // are the same one twice.
        if (instances.size() != 1) {
            throw invalid(function, std::to_string(instances.size()) +
                                        " instances of a key on the GPU, which proves one at a time");
        }
        device = OpeningGpu::ofInstance(*instances[0]);
        shplonk = std::make_unique<ShplonkProver>(
            std::move(opening), [this](const ShplonkProver &prover) { return device->evaluate(prover); });
    }
    if (device == nullptr)
#endif
    {
        shplonk = std::make_unique<ShplonkProver>(std::move(opening));
    }
    TimerStopAndLog(PILFFLONK_EVALUATIONS);

    Engine::Fr &fr = Engine::engine.fr;
    FrElement zh;
    fr.sub(zh, power(shplonk->xi(), uint64_t(1) << nBits), fr.one());
    if (fr.isZero(zh)) {
        throw std::runtime_error("Opening: ξ is in H, where Z_H vanishes");
    }

    // The evaluations, in the order of the transcript (pilfflonk/docs/protocol.md#transcript,
    // step 4): each at p_j(ξ·ω^prime) of its f.
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

    // The Q_i(ξ) of each instance whose Q is split, in the order of its layout
    // (pilfflonk/docs/protocol.md#transcript, step 4.3); and Q(ξ) of every instance,
    // Σ_i ξ^(i·S)·Q_i(ξ), by Horner from the last piece (Q_0(ξ) = Q(ξ) unsplit).
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

Opening::~Opening() = default;

FrElement Opening::q(uint64_t instance) const {
    if (instance >= qValues.size()) {
        throw invalid("Opening::q", "there is no instance " + std::to_string(instance));
    }
    return qValues[instance];
}

Opening::Proof Opening::open(Transcript &transcript) const {
    Engine::Fr &fr = Engine::engine.fr;
    Proof proof;
#ifdef __USE_CUDA__
    const CopyLog copies(pk->gpuKey(), "OPEN");
#endif
    TimerStart(PILFFLONK_OPEN);
#ifdef __USE_CUDA__
    if (device != nullptr) {
        proof.shplonk = shplonk->open(pk->srs(), transcript, *device);
    }
    if (device == nullptr)
#endif
    {
        proof.shplonk = shplonk->open(pk->srs(), transcript);
    }
    TimerStopAndLog(PILFFLONK_OPEN);
    proof.inv = verifierInverse(*shplonk, proof.shplonk.y);
    FrElement zh;
    fr.sub(zh, power(shplonk->xi(), uint64_t(1) << nBits), fr.one());
    fr.inv(proof.invZh, zh);
    return proof;
}

} // namespace PilFflonk
