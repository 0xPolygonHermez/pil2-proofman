// The device side of an Opening (pilfflonk_opening_gpu.hpp): compiled into the GPU library only (the
// Makefile's %_gpu.cpp rule), with g++. It calls the kernels of pilfflonk_shplonk.cu and
// pilfflonk_kernels.cu, and the PLONK GPU prover's helpers (rapidsnark/plonk_prover.cu), through
// their C linkage.
#include "pilfflonk_opening_gpu.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include "pilfflonk_kernels.hpp"
#include "pilfflonk_prover.hpp"
#include "pilfflonk_proving_key.hpp"
#include "pilfflonk_shplonk.hpp"

// The PLONK GPU prover's helpers (rapidsnark/plonk_prover.cu), declared as plonk_prover_gpu.c.cuh
// declares them.
extern "C" void gpu_plonk_compute_div_zerofier(void *dCoefs, uint64_t length, const void *invBetaPtr,
                                               const void *y0Ptr, void *dPairWork);
extern "C" void gpu_plonk_precompute_omega_tables_async(void *dBases, void *dTid, const void *omega4xPtr,
                                                        uint32_t blockSize, uint32_t numBlocks, void *stream);

namespace PilFflonk {

namespace {

using Engine = AltBn128::Engine;

// The block of gpu_plonk_precompute_omega_tables_async's tables: x^(256·b), and x^t for t < 256.
constexpr uint64_t TABLE_BLOCK = 256;
// gpu_plonk_compute_div_zerofier's scan: pairs (a, b) of two elements, in blocks of 1024 (256 threads
// of 4), the totals of each level's blocks after them while there are more than one.
constexpr uint64_t PAIR_BYTES = 2 * sizeof(FrElement);
constexpr uint64_t SCAN_BLOCK = 1024;

// The elements of the tables of the powers of one point, for evaluations of up to n coefficients:
// x^(256·b) for b <= (n − 1)/256, then x^t for t < 256.
uint64_t tableElements(uint64_t n) { return (n - 1) / TABLE_BLOCK + 1 + TABLE_BLOCK; }

// The pairs gpu_plonk_compute_div_zerofier uses for n coefficients.
uint64_t affinePairs(uint64_t n) {
    if (n < 2) {
        return 0;
    }
    uint64_t level = n - 1, pairs = level;
    while (level > SCAN_BLOCK) {
        level = (level + SCAN_BLOCK - 1) / SCAN_BLOCK;
        pairs += level;
    }
    return pairs;
}

// What fixDegree finds of a polynomial whose coefficients count() counts.
uint64_t degreeOf(uint64_t count) { return count == 0 ? 0 : count - 1; }

// Where an OpeningGpu keeps its data in its workspace: byte offsets, each a whole number of elements.
struct Workspace {
    uint64_t quotient;     // bounds.length elements: W, then L and W'
    uint64_t interpolants; // bounds.nEvaluations elements: each r_i in |T_i|, zero above its own
    uint64_t values;       // bounds.nEvaluations elements: the evaluations
    uint64_t descriptors;  // bounds.nEvaluations PilfflonkGpuEvaluation
    uint64_t zero;         // a 64-bit 0: the offset of the one polynomial of a commit or a count
    uint64_t found;        // a 64-bit count of coefficients
    // What is not needed at once: the tables of the distinct points and pilfflonk_gpu_evaluate's
    // scratch; a component, and its division's pairs after it; L's division's pairs; an MSM's
    // scalars (GpuKey::commit's work).
    uint64_t scratch;
    uint64_t bytes;
};

Workspace workspaceOf(const ShplonkBounds &b) {
    uint64_t end = 0;
    auto take = [&end](uint64_t bytes) {
        const uint64_t start = end;
        end += (bytes + sizeof(FrElement) - 1) / sizeof(FrElement) * sizeof(FrElement);
        return start;
    };
    const uint64_t evaluation =
        (b.nPoints * tableElements(b.component) + pilfflonk_gpu_evaluation_scratch(b.nEvaluations)) * sizeof(FrElement);
    const uint64_t component = b.component * sizeof(FrElement) + affinePairs(b.component) * PAIR_BYTES;
    const uint64_t l = affinePairs(b.length) * PAIR_BYTES;
    const uint64_t msm = std::max(b.wMsm, b.wpMsm) * sizeof(FrElement);
    Workspace w;
    w.quotient = take(b.length * sizeof(FrElement));
    w.interpolants = take(b.nEvaluations * sizeof(FrElement));
    w.values = take(b.nEvaluations * sizeof(FrElement));
    w.descriptors = take(b.nEvaluations * sizeof(PilfflonkGpuEvaluation));
    w.zero = take(sizeof(uint64_t));
    w.found = take(sizeof(uint64_t));
    w.scratch = take(std::max({evaluation, component, l, msm}));
    w.bytes = end;
    return w;
}

} // namespace

ShplonkBounds shplonkBounds(const AirKey &air) {
    const PilfflonkInfo &info = air.info();
    const uint64_t N = air.n();
    ShplonkBounds b;
    uint64_t longest = 1;
    for (uint64_t f = 0; f < info.layout.size(); ++f) {
        const LayoutEntry &entry = info.layout[f];
        const uint64_t roots = entry.k * entry.offsets.size();
        b.nEvaluations += roots;
        b.nPoints += entry.offsets.size();
        b.length = std::max({b.length, entry.degree, roots});
        if (entry.degree > roots) {
            b.wMsm = std::max(b.wMsm, entry.degree - roots);
        }
        longest = std::max(longest, entry.degree);
        b.component = std::max<uint64_t>(b.component, entry.offsets.size());
        for (const LayoutPol &pol : entry.pols) {
            const uint64_t coefficients = entry.stage == info.qStage()
                                              ? air.degrees().qPieceCoefficients[info.cmPolsMap[pol.id].stagePos]
                                              : N + air.blindLength(f);
            b.component = std::max(b.component, coefficients);
        }
    }
    b.wMsm = std::max<uint64_t>(b.wMsm, 1);
    b.wpMsm = std::max<uint64_t>(longest - 1, 1);
    return b;
}

uint64_t shplonkWorkspaceBytes(const ShplonkBounds &bounds) { return workspaceOf(bounds).bytes; }

OpeningGpu::OpeningGpu(const GpuKey &_key, Components _components, const ShplonkBounds &_bounds, uint8_t *_workspace)
    : key(_key), components(std::move(_components)), bounds(_bounds), workspace(_workspace) {
    pilfflonk_gpu_memset_zero(at<uint64_t>(workspaceOf(bounds).zero), sizeof(uint64_t));
}

std::unique_ptr<OpeningGpu> OpeningGpu::ofInstance(const Instance &instance) {
    const AirKey &key = instance.air();
    if (key.device() == nullptr) {
        throw std::logic_error("OpeningGpu: " + key.name() + "'s key is not on the GPU");
    }
    const GpuAirKey &air = *key.device();
    const GpuKey &gpu = air.gpuKey();
    const ArenaLayout &layout = air.arena();
    const PilfflonkInfo &info = key.info();
    FrElement *pieces = reinterpret_cast<FrElement *>(gpu.arena() + layout.qPieces);
    const FrElement *polys = reinterpret_cast<const FrElement *>(gpu.arena() + layout.polys);

    // Q's pieces, piece i after the bounds of those before it, up to its degree: the kernels read no
    // coefficient above it.
    std::vector<const FrElement *> piece(key.nQPieces());
    uint64_t start = 0;
    for (uint64_t i = 0; i < key.nQPieces(); ++i) {
        const Poly &q = *instance.qPiece(i);
        piece[i] = pieces + start;
        gpu.staging().toDevice(pieces + start, q.coef, (q.getDegree() + 1) * sizeof(FrElement));
        start += key.degrees().qPieceCoefficients[i];
    }

    Components components(info.layout.size());
    for (uint64_t f = 0; f < info.layout.size(); ++f) {
        const LayoutEntry &entry = info.layout[f];
        for (uint64_t j = 0; j < entry.k; ++j) {
            if (entry.stage == info.qStage()) {
                components[f].push_back(piece[info.cmPolsMap[entry.pols[j].id].stagePos]);
            } else {
                const FrElement *base = entry.stage == 0 ? air.fixedCoefficients() : polys;
                components[f].push_back(base + componentOffset(key, layout, f, j));
            }
        }
    }
    return std::make_unique<OpeningGpu>(gpu, std::move(components), shplonkBounds(key),
                                        gpu.arena() + layout.shplonk);
}

void OpeningGpu::requireShape(const ShplonkProver &prover) const {
    bool fits = components.size() == prover.size() && prover.workLength() <= bounds.length;
    uint64_t nEvaluations = 0, nPoints = 0;
    for (uint64_t i = 0; fits && i < prover.size(); ++i) {
        const uint64_t nOffsets = prover.points(i).size();
        fits = components[i].size() == prover.k(i) && nOffsets <= bounds.component;
        for (const Poly *p : prover.components(i)) {
            fits = fits && p->getDegree() < bounds.component;
        }
        nEvaluations += prover.roots(i).size();
        nPoints += nOffsets;
    }
    if (!fits || nEvaluations > bounds.nEvaluations || nPoints > bounds.nPoints) {
        throw std::logic_error("OpeningGpu: the opening is not of its components, or does not fit its bounds");
    }
}

FrElement OpeningGpu::read(const FrElement *src) const {
    FrElement value;
    key.staging().toHost(&value, src, sizeof(value));
    return value;
}

uint64_t OpeningGpu::count(const FrElement *data, uint64_t n) const {
    const Workspace w = workspaceOf(bounds);
    uint64_t *found = at<uint64_t>(w.found);
    pilfflonk_gpu_count_coefficients(found, data, at<uint64_t>(w.zero), 1, n);
    uint64_t coefficients = 0;
    key.staging().toHost(&coefficients, found, sizeof(coefficients));
    return coefficients;
}

bool OpeningGpu::divide(FrElement *data, uint64_t n, const FrElement &beta, void *pairs) const {
    Engine::Fr &fr = Engine::engine.fr;
    const FrElement low = read(data);
    if (n == 1) {
        // A constant: divisible by X − β only if it is 0, which is then its quotient.
        return fr.isZero(low);
    }
    // The scan's q_{c+1} = (q_c − a_{c+1})/β from q_0 = −a_0/β.
    FrElement inverse, first;
    fr.inv(inverse, beta);
    fr.mul(first, fr.neg(inverse), low);
    gpu_plonk_compute_div_zerofier(data, n, &inverse, &first, pairs);
    return fr.isZero(read(data + n - 1));
}

void OpeningGpu::uploadInterpolants(const ShplonkProver &prover, const ShplonkProver::Interpolants &r) const {
    bool interpolants = r.size() == prover.size();
    std::vector<FrElement> coefs;
    for (uint64_t i = 0; interpolants && i < prover.size(); ++i) {
        const uint64_t nRoots = prover.roots(i).size();
        interpolants = r[i] != nullptr && r[i]->getLength() <= nRoots;
        if (interpolants) {
            coefs.insert(coefs.end(), r[i]->coef, r[i]->coef + r[i]->getLength());
            coefs.resize(coefs.size() + nRoots - r[i]->getLength(), Engine::engine.fr.zero());
        }
    }
    if (!interpolants) {
        throw std::logic_error("OpeningGpu: r is not one interpolant of at most |T_i| coefficients per f_i");
    }
    key.staging().toDevice(at<FrElement>(workspaceOf(bounds).interpolants), coefs.data(),
                           coefs.size() * sizeof(FrElement));
}

G1Point OpeningGpu::commit(uint64_t coefficients, uint64_t n) const {
    if (coefficients > n) {
        throw std::logic_error("OpeningGpu: a quotient of " + std::to_string(coefficients) +
                               " coefficients, committed with an MSM of " + std::to_string(n));
    }
    const Workspace w = workspaceOf(bounds);
    return key.commit(at<FrElement>(w.quotient), at<uint64_t>(w.zero), 1, coefficients, n, at<FrElement>(w.scratch));
}

ShplonkProver::Evaluations OpeningGpu::evaluate(const ShplonkProver &prover) const {
    requireShape(prover);
    Engine::Fr &fr = Engine::engine.fr;
    const Workspace w = workspaceOf(bounds);

    // The distinct points, each with its tables, for evaluations of up to `longest` coefficients.
    std::vector<FrElement> points;
    std::vector<std::vector<uint64_t>> pointOf(prover.size());
    uint64_t longest = 1;
    for (uint64_t i = 0; i < prover.size(); ++i) {
        for (const FrElement &x : prover.points(i)) {
            const auto known =
                std::find_if(points.begin(), points.end(), [&](const FrElement &p) { return fr.eq(p, x); });
            pointOf[i].push_back(known - points.begin());
            if (known == points.end()) {
                points.push_back(x);
            }
        }
        for (const Poly *p : prover.components(i)) {
            longest = std::max(longest, p->getDegree() + 1);
        }
    }
    const uint64_t perPoint = tableElements(longest), nBlocks = perPoint - TABLE_BLOCK;
    FrElement *tables = at<FrElement>(w.scratch);
    for (uint64_t p = 0; p < points.size(); ++p) {
        FrElement *blocks = tables + p * perPoint;
        gpu_plonk_precompute_omega_tables_async(blocks, blocks + nBlocks, &points[p], TABLE_BLOCK,
                                                static_cast<uint32_t>(nBlocks), nullptr);
    }

    // Offset-major, as the prover's evaluations.
    std::vector<PilfflonkGpuEvaluation> evaluations;
    for (uint64_t i = 0; i < prover.size(); ++i) {
        const uint64_t k = prover.k(i);
        for (uint64_t m = 0; m < pointOf[i].size(); ++m) {
            const FrElement *blocks = tables + pointOf[i][m] * perPoint;
            for (uint64_t j = 0; j < k; ++j) {
                evaluations.push_back(PilfflonkGpuEvaluation{components[i][j], prover.components(i)[j]->getDegree() + 1,
                                                             blocks, blocks + nBlocks});
            }
        }
    }
    Staging &staging = key.staging();
    staging.toDevice(at<PilfflonkGpuEvaluation>(w.descriptors), evaluations.data(),
                     evaluations.size() * sizeof(PilfflonkGpuEvaluation));
    pilfflonk_gpu_evaluate(at<FrElement>(w.values), at<PilfflonkGpuEvaluation>(w.descriptors), evaluations.size(),
                           tables + points.size() * perPoint);
    std::vector<FrElement> values(evaluations.size());
    staging.toHost(values.data(), at<FrElement>(w.values), values.size() * sizeof(FrElement));

    ShplonkProver::Evaluations evals(prover.size());
    uint64_t next = 0;
    for (uint64_t i = 0; i < prover.size(); ++i) {
        const uint64_t n = prover.points(i).size() * prover.k(i);
        evals[i].assign(values.begin() + next, values.begin() + next + n);
        next += n;
    }
    return evals;
}

void OpeningGpu::computeW(const ShplonkProver &prover, const ShplonkProver::Interpolants &r, const FrElement &alpha) {
    requireShape(prover);
    uploadInterpolants(prover, r);
    Engine::Fr &fr = Engine::engine.fr;
    const Workspace w = workspaceOf(bounds);
    const uint64_t length = prover.workLength();
    FrElement *W = at<FrElement>(w.quotient);
    const FrElement *interpolants = at<FrElement>(w.interpolants);
    // A component's p_j − r^(j), then its quotients, with its division's pairs after it.
    FrElement *buffer = at<FrElement>(w.scratch);
    void *pairs = buffer + bounds.component;
    pilfflonk_gpu_memset_zero(W, length * sizeof(FrElement));
    FrElement alphaPower = fr.one();
    uint64_t rStart = 0;
    for (uint64_t i = 0; i < prover.size(); ++i) {
        const uint64_t k = prover.k(i), nRoots = prover.roots(i).size();
        const std::vector<FrElement> &points = prover.points(i);
        for (uint64_t j = 0; j < k; ++j) {
            const uint64_t coefs = prover.components(i)[j]->getDegree() + 1;
            uint64_t n = std::max<uint64_t>(coefs, points.size());
            pilfflonk_gpu_component_minus(buffer, n, components[i][j], coefs, interpolants + rStart, nRoots, k, j);
            // Z_{T_i} = Π_{s in O_i} (Y − ξ·ω_N^s) in Y = X^k, one factor at a time.
            for (const FrElement &point : points) {
                if (!divide(buffer, n, point, pairs)) {
                    throw ShplonkProver::remainderNotDivisible(i);
                }
                n = std::max<uint64_t>(n - 1, 1);
            }
            pilfflonk_gpu_add_component(W, buffer, n, k, j, &alphaPower);
        }
        fr.mul(alphaPower, alphaPower, alpha);
        rStart += nRoots;
    }
    wCount = count(W, length);
    prover.checkW(degreeOf(wCount));
}

G1Point OpeningGpu::commitW() { return commit(wCount, bounds.wMsm); }

void OpeningGpu::computeWp(const ShplonkProver &prover, const ShplonkProver::Interpolants &r, const FrElement &alpha,
                           const FrElement &y) {
    requireShape(prover);
    if (r.size() != prover.size() || std::find(r.begin(), r.end(), nullptr) != r.end()) {
        throw std::logic_error("OpeningGpu: r is not one interpolant per f_i");
    }
    Engine::Fr &fr = Engine::engine.fr;
    const ShplonkProver::LScalars scalars = prover.lScalars(alpha, y);
    // L's constant coefficient, but for the f_i's: −Σ_i f[i]·r_i(y).
    FrElement constant = fr.zero();
    for (uint64_t i = 0; i < prover.size(); ++i) {
        fr.sub(constant, constant, fr.mul(scalars.f[i], r[i]->evaluate(y)));
    }
    const Workspace w = workspaceOf(bounds);
    const uint64_t length = prover.workLength();
    FrElement *L = at<FrElement>(w.quotient);
    // L = w·W + Σ_i f[i]·(f_i − r_i(y)), over W, which is zero above its degree.
    pilfflonk_gpu_scale_add_constant(L, length, &scalars.w, &constant);
    for (uint64_t i = 0; i < prover.size(); ++i) {
        for (uint64_t j = 0; j < prover.k(i); ++j) {
            pilfflonk_gpu_add_component(L, components[i][j], prover.components(i)[j]->getDegree() + 1, prover.k(i), j,
                                        &scalars.f[i]);
        }
    }
    // W' = L/(Z_{T∖T_0}(y)·(X − y)), the 1/Z_{T∖T_0}(y) in L's scalars.
    if (!divide(L, length, y, at<uint8_t>(w.scratch))) {
        throw ShplonkProver::lNotDivisible();
    }
    wpCount = count(L, length);
    prover.checkWp(degreeOf(wpCount));
}

G1Point OpeningGpu::commitWp() { return commit(wpCount, bounds.wpMsm); }

const FrElement *OpeningGpu::quotient() const { return at<FrElement>(workspaceOf(bounds).quotient); }

} // namespace PilFflonk
