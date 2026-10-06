// The device side of an Opening (pilfflonk_opening_gpu.hpp): compiled into the GPU library only (the
// Makefile's %_gpu.cpp rule), with g++. It calls the kernels of pilfflonk_shplonk.cu and
// pilfflonk_kernels.cu, and the PLONK GPU prover's helpers (rapidsnark/plonk_prover.cu), through
// their C linkage.
#include "pilfflonk_opening_gpu.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include "pilfflonk_gpu.hpp"
#include "pilfflonk_kernels.hpp"
#include "pilfflonk_lde_kernels.hpp"
#include "pilfflonk_prover.hpp"
#include "pilfflonk_proving_key.hpp"
#include "pilfflonk_shplonk.hpp"

namespace PilFflonk {

namespace {

using Engine = AltBn128::Engine;

// The block of gpu_plonk_precompute_omega_tables_async's tables: x^(256·b), and x^t for t < 256.
constexpr uint64_t TABLE_BLOCK = 256;
// From this many offsets an f_i's components are divided by Z_{T_i} on a coset (two NTTs each), not by
// one scan per offset.
constexpr uint64_t NTT_MIN_OFFSETS = 4;

uint64_t nextPowerOfTwo(uint64_t n) {
    uint64_t p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

// The elements of the tables of the powers of one point, for evaluations of up to n coefficients:
// x^(256·b) for b <= (n − 1)/256, then x^t for t < 256.
uint64_t tableElements(uint64_t n) { return (n - 1) / TABLE_BLOCK + 1 + TABLE_BLOCK; }

// What fixDegree finds of a polynomial whose coefficients count() counts.
uint64_t degreeOf(uint64_t count) { return count == 0 ? 0 : count - 1; }

// divideOnCoset's scratch: a component and 1/Z on the coset, the tables of the shift and of its
// inverse, and a flag.
struct CosetScratch {
    FrElement *buffer, *zInverse, *blocks, *powers, *inverseBlocks, *inversePowers;
    uint32_t *zero;
};

CosetScratch cosetScratch(FrElement *scratch, uint64_t ext) {
    const uint64_t table = tableElements(ext), nBlocks = table - TABLE_BLOCK;
    CosetScratch c;
    c.buffer = scratch;
    c.zInverse = scratch + ext;
    c.blocks = c.zInverse + ext;
    c.powers = c.blocks + nBlocks;
    c.inverseBlocks = c.blocks + table;
    c.inversePowers = c.inverseBlocks + nBlocks;
    c.zero = reinterpret_cast<uint32_t *>(c.blocks + 2 * table);
    return c;
}

// The scan path's scratch: a component and its quotient (alternately), and the chunks' T and S.
struct LinearScratch {
    FrElement *a, *b, *T, *S;
};

LinearScratch linearScratch(FrElement *scratch, uint64_t component) {
    const uint64_t chunks = pilfflonk_gpu_division_chunks(component) + 1;
    LinearScratch l;
    l.a = scratch;
    l.b = l.a + component;
    l.T = l.b + component;
    l.S = l.T + chunks;
    return l;
}

uint64_t linearScratchBytes(uint64_t component) {
    return (2 * component + 2 * (pilfflonk_gpu_division_chunks(component) + 1)) * sizeof(FrElement);
}

uint64_t cosetScratchBytes(uint64_t ext) {
    return ext == 0 ? 0 : (2 * ext + 2 * tableElements(ext)) * sizeof(FrElement) + sizeof(uint64_t);
}

// Where an OpeningGpu keeps its data in its workspace: byte offsets, each a whole number of elements.
struct Workspace {
    uint64_t quotient;     // bounds.length elements: W, then L and W'
    uint64_t interpolants; // bounds.nEvaluations elements: each r_i in |T_i|, zero above its own
    uint64_t values;       // bounds.nEvaluations elements: the evaluations
    uint64_t descriptors;  // bounds.nEvaluations PilfflonkGpuEvaluation
    uint64_t zero;         // a 64-bit 0: the offset of the one polynomial of a commit or a count
    uint64_t found;        // a 64-bit count of coefficients
    uint64_t flags;        // bounds.nPoints 32-bit flags: computeW's inexact divisions, by f
    // What is not needed at once: the tables of the distinct points and pilfflonk_gpu_evaluate's
    // scratch; the components' divisions; L and its division's T and S; an MSM's scalars
    // (GpuKey::commit's work).
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
    const uint64_t l = (b.length + 2 * (pilfflonk_gpu_division_chunks(b.length) + 1)) * sizeof(FrElement);
    const uint64_t msm = std::max(b.wMsm, b.wpMsm) * sizeof(FrElement);
    const uint64_t coset = cosetScratchBytes(b.nttLength);
    const uint64_t linear = linearScratchBytes(b.component);
    Workspace w;
    w.quotient = take(b.length * sizeof(FrElement));
    w.interpolants = take(b.nEvaluations * sizeof(FrElement));
    w.values = take(b.nEvaluations * sizeof(FrElement));
    w.descriptors = take(b.nEvaluations * sizeof(PilfflonkGpuEvaluation));
    w.zero = take(sizeof(uint64_t));
    w.found = take(sizeof(uint64_t));
    w.flags = take(b.nPoints * sizeof(uint32_t));
    w.scratch = take(std::max({evaluation, l, msm, coset, linear}));
    w.bytes = end;
    return w;
}

} // namespace

ShplonkBounds shplonkBounds(const AirKey &air) {
    const PilfflonkInfo &info = air.info();
    const uint64_t N = air.n();
    ShplonkBounds b;
    for (uint64_t f = 0; f < info.layout.size(); ++f) {
        const LayoutEntry &entry = info.layout[f];
        const uint64_t roots = entry.k * entry.offsets.size();
        b.nEvaluations += roots;
        b.nPoints += entry.offsets.size();
        b.length = std::max({b.length, entry.degree, roots});
        b.component = std::max<uint64_t>(b.component, entry.offsets.size());
        uint64_t widest = entry.offsets.size() + 1; // Z's coefficients
        for (const LayoutPol &pol : entry.pols) {
            const uint64_t coefficients = entry.stage == info.qStage()
                                              ? air.degrees().qPieceCoefficients[info.cmPolsMap[pol.id].stagePos]
                                              : N + air.blindLength(f);
            b.component = std::max(b.component, coefficients);
            widest = std::max(widest, coefficients);
        }
        if (entry.offsets.size() >= NTT_MIN_OFFSETS) {
            b.nttLength = std::max(b.nttLength, nextPowerOfTwo(widest));
        }
    }
    b.wMsm = air.wMsm();
    b.wpMsm = air.wpMsm();
    return b;
}

uint64_t shplonkWorkspaceBytes(const ShplonkBounds &bounds) { return workspaceOf(bounds).bytes; }

OpeningGpu::OpeningGpu(const GpuKey &_key, Components _components, const ShplonkBounds &_bounds, uint8_t *_workspace)
    : key(_key), components(std::move(_components)), bounds(_bounds), workspace(_workspace) {
    const ProofCall call(key);
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
    const FrElement *polys = reinterpret_cast<const FrElement *>(gpu.arena() + layout.polys);
    const FrElement *pieces = reinterpret_cast<const FrElement *>(gpu.arena() + layout.qPieces);

    // Each p_j where the device keeps it: Q's pieces where InstanceGpu::commitQ left them.
    Components components(info.layout.size());
    for (uint64_t f = 0; f < info.layout.size(); ++f) {
        const LayoutEntry &entry = info.layout[f];
        const FrElement *base =
            entry.stage == 0 ? air.fixedCoefficients() : entry.stage == info.qStage() ? pieces : polys;
        for (uint64_t j = 0; j < entry.k; ++j) {
            components[f].push_back(base + componentOffset(key, layout, f, j));
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
        for (const ShplonkComponent &p : prover.components(i)) {
            fits = fits && p.degree() < bounds.component;
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
    const ProofCall call(key);
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
        for (const ShplonkComponent &p : prover.components(i)) {
            longest = std::max(longest, p.degree() + 1);
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
                evaluations.push_back(PilfflonkGpuEvaluation{components[i][j], prover.components(i)[j].degree() + 1,
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
    const ProofCall call(key);
    uploadInterpolants(prover, r);
    Engine::Fr &fr = Engine::engine.fr;
    const Workspace w = workspaceOf(bounds);
    const uint64_t length = prover.workLength();
    FrElement *W = at<FrElement>(w.quotient);
    const FrElement *interpolants = at<FrElement>(w.interpolants);
    // A component's p_j − r^(j), then its quotients, alternately in two buffers.
    const LinearScratch scan = linearScratch(at<FrElement>(w.scratch), bounds.component);
    uint32_t *flags = at<uint32_t>(w.flags);
    pilfflonk_gpu_memset_zero(W, length * sizeof(FrElement));
    pilfflonk_gpu_memset_zero(flags, prover.size() * sizeof(uint32_t));
    FrElement alphaPower = fr.one();
    uint64_t rStart = 0;
    for (uint64_t i = 0; i < prover.size(); ++i) {
        const uint64_t k = prover.k(i), nRoots = prover.roots(i).size();
        const std::vector<FrElement> &points = prover.points(i);
        uint64_t widest = points.size() + 1; // Z's coefficients
        for (const ShplonkComponent &p : prover.components(i)) {
            widest = std::max(widest, p.degree() + 1);
        }
        const uint64_t ext = nextPowerOfTwo(widest);
        if (points.size() >= NTT_MIN_OFFSETS && ext <= bounds.nttLength &&
            divideOnCoset(prover, i, ext, interpolants + rStart, alphaPower)) {
            fr.mul(alphaPower, alphaPower, alpha);
            rStart += nRoots;
            continue;
        }
        for (uint64_t j = 0; j < k; ++j) {
            const uint64_t coefs = prover.components(i)[j].degree() + 1;
            uint64_t n = std::max<uint64_t>(coefs, points.size());
            FrElement *dividend = scan.a, *quotient = scan.b;
            pilfflonk_gpu_component_minus(dividend, n, components[i][j], coefs, interpolants + rStart, nRoots, k, j);
            // Z_{T_i} = Π_{s in O_i} (Y − ξ·ω_N^s) in Y = X^k, one factor at a time; the flags read below.
            for (const FrElement &point : points) {
                pilfflonk_gpu_divide_linear(quotient, dividend, n, &point, scan.T, scan.S, flags + i);
                if (n == 1) {
                    // A constant: its quotient is 0 (the flag says whether it was).
                    pilfflonk_gpu_memset_zero(quotient, sizeof(FrElement));
                }
                std::swap(dividend, quotient);
                n = std::max<uint64_t>(n - 1, 1);
            }
            pilfflonk_gpu_add_component(W, dividend, n, k, j, &alphaPower);
        }
        fr.mul(alphaPower, alphaPower, alpha);
        rStart += nRoots;
    }
    std::vector<uint32_t> inexact(prover.size());
    key.staging().toHost(inexact.data(), flags, inexact.size() * sizeof(uint32_t));
    for (uint64_t i = 0; i < prover.size(); ++i) {
        if (inexact[i] != 0) {
            throw ShplonkProver::remainderNotDivisible(i);
        }
    }
    wCount = count(W, length);
    prover.checkW(degreeOf(wCount));
}

bool OpeningGpu::divideOnCoset(const ShplonkProver &prover, uint64_t i, uint64_t ext, const FrElement *interpolants,
                               const FrElement &alphaPower) const {
    Engine::Fr &fr = Engine::engine.fr;
    const Workspace w = workspaceOf(bounds);
    const CosetScratch c = cosetScratch(at<FrElement>(w.scratch), ext);
    const uint64_t k = prover.k(i), nRoots = prover.roots(i).size(), bits = __builtin_ctzll(ext);
    const std::vector<FrElement> &points = prover.points(i);
    const uint64_t m = points.size(), nBlocks = tableElements(ext) - TABLE_BLOCK;
    FrElement shift, shiftInverse;
    fr.set(shift, COSET_SHIFT); // pilfflonk_lde.hpp
    fr.inv(shiftInverse, shift);
    gpu_plonk_precompute_omega_tables_async(c.blocks, c.powers, &shift, TABLE_BLOCK, static_cast<uint32_t>(nBlocks),
                                            nullptr);
    gpu_plonk_precompute_omega_tables_async(c.inverseBlocks, c.inversePowers, &shiftInverse, TABLE_BLOCK,
                                            static_cast<uint32_t>(nBlocks), nullptr);

    // 1/Z(Y) on the coset, Z = Π_s (Y − ξ·ω_N^s).
    std::vector<FrElement> z(m + 1, fr.zero());
    z[0] = fr.one();
    for (uint64_t s = 0; s < m; ++s) {
        for (uint64_t d = s + 1; d > 0; --d) {
            z[d] = fr.sub(z[d - 1], fr.mul(points[s], z[d]));
        }
        z[0] = fr.neg(fr.mul(points[s], z[0]));
    }
    Staging &staging = key.staging();
    pilfflonk_gpu_memset_zero(c.zInverse, ext * sizeof(FrElement));
    staging.toDevice(c.zInverse, z.data(), z.size() * sizeof(FrElement));
    pilfflonk_gpu_mul_by_powers(c.zInverse, ext, c.blocks, c.powers);
    transformOnDevice(c.zInverse, bits, false);
    pilfflonk_gpu_memset_zero(c.zero, sizeof(uint32_t));
    pilfflonk_gpu_batch_inverse(c.zInverse, c.buffer, ext, c.zero);
    uint32_t vanishes = 0;
    staging.toHost(&vanishes, c.zero, sizeof(vanishes));
    if (vanishes != 0) {
        return false;
    }
    // A guard on the pipeline: 1/Z at the coset's first point, the shift.
    FrElement zShift = fr.zero();
    for (uint64_t d = m + 1; d > 0; --d) {
        zShift = fr.add(fr.mul(zShift, shift), z[d - 1]);
    }
    if (!fr.eq(fr.mul(zShift, read(c.zInverse)), fr.one())) {
        throw std::logic_error("OpeningGpu: 1/Z on the coset is wrong");
    }

    FrElement *W = at<FrElement>(w.quotient);
    for (uint64_t j = 0; j < k; ++j) {
        const uint64_t coefs = prover.components(i)[j].degree() + 1;
        const uint64_t quotient = std::max<uint64_t>(coefs, m) - m;
        pilfflonk_gpu_component_minus(c.buffer, ext, components[i][j], coefs, interpolants, nRoots, k, j);
        pilfflonk_gpu_mul_by_powers(c.buffer, ext, c.blocks, c.powers);
        transformOnDevice(c.buffer, bits, false);
        pilfflonk_gpu_mul_pointwise(c.buffer, c.zInverse, ext);
        transformOnDevice(c.buffer, bits, true);
        pilfflonk_gpu_mul_by_powers(c.buffer, ext, c.inverseBlocks, c.inversePowers);
        // Exact iff the result has degree below n − |O|: then it times Z is f − r (both of degree
        // below ext, equal on the coset).
        if (count(c.buffer, ext) > quotient) {
            throw ShplonkProver::remainderNotDivisible(i);
        }
        pilfflonk_gpu_add_component(W, c.buffer, std::max<uint64_t>(quotient, 1), k, j, &alphaPower);
    }
    return true;
}

G1Point OpeningGpu::commitW() {
    const ProofCall call(key);
    return commit(wCount, bounds.wMsm);
}

void OpeningGpu::computeWp(const ShplonkProver &prover, const ShplonkProver::Interpolants &r, const FrElement &alpha,
                           const FrElement &y) {
    requireShape(prover);
    const ProofCall call(key);
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
    // L in the scratch, so that W' = L/(X − y) goes over W, where it is committed.
    FrElement *L = at<FrElement>(w.scratch), *Wp = at<FrElement>(w.quotient);
    FrElement *T = L + length, *S = T + pilfflonk_gpu_division_chunks(length) + 1;
    uint32_t *inexact = at<uint32_t>(w.flags);
    // L = w·W + Σ_i f[i]·(f_i − r_i(y)), over W, which is zero above its degree.
    pilfflonk_gpu_memset_zero(L, length * sizeof(FrElement));
    pilfflonk_gpu_add_component(L, Wp, length, 1, 0, &scalars.w);
    const FrElement one = fr.one();
    pilfflonk_gpu_scale_add_constant(L, 1, &one, &constant);
    for (uint64_t i = 0; i < prover.size(); ++i) {
        for (uint64_t j = 0; j < prover.k(i); ++j) {
            pilfflonk_gpu_add_component(L, components[i][j], prover.components(i)[j].degree() + 1, prover.k(i), j,
                                        &scalars.f[i]);
        }
    }
    // W' = L/(Z_{T∖T_0}(y)·(X − y)), the 1/Z_{T∖T_0}(y) in L's scalars; its top coefficient is W's.
    pilfflonk_gpu_memset_zero(inexact, sizeof(uint32_t));
    pilfflonk_gpu_divide_linear(Wp, L, length, &y, T, S, inexact);
    pilfflonk_gpu_memset_zero(Wp + length - 1, sizeof(FrElement));
    uint32_t notDivisible = 0;
    key.staging().toHost(&notDivisible, inexact, sizeof(notDivisible));
    if (notDivisible != 0) {
        throw ShplonkProver::lNotDivisible();
    }
    wpCount = count(Wp, length);
    prover.checkWp(degreeOf(wpCount));
}

G1Point OpeningGpu::commitWp() {
    const ProofCall call(key);
    return commit(wpCount, bounds.wpMsm);
}

const FrElement *OpeningGpu::quotient() const { return at<FrElement>(workspaceOf(bounds).quotient); }

} // namespace PilFflonk
