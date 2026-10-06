#include "pilfflonk_msm.hpp"

#include <omp.h>
#include <x86intrin.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace PilFflonk {

namespace {

using Engine = AltBn128::Engine;
using Point = Engine::G1Point;
using Affine = Engine::G1PointAffine;
using Fe = Engine::F1Element;


// q, the BN128 base field modulus, little-endian limbs.
constexpr uint64_t Q[4] = {0x3c208c16d87cfd47ULL, 0x97816a916871ca8dULL, 0xb85045b68181585dULL,
                           0x30644e72e131a029ULL};

inline void addq(Fe &r, const Fe &a, const Fe &b) {
    unsigned long long s[4], d[4];
    unsigned char c = 0;
    c = _addcarry_u64(c, a.v[0], b.v[0], &s[0]);
    c = _addcarry_u64(c, a.v[1], b.v[1], &s[1]);
    c = _addcarry_u64(c, a.v[2], b.v[2], &s[2]);
    _addcarry_u64(c, a.v[3], b.v[3], &s[3]); // q < 2^254: no carry out
    unsigned char br = 0;
    br = _subborrow_u64(br, s[0], Q[0], &d[0]);
    br = _subborrow_u64(br, s[1], Q[1], &d[1]);
    br = _subborrow_u64(br, s[2], Q[2], &d[2]);
    br = _subborrow_u64(br, s[3], Q[3], &d[3]);
    const uint64_t keep = -uint64_t(br); // all ones if s < q
    for (int i = 0; i < 4; ++i) {
        r.v[i] = (s[i] & keep) | (d[i] & ~keep);
    }
}

inline void subq(Fe &r, const Fe &a, const Fe &b) {
    unsigned long long d[4], e[4];
    unsigned char br = 0;
    br = _subborrow_u64(br, a.v[0], b.v[0], &d[0]);
    br = _subborrow_u64(br, a.v[1], b.v[1], &d[1]);
    br = _subborrow_u64(br, a.v[2], b.v[2], &d[2]);
    br = _subborrow_u64(br, a.v[3], b.v[3], &d[3]);
    const uint64_t mask = -uint64_t(br);
    unsigned char c = 0;
    c = _addcarry_u64(c, d[0], Q[0] & mask, &e[0]);
    c = _addcarry_u64(c, d[1], Q[1] & mask, &e[1]);
    c = _addcarry_u64(c, d[2], Q[2] & mask, &e[2]);
    _addcarry_u64(c, d[3], Q[3] & mask, &e[3]);
    for (int i = 0; i < 4; ++i) {
        r.v[i] = e[i];
    }
}

inline void mulq(Fe &r, const Fe &a, const Fe &b) { Fq_rawMMul(r.v, a.v, b.v); }
inline void sqrq(Fe &r, const Fe &a) { Fq_rawMSquare(r.v, a.v); }
inline bool eqq(const Fe &a, const Fe &b) {
    return ((a.v[0] ^ b.v[0]) | (a.v[1] ^ b.v[1]) | (a.v[2] ^ b.v[2]) | (a.v[3] ^ b.v[3])) == 0;
}
inline bool zeroq(const Fe &a) { return (a.v[0] | a.v[1] | a.v[2] | a.v[3]) == 0; }
inline void negq(Fe &r, const Fe &a) {
    if (zeroq(a)) {
        r = a;
        return;
    }
    Fe q;
    std::memcpy(q.v, Q, sizeof(Q));
    subq(r, q, a);
}

// 1/a = a^(q−2) (a != 0), by a fixed window of 4 bits: no allocation, unlike ffiasm's mpz inverse.
void invq(Fe &r, const Fe &a) {
    constexpr uint64_t E[4] = {Q[0] - 2, Q[1], Q[2], Q[3]};
    Fe table[16];
    table[0] = Engine::engine.f1.one();
    table[1] = a;
    for (int i = 2; i < 16; ++i) {
        mulq(table[i], table[i - 1], a);
    }
    Fe acc = table[0];
    bool started = false;
    for (int nib = 63; nib >= 0; --nib) {
        const uint64_t d = (E[nib / 16] >> (4 * (nib % 16))) & 15;
        if (started) {
            sqrq(acc, acc);
            sqrq(acc, acc);
            sqrq(acc, acc);
            sqrq(acc, acc);
        }
        if (d != 0) {
            if (started) {
                mulq(acc, acc, table[d]);
            } else {
                acc = table[d];
                started = true;
            }
        }
    }
    r = acc;
}

// The signed-digit windows of c bits: digit w of a scalar is raw_w + carry_w, made negative (−2^c,
// carry out) from half = 2^(c−1) on, but in the top window, which absorbs the last carry.
struct Digits {
    uint32_t c;
    uint32_t windows;
    uint64_t mask;
    uint64_t half;

    explicit Digits(uint32_t _c) : c(_c), windows((255 + _c - 1) / _c), mask((uint64_t(1) << _c) - 1), half(uint64_t(1) << (_c - 1)) {}

    uint64_t raw(const uint64_t *s, uint32_t w) const {
        const uint32_t o = w * c;
        if (o >= 256) {
            return 0;
        }
        const uint32_t word = o >> 6, sh = o & 63;
        uint64_t v = s[word] >> sh;
        if (sh + c > 64 && word + 1 < 4) {
            v |= s[word + 1] << (64 - sh);
        }
        return v & mask;
    }

    // The digit, in [−half, half), or [0, half] in the top window.
    int64_t digit(const uint64_t *s, uint32_t w) const {
        uint64_t carry = 0;
        // The carry into w is 1 if raw_{w−1} >= half, 0 below half − 1, and that of w − 1 at half − 1.
        for (uint32_t v = w; v-- > 0;) {
            const uint64_t r = raw(s, v);
            if (r >= half) {
                carry = 1;
                break;
            }
            if (r + 1 < half) {
                break;
            }
        }
        int64_t d = int64_t(raw(s, w) + carry);
        if (w + 1 < windows && uint64_t(d) >= half) {
            d -= int64_t(uint64_t(1) << c);
        }
        return d;
    }
};

constexpr uint32_t MAX_BATCH = 2048;

// One thread's buckets of a window over a range of points: affine, accumulated in batches of
// affine additions with one inversion for the batch (Montgomery's trick). A bucket with an
// addition pending in the batch defers another point for it to the next batch.
struct Buckets {
    uint32_t nb;
    // A bucket on a cache line of its own.
    struct alignas(64) Bucket {
        Affine p;
    };
    std::vector<Bucket> b;
    std::vector<uint8_t> occupied;
    std::vector<uint32_t> stamp;
    uint32_t epoch = 1;

    // A point to add to a bucket: bases[index], negated if neg.
    struct Pending {
        uint32_t bucket;
        uint32_t neg;
        uint64_t index;
    };
    const Affine *bases;
    uint32_t K;
    std::vector<Pending> batch;
    std::vector<Fe> dx, prefix;
    std::vector<Pending> deferred, draining;
    // Allocated on first use; used lists the buckets with an overflow.
    std::vector<Point> overflow;
    std::vector<uint8_t> overflowUsed;
    std::vector<uint32_t> used;

    Buckets(uint32_t _nb, const Affine *_bases)
        : nb(_nb), b(_nb), occupied(_nb, 0), stamp(_nb, 0), bases(_bases),
          K(std::max<uint32_t>(16, std::min<uint32_t>(MAX_BATCH, _nb / 8))) {
        batch.reserve(K);
        overflow.resize(nb);
        overflowUsed.assign(nb, 0);
        dx.resize(K);
        prefix.resize(K);
    }

    Affine point(const Pending &p) const {
        Affine a = bases[p.index];
        if (p.neg) {
            negq(a.y, a.y);
        }
        return a;
    }

    void reset() {
        std::fill(occupied.begin(), occupied.end(), 0);
        for (uint32_t i : used) {
            overflowUsed[i] = 0;
        }
        used.clear();
    }

    // b = 2·b, affine (rare: a point equal to its bucket).
    static void doubleAffine(Affine &p) {
        Fe x2, num, den, inv, lambda, l2, t;
        sqrq(x2, p.x);
        addq(num, x2, x2);
        addq(num, num, x2);
        addq(den, p.y, p.y);
        invq(inv, den);
        mulq(lambda, num, inv);
        sqrq(l2, lambda);
        subq(l2, l2, p.x);
        subq(l2, l2, p.x);
        subq(t, p.x, l2);
        mulq(t, t, lambda);
        subq(p.y, t, p.y);
        p.x = l2;
    }

    // Returns whether the batch is full.
    bool schedule(const Pending &e) {
        const uint32_t i = e.bucket;
        if (stamp[i] == epoch) {
            // Few buckets in use (a window of small digits) would defer most points: past K/2, a
            // projective addition into the bucket's overflow instead.
            if (deferred.size() < K / 2) {
                deferred.push_back(e);
            } else {
                Affine p = point(e);
                if (!overflowUsed[i]) {
                    Engine::engine.g1.copy(overflow[i], Engine::engine.g1.zero());
                    overflowUsed[i] = 1;
                    used.push_back(i);
                }
                Engine::engine.g1.add(overflow[i], overflow[i], p);
            }
            return false;
        }
        if (!occupied[i]) {
            b[i].p = point(e);
            occupied[i] = 1;
            return false;
        }
        Affine &q = b[i].p;
        const Affine p = point(e);
        if (eqq(q.x, p.x)) {
            if (eqq(q.y, p.y)) {
                doubleAffine(q);
            } else {
                occupied[i] = 0;
            }
            return false;
        }
        stamp[i] = epoch;
        batch.push_back(e);
        return batch.size() == K;
    }

    void flush() {
        const uint32_t k = batch.size();
        if (k == 0) {
            return;
        }
        Engine &E = Engine::engine;
        Fe acc = E.f1.one();
        for (uint32_t j = 0; j < k; ++j) {
            subq(dx[j], bases[batch[j].index].x, b[batch[j].bucket].p.x);
            prefix[j] = acc;
            mulq(acc, acc, dx[j]);
        }
        Fe inv;
        invq(inv, acc);
        for (uint32_t j = k; j-- > 0;) {
            Fe ij, lambda, x3, t;
            mulq(ij, inv, prefix[j]);
            mulq(inv, inv, dx[j]);
            Affine &q = b[batch[j].bucket].p;
            const Affine p = point(batch[j]);
            subq(t, p.y, q.y);
            mulq(lambda, t, ij);
            sqrq(x3, lambda);
            subq(x3, x3, q.x);
            subq(x3, x3, p.x);
            subq(t, q.x, x3);
            mulq(t, t, lambda);
            subq(q.y, t, q.y);
            q.x = x3;
        }
        batch.clear();
        ++epoch;
    }

    void drain() {
        draining.swap(deferred);
        for (const Pending &d : draining) {
            if (schedule(d)) {
                flush();
            }
        }
        draining.clear();
    }

    void finish() {
        while (!batch.empty() || !deferred.empty()) {
            flush();
            drain();
        }
    }

    // Σ_i (i + 1)·b[i], by running sums from the top bucket down.
    Point sum() {
        Engine &E = Engine::engine;
        Point running, total;
        E.g1.copy(running, E.g1.zero());
        E.g1.copy(total, E.g1.zero());
        for (uint32_t i = nb; i-- > 0;) {
            if (occupied[i]) {
                E.g1.add(running, running, b[i].p);
            }
            if (overflowUsed[i]) {
                E.g1.add(running, running, overflow[i]);
            }
            if (!E.g1.isZero(running)) {
                E.g1.add(total, total, running);
            }
        }
        return total;
    }
};

// The window that minimises the additions into buckets (with the batch's inversion) plus their
// sums, ~4 times as costly each, with buckets of at most 2 MB.
uint32_t chooseWindow(uint64_t n, uint64_t threads) {
    uint32_t best = 2;
    double bestCost = 1e300;
    for (uint32_t c = 2; c <= 16; ++c) {
        const uint64_t W = (255 + c - 1) / c;
        const uint64_t P = std::max<uint64_t>(1, (4 * threads + W - 1) / W);
        // An inversion per batch, ~60 additions' worth.
        const double K = std::max<double>(16, std::min<double>(MAX_BATCH, double(uint64_t(1) << (c - 1)) / 8));
        const double cost = double(n) * W * (1 + 60 / K) + 4.0 * double(W * P) * double(uint64_t(1) << (c - 1));
        if (cost < bestCost) {
            bestCost = cost;
            best = c;
        }
    }
    return best;
}

} // namespace

Point msm(const Affine *bases, const uint64_t *scalars, uint64_t n) {
    Engine &E = Engine::engine;
    Point result;
    E.g1.copy(result, E.g1.zero());
    if (n == 0) {
        return result;
    }
    // The signed digits index buckets for scalars below 2^254 only: one above would write past them.
    bool wide = false;
#pragma omp parallel for reduction(|| : wide)
    for (uint64_t i = 0; i < n; ++i) {
        wide = wide || (scalars[4 * i + 3] >> 62) != 0;
    }
    if (wide) {
        throw std::invalid_argument("msm: a scalar is not canonical (not below 2^254)");
    }
    const uint64_t threads = omp_get_max_threads();
    if (n < 256) {
        E.g1.multiMulByScalar(result, const_cast<Affine *>(bases),
                              reinterpret_cast<uint8_t *>(const_cast<uint64_t *>(scalars)), 32,
                              static_cast<unsigned int>(n));
        return result;
    }
    const Digits digits(chooseWindow(n, threads));
    const uint32_t W = digits.windows;
    const uint64_t P = std::max<uint64_t>(1, (4 * threads + W - 1) / W);
    const uint64_t chunk = (n + P - 1) / P;
    const uint64_t nTasks = W * P;
    std::vector<Point> partial(nTasks);

#pragma omp parallel
    {
        Buckets buckets(uint32_t(digits.half), bases);
        // Range-major: the threads working at once read the same points, for different windows.
#pragma omp for schedule(dynamic, 1)
        for (uint64_t t = 0; t < nTasks; ++t) {
            const uint32_t w = t % W;
            const uint64_t lo = (t / W) * chunk, hi = std::min(n, lo + chunk);
            buckets.reset();
            for (uint64_t i = lo; i < hi; ++i) {
                const int64_t d = digits.digit(scalars + 4 * i, w);
                if (d == 0) {
                    continue;
                }
                const Affine &base = bases[i];
                if (zeroq(base.x) && zeroq(base.y)) {
                    continue;
                }
                const bool full = buckets.schedule(d > 0 ? Buckets::Pending{uint32_t(d - 1), 0, i}
                                                         : Buckets::Pending{uint32_t(-d - 1), 1, i});
                if (full) {
                    buckets.flush();
                    buckets.drain();
                }
            }
            buckets.finish();
            partial[t] = buckets.sum();
        }
    }

    std::vector<Point> windowSum(W);
    for (uint32_t w = 0; w < W; ++w) {
        E.g1.copy(windowSum[w], E.g1.zero());
    }
    for (uint64_t t = 0; t < nTasks; ++t) {
        E.g1.add(windowSum[t % W], windowSum[t % W], partial[t]);
    }
    E.g1.copy(result, windowSum[W - 1]);
    for (uint32_t w = W - 1; w-- > 0;) {
        for (uint32_t k = 0; k < digits.c; ++k) {
            E.g1.dbl(result, result);
        }
        E.g1.add(result, result, windowSum[w]);
    }
    return result;
}

} // namespace PilFflonk
