// The parts of the SHPLONK check that need no prover: the roots of A.2.5, the layouts the prover
// refuses, and the openings verifyOpening rejects before any pairing. The openings themselves are
// checked against the C++ prover in fixtures.test.js.

import assert from "node:assert/strict";
import { before, test } from "node:test";

import { checkLayout, computeRoots, rootOfUnity, verifyOpening } from "../src/shplonk.js";
import { newCurve } from "./support.js";

let curve;
before(async () => {
    curve = await newCurve();
});

// A fixed xiSeed: any non-zero element.
const XI_SEED = "4417881134626180770308697923359573201005643519861877412381846989312604493735";

test("roots of unity: 5^((r-1)/n) has order exactly n, and ω_N is ffjavascript's (and ffiasm's) generator of H", () => {
    const Fr = curve.Fr;
    // Each order n with the primes that divide it.
    const orders = [
        [1n, []],
        [2n, [2n]],
        [3n, [3n]],
        [4n, [2n]],
        [6n, [2n, 3n]],
        [12n, [2n, 3n]],
        [2n ** 28n, [2n]],
        [3n * 2n ** 20n, [2n, 3n]],
    ];
    for (const [n, primes] of orders) {
        const w = rootOfUnity(curve, n);
        assert.ok(Fr.eq(Fr.exp(w, n), Fr.one), `w_${n}^${n} = 1`);
        for (const p of primes) {
            assert.ok(!Fr.eq(Fr.exp(w, n / p), Fr.one), `w_${n} is primitive`);
        }
    }
    for (let nBits = 0; nBits <= 28; nBits++) {
        assert.ok(Fr.eq(rootOfUnity(curve, 2n ** BigInt(nBits)), Fr.w[nBits]), `ω_{2^${nBits}}`);
    }
    for (const n of [5n, 7n, 27n, 2n ** 29n]) {
        assert.throws(() => rootOfUnity(curve, n), /does not divide r - 1/);
    }
});

test("roots of A.2.5: x^k = ξ·ω_N^s for every offset, negative ones too, k·|O| distinct roots", () => {
    const Fr = curve.Fr;
    const xiSeed = Fr.e(XI_SEED);
    for (const nBits of [1, 2, 4]) {
        const N = 2 ** nBits;
        const f = [
            { k: 1, offsets: [0] },
            // N = 2: ω_N = -1, and s = -1 is the row of s = 1.
            { k: 3, offsets: N > 2 ? [-1, 0, 1, 2] : [-1, 0] },
            { k: 4, offsets: N > 2 ? [-1, 1] : [-1] },
            { k: 12, offsets: [0, -1] },
            { k: 2, offsets: [N - 1] },
        ];
        const layout = { nBits, powerW: 12, f };
        checkLayout(curve, layout);
        const { xi, f: roots } = computeRoots(curve, layout, xiSeed);
        assert.ok(Fr.eq(xi, Fr.exp(xiSeed, 12n)));
        const omegaN = Fr.w[nBits];
        f.forEach(({ k, offsets }, i) => {
            const { points, roots: T } = roots[i];
            assert.equal(points.length, offsets.length);
            assert.equal(T.length, k * offsets.length);
            // ω_{kN}^s by s mod kN, as the prover takes it, and ξ·ω_N^s by s mod N.
            const omegaKN = rootOfUnity(curve, BigInt(k * N));
            const seed = Fr.exp(xiSeed, BigInt(12 / k));
            const mod = (s, n) => BigInt(((s % n) + n) % n);
            offsets.forEach((s, m) => {
                assert.ok(Fr.eq(points[m], Fr.mul(xi, Fr.exp(omegaN, mod(s, N)))), `f_${i}: point of ${s}`);
                assert.ok(Fr.eq(T[m * k], Fr.mul(seed, Fr.exp(omegaKN, mod(s, k * N)))), `f_${i}: x_0 of ${s}`);
                for (let j = 0; j < k; j++) {
                    assert.ok(Fr.eq(Fr.exp(T[m * k + j], BigInt(k)), points[m]), `f_${i}: x_${j}^k of ${s}`);
                }
            });
            const distinct = new Set(T.map((x) => Fr.toString(x)));
            assert.equal(distinct.size, T.length, `f_${i}: distinct roots`);
        });
    }
});

test("roots: a negative offset is not its absolute value", () => {
    const Fr = curve.Fr;
    const layout = { nBits: 4, powerW: 3, f: [{ k: 3, offsets: [-1, 1] }] };
    const { xi, f: roots } = computeRoots(curve, layout, Fr.e(XI_SEED));
    const [minus, plus] = roots[0].points;
    assert.ok(!Fr.eq(minus, plus));
    assert.ok(Fr.eq(Fr.mul(minus, plus), Fr.square(xi)));
    assert.ok(Fr.eq(Fr.exp(roots[0].roots[0], 3n), Fr.mul(xi, Fr.inv(Fr.w[4]))));
});

test("refuses the layouts the prover refuses", () => {
    const valid = { nBits: 4, powerW: 3, f: [{ k: 3, offsets: [0, 1] }, { k: 1, offsets: [-1] }] };
    checkLayout(curve, valid);
    const cases = [
        [{ f: [] }, /no polynomials to open/],
        [{ nBits: 29 }, /nBits = 29 is not an integer between 0 and 28/],
        [{ nBits: -1 }, /nBits = -1/],
        [{ nBits: 1.5 }, /nBits = 1.5/],
        [{ powerW: 6 }, /powerW = 6 is not 3, the lcm of every k/],
        [{ powerW: 0 }, /powerW = 0 is not 3/],
        [{ f: [{ k: 5, offsets: [0] }], powerW: 5 }, /f_0: kN = 5·2\^4 does not divide r - 1/],
        [{ f: [{ k: 7, offsets: [0] }], powerW: 7 }, /f_0: kN = 7·2\^4 does not divide r - 1/],
        [{ f: [{ k: 27, offsets: [0] }], powerW: 27 }, /f_0: kN = 27·2\^4 does not divide r - 1/],
        [{ f: [{ k: 0, offsets: [0] }], powerW: 1 }, /f_0: k = 0 is not a positive integer/],
        [{ nBits: 27, f: [{ k: 4, offsets: [0] }], powerW: 4 }, /f_0: kN = 4·2\^27 does not divide r - 1/],
        [{ f: [{ k: 3, offsets: [] }], powerW: 3 }, /f_0 has no offsets/],
        [{ f: [{ k: 3, offsets: [1, 1] }], powerW: 3 }, /f_0: two offsets are the same row modulo N = 16/],
        [{ f: [{ k: 3, offsets: [-1, 15] }], powerW: 3 }, /f_0: two offsets are the same row modulo N = 16/],
        [{ f: [{ k: 1, offsets: [16] }], powerW: 1 }, /f_0: offset 16 is not an integer below N = 16/],
        [{ f: [{ k: 1, offsets: [-16] }], powerW: 1 }, /offset -16 is not an integer below/],
        [{ f: [{ k: 1, offsets: [0.5] }], powerW: 1 }, /offset 0.5 is not an integer/],
    ];
    for (const [change, message] of cases) {
        assert.throws(() => checkLayout(curve, { ...valid, ...change }), message, JSON.stringify(change));
    }
    // k = 3 at N = 2^27 is fine: 3 is odd.
    checkLayout(curve, { nBits: 27, powerW: 3, f: [{ k: 3, offsets: [0] }] });
});

// An opening of the right shape, not a valid one: verifyOpening must reject it before the pairing
// when a piece is missing or of the wrong kind.
function shapedOpening() {
    const { Fr, G1, G2 } = curve;
    const e = (v) => Fr.e(v);
    return {
        nBits: 4,
        powerW: 3,
        f: [{ k: 3, offsets: [0, 1] }, { k: 1, offsets: [-1] }],
        fixedCommitments: [G1.toAffine(G1.timesScalar(G1.one, 7))],
        commitments: [G1.toAffine(G1.timesScalar(G1.one, 11))],
        evaluations: [
            [[e(1), e(2), e(3)], [e(4), e(5), e(6)]],
            [[e(7)]],
        ],
        xiSeed: e(XI_SEED),
        alpha: e(13),
        y: e(17),
        W: G1.toAffine(G1.timesScalar(G1.one, 19)),
        Wp: G1.toAffine(G1.timesScalar(G1.one, 23)),
        X2: { one: G2.toAffine(G2.one), tau: G2.toAffine(G2.timesScalar(G2.one, 29)) },
    };
}

function recorder() {
    const messages = { info: [], warn: [], error: [] };
    return {
        messages,
        info: (m) => messages.info.push(m),
        warn: (m) => messages.warn.push(m),
        error: (m) => messages.error.push(m),
    };
}

test("rejects, and logs why, an opening of the wrong shape", async () => {
    const { Fr, G1 } = curve;
    const logger = recorder();
    assert.equal(await verifyOpening(curve, shapedOpening(), logger), false);
    assert.deepEqual(logger.messages.error, []);
    assert.deepEqual(logger.messages.warn, ["Invalid SHPLONK opening"]);

    const cases = [
        [(o) => o.fixedCommitments.pop(), /0 fixed and 1 other commitments for 2 polynomials/],
        [(o) => o.commitments.push(o.W), /1 fixed and 2 other commitments for 2 polynomials/],
        [(o) => (o.commitments[0] = G1.fromObject([1n, 3n])), /\[f_1\] is not a point of G1/],
        [(o) => (o.fixedCommitments[0] = Fr.one), /\[f_0\] is not a point of G1/],
        [(o) => o.evaluations.pop(), /evaluations for 1 polynomials, not 2/],
        [(o) => o.evaluations[0].pop(), /f_0: not one row of evaluations per offset/],
        [(o) => o.evaluations[0][1].pop(), /f_0: row 1 is not k = 3 elements of Fr/],
        [(o) => (o.evaluations[1][0][0] = 7n), /f_1: row 0 is not k = 1 elements of Fr/],
        [(o) => (o.alpha = 13n), /alpha is not an element of Fr/],
        [(o) => (o.xiSeed = Fr.zero), /xiSeed is zero/],
        [(o) => (o.Wp = G1.fromObject([1n, 3n])), /\[W\] or \[W'\] is not a point of G1/],
        [(o) => (o.X2 = { one: o.X2.one }), /\[1\]₂ or \[τ\]₂ is not a point of G2/],
        [(o) => (o.powerW = 1), /powerW = 1 is not 3/],
    ];
    for (const [change, message] of cases) {
        const opening = shapedOpening();
        change(opening);
        const log = recorder();
        assert.equal(await verifyOpening(curve, opening, log), false, String(message));
        assert.equal(log.messages.error.length, 1, String(message));
        assert.match(log.messages.error[0], message);
    }
});

test("rejects y on a root of some f_i, where q_i is not defined", async () => {
    const opening = shapedOpening();
    const { f: roots } = computeRoots(curve, opening, opening.xiSeed);
    opening.y = roots[1].roots[0];
    const logger = recorder();
    assert.equal(await verifyOpening(curve, opening, logger), false);
    assert.deepEqual(logger.messages.error, ["SHPLONK opening: y is a root of f_1"]);
});
