// The checks of snarkjs' steps 1-3 on what the verifier decodes (elements.js).

import assert from "node:assert/strict";
import { before, test } from "node:test";

import {
    PilFflonkInputError,
    decimalFromObject,
    frFromObject,
    frToObject,
    g1FromObject,
    g1ToObject,
    g2FromObject,
    toBigInt,
} from "../src/elements.js";
import { newCurve } from "./support.js";

let curve;
before(async () => {
    curve = await newCurve();
});

// [1]₂, the generator of G2: [[x.c0, x.c1], [y.c0, y.c1]].
const G2_ONE = [
    [
        "10857046999023057135944570762232829481370756359578518086990519993285655852781",
        "11559732032986387107991004021392285783925812861821192530917403151452391805634",
    ],
    [
        "8495653923123431417604973247489272438418190587263600148770280649306958101930",
        "4082367875863433681332203403145435568316851327593401208105741076214120093531",
    ],
];

test("integers: decimal strings, bigints and safe integers, never negative", () => {
    assert.equal(toBigInt("123", "v"), 123n);
    assert.equal(toBigInt(123n, "v"), 123n);
    assert.equal(toBigInt(123, "v"), 123n);
    for (const bad of ["-1", "0x10", "1.5", "", " 1", -1n, -1, 1.5, 2 ** 53, null, undefined, [1], {}]) {
        assert.throws(() => toBigInt(bad, "v"), PilFflonkInputError, String(bad));
    }
});

test("integers as pilfflonk's files spell them: decimal strings without sign or leading zeros", () => {
    assert.equal(decimalFromObject("0", "v"), 0n);
    assert.equal(decimalFromObject("1230", "v"), 1230n);
    for (const bad of ["01", "00", "-1", "+1", " 1", "1 ", "1e3", "0x1", "1.0", "", 1, 1n, null, ["1"]]) {
        assert.throws(() => decimalFromObject(bad, "v"), PilFflonkInputError, String(bad));
    }
});

test("scalars below r", () => {
    const r = curve.r;
    for (const v of [0n, 1n, r - 1n]) {
        assert.equal(frToObject(curve, frFromObject(curve, v.toString(), "s")), v.toString());
    }
    for (const v of [r, r + 1n, 2n ** 256n - 1n]) {
        assert.throws(() => frFromObject(curve, v.toString(), "s"), /is not below r/);
    }
});

test("G1 points: affine, canonical coordinates, on the curve, not at infinity", () => {
    const q = curve.q;
    const G1 = curve.G1;
    const two = g1ToObject(curve, G1.timesScalar(G1.one, 2));
    assert.ok(G1.eq(g1FromObject(curve, two, "P"), G1.timesScalar(G1.one, 2)));
    assert.ok(G1.eq(g1FromObject(curve, [...two, "1"], "P"), G1.timesScalar(G1.one, 2)));
    const refusals = [
        [[two[0], two[1], "2"], /z is not 1/],
        [[two[0], two[1], "0"], /z is not 1/],
        [[two[0]], /not an affine point/],
        ["12", /not an affine point/],
        [[two[0], (BigInt(two[1]) + q).toString()], /is not below q/],
        [[(BigInt(two[0]) + q).toString(), two[1]], /is not below q/],
        [[two[0], (BigInt(two[1]) + 1n).toString()], /is not on the curve/],
        [["0", "0"], /the point at infinity/],
    ];
    for (const [value, message] of refusals) {
        assert.throws(() => g1FromObject(curve, value, "P"), message, JSON.stringify(value));
    }
    assert.ok(G1.isZero(g1FromObject(curve, ["0", "0"], "P", { allowInfinity: true })));
    assert.deepEqual(g1ToObject(curve, G1.zero), ["0", "0"]);
});

test("G2 points: on the twist and in the r-torsion group", () => {
    const G2 = curve.G2;
    const F2 = G2.F;
    assert.ok(G2.eq(g2FromObject(curve, G2_ONE, "[1]₂"), G2.one));
    assert.ok(G2.eq(g2FromObject(curve, [...G2_ONE, ["1", "0"]], "[1]₂"), G2.one));
    assert.throws(() => g2FromObject(curve, [...G2_ONE, ["1", "1"]], "[1]₂"), /z is not 1/);
    const offTwist = [G2_ONE[0], [G2_ONE[1][0], "1"]];
    assert.throws(() => g2FromObject(curve, offTwist, "[1]₂"), /not on the twist/);
    assert.throws(() => g2FromObject(curve, [["0", "0"], ["0", "0"]], "[1]₂"), /the point at infinity/);
    // A point of the twist outside the r-torsion group: the twist has cofactor > 1.
    let x = F2.one;
    let rhs = F2.add(F2.mul(F2.square(x), x), G2.b);
    while (!F2.isSquare(rhs)) {
        x = F2.add(x, F2.one);
        rhs = F2.add(F2.mul(F2.square(x), x), G2.b);
    }
    const [xc0, xc1] = F2.toObject(x).map(String);
    const [yc0, yc1] = F2.toObject(F2.sqrt(rhs)).map(String);
    assert.throws(() => g2FromObject(curve, [[xc0, xc1], [yc0, yc1]], "P"), /not in the r-torsion group/);
});
