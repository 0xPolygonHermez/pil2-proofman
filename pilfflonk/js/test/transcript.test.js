// The transcript (pilfflonk/docs/protocol.md#transcript): the challenges pinned by the C++ and Rust
// transcript tests, the encoding and the squeeze written out by hand, and the points it refuses.

import assert from "node:assert/strict";
import { before, test } from "node:test";

import { keccak_256 } from "@noble/hashes/sha3";

import { PilFflonkInputError, frFromObject, g1FromObject } from "../src/elements.js";
import { Keccak256Transcript } from "../src/transcript.js";
import { newCurve } from "./support.js";

let curve;
before(async () => {
    curve = await newCurve();
});

const fromHex = (hex) => BigInt(`0x${hex}`);

// k·G for the generator G = (1, 2), from pil2-stark/test/pilfflonk/pilfflonk_transcript_test.cpp.
const P2 = [
    "030644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd3",
    "15ed738c0e0a7c92e7845f96b2ae9c0a68a6a449e3538fc7ff3ebf7a5a18a2c4",
];
const P3 = [
    "0769bf9ac56bea3ff40232bcb1b6bd159315d84715b8e679f2d355961915abf0",
    "2ab799bee0489429554fdb7c8d086475319e63b40b9c5b57cdf1ff3dd9fe2261",
];
const P5 = [
    "17c139df0efee0f766bc0204762b774362e4ded88953a39ce849a8a7fa163fa9",
    "01e0559bacb160664764a357af8a9fe70baa9258e0b959273ffc5718c6d4cc7c",
];
// Points on the curve with a coordinate below 2^192: G, -G, and one with y = 4.
const G = ["01", "02"];
const MINUS_G = ["01", "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd45"];
const SHORT_Y = ["06f01ee8d2eb05e922d444637d6bce667a5424edbe74fae5e74798cf68745a4b", "04"];

const R_MINUS_ONE = "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000";

// PINNED_HEX of testPinnedChallenges() in pilfflonk_transcript_test.cpp, and PINNED of
// transcript_reproduces_the_pinned_challenges in provers/starks-lib-c/src/ffi_pilfflonk.rs: absorb
// the scalars 1 and r - 1 and the points 2G and 3G, squeeze; absorb 5G, squeeze; squeeze.
const PINNED = [
    "26ecd6b31f13f24a81bb71f60ef433df1185225122b3b101c736948874b1a41f",
    "11dba4dd6851bff01335d5e5b90efe759da23fce9a86a1c050daefdcd6948c14",
    "1a94293663ade82c5745a10779392a00a28435f34adb91d46084d9e50bee7a69",
];

const scalar = (hex) => frFromObject(curve, fromHex(hex), hex);
const point = ([x, y]) => g1FromObject(curve, [fromHex(x), fromHex(y)], `(${x}, ${y})`);
const hexOf = (e) => curve.Fr.toString(e, 16).padStart(64, "0");

// The pinned sequence, calling `between` on the transcript before each of its steps.
function pinnedSequence(between = () => {}) {
    const t = new Keccak256Transcript(curve);
    between(t);
    t.addScalar(scalar("01"));
    t.addScalar(scalar(R_MINUS_ONE));
    between(t);
    t.addPolCommitment(point(P2));
    t.addPolCommitment(point(P3));
    between(t);
    const first = t.squeeze();
    between(t);
    t.addPolCommitment(point(P5));
    between(t);
    const second = t.squeeze();
    between(t);
    const third = t.squeeze();
    return [first, second, third].map(hexOf);
}

test("reproduces the challenges pinned by the C++ and Rust transcript tests", () => {
    assert.deepEqual(pinnedSequence(), PINNED);
});

// The transcript by hand: 32 big-endian bytes per scalar, x‖y per point, keccak256 mod r; a squeeze
// restarts the buffer as the challenge.
function bigEndian(value) {
    const bytes = new Uint8Array(32);
    for (let i = 31, v = value; i >= 0; i--, v >>= 8n) bytes[i] = Number(v & 0xffn);
    return bytes;
}

function hashToFr(values) {
    const buffer = new Uint8Array(32 * values.length);
    values.forEach((v, i) => buffer.set(bigEndian(v), 32 * i));
    const digest = keccak_256(buffer);
    const value = digest.reduce((acc, b) => (acc << 8n) | BigInt(b), 0n);
    return value % curve.r;
}

test("encodes big-endian, and a squeeze restarts the buffer from the challenge", () => {
    const first = hashToFr([1n, fromHex(R_MINUS_ONE), ...P2.map(fromHex), ...P3.map(fromHex)]);
    const second = hashToFr([first, ...P5.map(fromHex)]);
    const third = hashToFr([second]);
    assert.deepEqual([first, second, third].map((v) => v.toString(16).padStart(64, "0")), PINNED);
});

test("refuses the point at infinity, points off the curve and coordinates below 2^192", () => {
    const G1 = curve.G1;
    const refused = [
        ["the point at infinity (Jacobian)", G1.zero],
        ["the point at infinity (affine)", G1.zeroAffine],
        ["(1, 3), off the curve", G1.fromObject([1n, 3n])],
        ["G, x below 2^192", G1.fromObject(G.map(fromHex))],
        ["-G, x below 2^192", G1.fromObject(MINUS_G.map(fromHex))],
        ["a point with y = 4", G1.fromObject(SHORT_Y.map(fromHex))],
    ];
    // Refusals leave the transcript as it was.
    const challenges = pinnedSequence((t) => {
        for (const [name, p] of refused) {
            assert.throws(() => t.addPolCommitment(p), PilFflonkInputError, name);
        }
    });
    assert.deepEqual(challenges, PINNED);
    // The points below 2^192 are on the curve: only the transcript refuses them.
    assert.ok(G1.isValid(refused[3][1]) && G1.isValid(refused[4][1]) && G1.isValid(refused[5][1]));
    // Any representation of a point encodes as its affine x‖y.
    const jacobian = G1.timesScalar(G1.one, 2);
    const t = new Keccak256Transcript(curve);
    t.addPolCommitment(jacobian);
    t.addPolCommitment(point(P3));
    const u = new Keccak256Transcript(curve);
    u.addPolCommitment(point(P2));
    u.addPolCommitment(G1.toJacobian(point(P3)));
    assert.ok(curve.Fr.eq(t.getChallenge(), u.getChallenge()));
});

test("refuses to hash an empty transcript and values that are not curve elements", () => {
    const t = new Keccak256Transcript(curve);
    assert.throws(() => t.getChallenge(), /No data/);
    assert.throws(() => t.squeeze(), /No data/);
    assert.throws(() => t.addScalar(1n), TypeError);
    assert.throws(() => t.addScalar(new Uint8Array(31)), TypeError);
    assert.throws(() => t.addPolCommitment([1n, 2n]), TypeError);
    assert.ok(t.isEmpty());
    t.addScalar(curve.Fr.one);
    t.squeeze();
    // A squeeze leaves the challenge in the buffer.
    assert.ok(!t.isEmpty());
});
