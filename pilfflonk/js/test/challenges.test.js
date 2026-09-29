// The transcript sequence of A.4 (challenges.js), observed through a recording transcript: what is
// absorbed and squeezed, in order, written out here from the proof's names rather than derived as
// challenges.js derives it; and the challenges recomputed by hand from what was absorbed.

import assert from "node:assert/strict";
import { before, test } from "node:test";

import { keccak_256 } from "@noble/hashes/sha3";

import { computeChallenges } from "../src/challenges.js";
import { fromObjectProof, fromObjectPublics } from "../src/proof.js";
import { fromObjectVk } from "../src/vkey.js";
import { RecordingTranscript, forgeProof, randomPublics, sampleVkey, syntheticVkey } from "./proofs.js";
import { newCurve } from "./support.js";

let curve;
before(async () => {
    curve = await newCurve();
});

// The challenges of a transcript log, by hand (A.4): 32 big-endian bytes per scalar, x‖y per
// point, keccak256 mod r, and after a squeeze the buffer is the challenge alone.
function replayByHand(log) {
    const bigEndian = (v) => {
        const bytes = new Uint8Array(32);
        for (let i = 31, x = BigInt(v); i >= 0; i--, x >>= 8n) bytes[i] = Number(x & 0xffn);
        return bytes;
    };
    let buffer = [];
    const out = [];
    for (const [kind, value] of log) {
        if (kind === "fr") buffer.push(bigEndian(value));
        else if (kind === "g1") buffer.push(bigEndian(value[0]), bigEndian(value[1]));
        else {
            const data = new Uint8Array(buffer.length * 32);
            buffer.forEach((b, i) => data.set(b, 32 * i));
            const h = BigInt(`0x${Buffer.from(keccak_256(data)).toString("hex")}`) % curve.r;
            out.push(h.toString());
            buffer = [bigEndian(h)];
        }
    }
    return out;
}

function record(vkey, publics, proofObject) {
    const vk = fromObjectVk(curve, vkey);
    const proof = fromObjectProof(curve, proofObject, vk);
    const transcript = new RecordingTranscript(curve);
    const challenges = computeChallenges(curve, vk, fromObjectPublics(curve, publics, vk), proof, null, transcript);
    return { challenges, log: transcript.log, vk };
}

const s = (e) => curve.Fr.toString(e);
const point = (p) => p.slice(0, 2);

test("two stages: A.4's sequence, item by item", () => {
    const vkey = syntheticVkey(curve);
    const publics = randomPublics(curve, vkey);
    const proof = forgeProof(curve, vkey, publics);
    const { challenges, log } = record(vkey, publics, proof);
    const P = proof.polynomials;
    const E = proof.evaluations;
    assert.deepEqual(log, [
        // 1: digest mod r, one instance, the publics
        ["fr", (BigInt(vkey.digest) % curve.r).toString()],
        ["fr", "1"],
        ["fr", publics[0]],
        ["fr", publics[1]],
        // 2, s = 1: f2 and f3; one challenge of stage 2
        ["g1", point(P.f2)],
        ["g1", point(P.f3)],
        ["squeeze", s(challenges.stages[2][0])],
        // 2, s = 2 = nStages: f4, and no challenge
        ["g1", point(P.f4)],
        // 3: std_vc, Q's f5, xiSeed
        ["squeeze", s(challenges.stdVc)],
        ["g1", point(P.f5)],
        ["squeeze", s(challenges.xiSeed)],
        // 4: the fixed columns' evaluations in evMap order, then the others'
        ...["Syn.K", "Syn.P", "Syn.Kw", "aw-1", "bw-1", "a", "b", "c", "d", "dw2"].map((name) => ["fr", E[name]]),
        // 5: α_S, [W]₁, y
        ["squeeze", s(challenges.alpha)],
        ["g1", point(P.W)],
        ["squeeze", s(challenges.y)],
    ]);
    assert.equal(challenges.stages[2].length, 1);
    assert.deepEqual(
        replayByHand(log),
        log.filter(([kind]) => kind === "squeeze").map(([, v]) => v),
    );
});

test("one stage, as the Fibonacci: no challenge before std_vc", () => {
    const vkey = sampleVkey();
    const publics = randomPublics(curve, vkey);
    const proof = forgeProof(curve, vkey, publics);
    const { challenges, log } = record(vkey, publics, proof);
    const P = proof.polynomials;
    const E = proof.evaluations;
    assert.deepEqual(log, [
        ["fr", BigInt(vkey.digest).toString()],
        ["fr", "1"],
        ["fr", publics[0]],
        ["fr", publics[1]],
        ["g1", point(P.f1)],
        ["g1", point(P.f2)],
        ["squeeze", s(challenges.stdVc)],
        ["g1", point(P.f3)],
        ["squeeze", s(challenges.xiSeed)],
        ...["L1", "aw-1", "a", "b", "c"].map((name) => ["fr", E[name]]),
        ["squeeze", s(challenges.alpha)],
        ["g1", point(P.W)],
        ["squeeze", s(challenges.y)],
    ]);
    assert.deepEqual(challenges.stages, []);
});

test("a split Q: its pieces after the other evaluations, in the proof's order", () => {
    const vkey = syntheticVkey(curve, { maxQDegree: 1, qDeg: 2 });
    const publics = randomPublics(curve, vkey);
    const proof = forgeProof(curve, vkey, publics);
    const { challenges, log } = record(vkey, publics, proof);
    const E = proof.evaluations;
    const afterXiSeed = log.findIndex(([kind, v]) => kind === "squeeze" && v === s(challenges.xiSeed)) + 1;
    const evaluations = log.slice(afterXiSeed, -3);
    assert.deepEqual(evaluations, [
        ...["Syn.K", "Syn.P", "Syn.Kw", "aw-1", "bw-1", "a", "b", "c", "d", "dw2"].map((name) => ["fr", E[name]]),
        ["fr", E.Q1],
        ["fr", E.Q0],
    ]);
    assert.deepEqual(log.slice(-3, -2), [["squeeze", s(challenges.alpha)]]);
});

test("the challenges depend on everything absorbed, and on nothing else", () => {
    const vkey = syntheticVkey(curve);
    const publics = randomPublics(curve, vkey);
    const proof = forgeProof(curve, vkey, publics);
    const base = record(vkey, publics, proof).challenges;
    const y = (v, p, pr) => s(record(v, p, pr).challenges.y);

    const changed = structuredClone(proof);
    changed.evaluations.dw2 = String((BigInt(changed.evaluations.dw2) + 1n) % curve.r);
    assert.notEqual(y(vkey, publics, changed), s(base.y));
    assert.equal(
        s(record(vkey, publics, changed).challenges.xiSeed),
        s(base.xiSeed),
        "xiSeed is before the evaluations",
    );

    // inv, invZh and W' are not absorbed (A.4).
    const unabsorbed = structuredClone(proof);
    unabsorbed.evaluations.inv = "1";
    unabsorbed.evaluations.invZh = "2";
    unabsorbed.polynomials.Wp = proof.polynomials.W;
    assert.equal(y(vkey, publics, unabsorbed), s(base.y));

    const otherPublics = [publics[1], publics[0]];
    assert.notEqual(s(record(vkey, otherPublics, proof).challenges.stages[2][0]), s(base.stages[2][0]));
});
