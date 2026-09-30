// Reading the vkey (vkey.js): what it decodes, and every vkey it refuses, each with the reason.
// The refusals are Rust's (Vkey::validate, Layout::check, the serde types), snarkjs' checks on the
// points, and what the verifier needs to run (see vkey.js).

import assert from "node:assert/strict";
import { before, test } from "node:test";

import { PilFflonkInputError, frToObject, g1ToObject } from "../src/elements.js";
import { fromObjectVk } from "../src/vkey.js";
import { G2_GENERATOR, SAMPLE_DIGEST, sampleVkey, syntheticVkey } from "./proofs.js";
import { newCurve } from "./support.js";

let curve;
before(async () => {
    curve = await newCurve();
});

test("M15's sample vkey decodes", () => {
    const vk = fromObjectVk(curve, sampleVkey());
    assert.equal(vk.power, 3);
    assert.equal(vk.N, 8);
    assert.equal(vk.nStages, 1);
    assert.equal(vk.qStage, 2);
    assert.equal(vk.qPieces, 1);
    assert.deepEqual(vk.challenges, [
        { stage: 2, stageId: 0 },
        { stage: 3, stageId: 0 },
    ]);
    assert.deepEqual(
        vk.evMap.map((e) => e.name),
        ["L1", "aw-1", "a", "b", "c"],
    );
    assert.deepEqual(vk.evaluationOrder, [0, 1, 2, 3, 4]);
    assert.deepEqual(vk.commitmentNames, ["f1", "f2", "f3"]);
    assert.deepEqual(vk.qPieceNames, []);
    assert.equal(vk.fixedCommitments.length, 1);
    assert.ok(curve.G1.eq(vk.fixedCommitments[0], curve.G1.g));
    assert.ok(curve.G2.eq(vk.X2.tau, curve.G2.g) && curve.G2.eq(vk.X2.one, curve.G2.g));
    assert.equal(vk.digest, SAMPLE_DIGEST);
    assert.equal(curve.Fr.toString(vk.digestFr, 16), SAMPLE_DIGEST.slice(3), "below r: itself");
});

test("the synthetic vkey decodes, the fixed columns first in the proof's order", () => {
    const vk = fromObjectVk(curve, syntheticVkey(curve));
    assert.equal(vk.nStages, 2);
    assert.deepEqual(
        vk.evaluationOrder.map((i) => vk.evMap[i].name),
        ["Syn.K", "Syn.P", "Syn.Kw", "aw-1", "bw-1", "a", "b", "c", "d", "dw2"],
    );
    assert.deepEqual(vk.commitmentNames, ["f2", "f3", "f4", "f5"]);
    assert.deepEqual(vk.challenges, [
        { stage: 2, stageId: 0 },
        { stage: 3, stageId: 0 },
        { stage: 4, stageId: 0 },
    ]);
    const split = fromObjectVk(curve, syntheticVkey(curve, { maxQDegree: 1, qDeg: 2 }));
    assert.equal(split.qPieces, 2);
    assert.deepEqual(split.qPieceNames, ["Q1", "Q0"], "in the layout's order, as ProofNames");
});

test("digest mod r is what the transcript absorbs", () => {
    const vk = fromObjectVk(curve, { ...sampleVkey(), digest: `0x${"f".repeat(64)}` });
    assert.equal(curve.Fr.toString(vk.digestFr), ((2n ** 256n - 1n) % curve.r).toString());
});

test("a fixed commitment may be the point at infinity, (0, 0)", () => {
    const vk = fromObjectVk(curve, { ...sampleVkey(), f0: ["0", "0"] });
    assert.ok(curve.G1.isZero(vk.fixedCommitments[0]));
});

// A point of the twist outside the r-torsion group (as elements.test.js builds it).
function twistPointOutsideTorsion() {
    const G2 = curve.G2;
    const F2 = G2.F;
    let x = F2.one;
    let rhs = F2.add(F2.mul(F2.square(x), x), G2.b);
    while (!F2.isSquare(rhs)) {
        x = F2.add(x, F2.one);
        rhs = F2.add(F2.mul(F2.square(x), x), G2.b);
    }
    return [F2.toObject(x).map(String), F2.toObject(F2.sqrt(rhs)).map(String)];
}

test("every vkey the verifier refuses, with the reason", () => {
    const ev = (type, id, prime, openingPos) => ({ type, id, prime, openingPos });
    const refusals = {
        "another protocol": [(v) => (v.protocol = "fflonk"), /protocol "fflonk" is not pilfflonk/],
        "another curve": [(v) => (v.curve = "bls12381"), /curve "bls12381" is not bn128/],
        "another format version": [(v) => (v.formatVersion = 2), /formatVersion 2 is not supported/],
        "an unknown field": [(v) => (v.extra = 1), /unknown fields extra/],
        "f01 for f1": [(v) => (v.f01 = ["1", "2"]), /unknown fields f01/],
        "no qVerifier": [(v) => delete v.qVerifier, /no qVerifier/],
        "no fixed commitment": [(v) => delete v.f0, /fixed commitments are none/],
        "a fixed commitment too many": [(v) => (v.f1 = ["1", "2"]), /fixed commitments are f0, f1/],
        "a fixed commitment off the curve": [(v) => (v.f0 = ["1", "3"]), /not on the curve/],
        "a fixed commitment of three coordinates": [(v) => (v.f0 = ["1", "2", "1"]), /f0 is not a point/],
        "a coordinate with a leading zero": [(v) => (v.f0 = ["01", "2"]), /without sign or leading zeros/],
        "a coordinate as a number": [(v) => (v.f0 = [1, "2"]), /without sign or leading zeros/],
        "a coordinate not below q": [(v) => (v.f0 = [curve.q.toString(), "2"]), /not below q/],
        "X_2 off the twist": [(v) => (v.X_2 = [G2_GENERATOR[0], [G2_GENERATOR[1][0], "1"]]), /not on the twist/],
        "X_2 outside the r-torsion": [(v) => (v.X_2 = twistPointOutsideTorsion()), /not in the r-torsion group/],
        "X_2 at infinity": [
            (v) =>
                (v.X_2 = [
                    ["0", "0"],
                    ["0", "0"],
                ]),
            /the point at infinity/,
        ],
        "power above 28": [(v) => (v.power = 29), /power 29 is above 28/],
        "a power that is not an integer": [(v) => (v.power = 3.5), /power = 3.5 is not a non-negative integer/],
        "powerW that is not the lcm of the k": [(v) => (v.powerW = 1), /powerW = 1 is not 2/],
        "k that is not the number of polynomials": [(v) => (v.layout[2].k = 1), /f2 has k = 1 and 2 polynomials/],
        "k such that kN does not divide r - 1": [
            (v) => {
                v.layout[2].pols.push({ id: 4, name: "d" }, { id: 5, name: "e" }, { id: 6, name: "g" });
                v.layout[2].k = 5;
                v.evMap.push(ev("cm", 4, 0, 1), ev("cm", 5, 0, 1), ev("cm", 6, 0, 1));
            },
            /kN = 5·2\^3 does not divide r - 1/,
        ],
        "offsets that do not increase": [(v) => (v.layout[1].offsets = [0, -1]), /offsets \[0,-1\]/],
        "Q opened at another point": [
            (v) => {
                v.layout[3].offsets = [0, 1];
            },
            /holds Q, which is opened at ξ only/,
        ],
        "a stage that goes down": [(v) => (v.layout[2].stage = 0), /f2 is of stage 0 after one of stage 1/],
        "a column in two f": [(v) => (v.layout[2].pols[1].id = 1), /cm 1 is in two f/],
        "an evaluation no f opens": [(v) => v.evMap.push(ev("cm", 1, 1, 2)), /evMap has cm 1 at offset 1, which no f/],
        "an opening with no evaluation": [
            (v) => v.evMap.pop(),
            /layout opens cm 2 at offset 0, which the evMap does not have/,
        ],
        "an evaluation twice": [(v) => v.evMap.push(ev("cm", 2, 0, 1)), /evMap has cm 2 at offset 0 twice/],
        "an evaluation of Q": [(v) => v.evMap.push(ev("cm", 3, 0, 1)), /no f of the layout opens/],
        "an evMap entry of another type": [(v) => (v.evMap[0].type = "custom"), /of type "custom", not cm or const/],
        "an evMap entry with more fields": [
            (v) => (v.evMap[0].commitId = 0),
            /evMap\[0\] has the unknown fields commitId/,
        ],
        "a layout entry with fewer fields": [(v) => delete v.layout[0].degree, /layout\[0\] has no degree/],
        "a layout without Q": [(v) => (v.layout = v.layout.slice(0, 1)), /no f for Q/],
        "Q of another number of pieces": [
            (v) => (v.maxQDegree = 1),
            /the layout packs 1 polynomials of Q, and Q is made of 2/,
        ],
        "a maxQDegree that does not split Q": [
            (v) => (v.maxQDegree = 2),
            /maxQDegree is 2 and qDeg 2: Q is not split, and then maxQDegree is 0/,
        ],
        "challenges for no stage": [(v) => (v.numChallenges = [0, 1]), /numChallenges has 2 stages, and the layout 1/],
        "challenges of stage 1": [(v) => (v.numChallenges = [1]), /stage 1 has challenges, which A.4 never squeezes/],
        "no boundary": [(v) => (v.boundaries = []), /boundaries\[0\] is not everyRow/],
        "a first boundary that is not everyRow": [
            (v) => v.boundaries.unshift({ name: "lastRow" }),
            /boundaries\[0\] is not everyRow/,
        ],
        "a boundary twice": [(v) => v.boundaries.push({ name: "everyRow" }), /is there twice/],
        "an everyFrame that leaves no row": [
            (v) => v.boundaries.push({ name: "everyFrame", offsetMin: 4, offsetMax: 4 }),
            /no row of 8 is left/,
        ],
        "an everyFrame without offsets": [
            (v) => v.boundaries.push({ name: "everyFrame" }),
            /has no offsetMin, offsetMax/,
        ],
        "a firstRow with offsets": [
            (v) => v.boundaries.push({ name: "firstRow", offsetMin: 1 }),
            /has the unknown fields offsetMin/,
        ],
        "another domain": [(v) => v.boundaries.push({ name: "someRow" }), /"someRow", not a domain of A.1/],
        "two evaluations of the same name": [
            (v) => (v.layout[2].pols[0].name = "aw-1"),
            /two values of the proof are named "aw-1"/,
        ],
        "an evaluation named inv": [(v) => (v.layout[2].pols[1].name = "inv"), /named "inv"/],
        "a digest in capitals": [
            (v) => (v.digest = v.digest.toUpperCase().replace("0X", "0x")),
            /is not "0x" and 64 lowercase/,
        ],
        "a digest too short": [(v) => (v.digest = v.digest.slice(0, 65)), /is not "0x" and 64 lowercase/],
        "a qVerifier that is not code": [(v) => (v.qVerifier = []), /qVerifier: not a code block/],
        "a qVerifier reading a missing evaluation": [
            (v) => (v.qVerifier.code[0].src[0].id = 5),
            /eval 5 is not one of the 5 entries of the evMap/,
        ],
    };
    for (const [name, [change, message]] of Object.entries(refusals)) {
        const vkey = sampleVkey();
        change(vkey);
        assert.throws(() => fromObjectVk(curve, vkey), PilFflonkInputError, name);
        assert.throws(() => fromObjectVk(curve, vkey), message, name);
    }
    assert.throws(() => fromObjectVk(curve, []), /vkey: not an object/);
    assert.throws(() => fromObjectVk(curve, null), /vkey: not an object/);
});

test("a split Q needs pieces named Q0 … Q<m-1>", () => {
    const vkey = syntheticVkey(curve, { maxQDegree: 1, qDeg: 2 });
    vkey.layout[5].pols[0].name = "Q2";
    assert.throws(() => fromObjectVk(curve, vkey), /the pieces of Q are named Q2, Q0, not Q0, Q1/);
});

test("the synthetic vkey's points are the multiples of G it says", () => {
    const vk = fromObjectVk(curve, syntheticVkey(curve));
    const G1 = curve.G1;
    assert.deepEqual(g1ToObject(curve, vk.fixedCommitments[1]), g1ToObject(curve, G1.timesScalar(G1.g, 5)));
    assert.equal(frToObject(curve, vk.qVerifier.code[3].src[1].value), "5");
});
