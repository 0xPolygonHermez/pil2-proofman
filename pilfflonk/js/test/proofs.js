// Keys and proofs for the verifier's tests, independent of the prover (M18):
// - sampleVkey(): the vkey of M15's pinned digest vector (setup/pilfflonk/tests/setup/digest.rs);
// - syntheticVkey(): a vkey of two stages, packing, signed offsets, every domain of A.1 and every
//   operand of the qVerifier;
// - forgeProof(): a proof that verifies against a vkey whose [τ]₂ is [1]₂, the vkeys of the ptau
//   with τ = 1 of the tests (plan N13). Knowing τ, anyone can open anything: random commitments and
//   evaluations, the challenges they give, W' = (F − E − J)/(1 − y), which makes the pairing of
//   A.5 hold, e(F − E − J + y·W', [1]₂) = e(W', [1]₂), and the inv those give. It is not a prover
//   and says nothing about the prover's agreement with the verifier (M18's and M20's end-to-end
//   tests do): it runs the whole verifier on a proof it accepts, so that each change to it can be
//   shown to be rejected;
// - RecordingTranscript and recordingLogger, to observe what the verifier does.

import { keccak_256 } from "@noble/hashes/sha3";
import { bytesToHex } from "@noble/hashes/utils";

import { computeChallenges } from "../src/challenges.js";
import { frToObject, g1ToObject } from "../src/elements.js";
import { INV, INV_ZH, W, WP } from "../src/names.js";
import { fromObjectProof, fromObjectPublics } from "../src/proof.js";
import {
    computeE,
    computeF,
    computeInverseDenominators,
    computeJ,
    computeQuotients,
    computeR,
    computeRoots,
    computeZerofiers,
} from "../src/shplonk.js";
import { Keccak256Transcript } from "../src/transcript.js";
import { computeQ, openingEvaluations } from "../src/verify.js";
import { fromObjectVk, vkeyDigest } from "../src/vkey.js";

// [1]₂: the X_2 of a ptau with τ = 1 (setup/pilfflonk/tests/setup/common.rs, G2_GENERATOR).
export const G2_GENERATOR = [
    [
        "10857046999023057135944570762232829481370756359578518086990519993285655852781",
        "11559732032986387107991004021392285783925812861821192530917403151452391805634",
    ],
    [
        "8495653923123431417604973247489272438418190587263600148770280649306958101930",
        "4082367875863433681332203403145435568316851327593401208105741076214120093531",
    ],
];

// SAMPLE_DIGEST of setup/pilfflonk/tests/setup/digest.rs.
export const SAMPLE_DIGEST = "0x0d0a871547df85cf3f811d77135800ef101a2538ac40e315441369ee3ae0eb42";

// sample_vkey() of setup/pilfflonk/tests/setup/digest.rs as its file has it: an AIR of 8 rows, a
// fixed f, two of stage 1 (one packing two columns) and Q's.
export function sampleVkey() {
    const ev = (type, id, prime, openingPos) => ({ type, id, prime, openingPos });
    const f = (stage, pols, offsets, degree) => ({
        stage,
        pols: pols.map(([id, name]) => ({ id, name })),
        k: pols.length,
        offsets,
        degree,
    });
    return {
        protocol: "pilfflonk",
        curve: "bn128",
        formatVersion: 1,
        nPublic: 2,
        power: 3,
        powerW: 2,
        X_2: G2_GENERATOR,
        numChallenges: [0],
        evMap: [ev("const", 0, 0, 1), ev("cm", 0, -1, 0), ev("cm", 0, 0, 1), ev("cm", 1, 0, 1), ev("cm", 2, 0, 1)],
        layout: [
            f(0, [[0, "L1"]], [0], 8),
            f(1, [[0, "a"]], [-1, 0], 11),
            f(
                1,
                [
                    [1, "b"],
                    [2, "c"],
                ],
                [0],
                21,
            ),
            f(2, [[3, "Q"]], [0], 21),
        ],
        boundaries: [{ name: "everyRow" }],
        f0: ["1", "2"],
        qDeg: 2,
        maxQDegree: 0,
        qVerifier: {
            tmpUsed: 1,
            code: [{ op: "copy", dest: { type: "tmp", id: 0, dim: 1 }, src: [{ type: "eval", id: 1, dim: 1 }] }],
        },
        digest: SAMPLE_DIGEST,
    };
}

// A deterministic stream of elements of Fr and of multiples of G1's generator.
export function randomness(curve, seed) {
    let n = 0;
    const fr = () => {
        const bytes = keccak_256(new TextEncoder().encode(`${seed}:${n++}`));
        return curve.Fr.e(BigInt(`0x${bytesToHex(bytes)}`) % curve.r);
    };
    const g1 = () => curve.G1.toAffine(curve.G1.timesFr(curve.G1.g, fr()));
    return { fr, g1 };
}

const operand = (type, fields) => ({ type, ...fields, dim: 1 });
const tmp = (id) => operand("tmp", { id });
const evalOf = (id) => operand("eval", { id });

// A vkey of 8 rows and two stages, sealed with its digest. X_2 is [1]₂ (τ = 1), the fixed
// commitments are 3·G and 5·G.
//   f0  K (const)        offsets {0, 1}
//   f1  P (const)        offsets {0}
//   f2  a, b (k = 2)     offsets {-1, 0}   stage 1
//   f3  c                offsets {0}       stage 1
//   f4  d                offsets {0, 2}    stage 2, after one stage-2 challenge
//   f5  Q0               offsets {0}       Q, stage 3
// The evMap is in pil-info's order (by opening point, const before cm, by id), with K at offset 1
// after the committed columns at offset 0: the proof and the transcript list the fixed ones first.
// The qVerifier reads every kind of operand.
export function syntheticVkey(curve, { maxQDegree = 0, qDeg = 1 } = {}) {
    const ev = (type, id, prime, openingPos) => ({ type, id, prime, openingPos });
    const point = (m) => g1ToObject(curve, curve.G1.timesScalar(curve.G1.g, m));
    const split = maxQDegree > 0 && qDeg > maxQDegree;
    const qPols = split
        ? [
              { id: 5, name: "Q1" },
              { id: 4, name: "Q0" },
          ]
        : [{ id: 4, name: "Q0" }];
    const vkey = {
        protocol: "pilfflonk",
        curve: "bn128",
        formatVersion: 1,
        nPublic: 2,
        power: 3,
        powerW: 2,
        X_2: G2_GENERATOR,
        numChallenges: [0, 1],
        evMap: [
            ev("cm", 0, -1, 0),
            ev("cm", 1, -1, 0),
            ev("const", 0, 0, 1),
            ev("const", 1, 0, 1),
            ev("cm", 0, 0, 1),
            ev("cm", 1, 0, 1),
            ev("cm", 2, 0, 1),
            ev("cm", 3, 0, 1),
            ev("const", 0, 1, 2),
            ev("cm", 3, 2, 3),
        ],
        layout: [
            { stage: 0, pols: [{ id: 0, name: "Syn.K" }], k: 1, offsets: [0, 1], degree: 8 },
            { stage: 0, pols: [{ id: 1, name: "Syn.P" }], k: 1, offsets: [0], degree: 8 },
            {
                stage: 1,
                pols: [
                    { id: 0, name: "a" },
                    { id: 1, name: "b" },
                ],
                k: 2,
                offsets: [-1, 0],
                degree: 23,
            },
            { stage: 1, pols: [{ id: 2, name: "c" }], k: 1, offsets: [0], degree: 10 },
            { stage: 2, pols: [{ id: 3, name: "d" }], k: 1, offsets: [0, 2], degree: 11 },
            { stage: 3, pols: qPols, k: qPols.length, offsets: [0], degree: 24 },
        ],
        boundaries: [
            { name: "everyRow" },
            { name: "firstRow" },
            { name: "lastRow" },
            { name: "everyFrame", offsetMin: 1, offsetMax: 2 },
        ],
        f0: point(3),
        f1: point(5),
        qDeg,
        maxQDegree,
        qVerifier: {
            tmpUsed: 12,
            code: [
                { op: "mul", dest: tmp(0), src: [evalOf(4), evalOf(0)] },
                { op: "sub", dest: tmp(1), src: [tmp(0), operand("public", { id: 1 })] },
                { op: "mul", dest: tmp(2), src: [tmp(1), operand("challenge", { id: 0, stage: 2, stageId: 0 })] },
                { op: "add", dest: tmp(3), src: [tmp(2), operand("number", { value: "5" })] },
                { op: "copy", dest: tmp(4), src: [evalOf(9)] },
                { op: "mul", dest: tmp(5), src: [tmp(3), operand("Zi", { boundaryId: 1 })] },
                { op: "mul", dest: tmp(6), src: [tmp(4), operand("Zi", { boundaryId: 2 })] },
                { op: "add", dest: tmp(7), src: [tmp(5), tmp(6)] },
                { op: "mul", dest: tmp(8), src: [tmp(7), operand("challenge", { id: 1, stage: 3, stageId: 0 })] },
                { op: "mul", dest: tmp(9), src: [evalOf(8), operand("Zi", { boundaryId: 3 })] },
                { op: "add", dest: tmp(10), src: [tmp(8), tmp(9)] },
                { op: "mul", dest: tmp(11), src: [tmp(10), operand("Zi", { boundaryId: 0 })] },
            ],
            line: "",
        },
        digest: "",
    };
    return seal(vkey);
}

// The vkey with its digest set.
export function seal(vkey) {
    return { ...vkey, digest: vkeyDigest(vkey) };
}

export function randomPublics(curve, vkey, seed = "publics") {
    const rand = randomness(curve, seed);
    return Array.from({ length: vkey.nPublic }, () => frToObject(curve, rand.fr()));
}

// A proof that verifies against `vkeyObject` if its X_2 is [1]₂ (see the module): proof.json and
// what went into it.
export function forgeProof(curve, vkeyObject, publicsObject, seed = "proof") {
    const Fr = curve.Fr;
    const G1 = curve.G1;
    const rand = randomness(curve, seed);
    const vk = fromObjectVk(curve, vkeyObject);
    const publics = fromObjectPublics(curve, publicsObject, vk);
    const pointObject = (p) => [...g1ToObject(curve, p), "1"];

    const polynomials = {};
    vk.commitmentNames.forEach((name) => {
        polynomials[name] = pointObject(rand.g1());
    });
    polynomials[W] = pointObject(rand.g1());
    polynomials[WP] = pointObject(G1.g);
    const evaluations = {};
    for (const name of [...vk.evMap.map((e) => e.name), ...vk.qPieceNames, INV, INV_ZH]) {
        evaluations[name] = frToObject(curve, rand.fr());
    }
    const draft = { protocol: "pilfflonk", curve: "bn128", polynomials, evaluations };

    // ξ and Q(ξ) do not depend on what is absorbed after xiSeed: the pieces of a split Q can be
    // set to add up to Q(ξ), Q_0 = Q(ξ) − Σ_{i≥1} ξ^(i·M·N)·Q_i(ξ), before α_S and y are replayed.
    let proof = fromObjectProof(curve, draft, vk);
    let challenges = computeChallenges(curve, vk, publics, proof);
    const xi = Fr.exp(challenges.xiSeed, BigInt(vk.powerW));
    evaluations[INV_ZH] = frToObject(curve, Fr.inv(Fr.sub(Fr.exp(xi, BigInt(vk.N)), Fr.one)));
    const { q } = computeQ(curve, vk, publics, proof, challenges, xi);
    if (vk.qPieces > 1) {
        const shift = Fr.exp(xi, BigInt(vk.maxQDegree) * BigInt(vk.N));
        let q0 = q;
        for (let i = 1; i < vk.qPieces; i++) {
            const qi = Fr.e(BigInt(evaluations[`Q${i}`]));
            q0 = Fr.sub(q0, Fr.mul(Fr.exp(shift, BigInt(i)), qi));
        }
        evaluations.Q0 = frToObject(curve, q0);
        proof = fromObjectProof(curve, draft, vk);
        challenges = computeChallenges(curve, vk, publics, proof);
    }

    const f = vk.layout.map(({ k, offsets }) => ({ k, offsets }));
    const layout = { nBits: vk.power, powerW: vk.powerW, f };
    const { f: roots } = computeRoots(curve, layout, challenges.xiSeed);
    const zerofiers = computeZerofiers(curve, roots, challenges.y);
    const r = computeR(curve, f, roots, openingEvaluations(vk, proof, q), zerofiers, challenges.y);
    const quotients = computeQuotients(curve, zerofiers, challenges.alpha);
    const F = computeF(curve, [...vk.fixedCommitments, ...proof.commitments], quotients);
    const E = computeE(curve, r, quotients);
    const J = computeJ(curve, proof.W, quotients[0]);
    const FEJ = G1.sub(G1.sub(F, E), J);
    const Wp = G1.timesFr(FEJ, Fr.inv(Fr.sub(Fr.one, challenges.y)));
    polynomials[WP] = pointObject(Wp);
    // inv, which nothing absorbs: the inverse of the SHPLONK check's denominators (A.6).
    const denominators = computeInverseDenominators(curve, roots, zerofiers, challenges.y);
    evaluations[INV] = frToObject(curve, Fr.inv(denominators.reduce((acc, d) => Fr.mul(acc, d), Fr.one)));
    return draft;
}

// A Keccak256Transcript that records what is absorbed and squeezed, as ["fr", decimal],
// ["g1", [x, y]] and ["squeeze", decimal].
export class RecordingTranscript extends Keccak256Transcript {
    constructor(curve) {
        super(curve);
        this.curve = curve;
        this.log = [];
        this.squeezing = false;
    }

    addScalar(scalar) {
        if (!this.squeezing) this.log.push(["fr", this.Fr.toString(scalar)]);
        super.addScalar(scalar);
    }

    addPolCommitment(point) {
        this.log.push(["g1", g1ToObject(this.curve, point)]);
        super.addPolCommitment(point);
    }

    squeeze() {
        this.squeezing = true;
        try {
            const challenge = super.squeeze();
            this.log.push(["squeeze", this.Fr.toString(challenge)]);
            return challenge;
        } finally {
            this.squeezing = false;
        }
    }
}

// A logger that keeps every message, by level.
export function recordingLogger() {
    const messages = [];
    const at = (level) => (message) => messages.push({ level, message });
    return { messages, debug: at("debug"), info: at("info"), warn: at("warn"), error: at("error") };
}

// The errors a logger received.
export function errorsOf(logger) {
    return logger.messages.filter((m) => m.level === "error").map((m) => m.message);
}
