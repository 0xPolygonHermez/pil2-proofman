// verify(vkey, publics, proof, logger) and bin/verify.js on the synthetic vkeys (proofs.js), whose
// τ is 1: a forged proof verifies, and each change to it, to the publics or to the vkey is
// rejected -- the rejections M19 asks for, independently of the prover (whose proofs
// cli/tests/pilfflonk_prove.rs verifies, M18). Every malformed input gives
// false with a logged reason (verify.js), and the CLI exits with 0 only on a proof that verifies.

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { before, test } from "node:test";

import { g1ToObject } from "../src/elements.js";
import { verify } from "../src/verify.js";
import {
    errorsOf,
    forgeProof,
    randomPublics,
    randomness,
    recordingLogger,
    sampleVkey,
    seal,
    syntheticVkey,
} from "./proofs.js";
import { newCurve } from "./support.js";

const VERIFY_BIN = join(dirname(fileURLToPath(import.meta.url)), "..", "bin", "verify.js");

let curve;
let rand;
before(async () => {
    curve = await newCurve();
    rand = randomness(curve, "verify.test");
});

const plusOne = (v) => ((BigInt(v) + 1n) % curve.r).toString();
const anotherPoint = () => [...g1ToObject(curve, rand.g1()), "1"];

// verify, and the reasons it logged.
async function verdict(vkey, publics, proof) {
    const logger = recordingLogger();
    const result = await verify(vkey, publics, proof, logger);
    return { result, errors: errorsOf(logger), logger };
}

function forged(vkey) {
    const publics = randomPublics(curve, vkey);
    return { vkey, publics, proof: forgeProof(curve, vkey, publics) };
}

test("a forged proof verifies against each synthetic vkey", async () => {
    for (const vkey of [sampleVkey(), syntheticVkey(curve), syntheticVkey(curve, { maxQDegree: 1, qDeg: 2 })]) {
        const { publics, proof } = forged(vkey);
        const { result, errors, logger } = await verdict(vkey, publics, proof);
        assert.deepEqual(errors, []);
        assert.equal(result, true);
        assert.ok(logger.messages.some((m) => m.message === "PROOF VERIFIED SUCCESSFULLY"));
    }
});

test("it rejects the proof if any commitment, evaluation, W or W' changes", async () => {
    for (const vkey of [syntheticVkey(curve), syntheticVkey(curve, { maxQDegree: 1, qDeg: 2 })]) {
        const { publics, proof } = forged(vkey);
        for (const name of Object.keys(proof.polynomials)) {
            const changed = structuredClone(proof);
            changed.polynomials[name] = anotherPoint();
            assert.equal((await verdict(vkey, publics, changed)).result, false, name);
        }
        for (const name of Object.keys(proof.evaluations)) {
            const changed = structuredClone(proof);
            changed.evaluations[name] = plusOne(changed.evaluations[name]);
            assert.equal((await verdict(vkey, publics, changed)).result, false, name);
        }
    }
});

test("invZh must be 1/Z_H(ξ), and the pieces of a split Q add up to Q(ξ)", async () => {
    const { vkey, publics, proof } = forged(syntheticVkey(curve, { maxQDegree: 1, qDeg: 2 }));
    const changed = structuredClone(proof);
    changed.evaluations.invZh = plusOne(proof.evaluations.invZh);
    assert.deepEqual((await verdict(vkey, publics, changed)).errors, ["invZh is not 1/Z_H(ξ)"]);
    // Q0 changed: the transcript and the sum change; the sum is checked first.
    changed.evaluations.invZh = proof.evaluations.invZh;
    changed.evaluations.Q0 = plusOne(proof.evaluations.Q0);
    assert.deepEqual((await verdict(vkey, publics, changed)).errors, ["The pieces of Q do not add up to Q(ξ) (A.1)"]);
});

test("inv must be the inverse of the SHPLONK check's denominators: off by one or by any bit, it is rejected", async () => {
    for (const vkey of [sampleVkey(), syntheticVkey(curve), syntheticVkey(curve, { maxQDegree: 1, qDeg: 2 })]) {
        const { publics, proof } = forged(vkey);
        const changed = structuredClone(proof);
        changed.evaluations.inv = plusOne(proof.evaluations.inv);
        assert.deepEqual((await verdict(vkey, publics, changed)).errors, [
            "inv is not the inverse of the SHPLONK check's denominators (A.6)",
        ]);
    }
    // Every bit of the 254 of a scalar flipped: another inverse, or a value not below r.
    const { vkey, publics, proof } = forged(syntheticVkey(curve));
    const inv = BigInt(proof.evaluations.inv);
    for (let bit = 0n; bit < 254n; bit++) {
        const changed = structuredClone(proof);
        changed.evaluations.inv = (inv ^ (1n << bit)).toString();
        const { result, errors } = await verdict(vkey, publics, changed);
        assert.equal(result, false, `bit ${bit}`);
        assert.equal(errors.length, 1, `bit ${bit}`);
        assert.match(errors[0], /inv is not the inverse|inv is not an element of F|not below r/, `bit ${bit}`);
    }
});

test("it rejects the proof if any public changes", async () => {
    const { vkey, publics, proof } = forged(syntheticVkey(curve));
    for (let i = 0; i < publics.length; i++) {
        const changed = [...publics];
        changed[i] = plusOne(changed[i]);
        assert.equal((await verdict(vkey, changed, proof)).result, false, `publics[${i}]`);
    }
});

test("it rejects the proof if the vkey changes, sealed again or not", async () => {
    const { vkey, publics, proof } = forged(syntheticVkey(curve));
    const G1 = curve.G1;
    const changes = {
        "a fixed commitment": (v) => (v.f1 = g1ToObject(curve, G1.timesScalar(G1.g, 6))),
        "a constant of the qVerifier": (v) => (v.qVerifier.code[3].src[1].value = "6"),
        "a degree of the layout": (v) => (v.layout[3].degree = 11),
        "[τ]₂": (v) => (v.X_2 = [v.X_2[1].slice(), v.X_2[0].slice()]),
    };
    for (const [name, change] of Object.entries(changes)) {
        const changed = structuredClone(vkey);
        change(changed);
        const unsealed = await verdict(changed, publics, proof);
        assert.equal(unsealed.result, false, name);
        if (name !== "[τ]₂") {
            assert.deepEqual(unsealed.errors, ["The digest of the vkey is not the digest of its contents (A.6)"], name);
            assert.equal((await verdict(seal(changed), publics, proof)).result, false, `${name}, sealed`);
        }
    }
});

test("every malformed proof, publics or vkey gives false, with the reason", async () => {
    const { vkey, publics, proof } = forged(syntheticVkey(curve));
    const f2 = proof.polynomials.f2;
    const cases = {
        "another protocol": [(p) => (p.protocol = "fflonk"), /proof: protocol "fflonk" is not pilfflonk/],
        "another curve": [(p) => (p.curve = "bls12381"), /proof: curve "bls12381" is not bn128/],
        "a field too many": [(p) => (p.extra = {}), /the proof has extra, which the vkey does not/],
        "no evaluations": [(p) => delete p.evaluations, /the proof has no evaluations/],
        "a missing evaluation": [(p) => delete p.evaluations.a, /evaluations has no a/],
        "an evaluation too many": [(p) => (p.evaluations.aw = "1"), /evaluations has aw, which the vkey does not/],
        "a missing commitment": [(p) => delete p.polynomials.f3, /polynomials has no f3/],
        "a fixed commitment in the proof": [
            (p) => (p.polynomials.f0 = f2),
            /polynomials has f0, which the vkey does not/,
        ],
        "a point off the curve": [
            (p) => (p.polynomials.f2 = [f2[0], plusOne(f2[1]), "1"]),
            /f2: .* is not on the curve/,
        ],
        "the point at infinity": [(p) => (p.polynomials.W = ["0", "0", "1"]), /W: the point at infinity/],
        "a point with z = 2": [(p) => (p.polynomials.f2 = [f2[0], f2[1], "2"]), /f2 is not a point \[x, y, "1"\]/],
        "a point without z": [(p) => (p.polynomials.f2 = [f2[0], f2[1]]), /f2 is not a point \[x, y, "1"\]/],
        "a coordinate not below q": [
            (p) => (p.polynomials.f2 = [(BigInt(f2[0]) + curve.q).toString(), f2[1], "1"]),
            /f2.x: .* is not below q/,
        ],
        "a point the transcript never absorbs": [
            (p) => (p.polynomials.f2 = ["1", "2", "1"]),
            /coordinate below 2\^192/,
        ],
        "a scalar not below r": [(p) => (p.evaluations.c = curve.r.toString()), /c: .* is not below r/],
        "a scalar with a leading zero": [
            (p) => (p.evaluations.c = `0${p.evaluations.c}`),
            /c: .* without sign or leading zeros/,
        ],
        "a scalar as a number": [(p) => (p.evaluations.c = 5), /c: 5 is not a decimal string/],
        "a scalar as a bigint, as unstringifyBigInts leaves it": [
            (p) => (p.evaluations.c = 5n),
            /c: "5n" is not a decimal/,
        ],
    };
    for (const [name, [change, message]] of Object.entries(cases)) {
        const changed = structuredClone(proof);
        change(changed);
        const { result, errors } = await verdict(vkey, publics, changed);
        assert.equal(result, false, name);
        assert.equal(errors.length, 1, `${name}: ${errors}`);
        assert.match(errors[0], message, name);
    }

    const badPublics = {
        "one public too few": [[publics[0]], /publics: 1 of them, and the vkey has 2/],
        "a public not below r": [[publics[0], curve.r.toString()], /publics\[1\]: .* is not below r/],
        "a public as a number": [[publics[0], 1], /publics\[1\]: 1 is not a decimal string/],
        "publics that are not an array": [{ 0: publics[0] }, /publics: not an array/],
    };
    for (const [name, [value, message]] of Object.entries(badPublics)) {
        const { result, errors } = await verdict(vkey, value, proof);
        assert.equal(result, false, name);
        assert.match(errors.join("\n"), message, name);
    }

    const notAnObject = await verdict(vkey, publics, "proof");
    assert.equal(notAnObject.result, false);
    assert.deepEqual(notAnObject.errors, ["proof: not an object"]);

    for (const [name, value, message] of [
        ["a vkey that is not an object", "vkey", /vkey: not an object/],
        ["a vkey of another protocol", { ...vkey, protocol: "fflonk" }, /vkey: protocol "fflonk"/],
        ["a vkey with a bigint", { ...vkey, nPublic: 2n }, /vkey: nPublic = "2n" is not a non-negative integer/],
    ]) {
        const { result, errors } = await verdict(value, publics, proof);
        assert.equal(result, false, name);
        assert.match(errors.join("\n"), message, name);
    }
});

test("bin/verify.js exits with 0 only on a proof that verifies", async (t) => {
    const dir = mkdtempSync(join(tmpdir(), "pilfflonk-verify-"));
    t.after(() => rmSync(dir, { recursive: true, force: true }));
    const { vkey, publics, proof } = forged(syntheticVkey(curve));
    const file = (name, value) => {
        const path = join(dir, name);
        writeFileSync(path, typeof value === "string" ? value : JSON.stringify(value, null, 1));
        return path;
    };
    const run = (...args) => spawnSync(process.execPath, [VERIFY_BIN, ...args], { encoding: "utf8" });
    const [vk, pub, good] = [file("vkey.json", vkey), file("publics.json", publics), file("proof.json", proof)];
    const tampered = structuredClone(proof);
    tampered.evaluations.a = plusOne(tampered.evaluations.a);
    const bad = file("tampered.json", tampered);

    await t.test("a proof that verifies: 0", () => {
        const out = run(vk, pub, good);
        assert.equal(out.status, 0, out.stderr);
        assert.match(out.stderr, /OK: the proof verifies/);
    });
    await t.test("a tampered proof: 1", () => {
        const out = run(vk, pub, bad);
        assert.equal(out.status, 1, out.stderr);
        assert.match(out.stderr, /INVALID/);
    });
    await t.test("a malformed proof: 1", () => {
        const out = run(vk, pub, file("malformed.json", { protocol: "pilfflonk" }));
        assert.equal(out.status, 1, out.stderr);
        assert.match(out.stderr, /\[ERROR\] proof: the proof has no/);
    });
    await t.test("wrong arguments or a file that is not JSON: 2", () => {
        assert.equal(run(vk, pub).status, 2);
        assert.equal(run(vk, pub, good, good).status, 2);
        assert.equal(run(vk, pub, join(dir, "missing.json")).status, 2);
        assert.equal(run(vk, pub, file("text.json", "not json")).status, 2);
    });
});
