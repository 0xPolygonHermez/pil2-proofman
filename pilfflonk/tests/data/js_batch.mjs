// The JS verifier on a batch of cases, for the differential fuzzer of the Solidity verifier
// (pilfflonk/docs/verifier.md#differential-fuzzer; pilfflonk/tests/data/fuzz.rs): one Node process
// for many cases, each verified by verify() of pilfflonk/js/src/verify.js, the reference, as
// bin/verify.js verifies one, from its parsed JSON:
//
//     node pilfflonk/tests/data/js_batch.mjs <pilfflonk.vkey.json> <requests.jsonl> <responses.jsonl>
//
// Each line of the requests is one of:
// - {"op": "verify", "proof": <proof.json>, "publics": <publics.json>}: the response is
//   {"verdict": true | false, "messages": [...]}, the messages the verifier logged as errors and
//   warnings, in order (why it rejected the proof), or {"threw": message} if verify() threw, which it
//   does only on a failure of its own (bin/verify.js would exit with 1, as for a rejection);
// - {"op": "forgeWp", "proof": ..., "publics": ...}: the W' = y^(-1)·(E + J - F) that cancels the
//   left side of the pairing (shplonk.js, isValidPairing: F - E - J + y·W' = 0), with F, E and J as
//   the verifier computes them for this proof, whose W' it ignores: {"Wp": [x, y]}, affine, in
//   decimal. The pairing check is then e(0, [1]_2) = e(W', [x]_2), which holds only if W' is the
//   point at infinity (F = E + J) or [x]_2 is: an X_2 that a vkey could otherwise have
//   (pilfflonk/docs/verifier.md#refused-vkeys), which Vkey::validate refuses. A proof with this W'
//   must fail at the pairing against an honest vkey.
//
// Every value of the proof's JSON is a decimal string, as proof.json writes them, whatever it is: a
// mutated value need not be below r or q, and the verifier checks it.

import { readFileSync, writeFileSync } from "node:fs";
import { argv, exit, stderr } from "node:process";

import { computeChallenges } from "../../js/src/challenges.js";
import { g1ToObject } from "../../js/src/elements.js";
import { fromObjectProof, fromObjectPublics } from "../../js/src/proof.js";
import {
    computeE,
    computeF,
    computeJ,
    computeQuotients,
    computeR,
    computeRoots,
    computeZerofiers,
} from "../../js/src/shplonk.js";
import { computeQ, getCurve, openingEvaluations, verify } from "../../js/src/verify.js";
import { fromObjectVk } from "../../js/src/vkey.js";

const args = argv.slice(2);
if (args.length !== 3) {
    stderr.write("usage: js_batch.mjs <pilfflonk.vkey.json> <requests.jsonl> <responses.jsonl>\n");
    exit(2);
}
const [vkeyPath, requestsPath, responsesPath] = args;
const vkeyText = readFileSync(vkeyPath, "utf8");

async function verifyCase(request) {
    const messages = [];
    const logger = {
        debug: () => {},
        info: () => {},
        warn: (message) => messages.push(message),
        error: (message) => messages.push(message),
    };
    try {
        // Each case its own parsed vkey, as bin/verify.js parses it for each proof.
        const verdict = await verify(JSON.parse(vkeyText), request.publics, request.proof, logger);
        return { verdict, messages };
    } catch (e) {
        return { threw: String(e?.message ?? e) };
    }
}

async function forgeWp(request) {
    const curve = await getCurve();
    const { Fr, G1 } = curve;
    const vk = fromObjectVk(curve, JSON.parse(vkeyText));
    const publics = fromObjectPublics(curve, request.publics, vk);
    const proof = fromObjectProof(curve, request.proof, vk);
    const challenges = computeChallenges(curve, vk, publics, proof);
    const xi = Fr.exp(challenges.xiSeed, BigInt(vk.powerW));
    const { q } = computeQ(curve, vk, publics, proof, challenges, xi);
    const f = vk.layout.map(({ k, offsets }) => ({ k, offsets }));
    const { f: roots } = computeRoots(curve, { nBits: vk.power, powerW: vk.powerW, f }, challenges.xiSeed);
    const zerofiers = computeZerofiers(curve, roots, challenges.y);
    const r = computeR(curve, f, roots, openingEvaluations(vk, proof, q), zerofiers, challenges.y);
    const quotients = computeQuotients(curve, zerofiers, challenges.alpha);
    const F = computeF(curve, [...vk.fixedCommitments, ...proof.commitments], quotients);
    const E = computeE(curve, r, quotients);
    const J = computeJ(curve, proof.W, quotients[0]);
    // In Jacobian coordinates: ffjavascript's G1.sub of an affine point and a Jacobian one returns
    // their difference the other way round (pilfflonk/docs/README.md#rapidsnark-and-ffiasm).
    const sum = G1.sub(G1.add(G1.toJacobian(E), G1.toJacobian(J)), G1.toJacobian(F));
    return { Wp: g1ToObject(curve, G1.timesFr(sum, Fr.inv(challenges.y))) };
}

const responses = [];
for (const line of readFileSync(requestsPath, "utf8").split("\n")) {
    if (line.trim() === "") continue;
    const request = JSON.parse(line);
    if (request.op === "verify") {
        responses.push(JSON.stringify(await verifyCase(request)));
    } else if (request.op === "forgeWp") {
        responses.push(JSON.stringify(await forgeWp(request)));
    } else {
        stderr.write(`js_batch.mjs: a request of op ${JSON.stringify(request.op)}\n`);
        exit(2);
    }
}
writeFileSync(responsesPath, responses.map((r) => `${r}\n`).join(""));
exit(0);
