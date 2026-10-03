// The JSON view of a proof and its publics (pilfflonk/docs/formats.md#proof), read against the vkey
// as pilfflonk/src/proof.rs reads them (ProofJson, ProofNames::from_json; Publics), with the checks
// of steps 1-3 of snarkjs' fflonk verifier (src/fflonk_verify.js): commitments in G1, evaluations
// and publics in Fr.
//
//   {"protocol": "pilfflonk", "curve": "bn128",
//    "polynomials": {name: [x, y, "1"]}, "evaluations": {name: value}}
//
// polynomials holds exactly the commitments of the non-fixed f_i (f<g>, g the global index:
// pilfflonk/docs/protocol.md#global-order), W and Wp; evaluations exactly the evaluations of the
// evMap, the pieces Q_i(ξ) if Q is split, inv and invZh (names.js); every value a decimal string
// without sign or leading zeros. A name that is missing or should not be there, a point off the
// curve or at infinity, or a scalar not below r is a PilFflonkInputError naming it.

import { PilFflonkInputError, decimalFromObject, frFromObject, g1FromObject, show } from "./elements.js";
import { INV, INV_ZH, W, WP } from "./names.js";
import { CURVE, PROTOCOL } from "./vkey.js";

function fail(message) {
    throw new PilFflonkInputError(`proof: ${message}`);
}

function isPlainObject(v) {
    return typeof v === "object" && v !== null && !Array.isArray(v);
}

// An object with exactly the keys `names`.
function checkNames(value, names, what) {
    if (!isPlainObject(value)) fail(`${what} is not an object`);
    const missing = names.filter((name) => !Object.hasOwn(value, name));
    const extra = Object.keys(value).filter((name) => !names.includes(name));
    if (missing.length > 0) fail(`${what} has no ${missing.join(", ")}`);
    if (extra.length > 0) fail(`${what} has ${extra.join(", ")}, which the vkey does not`);
}

function scalar(curve, value, what) {
    return frFromObject(curve, decimalFromObject(value, what), what);
}

// A point as the proof writes it, [x, y, "1"] (proof.rs, SnarkjsG1): never the point at infinity.
function point(curve, value, what) {
    if (!Array.isArray(value) || value.length !== 3 || value[2] !== "1") fail(`${what} is not a point [x, y, "1"]`);
    const x = decimalFromObject(value[0], `${what}.x`);
    const y = decimalFromObject(value[1], `${what}.y`);
    return g1FromObject(curve, [x, y], `proof: ${what}`);
}

// The proof against the vkey vk (vkey.js, fromObjectVk):
//   commitments   [f_i]₁ of the non-fixed f_i, in the global order
//   W, Wp         [W]₁ and [W']₁
//   evaluations   by evMap index
//   qPieces       the pieces Q_i(ξ) of a split Q, in the order of vk.qPieceNames, and [] otherwise
//   inv, invZh
export function fromObjectProof(curve, proofObject, vk) {
    if (!isPlainObject(proofObject)) fail("not an object");
    checkNames(proofObject, ["protocol", "curve", "polynomials", "evaluations"], "the proof");
    if (proofObject.protocol !== PROTOCOL) fail(`protocol ${show(proofObject.protocol)} is not ${PROTOCOL}`);
    if (proofObject.curve !== CURVE) fail(`curve ${show(proofObject.curve)} is not ${CURVE}`);

    const { polynomials, evaluations } = proofObject;
    checkNames(polynomials, [...vk.commitmentNames, W, WP], "polynomials");
    const scalarNames = [...vk.evMap.map((e) => e.name), ...vk.qPieceNames, INV, INV_ZH];
    checkNames(evaluations, scalarNames, "evaluations");

    return {
        commitments: vk.commitmentNames.map((name) => point(curve, polynomials[name], name)),
        W: point(curve, polynomials[W], W),
        Wp: point(curve, polynomials[WP], WP),
        evaluations: vk.evMap.map((e) => scalar(curve, evaluations[e.name], e.name)),
        qPieces: vk.qPieceNames.map((name) => scalar(curve, evaluations[name], name)),
        inv: scalar(curve, evaluations[INV], INV),
        invZh: scalar(curve, evaluations[INV_ZH], INV_ZH),
    };
}

// publics.json: nPublic decimal strings below r, in the order of the publicsMap
// (pilfflonk/docs/formats.md#publics).
export function fromObjectPublics(curve, publicsObject, vk) {
    if (!Array.isArray(publicsObject)) throw new PilFflonkInputError("publics: not an array");
    if (publicsObject.length !== vk.nPublic) {
        throw new PilFflonkInputError(`publics: ${publicsObject.length} of them, and the vkey has ${vk.nPublic}`);
    }
    return publicsObject.map((p, i) => scalar(curve, p, `publics[${i}]`));
}
