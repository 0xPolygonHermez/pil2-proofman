// Q(ξ) from the evaluations (pilfflonk/docs/verifier.md#steps, step 5): the vkey's qVerifier, the
// code block of <air>.verifierinfo.json in the STARK's format with every dimension 1 (pil-info's
// generate_constraint_polynomial_verifier_code), run over Fr as pil-stark's fflonk verifier runs
// its verifierCode (src/fflonk/helpers/fflonk_verify.js, executeCode).
//
// The code is a list of {op, dest, src}: op one of add, sub (src[0] - src[1]), mul and copy, dest a
// temporary, and the result the value of the last entry's dest. It is the STARK's fold
// (pilfflonk/docs/protocol.md#constraint-polynomial), acc = acc·std_vc + c_i·Zi(D_i) over the
// constraints and then the im pols' im - e, times Zi(everyRow) at the end. The operands it can hold
// in format version 1:
//   tmp        a temporary the code wrote before
//   eval       the evaluation of evMap[id] in the proof
//   public     publics[id]
//   number     a constant, a decimal string below r
//   challenge  the challenge of (stage, stageId): stageId-th of stage s ≤ nStages, std_vc for
//              nStages + 1, xiSeed (std_xi) for nStages + 2
//              (pilfflonk/docs/protocol.md#transcript); its id, the position in the AIR's
//              challengesMap, must agree
//   Zi         Zi(boundaries[boundaryId]) at ξ: 1/Z_H(ξ) for everyRow, which is boundary 0, and
//              Z_H(ξ)/Z_D(ξ) for any other domain D
// Air values, airgroup values and proof values do not exist in v1, nor custom commits
// (pilfflonk/docs/README.md#scope); xDivXSubXi is FRI's. A vkey whose code has any other op or
// operand is refused as malformed.

import { PilFflonkInputError, decimalFromObject, show } from "./elements.js";
import { rootOfUnity } from "./shplonk.js";

const BINARY_OPS = new Set(["add", "sub", "mul"]);

function isPlainObject(v) {
    return typeof v === "object" && v !== null && !Array.isArray(v);
}

function isIndex(v, n) {
    return Number.isSafeInteger(v) && v >= 0 && v < n;
}

// The (stage, stageId) of every challenge of an AIR, in the order of its challengesMap: the
// numChallenges[s - 1] challenges of each stage s = 2 … nStages, then std_vc (nStages + 1) and
// std_xi (nStages + 2), each with stageId 0 (pilfflonk/docs/protocol.md#transcript). Stage 1 has
// none: the transcript squeezes no challenge before the first commitments.
export function challengesMap(numChallenges, nStages) {
    const map = [];
    for (let s = 2; s <= nStages; s++) {
        for (let j = 0; j < numChallenges[s - 1]; j++) map.push({ stage: s, stageId: j });
    }
    map.push({ stage: nStages + 1, stageId: 0 }, { stage: nStages + 2, stageId: 0 });
    return map;
}

// The qVerifier of a vkey, checked and with its constants decoded: every operand one of the
// above, every index in range, every temporary written before it is read, dimension 1.
//   shape: {nEvaluations, nPublic, nBoundaries, challenges: challengesMap(...)}
// Returns {tmpUsed, code: [{op, dest, src: [{type, id | value | stage, stageId}]}]}.
export function checkQVerifier(curve, qVerifier, shape) {
    const fail = (message) => {
        throw new PilFflonkInputError(`qVerifier: ${message}`);
    };
    if (!isPlainObject(qVerifier) || !Array.isArray(qVerifier.code)) fail("not a code block {tmpUsed, code}");
    const { tmpUsed, code } = qVerifier;
    if (!Number.isSafeInteger(tmpUsed) || tmpUsed < 0) fail(`tmpUsed = ${tmpUsed} is not a count`);
    if (code.length === 0) fail("no code: it computes no Q(ξ)");

    const written = new Set();
    const operand = (ref, where) => {
        if (!isPlainObject(ref)) fail(`${where} is not an operand`);
        if (ref.dim !== 1) fail(`${where} has dimension ${ref.dim}; over BN128 every operand has 1`);
        const index = (n, name) => {
            if (!isIndex(ref.id, n)) fail(`${where}: ${ref.type} ${ref.id} is not one of the ${n} ${name}`);
            return { type: ref.type, id: ref.id };
        };
        switch (ref.type) {
            case "tmp":
                if (!written.has(ref.id)) fail(`${where} reads temporary ${ref.id} before it is written`);
                return { type: "tmp", id: ref.id };
            case "eval":
                return index(shape.nEvaluations, "entries of the evMap");
            case "public":
                return index(shape.nPublic, "publics");
            case "number": {
                const v = decimalFromObject(ref.value, `${where}: number`);
                if (v >= curve.r) fail(`${where}: the number ${v} is not below r`);
                return { type: "number", value: curve.Fr.e(v) };
            }
            case "challenge": {
                const at = shape.challenges.findIndex((c) => c.stage === ref.stage && c.stageId === ref.stageId);
                if (at < 0) fail(`${where}: no challenge of stage ${ref.stage} has stageId ${ref.stageId}`);
                if (ref.id !== at) {
                    fail(
                        `${where}: the challenge of stage ${ref.stage}, stageId ${ref.stageId} is ${at}, not ${ref.id}`,
                    );
                }
                return { type: "challenge", stage: ref.stage, stageId: ref.stageId };
            }
            case "Zi":
                if (!isIndex(ref.boundaryId, shape.nBoundaries)) {
                    fail(`${where}: Zi of boundary ${ref.boundaryId}, and there are ${shape.nBoundaries}`);
                }
                return { type: "Zi", id: ref.boundaryId };
            case "airvalue":
            case "airgroupvalue":
            case "proofvalue":
                return fail(`${where}: ${ref.type}s do not exist in format version 1 (pilfflonk/docs/README.md#scope)`);
            default:
                return fail(`${where}: operand type ${show(ref.type)} is not one of the qVerifier's`);
        }
    };

    const checked = code.map((entry, i) => {
        const where = `code[${i}]`;
        if (!isPlainObject(entry) || !Array.isArray(entry.src)) fail(`${where} is not {op, dest, src}`);
        const arity = BINARY_OPS.has(entry.op) ? 2 : entry.op === "copy" ? 1 : -1;
        if (arity < 0) fail(`${where}: op ${show(entry.op)} is not add, sub, mul or copy`);
        if (entry.src.length !== arity) fail(`${where}: ${entry.op} takes ${arity} operands`);
        const src = entry.src.map((ref, j) => operand(ref, `${where}.src[${j}]`));
        const dest = entry.dest;
        if (!isPlainObject(dest) || dest.type !== "tmp" || dest.dim !== 1 || !isIndex(dest.id, tmpUsed)) {
            fail(`${where}: the destination is not a temporary below tmpUsed = ${tmpUsed}`);
        }
        written.add(dest.id);
        return { op: entry.op, dest: dest.id, src };
    });
    return { tmpUsed, code: checked };
}

// Zi(D) at ξ for each boundary D (pilfflonk/docs/protocol.md#constraint-polynomial), N = 2^nBits:
// 1/Z_H(ξ) for everyRow, and Z_H(ξ)/Z_D(ξ) otherwise, in closed form: Z_H(ξ)/(ξ - 1) for firstRow,
// Z_H(ξ)/(ξ - ω^(N-1)) for lastRow, and Π_j (ξ - ω^j) over the rows j an everyFrame excludes, the
// first offsetMin and the last offsetMax. ξ must not be in H: Z_H(ξ) ≠ 0, which also keeps ξ off
// every other domain.
export function computeZi(curve, boundaries, nBits, xi) {
    const Fr = curve.Fr;
    const N = 2 ** nBits;
    const omega = rootOfUnity(curve, BigInt(N));
    const zh = Fr.sub(Fr.exp(xi, BigInt(N)), Fr.one);
    if (Fr.isZero(zh)) {
        throw new PilFflonkInputError("ξ is in H: Z_H(ξ) = 0 and Q(ξ) is not defined");
    }
    const minusRow = (j) => Fr.sub(xi, Fr.exp(omega, BigInt(j)));
    return boundaries.map((b) => {
        switch (b.name) {
            case "everyRow":
                return Fr.inv(zh);
            case "firstRow":
                return Fr.div(zh, minusRow(0));
            case "lastRow":
                return Fr.div(zh, minusRow(N - 1));
            case "everyFrame": {
                let zi = Fr.one;
                for (let j = 0; j < b.offsetMin; j++) zi = Fr.mul(zi, minusRow(j));
                for (let j = N - b.offsetMax; j < N; j++) zi = Fr.mul(zi, minusRow(j));
                return zi;
            }
            default:
                throw new PilFflonkInputError(
                    `boundary ${show(b.name)} is not a domain (pilfflonk/docs/protocol.md#constraint-polynomial)`,
                );
        }
    });
}

// Runs checked code (checkQVerifier) and returns the value of its last destination.
//   ctx: {evaluations: [Fr] by evMap index, publics: [Fr], challenge(stage, stageId) → Fr,
//         zi: [Fr] by boundary}
export function executeCode(curve, code, ctx) {
    const Fr = curve.Fr;
    const tmp = [];
    const value = (ref) => {
        switch (ref.type) {
            case "tmp":
                return tmp[ref.id];
            case "eval":
                return ctx.evaluations[ref.id];
            case "public":
                return ctx.publics[ref.id];
            case "number":
                return ref.value;
            case "challenge":
                return ctx.challenge(ref.stage, ref.stageId);
            case "Zi":
                return ctx.zi[ref.id];
            default:
                throw new Error(`executeCode: operand ${ref.type} was not checked`);
        }
    };
    for (const { op, dest, src } of code) {
        const [a, b] = src.map(value);
        switch (op) {
            case "add":
                tmp[dest] = Fr.add(a, b);
                break;
            case "sub":
                tmp[dest] = Fr.sub(a, b);
                break;
            case "mul":
                tmp[dest] = Fr.mul(a, b);
                break;
            case "copy":
                tmp[dest] = a;
                break;
            default:
                throw new Error(`executeCode: op ${op} was not checked`);
        }
    }
    return tmp[code[code.length - 1].dest];
}

// Σ_i ξ^(i·M·N)·Q_i(ξ): Q(ξ) from the pieces of a split Q, M = maxQDegree
// (pilfflonk/docs/protocol.md#q-pieces).
export function joinQPieces(curve, pieces, xi, nBits, maxQDegree) {
    const Fr = curve.Fr;
    const shift = Fr.exp(xi, BigInt(maxQDegree) * (1n << BigInt(nBits)));
    let q = Fr.zero;
    for (let i = pieces.length - 1; i >= 0; i--) {
        q = Fr.add(Fr.mul(q, shift), pieces[i]);
    }
    return q;
}
