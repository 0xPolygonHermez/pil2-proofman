// pilfflonk.vkey.json (spec-seed.md A.6, §4.5 step 1): read and checked as snarkjs' fflonk verifier
// reads its verification key (src/fflonk_verify.js, fromObjectVk), and its digest.
//
// The format is pilfflonk/src/vkey.rs's (M12), format version 1, for one AIR (D2):
//   protocol "pilfflonk", curve "bn128", formatVersion 1, nPublic, power (nBits), powerW,
//   X_2 ([τ]₂ as [[x.c0, x.c1], [y.c0, y.c1]]), numChallenges, evMap, layout, boundaries,
//   f0 … f<n-1> (the fixed commitments, [x, y]), qDeg, maxQDegree, qVerifier, digest.
// A vkey is accepted only if the Rust side would read it (Vkey::validate, Layout::check), with the
// checks of snarkjs' steps 1-3 on its points (elements.js) and those the verifier needs to run:
//   - numChallenges has one entry per stage, and stage 1 none: A.4 squeezes no challenge before
//     the commitments of stage 1;
//   - boundaries[0] is everyRow, the Zi that A.6 makes 1/Z_H, and an everyFrame leaves a row;
//   - the qVerifier is code the verifier can run (qverifier.js);
//   - the names of the evaluations (names.js) do not collide.
// The digest (A.6) is keccak256("pilfflonk-v1" ‖ canonical(vkey without digest)), json.js the
// canonical JSON. Anything else is a PilFflonkInputError naming what is wrong.

import { keccak_256 } from "@noble/hashes/sha3";
import { bytesToHex } from "@noble/hashes/utils";

import { PilFflonkInputError, decimalFromObject, g1FromObject, g2FromObject, show } from "./elements.js";
import { canonicalJson } from "./json.js";
import { INV, INV_ZH, commitmentName, evaluationName } from "./names.js";
import { challengesMap, checkQVerifier } from "./qverifier.js";
import { checkLayout } from "./shplonk.js";

export const PROTOCOL = "pilfflonk";
export const CURVE = "bn128";
export const FORMAT_VERSION = 1;
export const DIGEST_DOMAIN = "pilfflonk-v1";

// The 2-adicity of r - 1, the largest nBits (global_info.rs, MAX_NBITS).
const MAX_NBITS = 28;

const FIELDS = [
    "protocol",
    "curve",
    "formatVersion",
    "nPublic",
    "power",
    "powerW",
    "X_2",
    "numChallenges",
    "evMap",
    "layout",
    "boundaries",
    "qDeg",
    "maxQDegree",
    "qVerifier",
    "digest",
];

// f<i>, i in decimal without leading zeros (vkey.rs, fixed_index).
const FIXED_KEY = /^f(0|[1-9][0-9]*)$/;
const DIGEST = /^0x[0-9a-f]{64}$/;

function fail(message) {
    throw new PilFflonkInputError(`vkey: ${message}`);
}

function isPlainObject(v) {
    return typeof v === "object" && v !== null && !Array.isArray(v);
}

// An object with exactly these keys, as serde's deny_unknown_fields reads it.
function checkKeys(value, keys, what) {
    if (!isPlainObject(value)) fail(`${what} is not an object`);
    const extra = Object.keys(value).filter((k) => !keys.includes(k));
    const missing = keys.filter((k) => !Object.hasOwn(value, k));
    if (extra.length > 0) fail(`${what} has the unknown fields ${extra.join(", ")}`);
    if (missing.length > 0) fail(`${what} has no ${missing.join(", ")}`);
}

// A u64 of the JSON, as a number: the canonical JSON has no larger integers (json.js).
function count(value, what) {
    if (!Number.isSafeInteger(value) || value < 0) fail(`${what} = ${show(value)} is not a non-negative integer`);
    return value;
}

function signed(value, what) {
    if (!Number.isSafeInteger(value) || Object.is(value, -0)) fail(`${what} = ${show(value)} is not an integer`);
    return value;
}

function array(value, what) {
    if (!Array.isArray(value)) fail(`${what} is not an array`);
    return value;
}

// The decimal strings of a point, each canonical (elements.js, decimalFromObject).
function coordinates(value, shape, what) {
    const check = (v, s, path) => {
        if (Array.isArray(s)) {
            if (!Array.isArray(v) || v.length !== s.length) fail(`${path} is not a point in the form ${show(shape)}`);
            return v.map((c, i) => check(c, s[i], `${path}[${i}]`));
        }
        return decimalFromObject(v, path);
    };
    return check(value, shape, what);
}

// keccak256("pilfflonk-v1" ‖ canonical(vkey without digest)), as "0x" and 64 lowercase
// hexadecimal digits, of the vkey as JSON.parse returns it (A.6).
export function vkeyDigest(vkObject) {
    if (!isPlainObject(vkObject)) fail("not an object");
    const { digest: _digest, ...rest } = vkObject;
    const preimage = new TextEncoder().encode(DIGEST_DOMAIN + canonicalJson(rest));
    return `0x${bytesToHex(keccak_256(preimage))}`;
}

// The number of pieces Q is split into (layout.rs, q_pieces): ⌈qDeg / maxQDegree⌉ if maxQDegree > 0
// and qDeg > maxQDegree, and 1 otherwise, Q whole (A.1).
export function qPieces(qDeg, maxQDegree) {
    return maxQDegree > 0 && qDeg > maxQDegree ? Math.ceil(qDeg / maxQDegree) : 1;
}

function readBoundary(b, i) {
    const what = `boundaries[${i}]`;
    if (!isPlainObject(b)) fail(`${what} is not an object`);
    if (b.name === "everyFrame") {
        checkKeys(b, ["name", "offsetMin", "offsetMax"], what);
        return {
            name: b.name,
            offsetMin: count(b.offsetMin, `${what}.offsetMin`),
            offsetMax: count(b.offsetMax, `${what}.offsetMax`),
        };
    }
    if (!["everyRow", "firstRow", "lastRow"].includes(b.name)) fail(`${what} is ${show(b.name)}, not a domain of A.1`);
    checkKeys(b, ["name"], what);
    return { name: b.name };
}

// Layout::check of pilfflonk/src/layout.rs without the pol maps, which the vkey does not carry:
// by ascending stage, fixed first and Q's last (qStage is the last f's); k = |pols|; increasing
// offsets; Q opened at ξ only and made of qPieces polynomials; no column in two f; the evMap is
// every (column, offset) the layout opens, but Q's. The roots' conditions (kN | r - 1, offsets and
// powerW) are shplonk.js'.
function checkLayoutOf(curve, vk) {
    const { layout, evMap, qStage, power, powerW } = vk;
    let previous = 0;
    let nQ = 0;
    const packed = new Map();
    const opened = new Set();
    layout.forEach((f, i) => {
        if (f.stage < previous) fail(`f${i} is of stage ${f.stage} after one of stage ${previous} (A.5)`);
        previous = f.stage;
        if (f.pols.length === 0 || f.k !== f.pols.length) fail(`f${i} has k = ${f.k} and ${f.pols.length} polynomials`);
        if (f.offsets.length === 0 || f.offsets.some((s, j) => j > 0 && f.offsets[j - 1] >= s)) {
            fail(`f${i} has offsets ${show(f.offsets)}: at least one, increasing`);
        }
        if (f.degree === 0) fail(`f${i} has degree 0`);
        const isQ = f.stage === qStage;
        if (isQ) {
            if (f.offsets.length !== 1 || f.offsets[0] !== 0) fail(`f${i} holds Q, which is opened at ξ only`);
            nQ += f.k;
        }
        const type = f.stage === 0 ? "const" : "cm";
        for (const pol of f.pols) {
            const key = `${type} ${pol.id}`;
            if (packed.has(key)) fail(`${key} is in two f of the layout`);
            packed.set(key, pol.name);
            if (!isQ) for (const s of f.offsets) opened.add(`${key} ${s}`);
        }
    });
    if (nQ !== vk.qPieces) fail(`the layout packs ${nQ} polynomials of Q, and Q is made of ${vk.qPieces}`);

    const evaluated = new Set();
    for (const e of evMap) {
        const key = `${e.type} ${e.id} ${e.prime}`;
        if (evaluated.has(key)) fail(`the evMap has ${e.type} ${e.id} at offset ${e.prime} twice`);
        evaluated.add(key);
        if (!opened.has(key))
            fail(`the evMap has ${e.type} ${e.id} at offset ${e.prime}, which no f of the layout opens`);
    }
    for (const key of opened) {
        if (!evaluated.has(key))
            fail(`the layout opens ${key.replace(/ (-?\d+)$/, " at offset $1")}, which the evMap does not have`);
    }

    try {
        checkLayout(curve, { nBits: power, powerW, f: layout });
    } catch (e) {
        if (e instanceof PilFflonkInputError) fail(e.message);
        throw e;
    }
    return packed;
}

// The vkey, checked and decoded (see the module):
//   nPublic, power, N, powerW, qDeg, maxQDegree, qPieces, numChallenges, nStages, qStage
//   X2            {one: [1]₂, tau: [τ]₂}
//   evMap         [{type, id, prime, openingPos, name}]: name, the evaluation's in the proof
//   layout        [{stage, pols: [{id, name}], k, offsets, degree}], f_i at position i
//   boundaries    [{name, offsetMin?, offsetMax?}]
//   fixedCommitments  [f_i]₁ of the fixed f_i, the first of the layout
//   commitmentNames   the proof's names of the other f_i, in the global order of A.5
//   evaluationOrder   the evMap indices in the order of the proof and of A.4 step 4: the fixed
//                     columns', then the others', each in the order of the evMap
//   qPieceNames       the names of the pieces Q_i(ξ) of a split Q in the order of the proof, and
//                     [] otherwise
//   qVerifier     {tmpUsed, code}, checked (qverifier.js)
//   challenges    the (stage, stageId) of each challenge, in challengesMap order
//   digest        the hexadecimal string; digestFr, digest mod r (A.4, A.6)
export function fromObjectVk(curve, vkObject) {
    if (!isPlainObject(vkObject)) fail("not an object");
    const extra = Object.keys(vkObject).filter((k) => !FIELDS.includes(k) && !FIXED_KEY.test(k));
    if (extra.length > 0) fail(`unknown fields ${extra.join(", ")}`);
    const missing = FIELDS.filter((k) => !Object.hasOwn(vkObject, k));
    if (missing.length > 0) fail(`no ${missing.join(", ")}`);

    if (vkObject.protocol !== PROTOCOL) fail(`protocol ${show(vkObject.protocol)} is not ${PROTOCOL}`);
    if (vkObject.curve !== CURVE) fail(`curve ${show(vkObject.curve)} is not ${CURVE}`);
    if (vkObject.formatVersion !== FORMAT_VERSION) {
        fail(`formatVersion ${show(vkObject.formatVersion)} is not supported; this verifier reads ${FORMAT_VERSION}`);
    }

    const vk = {
        nPublic: count(vkObject.nPublic, "nPublic"),
        power: count(vkObject.power, "power"),
        powerW: count(vkObject.powerW, "powerW"),
        qDeg: count(vkObject.qDeg, "qDeg"),
        maxQDegree: count(vkObject.maxQDegree, "maxQDegree"),
    };
    if (vk.power > MAX_NBITS) fail(`power ${vk.power} is above ${MAX_NBITS}`);
    vk.N = 2 ** vk.power;
    vk.qPieces = qPieces(vk.qDeg, vk.maxQDegree);

    const X2 = coordinates(
        vkObject.X_2,
        [
            ["x.c0", "x.c1"],
            ["y.c0", "y.c1"],
        ],
        "X_2",
    );
    vk.X2 = { one: curve.G2.g, tau: g2FromObject(curve, X2, "vkey: X_2") };

    vk.evMap = array(vkObject.evMap, "evMap").map((e, i) => {
        const what = `evMap[${i}]`;
        checkKeys(e, ["type", "id", "prime", "openingPos"], what);
        if (e.type !== "cm" && e.type !== "const") fail(`${what} is of type ${show(e.type)}, not cm or const`);
        return {
            type: e.type,
            id: count(e.id, `${what}.id`),
            prime: signed(e.prime, `${what}.prime`),
            openingPos: count(e.openingPos, `${what}.openingPos`),
        };
    });

    vk.layout = array(vkObject.layout, "layout").map((f, i) => {
        const what = `layout[${i}]`;
        checkKeys(f, ["stage", "pols", "k", "offsets", "degree"], what);
        return {
            stage: count(f.stage, `${what}.stage`),
            pols: array(f.pols, `${what}.pols`).map((p, j) => {
                checkKeys(p, ["id", "name"], `${what}.pols[${j}]`);
                if (typeof p.name !== "string") fail(`${what}.pols[${j}].name is not a string`);
                return { id: count(p.id, `${what}.pols[${j}].id`), name: p.name };
            }),
            k: count(f.k, `${what}.k`),
            offsets: array(f.offsets, `${what}.offsets`).map((s, j) => signed(s, `${what}.offsets[${j}]`)),
            degree: count(f.degree, `${what}.degree`),
        };
    });
    vk.qStage = vk.layout.length > 0 ? vk.layout[vk.layout.length - 1].stage : 0;
    if (vk.qStage === 0) fail("the layout has no f for Q");
    vk.nStages = vk.qStage - 1;
    const packed = checkLayoutOf(curve, vk);

    vk.numChallenges = array(vkObject.numChallenges, "numChallenges").map((n, i) => count(n, `numChallenges[${i}]`));
    if (vk.numChallenges.length !== vk.nStages) {
        fail(`numChallenges has ${vk.numChallenges.length} stages, and the layout ${vk.nStages}`);
    }
    if (vk.numChallenges[0] !== 0) fail("stage 1 has challenges, which A.4 never squeezes");
    vk.challenges = challengesMap(vk.numChallenges, vk.nStages);

    vk.boundaries = array(vkObject.boundaries, "boundaries").map(readBoundary);
    if (vk.boundaries.length === 0 || vk.boundaries[0].name !== "everyRow") {
        fail("boundaries[0] is not everyRow, whose Zi is 1/Z_H (A.6)");
    }
    vk.boundaries.forEach((b, i) => {
        const same = (c) => c.name === b.name && c.offsetMin === b.offsetMin && c.offsetMax === b.offsetMax;
        if (vk.boundaries.slice(0, i).some(same)) fail(`boundary ${show(b)} is there twice`);
        if (b.name === "everyFrame" && b.offsetMin + b.offsetMax >= vk.N) {
            fail(`boundaries[${i}] is everyFrame {${b.offsetMin}, ${b.offsetMax}}: no row of ${vk.N} is left`);
        }
    });

    const nFixed = vk.layout.filter((f) => f.stage === 0).length;
    const fixedKeys = Object.keys(vkObject).filter((k) => FIXED_KEY.test(k));
    if (fixedKeys.length !== nFixed || fixedKeys.some((k) => Number(k.slice(1)) >= nFixed)) {
        const have = fixedKeys.join(", ") || "none";
        const expected = nFixed === 0 ? "none" : `f0 … f${nFixed - 1}`;
        fail(`the fixed commitments are ${have}, and the layout has ${nFixed} fixed f: ${expected}`);
    }
    vk.fixedCommitments = vk.layout.slice(0, nFixed).map((_, i) => {
        const [x, y] = coordinates(vkObject[`f${i}`], ["x", "y"], `f${i}`);
        // A fixed column that vanishes at τ commits to the point at infinity, (0, 0) (field.rs).
        return g1FromObject(curve, [x, y], `vkey: f${i}`, { allowInfinity: true });
    });
    vk.commitmentNames = vk.layout.slice(nFixed).map((_, i) => commitmentName(nFixed + i));

    // The evaluations' names in the proof (names.js), and their order in it: A.4 step 4.
    vk.evMap.forEach((e) => {
        e.name = evaluationName(packed.get(`${e.type} ${e.id}`), e.prime);
    });
    const indices = vk.evMap.map((_, i) => i);
    vk.evaluationOrder = [
        ...indices.filter((i) => vk.evMap[i].type === "const"),
        ...indices.filter((i) => vk.evMap[i].type === "cm"),
    ];
    vk.qPieceNames = vk.qPieces > 1 ? qPieceNames(vk) : [];
    const names = [...vk.evMap.map((e) => e.name), ...vk.qPieceNames, INV, INV_ZH];
    const repeated = names.find((name, i) => names.indexOf(name) !== i);
    if (repeated !== undefined) fail(`two values of the proof are named ${show(repeated)}`);

    vk.qVerifier = checkQVerifier(curve, vkObject.qVerifier, {
        nEvaluations: vk.evMap.length,
        nPublic: vk.nPublic,
        nBoundaries: vk.boundaries.length,
        challenges: vk.challenges,
    });

    if (typeof vkObject.digest !== "string" || !DIGEST.test(vkObject.digest)) {
        fail(`digest ${show(vkObject.digest)} is not "0x" and 64 lowercase hexadecimal digits`);
    }
    vk.digest = vkObject.digest;
    vk.digestFr = curve.Fr.e(BigInt(vk.digest) % curve.r);
    return vk;
}

// The names of the pieces of a split Q, in the order of the proof: the polynomials of Q's f in the
// order of the layout, as ProofNames lists them (proof.rs). A.6 names the pieces Q0 … Q<m-1>, and
// piece i, the one multiplied by ξ^(i·M·N) (A.1), is the one named Q<i> (qPieceIndex).
function qPieceNames(vk) {
    const names = vk.layout.filter((f) => f.stage === vk.qStage).flatMap((f) => f.pols.map((p) => p.name));
    const expected = Array.from({ length: vk.qPieces }, (_, i) => `Q${i}`);
    if (JSON.stringify([...names].sort()) !== JSON.stringify([...expected].sort())) {
        fail(`the pieces of Q are named ${names.join(", ")}, not ${expected.join(", ")}`);
    }
    return names;
}

// The index i of the piece Q_i of a split Q named `name` (qPieceNames).
export function qPieceIndex(name) {
    return Number(name.slice(1));
}
