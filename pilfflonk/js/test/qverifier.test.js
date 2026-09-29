// Q(ξ) from the qVerifier (qverifier.js): the code it accepts and runs, the zerofiers of A.1 against
// a product over the rows of each domain and against the Rust oracle's and the C++'s at one point,
// and the check of a split Q. The comparison with the Rust oracle's Q(ξ) on the Fibonacci's real
// vkey is in setup.test.js.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { before, test } from "node:test";

import { PilFflonkInputError } from "../src/elements.js";
import { challengesMap, checkQVerifier, computeZi, executeCode, joinQPieces } from "../src/qverifier.js";
import { rootOfUnity } from "../src/shplonk.js";
import { randomness } from "./proofs.js";
import { newCurve } from "./support.js";

let curve;
let rand;
before(async () => {
    curve = await newCurve();
    rand = randomness(curve, "qverifier");
});

const ref = (type, fields = {}) => ({ type, dim: 1, ...fields });
const tmp = (id) => ref("tmp", { id });
const entry = (op, dest, ...src) => ({ op, dest: tmp(dest), src });

const SHAPE = {
    nEvaluations: 3,
    nPublic: 2,
    nBoundaries: 2,
    challenges: challengesMap([0, 2], 2),
};

test("the challengesMap: the stages' challenges, then std_vc and std_xi", () => {
    assert.deepEqual(challengesMap([0], 1), [
        { stage: 2, stageId: 0 },
        { stage: 3, stageId: 0 },
    ]);
    assert.deepEqual(challengesMap([0, 2, 1], 3), [
        { stage: 2, stageId: 0 },
        { stage: 2, stageId: 1 },
        { stage: 3, stageId: 0 },
        { stage: 4, stageId: 0 },
        { stage: 5, stageId: 0 },
    ]);
});

test("every op and every operand, as the code says", () => {
    const Fr = curve.Fr;
    const evaluations = [rand.fr(), rand.fr(), rand.fr()];
    const publics = [rand.fr(), rand.fr()];
    const zi = [rand.fr(), rand.fr()];
    const challenges = { "2/0": rand.fr(), "2/1": rand.fr(), "3/0": rand.fr() };
    const code = [
        entry("add", 0, ref("eval", { id: 0 }), ref("public", { id: 1 })),
        entry("sub", 1, tmp(0), ref("number", { value: "7" })),
        entry("mul", 2, tmp(1), ref("challenge", { id: 1, stage: 2, stageId: 1 })),
        entry("copy", 3, ref("Zi", { boundaryId: 1 })),
        entry("mul", 3, tmp(3), tmp(2)),
        entry("sub", 4, ref("challenge", { id: 2, stage: 3, stageId: 0 }), ref("eval", { id: 2 })),
        entry("add", 5, tmp(3), tmp(4)),
    ];
    const { code: checked, tmpUsed } = checkQVerifier(curve, { tmpUsed: 6, code, line: "" }, SHAPE);
    assert.equal(tmpUsed, 6);
    const value = executeCode(curve, checked, {
        evaluations,
        publics,
        challenge: (stage, stageId) => challenges[`${stage}/${stageId}`],
        zi,
    });
    const t2 = Fr.mul(Fr.sub(Fr.add(evaluations[0], publics[1]), Fr.e(7)), challenges["2/1"]);
    const expected = Fr.add(Fr.mul(zi[1], t2), Fr.sub(challenges["3/0"], evaluations[2]));
    assert.ok(Fr.eq(value, expected));
});

test("the code it refuses, with the reason", () => {
    const good = () => [entry("copy", 0, ref("eval", { id: 0 }))];
    const refusals = {
        "no code": [() => [], /no code/],
        "an unknown op": [() => [entry("div", 0, tmp(0), tmp(0))], /op "div" is not add, sub, mul or copy/],
        "sub_swap, which has no dimension to swap": [() => [entry("sub_swap", 0, tmp(0), tmp(0))], /op "sub_swap"/],
        "an op with too few operands": [() => [entry("mul", 0, ref("eval", { id: 0 }))], /mul takes 2 operands/],
        "a copy with two operands": [
            () => [entry("copy", 0, ref("eval", { id: 0 }), ref("eval", { id: 1 }))],
            /copy takes 1/,
        ],
        "a temporary read before it is written": [
            () => [entry("copy", 0, tmp(1))],
            /reads temporary 1 before it is written/,
        ],
        "a temporary read by the op that writes it first": [
            () => [entry("add", 0, tmp(0), tmp(0))],
            /reads temporary 0/,
        ],
        "a destination that is not a temporary": [
            () => [{ op: "copy", dest: ref("eval", { id: 0 }), src: [ref("eval", { id: 0 })] }],
            /destination is not a temporary/,
        ],
        "a destination above tmpUsed": [() => [entry("copy", 9, ref("eval", { id: 0 }))], /below tmpUsed = 4/],
        "an evaluation out of the evMap": [
            () => [entry("copy", 0, ref("eval", { id: 3 }))],
            /eval 3 is not one of the 3/,
        ],
        "a public out of range": [() => [entry("copy", 0, ref("public", { id: 2 }))], /public 2 is not one of the 2/],
        "a number not below r": [() => [entry("copy", 0, ref("number", { value: curve.r.toString() }))], /not below r/],
        "a number that is not a decimal string": [
            () => [entry("copy", 0, ref("number", { value: 7 }))],
            /decimal string/,
        ],
        "a number with a sign": [() => [entry("copy", 0, ref("number", { value: "-1" }))], /decimal string/],
        "an operand of dimension 3": [() => [entry("copy", 0, { type: "eval", id: 0, dim: 3 })], /dimension 3/],
        "a challenge of no stage": [
            () => [entry("copy", 0, ref("challenge", { id: 0, stage: 1, stageId: 0 }))],
            /no challenge of stage 1 has stageId 0/,
        ],
        "a challenge whose id is not its position": [
            () => [entry("copy", 0, ref("challenge", { id: 0, stage: 3, stageId: 0 }))],
            /the challenge of stage 3, stageId 0 is 2, not 0/,
        ],
        "a Zi of no boundary": [
            () => [entry("copy", 0, ref("Zi", { boundaryId: 2 }))],
            /Zi of boundary 2, and there are 2/,
        ],
        "an air value (D2)": [
            () => [entry("copy", 0, ref("airvalue", { id: 0 }))],
            /airvalues do not exist in format version 1/,
        ],
        "a proof value (D2)": [() => [entry("copy", 0, ref("proofvalue", { id: 0 }))], /proofvalues do not exist/],
        "an airgroup value (D2)": [
            () => [entry("copy", 0, ref("airgroupvalue", { id: 0 }))],
            /airgroupvalues do not exist/,
        ],
        "FRI's xDivXSubXi": [
            () => [entry("copy", 0, ref("xDivXSubXi", { id: 0 }))],
            /"xDivXSubXi" is not one of the qVerifier's/,
        ],
        "a column, which the verifier reads as an eval": [
            () => [entry("copy", 0, ref("cm", { id: 0 }))],
            /"cm" is not one/,
        ],
    };
    for (const [name, [code, message]] of Object.entries(refusals)) {
        const qVerifier = { tmpUsed: 4, code: code() };
        assert.throws(() => checkQVerifier(curve, qVerifier, SHAPE), PilFflonkInputError, name);
        assert.throws(() => checkQVerifier(curve, qVerifier, SHAPE), message, name);
    }
    for (const bad of [null, [], { code: good() }, { tmpUsed: -1, code: good() }, { tmpUsed: 1, code: {} }]) {
        assert.throws(() => checkQVerifier(curve, bad, SHAPE), PilFflonkInputError, JSON.stringify(bad));
    }
});

// Z_D(ξ) = Π_{j ∈ D} (ξ − ω^j), over the rows of D: independent of computeZi's closed forms.
function zerofierByRows(rows, xi, nBits) {
    const Fr = curve.Fr;
    const omega = rootOfUnity(curve, BigInt(2 ** nBits));
    return rows.reduce((z, j) => Fr.mul(z, Fr.sub(xi, Fr.exp(omega, BigInt(j)))), Fr.one);
}

test("Zi of every domain: 1/Z_H(ξ) for everyRow and Z_H(ξ)/Z_D(ξ) for the others (A.1, A.6)", () => {
    const Fr = curve.Fr;
    const nBits = 3;
    const N = 8;
    const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);
    const boundaries = [
        [{ name: "everyRow" }, range(0, N)],
        [{ name: "firstRow" }, [0]],
        [{ name: "lastRow" }, [N - 1]],
        [{ name: "everyFrame", offsetMin: 1, offsetMax: 2 }, range(1, N - 2)],
        [{ name: "everyFrame", offsetMin: 0, offsetMax: 3 }, range(0, N - 3)],
        [{ name: "everyFrame", offsetMin: 2, offsetMax: 0 }, range(2, N)],
    ];
    for (const xi of [rand.fr(), rand.fr(), Fr.neg(Fr.e(3))]) {
        const zi = computeZi(
            curve,
            boundaries.map(([b]) => b),
            nBits,
            xi,
        );
        const zh = zerofierByRows(range(0, N), xi, nBits);
        assert.ok(Fr.eq(Fr.mul(zi[0], zh), Fr.one), "everyRow");
        boundaries.slice(1).forEach(([b, rows], i) => {
            const zd = zerofierByRows(rows, xi, nBits);
            assert.ok(Fr.eq(Fr.mul(zi[i + 1], zd), zh), JSON.stringify(b));
        });
    }
});

// The interpreter's fixture, written by setup/pilfflonk/tests/bytecode/interpreter.rs with the Rust
// oracle's values: its Zi at ξ for everyRow, firstRow, lastRow and everyFrame {1, 2}, which the C++
// zerofiersAt reproduces (pil2-stark/test/pilfflonk/pilfflonk_expressions_test.cpp).
const INTERPRETER_FIXTURE = new URL(
    "../../../setup/pilfflonk/tests/fixtures/bytecode/Sample.expected.json",
    import.meta.url,
);

test("Zi at the ξ of the interpreter's fixture: the Rust oracle's and the C++'s (plan M24)", () => {
    const fixture = JSON.parse(readFileSync(INTERPRETER_FIXTURE, "utf8"));
    assert.deepEqual(fixture.boundaries.map((b) => b.name), ["everyRow", "firstRow", "lastRow", "everyFrame"]);
    const zi = computeZi(curve, fixture.boundaries, fixture.nBits, curve.Fr.e(BigInt(fixture.xi)));
    assert.deepEqual(zi.map((z) => curve.Fr.toString(z, 10)), fixture.expected.xiZerofiers);
});

test("Q is not defined on H", () => {
    const omega = rootOfUnity(curve, 8n);
    for (const xi of [curve.Fr.one, curve.Fr.exp(omega, 5n)]) {
        assert.throws(() => computeZi(curve, [{ name: "everyRow" }], 3, xi), /ξ is in H/);
    }
});

test("the pieces of a split Q add up as Σ ξ^(i·M·N)·Q_i(ξ) (A.1)", () => {
    const Fr = curve.Fr;
    const xi = rand.fr();
    const pieces = [rand.fr(), rand.fr(), rand.fr()];
    // M = 2, N = 8: ξ^0, ξ^16, ξ^32.
    const expected = pieces.reduce((acc, q, i) => Fr.add(acc, Fr.mul(Fr.exp(xi, BigInt(16 * i)), q)), Fr.zero);
    assert.ok(Fr.eq(joinQPieces(curve, pieces, xi, 3, 2), expected));
});
