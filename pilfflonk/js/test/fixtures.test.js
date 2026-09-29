// The cross-check with the C++ prover (M8; validations 4 and 5 of Fase 0): on every opening M7's
// tests write,
// - the transcript script, replayed literally, gives the C++ challenges;
// - the verifier accepts the opening, with its challenges replayed from the proof;
// - it rejects every tampered copy: an evaluation, a commitment, [W] or [W'] changed by one or by a
//   bit, or [τ]₂ that is not τ's, with the challenges the changed proof gives and with the
//   original ones; and xiSeed, α or y changed.
// Skipped, saying why, unless PILFFLONK_SHPLONK_FIXTURES names the directory (test/fixtures.sh).

import assert from "node:assert/strict";
import { before, test } from "node:test";

import {
    PilFflonkInputError,
    frFromObject,
    frToObject,
    g1FromObject,
    g1ToObject,
    g2FromObject,
} from "../src/elements.js";
import {
    computeE,
    computeF,
    computeJ,
    computeQuotients,
    computeR,
    computeRoots,
    computeZerofiers,
} from "../src/shplonk.js";
import { Keccak256Transcript } from "../src/transcript.js";
import { EXPECTED_CASES, FIXTURES_DIR, NO_FIXTURES, loadFixtures, newCurve, openingOf, verifies } from "./support.js";

if (!FIXTURES_DIR) {
    test("C++ SHPLONK fixtures (M7)", { skip: NO_FIXTURES }, () => {});
} else {
    run(FIXTURES_DIR);
}

function run(dir) {
    const fixtures = loadFixtures(dir);
    let curve;
    before(async () => {
        curve = await newCurve();
    });

    test(`${dir} holds every case of pilfflonk_shplonk_test.cpp`, () => {
        const missing = EXPECTED_CASES.filter((name) => !fixtures.some((f) => f.name === name));
        assert.deepEqual(missing, [], `missing fixtures: ${missing.join(", ")}`);
    });

    for (const fixture of fixtures) {
        test(`fixture ${fixture.name}`, async (t) => {
            assert.equal(fixture.protocol, "pilfflonk-shplonk");
            assert.equal(fixture.curve, "bn128");

            await t.test("replaying its transcript script literally gives the C++ challenges", () => {
                assert.deepEqual(replayScript(curve, fixture.transcript), [
                    ["xiSeed", fixture.xiSeed],
                    ["alpha", fixture.alpha],
                    ["y", fixture.y],
                ]);
            });

            await t.test("the script absorbs the fixture's seed, commitments, evaluations and [W]", () => {
                const script = fixture.transcript;
                assert.deepEqual(
                    script.map((op) => (op.op === "absorb" ? `absorb ${op.kind}` : `squeeze ${op.name}`)),
                    [
                        "absorb fr",
                        "absorb g1",
                        "squeeze xiSeed",
                        "absorb fr",
                        "squeeze alpha",
                        "absorb g1",
                        "squeeze y",
                    ],
                );
                assert.equal(script[0].values.length, 1);
                assert.deepEqual(script[1].values, fixture.f.map((f) => f.commitment));
                assert.deepEqual(script[3].values, fixture.f.flatMap((f) => f.evaluations.flat()));
                assert.deepEqual(script[5].values, [fixture.W]);
                // The same challenges from the fields, as the verifier replays them.
                const opening = openingOf(curve, fixture);
                assert.deepEqual(
                    [opening.xiSeed, opening.alpha, opening.y].map((e) => frToObject(curve, e)),
                    [fixture.xiSeed, fixture.alpha, fixture.y],
                );
            });

            await t.test("ξ, the points ξ·ω_N^s and the roots are the prover's", () => {
                const Fr = curve.Fr;
                const opening = openingOf(curve, fixture);
                const { xi, f: roots } = computeRoots(curve, opening, opening.xiSeed);
                assert.equal(frToObject(curve, xi), fixture.xi);
                fixture.f.forEach((f, i) => {
                    assert.deepEqual(roots[i].points.map((p) => frToObject(curve, p)), f.points, `f_${i}`);
                    roots[i].roots.forEach((x, n) => {
                        const point = roots[i].points[Math.floor(n / f.k)];
                        assert.ok(Fr.eq(Fr.exp(x, BigInt(f.k)), point), `f_${i}: root ${n}`);
                    });
                });
            });

            await t.test("its SRS is τ's: [1]₂ is the generator and [τ]₂ = τ·[1]₂", () => {
                const G2 = curve.G2;
                const one = g2FromObject(curve, fixture.X2.one, "[1]₂");
                const tau = g2FromObject(curve, fixture.X2.tau, "[τ]₂");
                assert.ok(G2.eq(one, G2.one));
                assert.ok(G2.eq(tau, G2.timesFr(G2.one, frFromObject(curve, fixture.tau, "τ"))));
            });

            await t.test("is accepted, whichever of its f_i are the fixed ones", async () => {
                for (const nFixed of new Set([0, 1, fixture.f.length])) {
                    assert.equal(await verifies(curve, fixture, { nFixed }), true, `${nFixed} fixed`);
                }
            });

            await t.test("F - E - J + y·[W'] = τ·[W'], the pairing's equation with the known τ", () => {
                const G1 = curve.G1;
                const o = openingOf(curve, fixture);
                const { f: roots } = computeRoots(curve, o, o.xiSeed);
                const zerofiers = computeZerofiers(curve, roots, o.y);
                const r = computeR(curve, o.f, roots, o.evaluations, zerofiers, o.y);
                const q = computeQuotients(curve, zerofiers, o.alpha);
                const F = computeF(curve, o.commitments, q);
                const E = computeE(curve, r, q);
                const J = computeJ(curve, o.W, q[0]);
                const lhs = G1.add(G1.sub(G1.sub(F, E), J), G1.timesFr(o.Wp, o.y));
                assert.ok(G1.eq(lhs, G1.timesFr(o.Wp, frFromObject(curve, fixture.tau, "τ"))));
            });

            await t.test("every tampered proof or key is rejected", async () => {
                const original = { xiSeed: fixture.xiSeed, alpha: fixture.alpha, y: fixture.y };
                let count = 0;
                for (const { name, change, rejectedBy } of dataTampers(curve, fixture)) {
                    const tampered = structuredClone(fixture);
                    change(tampered);
                    const message = `${name}: rejected by ${rejectedBy}`;
                    assert.equal(decodes(curve, tampered), rejectedBy === "pairing", message);
                    assert.equal(await verifies(curve, tampered), false, `${name}, challenges replayed`);
                    const withOriginal = await verifies(curve, tampered, { challenges: original });
                    assert.equal(withOriginal, false, `${name}, the original challenges`);
                    count++;
                }
                t.diagnostic(`${count} tampered proofs or keys rejected`);
            });

            await t.test("every tampered challenge is rejected", async () => {
                let count = 0;
                for (const { name, change } of challengeTampers(curve, fixture)) {
                    const challenges = { xiSeed: fixture.xiSeed, alpha: fixture.alpha, y: fixture.y };
                    change(challenges);
                    assert.equal(await verifies(curve, fixture, { challenges }), false, name);
                    count++;
                }
                t.diagnostic(`${count} tampered challenges rejected`);
            });
        });
    }
}

// The fixture's transcript script, operation by operation, on a new transcript: the name and the
// value of each squeeze.
function replayScript(curve, script) {
    const transcript = new Keccak256Transcript(curve);
    const squeezed = [];
    for (const op of script) {
        if (op.op === "absorb" && op.kind === "fr") {
            op.values.forEach((v, i) => transcript.addScalar(frFromObject(curve, v, `fr ${i}`)));
        } else if (op.op === "absorb" && op.kind === "g1") {
            op.values.forEach((v, i) => transcript.addPolCommitment(g1FromObject(curve, v, `g1 ${i}`)));
        } else if (op.op === "squeeze") {
            squeezed.push([op.name, frToObject(curve, transcript.squeeze())]);
        } else {
            assert.fail(`unknown transcript operation ${JSON.stringify(op)}`);
        }
    }
    return squeezed;
}

function decodes(curve, fixture) {
    try {
        openingOf(curve, fixture);
        return true;
    } catch (e) {
        if (e instanceof PilFflonkInputError) return false;
        throw e;
    }
}

const plusOne = (curve, v) => ((BigInt(v) + 1n) % curve.r).toString();
const flip = (v, bit) => (BigInt(v) ^ (1n << BigInt(bit))).toString();

// Every change to the proof or the key a verifier must catch, and what catches it: "decoding" for
// a value that is no longer a scalar or a point, "pairing" for one that still is.
function* dataTampers(curve, fixture) {
    const G1 = curve.G1;
    const plusG = (p) => g1ToObject(curve, G1.add(g1FromObject(curve, p, "P"), G1.one));
    const scalarFlip = (v, bit) => (BigInt(flip(v, bit)) < curve.r ? "pairing" : "decoding");

    for (const [i, f] of fixture.f.entries()) {
        for (const [m, row] of f.evaluations.entries()) {
            for (const [j, e] of row.entries()) {
                yield {
                    name: `f_${i}: p_${j}(ξ·ω^${f.offsets[m]}) + 1`,
                    change: (c) => (c.f[i].evaluations[m][j] = plusOne(curve, e)),
                    rejectedBy: "pairing",
                };
            }
        }
        const m = f.evaluations.length - 1;
        const j = f.k - 1;
        const first = f.evaluations[0][0];
        const last = f.evaluations[m][j];
        yield {
            name: `f_${i}: bit 0 of its first evaluation`,
            change: (c) => (c.f[i].evaluations[0][0] = flip(first, 0)),
            rejectedBy: scalarFlip(first, 0),
        };
        yield {
            name: `f_${i}: bit 200 of its last evaluation`,
            change: (c) => (c.f[i].evaluations[m][j] = flip(last, 200)),
            rejectedBy: scalarFlip(last, 200),
        };
        yield {
            name: `f_${i}: bit 255 of its last evaluation`,
            change: (c) => (c.f[i].evaluations[m][j] = flip(last, 255)),
            rejectedBy: "decoding",
        };
        const [x, y] = f.commitment;
        yield { name: `[f_${i}] + G`, change: (c) => (c.f[i].commitment = plusG(f.commitment)), rejectedBy: "pairing" };
        yield {
            name: `[f_${i}]: bit 0 of x`,
            change: (c) => (c.f[i].commitment = [flip(x, 0), y]),
            rejectedBy: "decoding",
        };
    }
    for (const key of ["W", "Wp"]) {
        const [x, y] = fixture[key];
        yield { name: `[${key}] + G`, change: (c) => (c[key] = plusG(fixture[key])), rejectedBy: "pairing" };
        yield { name: `[${key}] at infinity`, change: (c) => (c[key] = ["0", "0"]), rejectedBy: "decoding" };
        yield { name: `[${key}]: bit 0 of y`, change: (c) => (c[key] = [x, flip(y, 0)]), rejectedBy: "decoding" };
        yield { name: `[${key}]: bit 255 of x`, change: (c) => (c[key] = [flip(x, 255), y]), rejectedBy: "decoding" };
    }
    // [τ]₂ of another τ: [1]₂ itself.
    yield { name: "[τ]₂ = [1]₂", change: (c) => (c.X2.tau = c.X2.one), rejectedBy: "pairing" };
}

// Every change to {xiSeed, alpha, y}, the opening intact, where the challenge takes part in the
// check. With a single f there is nothing to batch: no q_i with i ≥ 1, so no α. With a single root
// ξ besides, the opening is KZG's: W' = W, and with E = f(ξ)·G1 and J = (y - ξ)·[W], F - E - J +
// y·[W'] does not depend on y.
function* challengeTampers(curve, fixture) {
    const names = ["xiSeed"];
    if (fixture.f.length > 1) names.push("alpha");
    if (fixture.f.length > 1 || fixture.f[0].k * fixture.f[0].offsets.length > 1) {
        names.push("y");
    } else {
        assert.deepEqual(fixture.Wp, fixture.W, "a single root: W' = W");
    }
    for (const name of names) {
        yield { name: `${name} + 1`, change: (c) => (c[name] = plusOne(curve, c[name])) };
        // Bit 0 of r - 1 gives r, not a scalar: 0 then.
        yield { name: `${name}: bit 0`, change: (c) => (c[name] = (BigInt(flip(c[name], 0)) % curve.r).toString()) };
    }
}
