// The C++ prover's SHPLONK fixtures as a verifier sees them. They are written by
// `PILFFLONK_SHPLONK_FIXTURES=<dir> make -C pil2-stark pilfflonk_test` as <dir>/shplonk_<case>.json,
// in the format documented above writeFixture() in pil2-stark/test/pilfflonk/pilfflonk_shplonk_test.cpp;
// test/fixtures.sh regenerates them and runs the tests.

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { buildBn128 } from "ffjavascript";

import { PilFflonkInputError, frFromObject, g1FromObject, g2FromObject } from "../src/elements.js";
import { verifyOpening } from "../src/shplonk.js";
import { Keccak256Transcript } from "../src/transcript.js";

export const FIXTURES_DIR = process.env.PILFFLONK_SHPLONK_FIXTURES || "";

export const NO_FIXTURES =
    "PILFFLONK_SHPLONK_FIXTURES is not set: run test/fixtures.sh to regenerate the C++ fixtures " +
    "and run these tests on them";

// The cases of pilfflonk_shplonk_test.cpp, cases().
export const EXPECTED_CASES = [
    "single",
    "everyk_0",
    "everyk_01",
    "everyk_m1012",
    "repeated",
    "short",
    "tiny",
    "k134_0",
    "k134_01",
    "k134_m1012",
];

export function loadFixtures(dir) {
    return readdirSync(dir)
        .filter((name) => /^shplonk_.*\.json$/.test(name))
        .sort()
        .map((name) => JSON.parse(readFileSync(join(dir, name), "utf8")));
}

// Single-threaded: no workers to terminate, and fast enough for these sizes.
export async function newCurve() {
    return await buildBn128(true);
}

// The fixture's transcript in terms of its fields rather than its script
// (pilfflonk/docs/protocol.md#transcript, in miniature): the seed standing in for the digest, every
// [f_i]₁; squeeze xiSeed; every evaluation, f by f, offset-major, p_0 first; squeeze α; [W]₁;
// squeeze y.
export function replayChallenges(curve, { seed, commitments, evaluations, W }) {
    const transcript = new Keccak256Transcript(curve);
    transcript.addScalar(seed);
    commitments.forEach((c) => transcript.addPolCommitment(c));
    const xiSeed = transcript.squeeze();
    evaluations.flat(2).forEach((e) => transcript.addScalar(e));
    const alpha = transcript.squeeze();
    transcript.addPolCommitment(W);
    const y = transcript.squeeze();
    return { xiSeed, alpha, y };
}

// The fixture decoded as a verifier decodes a proof and a key (elements.js). The first `nFixed`
// commitments play the fixed ones, taken from the key. The challenges are replayed from the decoded
// values, or taken from `challenges` ({xiSeed, alpha, y}, decimal strings) as they are.
export function openingOf(curve, fixture, { nFixed = 0, challenges } = {}) {
    const all = fixture.f.map((f, i) => g1FromObject(curve, f.commitment, `[f_${i}]`));
    const evaluations = fixture.f.map((f, i) =>
        f.evaluations.map((row, m) => row.map((e, j) => frFromObject(curve, e, `f_${i}: p_${j}(ξ·ω^${f.offsets[m]})`))),
    );
    const W = g1FromObject(curve, fixture.W, "[W]");
    const Wp = g1FromObject(curve, fixture.Wp, "[W']");
    const X2 = { one: g2FromObject(curve, fixture.X2.one, "[1]₂"), tau: g2FromObject(curve, fixture.X2.tau, "[τ]₂") };
    let replayed;
    if (challenges) {
        replayed = {
            xiSeed: frFromObject(curve, challenges.xiSeed, "xiSeed"),
            alpha: frFromObject(curve, challenges.alpha, "alpha"),
            y: frFromObject(curve, challenges.y, "y"),
        };
    } else {
        const seed = frFromObject(curve, fixture.transcript[0].values[0], "seed");
        replayed = replayChallenges(curve, { seed, commitments: all, evaluations, W });
    }
    return {
        nBits: fixture.nBits,
        powerW: fixture.powerW,
        f: fixture.f.map(({ k, offsets }) => ({ k, offsets })),
        fixedCommitments: all.slice(0, nFixed),
        commitments: all.slice(nFixed),
        evaluations,
        ...replayed,
        W,
        Wp,
        X2,
    };
}

// What a verifier concludes from the fixture: a value it cannot decode rejects it.
export async function verifies(curve, fixture, options = {}) {
    let opening;
    try {
        opening = openingOf(curve, fixture, options);
    } catch (e) {
        if (e instanceof PilFflonkInputError) return false;
        throw e;
    }
    return await verifyOpening(curve, opening);
}
