// The pilfflonk verifier (pilfflonk/docs/verifier.md#js-verifier): verify(vkey, publics, proof,
// logger) → true or false, in the shape of snarkjs' fflonk verifier (src/fflonk_verify.js), whose
// steps it keeps (pilfflonk/docs/verifier.md#steps):
//
// 1-3. decode and check the vkey (vkey.js; its [τ]₂ in the r-torsion), its digest
//      (pilfflonk/docs/formats.md#digest), the publics and the proof (proof.js): commitments in G1,
//      evaluations and publics in Fr, and exactly the values the vkey says;
// 4.   replay the transcript (challenges.js, pilfflonk/docs/protocol.md#transcript);
// 5.   Z_H(ξ) with ξ = xiSeed^powerW, checked against the proof's invZh as pil-stark's fflonk
//      verifier checks it; Q(ξ) from the evaluations with the vkey's qVerifier (qverifier.js),
//      and, if Q is split, Σ_i ξ^(i·M·N)·Q_i(ξ) = Q(ξ) (pilfflonk/docs/protocol.md#q-pieces);
// 6.   the proof's inv: the inverse of the product of the denominators the SHPLONK check inverts
//      (shplonk.js, computeInverseDenominators; pilfflonk/docs/protocol.md#inverses), as snarkjs'
//      Solidity verifier checks its own;
// 7.   the SHPLONK opening with a pairing (shplonk.js, pilfflonk/docs/protocol.md#pairing-check),
//      the fixed commitments always the vkey's, never the proof's, and Q(ξ) the evaluation of Q's f
//      at ξ.
//
// Arguments are the parsed JSON of pilfflonk.vkey.json, publics.json and proof.json. Every input it
// rejects, malformed ones included, gives false and a logged reason, as snarkjs does for what it
// checks; it throws only on a failure of its own. The logger, optional, has snarkjs' methods
// (debug, info, warn, error).
//
// Neither inv nor invZh is absorbed: each is checked against what it must be the inverse of.

import { buildBn128 } from "ffjavascript";

import { challengeOf, computeChallenges } from "./challenges.js";
import { PilFflonkInputError } from "./elements.js";
import { fromObjectProof, fromObjectPublics } from "./proof.js";
import { computeZi, executeCode, joinQPieces } from "./qverifier.js";
import {
    checkLayout,
    computeInverseDenominators,
    computeRoots,
    computeZerofiers,
    isValidInverse,
    verifyOpening,
} from "./shplonk.js";
import { fromObjectVk, qPieceIndex, vkeyDigest } from "./vkey.js";

let curvePromise;

// BN254, single-threaded: a verification is small, and no workers are left to terminate.
export function getCurve() {
    if (!curvePromise) curvePromise = buildBn128(true);
    return curvePromise;
}

// The evaluations of every f_i at its points, as verifyOpening takes them: evaluations[i][m][j] =
// p_j(ξ·ω^s) for s = layout[i].offsets[m], from the proof's, and for Q's f its value at ξ: Q(ξ)
// computed, or the pieces Q_i(ξ) of the proof if Q is split.
export function openingEvaluations(vk, proof, q) {
    const byColumn = new Map(vk.evMap.map((e, i) => [`${e.type} ${e.id} ${e.prime}`, proof.evaluations[i]]));
    const byPiece = new Map(vk.qPieceNames.map((name, i) => [name, proof.qPieces[i]]));
    return vk.layout.map((f) => {
        if (f.stage === vk.qStage) {
            return [f.pols.map((p) => (vk.qPieces > 1 ? byPiece.get(p.name) : q))];
        }
        const type = f.stage === 0 ? "const" : "cm";
        return f.offsets.map((s) => f.pols.map((p) => byColumn.get(`${type} ${p.id} ${s}`)));
    });
}

// The pieces Q_0(ξ) … Q_{m-1}(ξ) of a split Q, by piece (vkey.js, qPieceIndex).
function piecesByIndex(vk, proof) {
    const pieces = [];
    vk.qPieceNames.forEach((name, i) => {
        pieces[qPieceIndex(name)] = proof.qPieces[i];
    });
    return pieces;
}

// Q(ξ) from the proof's evaluations (step 5): the value of the vkey's qVerifier, and the Zi at ξ.
export function computeQ(curve, vk, publics, proof, challenges, xi) {
    const zi = computeZi(curve, vk.boundaries, vk.power, xi);
    const q = executeCode(curve, vk.qVerifier.code, {
        evaluations: proof.evaluations,
        publics,
        challenge: (stage, stageId) => challengeOf(vk, challenges, stage, stageId),
        zi,
    });
    return { q, zi };
}

export async function verify(vkObject, publicsObject, proofObject, logger) {
    if (logger) logger.info("PILFFLONK VERIFIER STARTED");
    const curve = await getCurve();
    try {
        const res = await run(curve, vkObject, publicsObject, proofObject, logger);
        if (logger) logger.info("PILFFLONK VERIFIER FINISHED");
        return res;
    } catch (e) {
        if (!(e instanceof PilFflonkInputError)) throw e;
        if (logger) logger.error(e.message);
        return false;
    }
}

async function run(curve, vkObject, publicsObject, proofObject, logger) {
    const Fr = curve.Fr;
    const info = (message) => {
        if (logger) logger.info(message);
    };

    info("> Checking the verification key");
    const vk = fromObjectVk(curve, vkObject);
    if (vkeyDigest(vkObject) !== vk.digest) {
        if (logger) {
            logger.error("The digest of the vkey is not the digest of its contents (pilfflonk/docs/formats.md#digest)");
        }
        return false;
    }

    if (logger) {
        logger.info("----------------------------");
        logger.info("  PILFFLONK VERIFY SETTINGS");
        logger.info(`  Curve:         ${curve.name}`);
        logger.info(`  Circuit power: ${vk.power}`);
        logger.info(`  Domain size:   ${vk.N}`);
        logger.info(`  Public vars:   ${vk.nPublic}`);
        logger.info(`  Polynomials:   ${vk.layout.length} f, ${vk.fixedCommitments.length} fixed`);
        logger.info("----------------------------");
    }

    // STEPS 1-3: the commitments in G1, the evaluations and the publics in Fr.
    info("> Checking the public inputs belong to F");
    const publics = fromObjectPublics(curve, publicsObject, vk);
    info("> Checking the commitments belong to G1 and the evaluations to F");
    const proof = fromObjectProof(curve, proofObject, vk);

    // STEP 4
    info("> Computing the challenges");
    const challenges = computeChallenges(curve, vk, publics, proof, logger);
    const xi = Fr.exp(challenges.xiSeed, BigInt(vk.powerW));

    // STEP 5
    info("> Computing Z_H(ξ) and Q(ξ)");
    const zh = Fr.sub(Fr.exp(xi, BigInt(vk.N)), Fr.one);
    if (!Fr.eq(Fr.mul(zh, proof.invZh), Fr.one)) {
        if (logger) logger.error("invZh is not 1/Z_H(ξ)");
        return false;
    }
    const { q } = computeQ(curve, vk, publics, proof, challenges, xi);
    if (vk.qPieces > 1 && !Fr.eq(joinQPieces(curve, piecesByIndex(vk, proof), xi, vk.power, vk.maxQDegree), q)) {
        if (logger) logger.error("The pieces of Q do not add up to Q(ξ) (pilfflonk/docs/protocol.md#q-pieces)");
        return false;
    }

    // STEP 6
    info("> Checking inv");
    const layout = { nBits: vk.power, powerW: vk.powerW, f: vk.layout.map(({ k, offsets }) => ({ k, offsets })) };
    checkLayout(curve, layout);
    const { f: roots } = computeRoots(curve, layout, challenges.xiSeed);
    const denominators = computeInverseDenominators(
        curve,
        roots,
        computeZerofiers(curve, roots, challenges.y),
        challenges.y,
    );
    if (!isValidInverse(curve, denominators, proof.inv)) {
        if (logger) {
            logger.error(
                "inv is not the inverse of the SHPLONK check's denominators (pilfflonk/docs/protocol.md#inverses)",
            );
        }
        return false;
    }

    // STEP 7
    info("> Checking the SHPLONK opening");
    const res = await verifyOpening(
        curve,
        {
            nBits: vk.power,
            powerW: vk.powerW,
            f: vk.layout.map(({ k, offsets }) => ({ k, offsets })),
            fixedCommitments: vk.fixedCommitments,
            commitments: proof.commitments,
            evaluations: openingEvaluations(vk, proof, q),
            xiSeed: challenges.xiSeed,
            alpha: challenges.alpha,
            y: challenges.y,
            W: proof.W,
            Wp: proof.Wp,
            X2: vk.X2,
        },
        logger,
    );

    if (logger) {
        if (res) {
            logger.info("PROOF VERIFIED SUCCESSFULLY");
        } else {
            logger.warn("Invalid Proof");
        }
    }
    return res;
}
