// The transcript sequence of spec-seed.md A.4 for a proof of one instance (D2), replayed from the
// vkey, the publics and the proof, as snarkjs' fflonk verifier replays its own
// (src/fflonk_verify.js, computeChallenges). One transcript for the whole proof (transcript.js):
//
// 1. absorb digest mod r, the number of instances of the AIR (1) and the publics;
// 2. for each stage s = 1 … nStages:
//    1. absorb the commitments of the non-fixed f_i of stage s, in the global order (A.5);
//    2. the air values, airgroup values and proof values of stage s: none in v1 (D2);
//    3. if s < nStages, squeeze the numChallenges[s] challenges of stage s + 1, one per call;
// 3. squeeze std_vc; absorb the commitments of Q; squeeze xiSeed (std_xi);
// 4. absorb the evaluations: the fixed columns', then the others', each in the order of the evMap;
//    then the pieces Q_i(ξ) if Q is split;
// 5. squeeze α_S; absorb [W]₁; squeeze y.
//
// The fixed commitments are not absorbed: the digest binds them, with the rest of the vkey.

import { Keccak256Transcript } from "./transcript.js";

// The challenges of a proof: stages[s][j] the j-th challenge of stage s, for 2 ≤ s ≤ nStages;
// stdVc and xiSeed, the challenges of stages nStages + 1 and nStages + 2; alpha (α_S) and y, the
// SHPLONK's (A.5). `transcript`, a Keccak256Transcript, is for tests to observe what is absorbed.
export function computeChallenges(curve, vk, publics, proof, logger, transcript = new Keccak256Transcript(curve)) {
    const Fr = curve.Fr;
    const debug = (name, value) => {
        if (logger) logger.debug(`··· challenges.${name}: ${Fr.toString(value)}`);
    };

    // Step 1.
    transcript.addScalar(vk.digestFr);
    transcript.addScalar(Fr.one);
    publics.forEach((p) => transcript.addScalar(p));

    // The commitments of the non-fixed f_i, in the order of the layout, and the stage of each.
    const nFixed = vk.layout.length - proof.commitments.length;
    const committed = proof.commitments.map((commitment, i) => ({ commitment, stage: vk.layout[nFixed + i].stage }));
    const absorbStage = (s) => {
        for (const { commitment, stage } of committed) {
            if (stage === s) transcript.addPolCommitment(commitment);
        }
    };

    // Step 2.
    const stages = [];
    for (let s = 1; s <= vk.nStages; s++) {
        absorbStage(s);
        if (s < vk.nStages) {
            stages[s + 1] = [];
            for (let j = 0; j < vk.numChallenges[s]; j++) {
                stages[s + 1].push(transcript.squeeze());
                debug(`stage${s + 1}[${j}]`, stages[s + 1][j]);
            }
        }
    }

    // Step 3.
    const stdVc = transcript.squeeze();
    debug("std_vc", stdVc);
    absorbStage(vk.qStage);
    const xiSeed = transcript.squeeze();
    debug("xiSeed", xiSeed);

    // Step 4.
    vk.evaluationOrder.forEach((i) => transcript.addScalar(proof.evaluations[i]));
    proof.qPieces.forEach((q) => transcript.addScalar(q));

    // Step 5.
    const alpha = transcript.squeeze();
    debug("alpha", alpha);
    transcript.addPolCommitment(proof.W);
    const y = transcript.squeeze();
    debug("y", y);

    return { stages, stdVc, xiSeed, alpha, y };
}

// The value of the challenge operand (stage, stageId) of the qVerifier (qverifier.js): the
// challenges of stage s ≤ nStages, std_vc for nStages + 1 and xiSeed for nStages + 2 (A.4).
export function challengeOf(vk, challenges, stage, stageId) {
    if (stage === vk.nStages + 1 && stageId === 0) return challenges.stdVc;
    if (stage === vk.nStages + 2 && stageId === 0) return challenges.xiSeed;
    const value = challenges.stages[stage]?.[stageId];
    if (value === undefined) throw new Error(`challengeOf: no challenge of stage ${stage}, stageId ${stageId}`);
    return value;
}
