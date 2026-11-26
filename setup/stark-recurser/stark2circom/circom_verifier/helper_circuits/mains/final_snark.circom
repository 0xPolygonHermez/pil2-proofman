pragma circom 2.1.0;
pragma custom_templates;

include "mains/sha256_inputs.circom";

/*
    The final SNARK's circuit: the recursivef's proof verified (StarkVerifierBN128, key 0) and its publics
    hashed (GetSha256Inputs over the publics definitions vd_DEFS()). Incoming publics:
    [rootCVadcopFinal(4) | is_vadcop_final_proof(1) | real publics]; the flag is verified, not hashed.
    vkPos: the program VK's (4) position in the real publics, -1 if none.
*/
template FinalSnark(nSha, nDefs, vkPos) {
    var nPub = svd_nPublics(0);
    var nProof = nPub - 5;
    assert(vkPos < 0 || nProof >= vkPos + 4);

    signal output publicsHash;
    signal input publics[nPub];
    signal input proof[svd_proofSize(0)];

    component sV = StarkVerifierBN128(0);
    sV.proof <== proof;
    sV.publics <== publics;

    signal rootC[4];
    for (var i = 0; i < 4; i++) {
        rootC[i] <== publics[i];
    }
    signal is_vadcop_final_proof;
    is_vadcop_final_proof <== publics[4];
    is_vadcop_final_proof * (1 - is_vadcop_final_proof) === 0;
    // An aggregate's program VK is the key it verified under, not a key it chose.
    if (vkPos >= 0) {
        for (var i = 0; i < 4; i++) {
            (1 - is_vadcop_final_proof) * (rootC[i] - publics[5 + vkPos + i]) === 0;
        }
    }
    signal publicsProof[nProof];
    for (var i = 0; i < nProof; i++) {
        publicsProof[i] <== publics[i + 5];
    }

    publicsHash <== GetSha256Inputs(nSha, nDefs, vd_DEFS())(publicsProof, rootC);
}
