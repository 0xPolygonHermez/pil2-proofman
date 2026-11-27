pragma circom 2.1.0;
pragma custom_templates;

// The compressed vadcop_final.

// The compressed vadcop_final: the incoming proof's publics are [is_vadcop_final_proof | publics];
// this proof re-exposes the flag-free rest.
template FinalCompressed(nPub, nReal) {
    signal input publics[nPub];
    signal output publics_stripped[nReal];
    signal input proof[key_proofSize(0)];

    component sV = StarkVerifierGL(0);
    sV.proof <== proof;
    if (nPub > 0) { sV.publics <== publics; }
    for (var i = 0; i < nReal; i++) { publics_stripped[i] <== publics[i + 1]; }
}
