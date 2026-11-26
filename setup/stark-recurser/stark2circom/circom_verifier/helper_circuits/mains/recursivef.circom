pragma circom 2.1.0;
pragma custom_templates;

// recursivef: a Goldilocks STARK that verifies one vadcop_final proof.

// A recursivef (Goldilocks): its verifier with rootC and the publics public.
template Recursivef() {
    signal input rootC[4];
    signal input publics[svd_nPublics(0)];
    signal input proof[svd_proofSize(0)];

    component sV = StarkVerifierGL(0);
    sV.proof <== proof;
    sV.publics <== publics;
    sV.rootCIn <== rootC;
}
