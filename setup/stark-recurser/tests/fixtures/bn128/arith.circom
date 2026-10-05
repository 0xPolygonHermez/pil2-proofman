pragma circom 2.1.0;

// plonk2pil's BN128 fixture, for tests/plonk2pil_bn128.rs: multiplications, additions and constants,
// several of them wider than 64 bits, so the r1cs carries 32-byte coefficients. Compiled with --O1,
// which keeps the linear constraints for plonk2pil's sum gates. No custom gates.
template Arith() {
    signal input a;
    signal input b;
    signal input c[4];
    signal output out[4];

    // Both factors are linear combinations with a constant; the right one's is r - 1, i.e. -1.
    signal p <== (a + 3 * b + 7) * (c[0] - c[1] + 21888242871839275222246405745257275088548364400416034343698204186575808495616);

    // A sum of more than three terms, which plonk2pil folds two at a time through additions.
    signal q <== (a + 2 * b + 3 * c[0] + 4 * c[1] + 5 * c[2] + 6 * c[3] + 18446744073709551616) * p;

    // Linear constraints with wide coefficients.
    signal s <== 340282366920938463463374607431768211457 * (a - b) + 12345678901234567890123456789 * c[3];
    signal t <== q + 7 * s - 99;

    out[0] <== p * q;
    out[1] <== a * a + 123456789012345678901234567890;
    out[2] <== (s + t) * (c[2] - 5);
    out[3] <== t;
}

component main {public [a, b]} = Arith();
