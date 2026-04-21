pragma circom 2.2.0;

include "bitify.circom";

/*
    Circuits that implement operations for the field Fp
*/

// Given a,c ∈ Fp and power_log >= 0, checks c == a^(2^power_log)
template ExpPowOfTwoFp(power_log) {
    signal input a;
    signal output c;
    
    signal im[power_log+1];
    im[0] <== a;
    for (var i = 1; i <= power_log; i++) {
        im[i] <== im[i-1] * im[i-1];
    }

    c <== im[power_log];
}

// Given NON-ZERO a ∈ Fp and is_square ∈ {0,1}, checks is_square == (a^((p-1)/2) == 1)
template IsSquareFp() {
    signal input a;
    signal output {binary} is_square;

    // Compute pow = a^((p-1)/2) ∈ {1,p-1}
    signal pow_31 <== ExpPowOfTwoFp(31)(a);
    signal pow_63 <== ExpPowOfTwoFp(32)(pow_31);
    signal legendre <-- pow_63 / pow_31;
    legendre * pow_31 === pow_63;

    // If legendre == 1, is_square = 1; else, is_square = 0
    is_square <-- (1 + legendre) / 2;
    2 * is_square === 1 + legendre;
}

// Given a ∈ Fp and sign ∈ {0,1}, checks sign == sign(a) := a % 2
template SignFp() {
    signal input a;
    signal output {binary} sign;

    signal a_bits[64] <== Num2Bits(64)(a);

    sign <== a_bits[0];
}

// Helper functions

function exp_pow_of_two_fp(a, power_log) {
    var c = a;
    for (var i = 0; i < power_log; i++) {
        c **= 2;
    }

    return c;
}

// (p-1)/2 = 2^63 - 2^31 -> x^((p-1)/2) = x^(2^63) / x^(2^31)
function is_square_fp(a) {
    var pow_31 = exp_pow_of_two_fp(a, 31);
    var pow_63 = exp_pow_of_two_fp(pow_31, 32);
    var pow = pow_63 / pow_31;

    return pow == 1;
}

function get_square_fp(a) {
    // 1] Compute x ∈ Fp such that x² - a is not a square
    var g = 7; // Goldilocks multiplicative generator
    var x = 1;
    var non_residue = x - a;
    var is_square = is_square_fp(non_residue);
    while (is_square) {
        x *= g;
        non_residue = x**2 - a;
        is_square = is_square_fp(non_residue);
    }

    // 2] Compute (x + sqrt(x² - a))^((p+1)/2)
    var c[2] = [x, 1];
    var res[2] = exp_cipolla(c, non_residue);

    return res[0];
}

/*
    Functions that implement operations for the extension field Fp² = F[X]/(X² - (x² - c)), 
    where x² - c is a non-residue in Fp.
    Functions are adapted from https://github.com/Plonky3/Plonky3/pull/439
*/

function mul_cipolla(a, b, nonresidue) {
    var real = a[0] * b[0] + nonresidue * a[1] * b[1];
    var imag = a[0] * b[1] + a[1] * b[0];

    return [real, imag];
}

function square_cipolla(a, nonresidue) {
    var real = a[0]**2 + nonresidue * a[1]**2;
    var imag = 2 * a[0] * a[1];

    return [real, imag];
}

function div_cipolla(a, b, nonresidue) {
    var denom = b[0]**2 - nonresidue * b[1]**2;
    var real = (a[0] * b[0] - nonresidue * a[1] * b[1]) / denom;
    var imag = (a[1] * b[0] - a[0] * b[1]) / denom;

    return [real, imag];
}

function exp_power_of_2_cipolla(a, power_log, nonresidue) {
    var res[2] = a;
    for (var i = 0; i < power_log; i++) {
        res = square_cipolla(res, nonresidue);
    }
    return res;
}

// Computes exponentiation by (p+1)/2 = 2^63 - 2^31 + 1
function exp_cipolla(a, nonresidue) {
    var pow_31[2] = exp_power_of_2_cipolla(a, 31, nonresidue);
    var pow_63[2] = exp_power_of_2_cipolla(pow_31, 32, nonresidue);
    var pow[2] = div_cipolla(pow_63, pow_31, nonresidue);
    var result[2] = mul_cipolla(a, pow, nonresidue);

    return result;
}