pragma circom 2.2.0;

include "iszero.circom";
include "fp.circom";

/*
    Circuits that implement operations for the extension field Fp⁵ = F[X]/(X⁵-3)
*/

// Given a ∈ Fp⁵, is_zero ∈ {0,1}, checks is_zero == (a == 0)
// Cost: 18 (6 sub, 12 mul)
template IsZeroFp5() {
    signal input a[5];
    signal output {binary} is_zero;

    signal is_zero_i[5];
    var is_zero_sum = 0;
    for (var i = 0; i < 5; i++) {
        is_zero_i[i] <== IsZero()(a[i]);
        is_zero_sum += is_zero_i[i];
    }

    is_zero <== IsZero()(5 - is_zero_sum);
}

// Given a,c ∈ Fp⁵, checks c == -a
// Cost: 5 (5 neg)
template NegFp5() {
    signal input a[5];
    signal output c[5];

    for (var i = 0; i < 5; i++) {
        c[i] <== -a[i];
    }
}

// Given a,b,c ∈ Fp⁵, checks c == a+b
// Cost: 5 (5 add)
template AddFp5() {
    signal input a[5];
    signal input b[5];
    signal output c[5];

    for (var i = 0; i < 5; i++) {
        c[i] <== a[i] + b[i];
    }
}

// Given a,b,c ∈ Fp⁵, checks c == a-b
// Cost: 5 (5 sub)
template SubFp5() {
    signal input a[5];
    signal input b[5];
    signal output c[5];

    for (var i = 0; i < 5; i++) {
        c[i] <== a[i] - b[i];
    }
}


// Given a,b,c ∈ Fp⁵, checks c == a·b
// Cost: 49 (25 mul, 20 add, 4 scalar mul)
template MulFp5() {
    signal input a[5];
    signal input b[5];
    signal output c[5];

    signal a1b4 <== a[1]*b[4];
    signal a2b3 <== a[2]*b[3];
    signal a3b2 <== a[3]*b[2];
    signal a4b1 <== a[4]*b[1];
    signal a1b0 <== a[1]*b[0];
    signal a2b4 <== a[2]*b[4];
    signal a3b3 <== a[3]*b[3];
    signal a4b2 <== a[4]*b[2];
    signal a1b1 <== a[1]*b[1];
    signal a2b0 <== a[2]*b[0];
    signal a3b4 <== a[3]*b[4];
    signal a4b3 <== a[4]*b[3];
    signal a1b2 <== a[1]*b[2];
    signal a2b1 <== a[2]*b[1];
    signal a3b0 <== a[3]*b[0];
    signal a4b4 <== a[4]*b[4];
    signal a1b3 <== a[1]*b[3];
    signal a2b2 <== a[2]*b[2];
    signal a3b1 <== a[3]*b[1];
    signal a4b0 <== a[4]*b[0];
    c[0] <== a[0]*b[0] + 3*(a1b4 + a2b3 + a3b2 + a4b1);
    c[1] <== a[0]*b[1] + a1b0 + 3*(a2b4 + a3b3 + a4b2);
    c[2] <== a[0]*b[2] + a1b1 + a2b0 + 3*(a3b4 + a4b3);
    c[3] <== a[0]*b[3] + a1b2 + a2b1 + a3b0 + 3*(a4b4);
    c[4] <== a[0]*b[4] + a1b3 + a2b2 + a3b1 + a4b0;
}

// Given a,b,c ∈ Fp⁵, checks c == a·b
// Cost: 49 (25 mul, 20 add, 4 scalar mul)
template ScalarMulFp5() {
    signal input a[5];
    signal input b;
    signal output c[5];

    for (var i = 0; i < 5; i++) {
        c[i] <== a[i] * b;
    }
}

// Given a,c ∈ Fp⁵, checks c == a²
// Cost: 34 (15 mul, 10 add, 9 scalar mul)
template SquareFp5() {
    signal input a[5];
    signal output c[5];

    signal a1a4 <== a[1]*a[4];
    signal a2a3 <== a[2]*a[3];
    signal a2a4 <== a[2]*a[4];
    signal a3a3 <== a[3]*a[3];
    signal a1a1 <== a[1]*a[1];
    signal a3a4 <== a[3]*a[4];
    signal a1a2 <== a[1]*a[2];
    signal a4a4 <== a[4]*a[4];
    signal a1a3 <== a[1]*a[3];
    signal a2a2 <== a[2]*a[2];
    c[0] <== a[0]*a[0] + 6*(a1a4 + a2a3);
    c[1] <== 2*a[0]*a[1] + 3*(2*a2a4 + a3a3);
    c[2] <== 2*a[0]*a[2] + a1a1 + 6*a3a4;
    c[3] <== 2*(a[0]*a[3] + a1a2) + 3*a4a4;
    c[4] <== 2*(a[0]*a[4] + a1a3) + a2a2;
}

// Given NON-ZERO a,c ∈ Fp⁵, checks c == 1/a
// Cost: 126 (72 mul, 44 add, 9 scalar mul, 1 div)
template InvFp5() {
    signal input a[5];
    signal output c[5];

    // Get the inverse of a
    c <-- get_inverse_fp5(a);

    // Check that a·c == 1
    signal ac[5] <== MulFp5()(a, c);
    ac === [1, 0, 0, 0, 0];
}

// Given a,c ∈ Fp⁵, checks c == 1/a
template Inv0Fp5() {
    signal input a[5];
    signal output {binary} a_is_zero;
    signal output c[5];

    // Check if a is zero
    a_is_zero <== IsZeroFp5()(a);

    // Get the inverse of a
    c <-- get_inverse_fp5(a);
    signal ac[5] <== MulFp5()(a, c);
    // If a is zero, check c == 0; otherwise, check a·c == 1
    for (var i = 0; i < 5; i++) {
        a_is_zero * (c[i] - (ac[i] - (i == 0))) + (ac[i] - (i == 0)) === 0;
    }
}

// Given a,c ∈ Fp⁵ and NON-ZERO b ∈ Fp⁵, checks c == a/b
// Cost: 175 (97 mul, 64 add, 13 scalar mul, 1 div)
template DivFp5() {
    signal input a[5];
    signal input b[5];
    signal output c[5];

    // Get the inverse of b
    signal b_inv[5] <== InvFp5()(b);

    // Check that c == a·b^(-1)
    c <== MulFp5()(a, b_inv);
}

// Given a,b,c ∈ Fp⁵, checks c == a/b
template Div0Fp5() {
    signal input a[5];
    signal input b[5];
    signal output c[5];

    // Get the inverse of b
    // If b is zero, b^(-1) is zero
    signal b_inv[5];
    (_, b_inv) <== Inv0Fp5()(b);

    // Check that c == a·b^(-1)
    c <== MulFp5()(a, b_inv);
}

// Given a,c ∈ Fp⁵, checks c == aᵖ
// Cost: 4 (4 mul)
template FirstFrobenius() {
    signal input a[5];
    signal output c[5];

    c[0] <== a[0];
    c[1] <== a[1] * 1041288259238279555;
    c[2] <== a[2] * 15820824984080659046;
    c[3] <== a[3] * 211587555138949697;
    c[4] <== a[4] * 1373043270956696022;
}

// Given a,c ∈ Fp⁵, checks c == aᵖ･ᵖ
// Cost: 4 (4 mul)
template SecondFrobenius() {
    signal input a[5];
    signal output c[5];

    c[0] <== a[0];
    c[1] <== a[1] * 15820824984080659046;
    c[2] <== a[2] * 1373043270956696022;
    c[3] <== a[3] * 1041288259238279555;
    c[4] <== a[4] * 211587555138949697;
}

// Given a ∈ Fp⁵ and c ∈ Fp, checks c == a^(p⁴ + p³ + p² + p + 1)
template ExpFifthCyclotomicFp5() {
    signal input a[5];
    signal output c;

    signal t0[5] <== FirstFrobenius()(a);   // a^p
    signal t1[5] <== SecondFrobenius()(a);  // a^(p²)
    signal t2[5] <== MulFp5()(t1, t0);      // a^(p² + p)
    signal t3[5] <== SecondFrobenius()(t2); // a^(p⁴ + p³)
    signal t4[5] <== MulFp5()(t2, t3);      // a^(p⁴ + p³ + p² + p)

    // c = a^(p⁴ + p³ + p² + p + 1)
    signal a1t44 <== a[1] * t4[4];
    signal a2t43 <== a[2] * t4[3];
    signal a3t42 <== a[3] * t4[2];
    signal a4t41 <== a[4] * t4[1];
    c <== a[0] * t4[0] + 3 * (a1t44 + a2t43 + a3t42 + a4t41);
}

// Given NON-ZERO a ∈ Fp⁵ and is_square ∈ {0,1}, checks is_square == (a^((p⁵-1)/2) == 1)
template IsSquareFp5() {
    signal input a[5];
    signal output pow;
    signal output {binary} is_square;

    // Compute pow = a^(p⁴ + p³ + p² + p + 1)
    pow <== ExpFifthCyclotomicFp5()(a);

    // Check if pow is a square over Fp
    is_square <== IsSquareFp()(pow);
}

// Given a SQUARE a ∈ Fp⁵ and c ∈ Fp⁵, checks c == sqrt(a)
template SqrtFp5() {
    signal input a[5];
    signal output c[5];

    // Get a square root of a
    c <-- get_sqrt_fp5(a);

    // Check that c² == a
    signal mul[5] <== MulFp5()(c, c);
    for (var i = 0; i < 5; i++) {
        mul[i] === a[i];
    }
}

template SignFp5() {
    signal input a[5];
    signal output {binary} sign;

    signal sign_i[5];
    signal is_zero_i[5];
    for (var i = 0; i < 5; i++) {
        sign_i[i] <== SignFp()(a[i]);
        is_zero_i[i] <== IsZero()(a[i]);
    }

    // sign(a[0]) || (a[0] == 0 && sign(a[1])) || (a[0] == 0 && a[1] == 0 && sign(a[2])) 
    // || (a[0] == 0 && a[1] == 0 && a[2] == 0 && sign(a[3])) 
    // || (a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0 && sign(a[4]))
    signal sign_3_sign_4 <== sign_i[3] + is_zero_i[3] * sign_i[4];
    signal sign_2_sign_3_sign_4 <== sign_i[2] + is_zero_i[2] * sign_3_sign_4;
    signal sign_1_sign_2_sign_3_sign_4 <== sign_i[1] + is_zero_i[1] * sign_2_sign_3_sign_4;
    sign <== sign_i[0] + is_zero_i[0] * sign_1_sign_2_sign_3_sign_4;
}

// Helper functions

function is_zero_fp5(a) {
    return ((a[0] == 0) && (a[1] == 0) && (a[2] == 0) && (a[3] == 0) && (a[4] == 0));
}

// Given a,b ∈ Fp⁵, computes c := a+b ∈ Fp⁵
function add_fp5(a, b) {
    var c[5];
    for (var i = 0; i < 5; i++) {
        c[i] = a[i] + b[i];
    }
    return c;
}

// Given a,b ∈ Fp⁵, computes c := a·b ∈ Fp⁵
function mul_fp5(a, b) {
    var c[5];

    c[0] = a[0]*b[0] + 3*(a[1]*b[4] + a[2]*b[3] + a[3]*b[2] + a[4]*b[1]);
    c[1] = a[0]*b[1] + a[1]*b[0] + 3*(a[2]*b[4] + a[3]*b[3] + a[4]*b[2]);
    c[2] = a[0]*b[2] + a[1]*b[1] + a[2]*b[0] + 3*(a[3]*b[4] + a[4]*b[3]);
    c[3] = a[0]*b[3] + a[1]*b[2] + a[2]*b[1] + a[3]*b[0] + 3*(a[4]*b[4]);
    c[4] = a[0]*b[4] + a[1]*b[3] + a[2]*b[2] + a[3]*b[1] + a[4]*b[0];

    return c;
}

function square_fp5(a) {
    var c[5];

    c[0] = a[0]*a[0] + 6*(a[1]*a[4] + a[2]*a[3]);
    c[1] = 2*a[0]*a[1] + 3*(2*a[2]*a[4] + a[3]*a[3]);
    c[2] = 2*a[0]*a[2] + a[1]*a[1] + 6*a[3]*a[4];
    c[3] = 2*(a[0]*a[3] + a[1]*a[2]) + 3*a[4]*a[4];
    c[4] = 2*(a[0]*a[4] + a[1]*a[3]) + a[2]*a[2];

    return c;
}

function exp_pow_of_two_fp5(a, power_log) {
    var c[5] = a;
    for (var i = 0; i < power_log; i++) {
        c = square_fp5(c);
    }

    return c;
}

// Given a ∈ Fp⁵, compute c := aᵖ ∈ Fp⁵
function first_frobenius(a) {
    var c[5];

    c[0] = a[0];
    c[1] = a[1] * 1041288259238279555;
    c[2] = a[2] * 15820824984080659046;
    c[3] = a[3] * 211587555138949697;
    c[4] = a[4] * 1373043270956696022;

    return c;
}

// Given a ∈ Fp⁵, compute c := aᵖ･ᵖ ∈ Fp⁵
function second_frobenius(a) {
    var c[5];

    c[0] = a[0];
    c[1] = a[1] * 15820824984080659046;
    c[2] = a[2] * 1373043270956696022;
    c[3] = a[3] * 1041288259238279555;
    c[4] = a[4] * 211587555138949697;

    return c;
}

function exp_fifth_cyclotomic_fp5(a) {
    var t0[5] = mul_fp5(second_frobenius(a), first_frobenius(a)); // a^(p² + p)
    var t1[5] = second_frobenius(t0);                             // a^(p⁴ + p³)
    var t2[5] = mul_fp5(t0, t1);                                  // a^(p⁴ + p³ + p² + p)

    return a[0] * t2[0] + 3 * (a[1] * t2[4] + a[2] * t2[3] + a[3] * t2[2] + a[4] * t2[1]); // a^x
}

// Given a ∈ Fp⁵, compute c := 1/a ∈ Fp⁵
// as 1/a = a^(x-1) / a^x, where x = p⁴ + p³ + p² + p + 1
// Cost: 126 (72 mul, 44 add, 9 scalar mul, 1 div)
function get_inverse_fp5(a) {
    if (is_zero_fp5(a)) {
        return [0, 0, 0, 0, 0];
    }

    var t0[5] = mul_fp5(second_frobenius(a), first_frobenius(a)); // a^(p² + p)
    var t1[5] = second_frobenius(t0);                             // a^(p⁴ + p³)
    var t2[5] = mul_fp5(t0, t1);                                  // a^(p⁴ + p³ + p² + p)

    var t3 = a[0] * t2[0] + 3 * (a[1] * t2[4] + a[2] * t2[3] + a[3] * t2[2] + a[4] * t2[1]); // a^x

    var t4 = 1 / t3; // 1 / a^x
    // var t4 =  (t3 != 0) ? 1 / t3 : 0;

    // a^(x-1) * a^(-x)
    var c[5];
    for (var i = 0; i < 5; i++) {
        c[i] = t2[i] * t4;
    }

    return c;
}

function div_fp5(a, b) {
    var b_inv[5] = get_inverse_fp5(b);
    var c[5] = mul_fp5(a, b_inv);

    return c;
}

function get_sqrt_fp5(a) {
    // We compute the square root using the identity:
    //      1     p⁴ + p³ + p² + p + 1       p+1          p+1
    //     --- + ----------------------  = (-----)·p³ + (-----)·p + 1
    //      2              2                  2            2

    // sqrt(0) = 0 and sqrt(1) = 1
    if (((a[0] == 0) && (a[1] == 0) && (a[2] == 0) && (a[3] == 0) && (a[4] == 0))
        || ((a[0] == 1) && (a[1] == 0) && (a[2] == 0) && (a[3] == 0) && (a[4] == 0))) {
        return a;
    }

    // First Part: Compute the square root of a^-(p⁴ + p³ + p² + p + 1) ∈ Fp
    var pow_x = exp_fifth_cyclotomic_fp5(a); // a^x
    var x = get_square_fp(1 / pow_x); // sqrt(1 / a^x)

    // Second Part: Compute a^(((p+1)/2)p³ + ((p+1)/2)p + 1)

    // 1] Compute a^((p+1)/2). Notice (p+1)/2 = 2^63 - 2^31 + 1
    var pow_31[5] = exp_pow_of_two_fp5(a, 31);
    var pow_63[5] = exp_pow_of_two_fp5(pow_31, 32);
    var pow[5] = mul_fp5(a, div_fp5(pow_63, pow_31));

    // 2] Compute the rest using Frobenius
    var pow_frob[5] = first_frobenius(pow);
    var y[5] = pow_frob;

    pow_frob = second_frobenius(pow_frob);
    y = mul_fp5(y, pow_frob);
    y = mul_fp5(y, a);

    return [x*y[0], x*y[1], x*y[2], x*y[3], x*y[4]];
}