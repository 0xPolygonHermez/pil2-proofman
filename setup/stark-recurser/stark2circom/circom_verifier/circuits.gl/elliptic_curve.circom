pragma circom 2.2.0;

include "mux1.circom";

include "fp5.circom";

/*
    Circuits that implement operations for the elliptic curve 
                    E: y² = x³ + Ax + B
    defined over the extension field Fp⁵ = F[X]/(X⁵-3)
*/

// TODO: Use buses when errors get corrected!
// bus PointFp5(){
//     signal x[5];
//     signal y[5];
// }

// Give P ∈ E(Fp⁵), checks if P == 𝒪
// It assumes that 𝒪 can be represented as (0,0) in the curve
template IsInfinityECFp5() {
    signal input P[2][5];
    signal output {binary} is_inf;

    signal {binary} is_inf_x <== IsZeroFp5()(P[0]);
    signal {binary} is_inf_y <== IsZeroFp5()(P[1]);
    is_inf <== is_inf_x * is_inf_y;
}

// Given P,Q,R ∈ E(Fp⁵), checks R == P+Q
// It supports the cases where P or Q are 𝒪 (represented as (0,0)) 
// or where Q == P,-P. However, the result will be incorrect in these cases.
// Exception: When P == Q == 𝒪, the result will correctly be 𝒪.
// So, extra care must be taken when using this circuit.
template AddECFp5() {
    signal input P[2][5];
    signal input Q[2][5];
    signal output R[2][5];

    var Px[5] = P[0];
    var Py[5] = P[1];
    var Qx[5] = Q[0];
    var Qy[5] = Q[1];

    signal slope_num[5] <== SubFp5()(Qy, Py);
    signal slope_den[5] <== SubFp5()(Qx, Px);
    signal slope[5] <== Div0Fp5()(slope_num, slope_den);

    signal slope_sq[5] <== SquareFp5()(slope);
    signal xRa[5] <== SubFp5()(slope_sq, Px);
    R[0] <== SubFp5()(xRa, Qx);

    signal xdiff[5] <== SubFp5()(Px, R[0]);
    signal yRa[5] <== MulFp5()(slope, xdiff);
    R[1] <== SubFp5()(yRa, Py);
}

// Given P,R ∈ E(Fp⁵), checks R == 2·P
// It assumes P,R != 𝒪, Py != 0
template DoubleECFp5(A) {
    signal input P[2][5];
    signal output R[2][5];

    var Px[5] = P[0];
    var Py[5] = P[1];

    signal x_sq[5] <== SquareFp5()(Px);
    signal slope_num_a[5] <== ScalarMulFp5()(x_sq, 3);
    signal slope_num[5] <== AddFp5()(slope_num_a, A);
    signal slope_den[5] <== ScalarMulFp5()(Py, 2);
    signal slope[5] <== DivFp5()(slope_num, slope_den);

    signal slope_sq[5] <== SquareFp5()(slope);
    signal xRa[5] <== SubFp5()(slope_sq, Px);
    R[0] <== SubFp5()(xRa, Px);

    signal xdiff[5] <== SubFp5()(Px, R[0]);
    signal yRa[5] <== MulFp5()(slope, xdiff);
    R[1] <== SubFp5()(yRa, Py);
}

// Given x,y ∈ Fp⁵ and S ∈ E(Fp⁵), checks S == hash_to_curve(x,y)
// It assumes x != y and S != 𝒪
template HashToCurve(A, B, Z, C1, C2) {
    signal input x[5];
    signal input y[5];
    signal output S[2][5];

    signal P[2][5] <== MapToCurve(A, B, Z, C1, C2)(x); // P != 𝒪
    signal Q[2][5] <== MapToCurve(A, B, Z, C1, C2)(y); // Q != 𝒪
    signal R[2][5] <== AddECFp5()(P, Q);               // (w.h.p. over x,y) Q != P,-P => R != 𝒪
    S <== ClearCofactor(A, B)(R);                      // (w.h.p. over x,y) ord(R) > h => S != 𝒪
}

// Given a ∈ Fp⁵ and R ∈ E(Fp⁵), checks R == map_to_curve(a)
// It assumes R != 𝒪
template MapToCurve(A, B, Z, C1, C2) {
    signal input u[5];
    signal output R[2][5];

    signal tv1a[5] <== SquareFp5()(u);      // tv1 = u²
    signal tv1[5] <== MulFp5()(Z, tv1a);    // tv1 = Z·u²
    signal tv2a[5] <== SquareFp5()(tv1);    // tv2 = Z²·u⁴

    signal x1a[5] <== AddFp5()(tv2a, tv1);  //  x1 = Z²·u⁴ + Z·u²
    signal {binary} e1;
    signal x1b[5];
    (e1, x1b) <== Inv0Fp5()(x1a); //  e1 = (x1 == 0), x1 = 1 / (Z²·u⁴ + Z·u²)
    signal x1c[5] <== MultiMux1(5)([[x1b[0] + 1, x1b[1], x1b[2], x1b[3], x1b[4]], C2], e1); // If (tv1 + tv2) == 0, set x1 = -1 / Z
    signal x1[5] <== MulFp5()(x1c, C1);   // x1 = (-B / A) * (1 + (1 / (Z²·u⁴ + Z·u²)))

    signal gx1a[5] <== SquareFp5()(x1);        // gx1 = x1²
    signal gx1b[5] <== AddFp5()(gx1a, A);      // gx1 = x1² + A
    signal gx1c[5] <== MulFp5()(gx1b, x1);     // gx1 = x1³ + A·x1
    signal gx1[5] <== AddFp5()(gx1c, B);       // gx1 = x1³ + A·x1 + B

    signal x2[5] <== MulFp5()(tv1, x1);        // x2 = Z·u²·x1
    signal tv2[5] <== MulFp5()(tv1, tv2a);     // tv2 = Z³·u⁶
    signal gx2[5] <== MulFp5()(gx1, tv2);      // gx2 = (Z·u²)³·gx1

    signal {binary} e2;
    (_, e2) <== IsSquareFp5()(gx1);

    signal x[5] <== MultiMux1(5)([x2, x1], e2); // If is_square(gx1), x = x1, else x = x2
    signal y2[5] <== MultiMux1(5)([gx2, gx1], e2); // If is_square(gx1), y2 = gx1, else y2 = gx2

    signal y[5] <== SqrtFp5()(y2); // y = sqrt(y2)

    // Fix the sign of y
    signal {binary} sign_u <== SignFp5()(u);
    signal {binary} sign_y <== SignFp5()(y);
    signal {binary} nor_sign <== (1 - sign_u) * (1 - sign_y);
    signal {binary} e3 <== sign_u * sign_y + nor_sign; // e3 = 1 if sign_u == sign_y, else e3 = 0
    signal y_neg[5] <== NegFp5()(y);

    R[0] <== x;
    R[1] <== MultiMux1(5)([y_neg, y], e3);
}

// Given P,R ∈ E(Fp⁵), checks R = h·P, where h is the cofactor of the curve
// It assumes P,R != 𝒪 and ord(P) > h
template ClearCofactor(A, B) {
    signal input P[2][5];
    signal output R[2][5];

    // If it is the EcGFp5 curve
    if ((A[0] == 6148914689804861439) && (A[1] == 263) && (A[2] == 0) && (A[3] == 0) && (A[4] == 0) 
         && (B[0] == 15713893096167979237) && (B[1] == 6148914689804861265) && (B[2] == 0) && (B[3] == 0) && (B[4] == 0)) 
    {
        // Cofactor is 2, just double P
        R <== DoubleECFp5(A)(P);
    } 
    // If it is the EcMasFp5 curve
    else if ((A[0] == 3) && (A[1] == 0) && (A[2] == 0) && (A[3] == 0) && (A[4] == 0) 
         && (B[0] == 0) && (B[1] == 0) && (B[2] == 0) && (B[3] == 0) && (B[4] == 8)) 
    {
        // Cofactor is 1
        R <== P;
    } 
    else 
    {
        log("ClearCofactor: Unsupported curve A = [", A[0], A[1], A[2], A[3], A[4], "], B = [", B[0], B[1], B[2], B[3], B[4], "]");
        assert(1 == 0);
    }
}