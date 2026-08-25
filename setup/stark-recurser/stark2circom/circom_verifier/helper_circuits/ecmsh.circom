pragma circom 2.1.0;
pragma custom_templates;

include "bitify.circom";
include "elliptic_curve.circom";

/*
    Classical one-point elliptic-curve multiset hash (ECMSH) over the curve
                    E: y² = x³ + Ax + B,
    defined over the extension field Fp⁵ = F[X]/(X⁵-3) over Goldilocks.

    Let G ⊆ E(Fp⁵) be a subgroup of prime order. Supported curves:
        - EcMasFp5: A = 3,        B = 8X⁴       (cofactor 1, G = E)
        - EcGFp5:   A = a₀ + a₁X, B = b₀ + b₁X  (cofactor 2)

    The digest of a multiset {m_1, ..., m_n} is
                    H = g + h(m_1) + ... + h(m_n),
    where h is a map-to-curve and g is an (fixed non-zero generator of G)
    offset away from 𝒪 and cancels when two digests are compared for equality.

    Instead of computing a hash-to-curve in-circuit, h is checked as a relation
    (the y-increment method): the prover natively finds the smallest tweak k < T
    such that y = m·T + k is a valid ordinate, solves the curve equation for the
    abscissa x, and supplies (k, x) as witnesses. The circuit only checks
        1. m < 2^MSG_BITS and k < T = 2^TWEAK_BITS   (range checks)
        2. y = m·T + k, y != 0                       (linear + one constraint)
        3. y² = x³ + Ax + B in Fp⁵                   (on-curve check)
        4. (x, y) ∈ G ⊆ E                            (subgroup check)
    Since y lives in the base field, y² is a single multiplication, A·x is a
    linear combination, and four of the five coordinate equalities of the
    on-curve check cost no multiplication.
    The subgroup check is realized by clearing the cofactor: h(m) is the mapped
    point itself for EcMasFp5 and the mapped point doubled once for EcGFp5. The
    y != 0 constraint excludes the 2-torsion ordinate, which would otherwise
    clear to 𝒪 and let an event contribute nothing to the digest.

    Inverse-freeness: all encoded ordinates satisfy 0 < y < 2^(MSG_BITS + TWEAK_BITS) < p/2,
    so a mapped point and its negation are never both reachable and inserts
    cannot cancel to 𝒪.
    Together with the offset g, this keeps the incomplete affine addition of
    AddECFp5 away from its exceptional cases (doubling, negation pairs, 𝒪) for
    honestly generated insert-only digests.
*/

// Fixed accumulator offset: the subgroup generator of the chosen curve.
function ECMSH_OFFSET(A, B) {
    // EcMasFp5 generator (0, 18446741870424883713·X²)
    if ((A[0] == 3) && (A[1] == 0) && (A[2] == 0) && (A[3] == 0) && (A[4] == 0)
         && (B[0] == 0) && (B[1] == 0) && (B[2] == 0) && (B[3] == 0) && (B[4] == 8))
    {
        return [[0, 0, 0, 0, 0], [0, 0, 18446741870424883713, 0, 0]];
    }
    // EcGFp5 generator (0, y_g)
    if ((A[0] == 6148914689804861439) && (A[1] == 263) && (A[2] == 0) && (A[3] == 0) && (A[4] == 0)
         && (B[0] == 15713893096167979237) && (B[1] == 6148914689804861265) && (B[2] == 0) && (B[3] == 0) && (B[4] == 0))
    {
        return [[0, 0, 0, 0, 0], [11002749681768771274, 11642892185553879191, 663487151061499164, 2764891638068209098, 2343917403129570002]];
    }
    assert(1 == 0); // unsupported curve
    return [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]];
}

// Constant-by-signal multiplication in Fp⁵ = F[X]/(X⁵-3): returns C·x as
// linear combinations of x, at no constraint cost.
function ECMSH_CONST_MUL_FP5(C, x) {
    var r[5] = [0, 0, 0, 0, 0];
    for (var i = 0; i < 5; i++) {
        for (var j = 0; j < 5; j++) {
            if (i + j < 5) {
                r[i + j] += C[i] * x[j];
            } else {
                r[i + j - 5] += 3 * C[i] * x[j];
            }
        }
    }
    return r;
}

// Given a message m and prover-supplied witnesses (k, x), checks that
// (x, m·T + k) lies on E and returns its cofactor-cleared image P ∈ G.
// The native counterpart scans k = 0, ..., T-1 and solves the depressed cubic
// x³ + Ax + (B - y²) = 0 over Fp⁵ for the first valid y; a tweak range of
// T = 256 fails for all k with probability below (1/3)^256.
template EcmshMapToCurve(A, B, MSG_BITS, TWEAK_BITS) {
    // y = m·T + k must stay below p/2 ≈ 2⁶³ to keep the image inverse-free
    assert(MSG_BITS + TWEAK_BITS <= 62);

    signal input m;       // message
    signal input k;       // tweak witness
    signal input x[5];    // abscissa witness
    signal output P[2][5];

    // 1. Range checks
    _ <== Num2Bits(MSG_BITS)(m);
    _ <== Num2Bits(TWEAK_BITS)(k);

    // 2. Encoded ordinate, in the base field Fp ⊂ Fp⁵; y != 0 excludes the
    // 2-torsion ordinate on cofactor-2 curves
    signal y <== m * (1 << TWEAK_BITS) + k;
    signal y_inv <-- y != 0 ? 1 / y : 0;
    y * y_inv === 1;

    // 3. On-curve check y² = x³ + Ax + B
    // Since y is a base-field element, y² only meets coordinate 0 of the RHS;
    // A·x and B are linear, so the remaining coordinates cost no multiplication.
    signal x_sq[5] <== SquareFp5()(x);
    signal x_cb[5] <== MulFp5()(x_sq, x);
    var ax[5] = ECMSH_CONST_MUL_FP5(A, x);

    y * y === x_cb[0] + ax[0] + B[0];
    x_cb[1] + ax[1] + B[1] === 0;
    x_cb[2] + ax[2] + B[2] === 0;
    x_cb[3] + ax[3] + B[3] === 0;
    x_cb[4] + ax[4] + B[4] === 0;

    // 4. Subgroup check: clear the cofactor so P ∈ G
    P <== ClearCofactor(A, B)([x, [y, 0, 0, 0, 0]]);
}

// Given an accumulator state and one message with its map-to-curve
// witnesses, checks the mapping and returns accIn + h(m).
// accIn must be a non-exceptional state (see header); this holds for any
// accumulator chain started at ECMSH_OFFSET(A, B) over an inverse-free image.
template EcmshInsert(A, B, MSG_BITS, TWEAK_BITS) {
    signal input accIn[2][5];
    signal input m;
    signal input k;
    signal input x[5];
    signal output accOut[2][5];

    signal P[2][5] <== EcmshMapToCurve(A, B, MSG_BITS, TWEAK_BITS)(m, k, x);
    accOut <== AddECFp5()(accIn, P);
}

// Insert-only digest of n messages: g + h(m[0]) + ... + h(m[n-1]).
// Two views of the same events are equal as multisets iff their digests are
// equal componentwise.
template EcmshDigest(A, B, n, MSG_BITS, TWEAK_BITS) {
    signal input m[n];
    signal input k[n];
    signal input x[n][5];
    signal output digest[2][5];

    var G[2][5] = ECMSH_OFFSET(A, B);

    signal acc[n + 1][2][5];
    acc[0] <== G;
    for (var i = 0; i < n; i++) {
        acc[i + 1] <== EcmshInsert(A, B, MSG_BITS, TWEAK_BITS)(acc[i], m[i], k[i], x[i]);
    }

    digest <== acc[n];
}
