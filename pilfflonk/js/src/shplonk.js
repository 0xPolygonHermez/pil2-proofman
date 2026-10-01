// The SHPLONK opening check of pilfflonk (pilfflonk/docs/protocol.md#pairing-check), with the
// structure of snarkjs' fflonk verifier (src/fflonk_verify.js: computeF, computeE, computeJ,
// isValidPairing). snarkjs has three fixed polynomials C0, C1 and C2 and their roots written out;
// here they are the list of f_i, each with its k and signed offsets, and r_i(y) and q_i are
// generalised as shplonkjs' verifyOpenings does (shplonkjs/src/helpers/verifier.js:107-138).
// It checks
//
//     e(F - E - J + y·[W'], [1]₂) = e([W'], [τ]₂)
//
//     F = [f_0] + Σ_{i≥1} q_i·[f_i],   E = (r_0(y) + Σ_{i≥1} q_i·r_i(y))·G1,   J = q_0·[W],
//     q_0 = Z_{T_0}(y),   q_i = α^i·Z_{T_0}(y)/Z_{T_i}(y),
//
// with T_i the roots of f_i (pilfflonk/docs/protocol.md#roots) and r_i the polynomial that
// interpolates f_i on T_i. The prover's Z_T = Π_i Z_{T_i} counts a root once per f_i that has it,
// repetitions included (pilfflonk_shplonk_prover.cpp, quotientWp; pil-fflonk/src/shplonk.cpp:83-101,
// 263-270), and it scales W' by 1/Z_{T∖T_0}(y) = 1/Π_{i≥1} Z_{T_i}(y): that leaves q_0 =
// Z_T(y)/Z_{T∖T_0}(y) = Z_{T_0}(y) and the q_i above, each Z_{T_i} over every root of T_i, shared
// with another f_j or not.
//
// Every input is explicit: the transcript replay that gives xiSeed, α and y
// (pilfflonk/docs/protocol.md#transcript) is the caller's.

import { PilFflonkInputError } from "./elements.js";

// The 2-adicity of r - 1.
const MAX_NBITS = 28;

function log(logger, message) {
    if (logger) logger.info(message);
}

function gcd(a, b) {
    while (b !== 0) [a, b] = [b, a % b];
    return a;
}

// 5^((r-1)/n), a primitive n-th root of unity for n (a bigint) dividing r - 1: w_k for n = k, ω_N
// for n = N and ω_{kN} for n = kN (pilfflonk/docs/protocol.md#roots). 5 is the smallest quadratic
// non-residue, the generator ffiasm's FFT and ffjavascript raise, so ω_N generates the H they use.
export function rootOfUnity(curve, n) {
    const order = curve.r - 1n;
    if (order % n !== 0n) {
        throw new PilFflonkInputError(`there is no root of unity of order ${n}: it does not divide r - 1`);
    }
    return curve.Fr.exp(curve.Fr.e(5), order / n);
}

// base^s, the inverse of base^|s| for s < 0.
function signedPower(Fr, base, s) {
    const power = Fr.exp(base, BigInt(Math.abs(s)));
    return s < 0 ? Fr.inv(power) : power;
}

function isFr(curve, e) {
    return e instanceof Uint8Array && e.byteLength === curve.Fr.n8;
}

// A point of `group` (curve.G1 or curve.G2), affine or Jacobian, on the curve: G1.isValid; for G2,
// the r-torsion is elements.js' to check, on what it decodes.
function isPoint(group, p) {
    const n8 = group.F.n8;
    return p instanceof Uint8Array && (p.byteLength === 2 * n8 || p.byteLength === 3 * n8) && group.isValid(p);
}

// The layout the roots derive from, as the prover refuses it (ShplonkProver's constructor): every
// k divides r - 1 with kN within 2^28, the offsets of an f are distinct rows below N in absolute
// value, and powerW is the lcm of every k.
export function checkLayout(curve, { nBits, powerW, f }) {
    const fail = (message) => {
        throw new PilFflonkInputError(`SHPLONK layout: ${message}`);
    };
    if (!Number.isSafeInteger(nBits) || nBits < 0 || nBits > MAX_NBITS) {
        fail(`nBits = ${nBits} is not an integer between 0 and ${MAX_NBITS}, the 2-adicity of r - 1`);
    }
    const N = 2 ** nBits;
    if (!Array.isArray(f) || f.length === 0) fail("no polynomials to open");
    let lcm = 1;
    f.forEach(({ k, offsets }, i) => {
        if (!Number.isSafeInteger(k) || k < 1) fail(`f_${i}: k = ${k} is not a positive integer`);
        if ((curve.r - 1n) % (BigInt(k) * BigInt(N)) !== 0n) {
            fail(`f_${i}: kN = ${k}·2^${nBits} does not divide r - 1`);
        }
        if (!Array.isArray(offsets) || offsets.length === 0) fail(`f_${i} has no offsets`);
        const rows = new Set();
        for (const s of offsets) {
            if (!Number.isSafeInteger(s) || Math.abs(s) >= N) {
                fail(`f_${i}: offset ${s} is not an integer below N = ${N} in absolute value`);
            }
            const row = ((s % N) + N) % N;
            if (rows.has(row)) fail(`f_${i}: two offsets are the same row modulo N = ${N}`);
            rows.add(row);
        }
        lcm = (lcm / gcd(lcm, k)) * k;
        if (!Number.isSafeInteger(lcm)) fail("the lcm of every k is not a safe integer");
    });
    if (powerW !== lcm) fail(`powerW = ${powerW} is not ${lcm}, the lcm of every k`);
}

// ξ = xiSeed^powerW and, for each f_i, the points ξ·ω_N^s of its offsets and its roots T_i,
// offset-major (pilfflonk/docs/protocol.md#roots): x_j = xiSeed^(powerW/k)·ω_{kN}^s·w_k^j, so that
// x_j^k = ξ·ω_N^s, also for s < 0.
export function computeRoots(curve, { nBits, powerW, f }, xiSeed) {
    const Fr = curve.Fr;
    const N = 2 ** nBits;
    const xi = Fr.exp(xiSeed, BigInt(powerW));
    const omegaN = rootOfUnity(curve, BigInt(N));
    const roots = f.map(({ k, offsets }) => {
        const wk = rootOfUnity(curve, BigInt(k));
        const omegaKN = rootOfUnity(curve, BigInt(k) * BigInt(N));
        const seed = Fr.exp(xiSeed, BigInt(powerW / k));
        const points = [];
        const T = [];
        for (const s of offsets) {
            points.push(Fr.mul(xi, signedPower(Fr, omegaN, s)));
            let x = Fr.mul(seed, signedPower(Fr, omegaKN, s));
            for (let j = 0; j < k; j++) {
                T.push(x);
                x = Fr.mul(x, wk);
            }
        }
        return { points, roots: T };
    });
    return { xi, f: roots };
}

// Z_{T_i}(y) = Π_{x ∈ T_i} (y - x) for each f_i.
export function computeZerofiers(curve, roots, y) {
    const Fr = curve.Fr;
    return roots.map(({ roots: T }) => T.reduce((z, x) => Fr.mul(z, Fr.sub(y, x)), Fr.one));
}

// r_i(y) for each f_i (pilfflonk/docs/protocol.md#pairing-check): on each root x of offset s,
// f_i(x) = Σ_j p_j(ξ·ω_N^s)·x^j, and r_i(y) = Σ_m f_i(x_m)·L_m(y) in the Lagrange basis of T_i,
// L_m(y) = Z_{T_i}(y) / ((y - x_m)·Π_{l≠m} (x_m - x_l)). y must not be in T_i: Z_{T_i}(y) ≠ 0.
export function computeR(curve, f, roots, evaluations, zerofiers, y) {
    const Fr = curve.Fr;
    return f.map(({ k }, i) => {
        const T = roots[i].roots;
        let r = Fr.zero;
        for (let m = 0; m < T.length; m++) {
            const row = evaluations[i][Math.floor(m / k)];
            let value = Fr.zero;
            for (let j = k - 1; j >= 0; j--) {
                value = Fr.add(Fr.mul(value, T[m]), row[j]);
            }
            let den = Fr.sub(y, T[m]);
            for (let l = 0; l < T.length; l++) {
                if (l !== m) den = Fr.mul(den, Fr.sub(T[m], T[l]));
            }
            r = Fr.add(r, Fr.mul(value, Fr.div(zerofiers[i], den)));
        }
        return r;
    });
}

// The denominators this verifier inverts in the SHPLONK check at y, in the order of the proof's inv
// (pilfflonk/docs/protocol.md#inverses), as the prover lists them (verifierDenominators,
// pil2-stark/src/pilfflonk/pilfflonk_shplonk_prover.hpp):
//   1. Z_{T_i}(y) for i = 1 … n − 1: those of the q_i (computeQuotients);
//   2. for each f_i, i = 0 … n − 1, and each root x_m of T_i in its order (offset-major, as
//      computeRoots gives them): (y − x_m)·Π_{l≠m} (x_m − x_l), those of the Lagrange basis of
//      r_i(y) (computeR).
export function computeInverseDenominators(curve, roots, zerofiers, y) {
    const Fr = curve.Fr;
    const denominators = zerofiers.slice(1);
    for (const { roots: T } of roots) {
        for (let m = 0; m < T.length; m++) {
            let den = Fr.sub(y, T[m]);
            for (let l = 0; l < T.length; l++) {
                if (l !== m) den = Fr.mul(den, Fr.sub(T[m], T[l]));
            }
            denominators.push(den);
        }
    }
    return denominators;
}

// Whether inv is the inverse of the product of the denominators, inv·Π = 1: the check snarkjs'
// Solidity fflonk verifier makes before it uses its own inv (templates/verifier_fflonk.sol.ejs,
// inverseArray), with which a verifier gets every inverse by Montgomery's trick.
export function isValidInverse(curve, denominators, inv) {
    const Fr = curve.Fr;
    const product = denominators.reduce((acc, d) => Fr.mul(acc, d), Fr.one);
    return Fr.eq(Fr.mul(product, inv), Fr.one);
}

// q_0 = Z_{T_0}(y), q_i = α^i·Z_{T_0}(y)/Z_{T_i}(y) (pilfflonk/docs/protocol.md#pairing-check): the
// i-th power of α for the i-th f of the global order.
export function computeQuotients(curve, zerofiers, alpha) {
    const Fr = curve.Fr;
    const quotients = [zerofiers[0]];
    let alphaPower = Fr.one;
    for (let i = 1; i < zerofiers.length; i++) {
        alphaPower = Fr.mul(alphaPower, alpha);
        quotients.push(Fr.mul(alphaPower, Fr.div(zerofiers[0], zerofiers[i])));
    }
    return quotients;
}

// F = [f_0] + Σ_{i≥1} q_i·[f_i], in Jacobian coordinates.
export function computeF(curve, commitments, quotients) {
    const G1 = curve.G1;
    let F = G1.toJacobian(commitments[0]);
    for (let i = 1; i < commitments.length; i++) {
        F = G1.add(F, G1.timesFr(commitments[i], quotients[i]));
    }
    return F;
}

// E = (r_0(y) + Σ_{i≥1} q_i·r_i(y))·G1.
export function computeE(curve, r, quotients) {
    const Fr = curve.Fr;
    let e = r[0];
    for (let i = 1; i < r.length; i++) {
        e = Fr.add(e, Fr.mul(quotients[i], r[i]));
    }
    return curve.G1.timesFr(curve.G1.one, e);
}

// J = q_0·[W].
export function computeJ(curve, W, quotient0) {
    return curve.G1.timesFr(W, quotient0);
}

// e(F - E - J + y·[W'], [1]₂) = e([W'], [τ]₂), as e(-(F - E - J + y·[W']), [1]₂)·e([W'], [τ]₂) = 1.
// F in Jacobian coordinates: ffjavascript 0.3.1's G1.sub(a, b) of an affine a and a Jacobian b
// returns b - a (wasm_curve.js, sub: _subMixed(b, a)). snarkjs never meets it, its F being a sum.
export async function isValidPairing(curve, F, E, J, Wp, y, X2) {
    const G1 = curve.G1;
    const A1 = G1.add(G1.sub(G1.sub(G1.toJacobian(F), E), J), G1.timesFr(Wp, y));
    return await curve.pairingEq(G1.neg(A1), X2.one, Wp, X2.tau);
}

// Everything verifyOpening takes, as it takes it; throws a PilFflonkInputError otherwise.
function checkOpening(curve, opening) {
    checkLayout(curve, opening);
    const { f, fixedCommitments, commitments, evaluations, xiSeed, alpha, y, W, Wp, X2 } = opening;
    const fail = (message) => {
        throw new PilFflonkInputError(`SHPLONK opening: ${message}`);
    };
    if (!Array.isArray(fixedCommitments) || !Array.isArray(commitments)) {
        fail("the commitments are not arrays");
    }
    if (fixedCommitments.length + commitments.length !== f.length) {
        const n = `${fixedCommitments.length} fixed and ${commitments.length} other commitments`;
        fail(`${n} for ${f.length} polynomials`);
    }
    [...fixedCommitments, ...commitments].forEach((p, i) => {
        if (!isPoint(curve.G1, p)) fail(`[f_${i}] is not a point of G1`);
    });
    if (!Array.isArray(evaluations) || evaluations.length !== f.length) {
        fail(`evaluations for ${evaluations?.length} polynomials, not ${f.length}`);
    }
    f.forEach(({ k, offsets }, i) => {
        const rows = evaluations[i];
        if (!Array.isArray(rows) || rows.length !== offsets.length) {
            fail(`f_${i}: not one row of evaluations per offset`);
        }
        rows.forEach((row, m) => {
            if (!Array.isArray(row) || row.length !== k || !row.every((e) => isFr(curve, e))) {
                fail(`f_${i}: row ${m} is not k = ${k} elements of Fr`);
            }
        });
    });
    for (const [name, e] of Object.entries({ xiSeed, alpha, y })) {
        if (!isFr(curve, e)) fail(`${name} is not an element of Fr`);
    }
    if (curve.Fr.isZero(xiSeed)) fail("xiSeed is zero");
    if (!isPoint(curve.G1, W) || !isPoint(curve.G1, Wp)) fail("[W] or [W'] is not a point of G1");
    if (!X2 || !isPoint(curve.G2, X2.one) || !isPoint(curve.G2, X2.tau)) fail("[1]₂ or [τ]₂ is not a point of G2");
}

// Whether the SHPLONK opening verifies (pilfflonk/docs/protocol.md#shplonk-opening):
//   nBits, powerW       N = 2^nBits; powerW, the lcm of every k
//   f                   [{k, offsets}]: the layout, in the global order, f_0 first
//   fixedCommitments    [f_i]₁ of the first fixedCommitments.length f_i, the fixed ones: from the
//                       vkey, never from the proof
//   commitments         [f_i]₁ of the others, in order: from the proof
//   evaluations         evaluations[i][m][j] = p_j(ξ·ω_N^s), with s = f[i].offsets[m]
//   xiSeed, alpha, y    the challenges, from the caller's transcript replay
//   W, Wp               [W]₁ and [W']₁, from the proof
//   X2                  {one: [1]₂, tau: [τ]₂}, from the vkey
// Scalars are elements of curve.Fr and points of curve.G1 and curve.G2 (see elements.js). An input
// of the wrong shape is logged and rejected, as snarkjs does.
export async function verifyOpening(curve, opening, logger) {
    try {
        checkOpening(curve, opening);
    } catch (e) {
        if (!(e instanceof PilFflonkInputError)) throw e;
        if (logger) logger.error(e.message);
        return false;
    }
    const { f, fixedCommitments, commitments, evaluations, xiSeed, alpha, y, W, Wp, X2 } = opening;

    log(logger, "> Computing the roots of every f_i");
    const { f: roots } = computeRoots(curve, opening, xiSeed);

    log(logger, "> Computing Z_{T_i}(y)");
    const zerofiers = computeZerofiers(curve, roots, y);
    const rootAtY = zerofiers.findIndex((z) => curve.Fr.isZero(z));
    if (rootAtY >= 0) {
        if (logger) logger.error(`SHPLONK opening: y is a root of f_${rootAtY}`);
        return false;
    }

    log(logger, "> Computing r_i(y)");
    const r = computeR(curve, f, roots, evaluations, zerofiers, y);

    log(logger, "> Computing the quotients q_i");
    const quotients = computeQuotients(curve, zerofiers, alpha);

    log(logger, "> Computing F");
    const F = computeF(curve, [...fixedCommitments, ...commitments], quotients);

    log(logger, "> Computing E");
    const E = computeE(curve, r, quotients);

    log(logger, "> Computing J");
    const J = computeJ(curve, W, quotients[0]);

    log(logger, "> Validating the opening with a pairing");
    const res = await isValidPairing(curve, F, E, J, Wp, y, X2);

    if (logger) {
        if (res) {
            logger.info("SHPLONK opening verified");
        } else {
            logger.warn("Invalid SHPLONK opening");
        }
    }
    return res;
}
