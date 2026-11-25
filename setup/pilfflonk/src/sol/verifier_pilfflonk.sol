// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity >=0.7.0 <0.9.0;

/// @title PilfflonkVerifier
/// @notice Verifies the pilfflonk proofs of one vkey (its DIGEST below): its pilfflonk.vkey.json,
///         from which `proofman-setup setup-pilfflonk --solidity` or
///         `proofman-setup pilfflonk-solidity` generated it.
/// @dev    It accepts exactly the proofs the JS verifier (pilfflonk/js, verify.js) accepts, and its
///         steps are the JS's. Calldata: `proof` is the proof's bytes (the commitments of the
///         non-fixed f_i, W, W', the evaluations, the Q_i(xi) if Q is split, inv and invZh) as
///         big-endian words, followed by the auxiliary inverses 1/(xi - w^j) of the firstRow and
///         lastRow boundaries, if any; `pubSignals`, the publics. Its structure is snarkjs
///         0.7.6's fflonk verifier (verifier_fflonk.sol.ejs).
// The static part of the contract: setup/pilfflonk/src/solidity.rs replaces each marker line, the
// tag "generated" and a part's name, with the vkey's constants or the straight-line Yul of its
// layout and qVerifier.
contract PilfflonkVerifier {
    // Scalar field size
    uint256 constant q    = 21888242871839275222246405745257275088548364400416034343698204186575808495617;
    // Base field size
    uint256 constant qf   = 21888242871839275222246405745257275088696311157297823662689037894645226208583;
    // [1]_1
    uint256 constant G1x  = 1;
    uint256 constant G1y  = 2;
    // [1]_2
    uint256 constant G2x1 = 10857046999023057135944570762232829481370756359578518086990519993285655852781;
    uint256 constant G2x2 = 11559732032986387107991004021392285783925812861821192530917403151452391805634;
    uint256 constant G2y1 = 8495653923123431417604973247489272438418190587263600148770280649306958101930;
    uint256 constant G2y2 = 4082367875863433681332203403145435568316851327593401208105741076214120093531;

    // @generated constants

    // `proof` has no name: the contract reads its words at the constant offsets above.
    // @generated signature
        assembly {
            // The proof is rejected: verifyProof returns false.
            function fail() {
                mstore(0, 0)
                return(0, 0x20)
            }

            // elements.js, frFromObject: every scalar below r. bad is whether one is not, accumulated
            // over them without a branch each.
            function checkFields(bad) {
                if bad {
                    fail()
                }
            }

            // elements.js, g1FromObject: a point of G1, affine, its coordinates below the base
            // field's q, not the point at infinity (0, 0), and on y^2 = x^3 + 3, which is G1: its
            // cofactor is 1.
            function checkPointBelongsToBN128Curve(p) {
                let x := calldataload(p)
                let y := calldataload(add(p, 32))
                if iszero(and(lt(x, qf), lt(y, qf))) {
                    fail()
                }
                if iszero(or(x, y)) {
                    fail()
                }
                if iszero(eq(mulmod(y, y, qf), addmod(mulmod(x, mulmod(x, x, qf), qf), 3, qf))) {
                    fail()
                }
            }

            // transcript.js, addPolCommitment: a point the transcript absorbs has no coordinate
            // below 2^192.
            function checkAbsorbed(p) {
                if or(iszero(shr(192, calldataload(p))), iszero(shr(192, calldataload(add(p, 32))))) {
                    fail()
                }
            }

            // STEPS 1-3 (verify.js; proof.js, fromObjectProof and fromObjectPublics): the
            // commitments, W and W' in G1; the evaluations, the pieces of Q, inv, invZh (and the
            // auxiliary inverses) and the publics in Fr. The lengths are the vkey's, fixed by the
            // types of the arguments; the vkey is this contract's constants.
            function checkInput() {
                // @generated checkInput
            }

            // STEP 4: challenges.js, computeChallenges, the transcript. Its buffer, pT,
            // holds what is absorbed since the last squeeze, which is keccak256 of it mod r and
            // then the buffer's only value (transcript.js, squeeze).
            function computeChallenges() {
                let pT := mload(0x40)
                let c
                // @generated computeChallenges
            }

            // b^e mod r.
            function powMod(b, e) -> result {
                result := 1
                for { } gt(e, 0) { e := shr(1, e) } {
                    if and(e, 1) {
                        result := mulmod(result, b, q)
                    }
                    b := mulmod(b, b, q)
                }
            }

            // STEP 5 (verify.js): xi = xiSeed^powerW, and Z_H(xi) = xi^N - 1 checked against the
            // proof's invZh.
            function computeZh() {
                let xi := powMod(mload(pXiSeed), powerW)
                mstore(pXi, xi)
                let xin := xi
                // @generated computeZh
                mstore(pXiN, xin)
                let zh := addmod(xin, sub(q, 1), q)
                mstore(pZh, zh)
                if iszero(eq(mulmod(zh, calldataload(pInvZh), q), 1)) {
                    fail()
                }
            }

            // STEP 5: qverifier.js, computeZi, Zi(D) at xi for each boundary D:
            // 1/Z_H(xi) for everyRow, Z_H(xi)/Z_D(xi) for the others.
            function computeZi() {
                let xi := mload(pXi)
                let zh := mload(pZh)
                // @generated computeZi
            }

            // STEP 5: verify.js, computeQ, and qverifier.js, executeCode: Q(xi) with the vkey's
            // qVerifier, the STARK's fold of the constraints, over the evaluations.
            function computeQ() {
                // @generated computeQ
            }

            // @generated checkQPieces

            // STEP 6 (verify.js); shplonk.js, computeRoots: the roots T_i of every f_i are, for each
            // offset s, the k-th roots of z_s = xi·w_N^s; the verifier needs only the z_s and y^k
            // (computeInversions, computeR), never the roots.
            function computeZ() {
                let xiSeed := mload(pXiSeed)
                // shplonk.js, checkOpening: xiSeed is not zero.
                if iszero(xiSeed) {
                    fail()
                }
                let xi := mload(pXi)
                let y := mload(pY)
                // @generated computeZ
            }

            // shplonk.js, isValidInverse: inv·A·B = 1, with invB = inv·A.
            function checkInv(invB, b) {
                if iszero(eq(mulmod(invB, b, q), 1)) {
                    fail()
                }
            }

            // The inverses of the len values from pVals, given acc, the inverse of their product:
            // snarkjs's inverseArray without its check, which computeInversions makes.
            function batchInverse(pVals, len, acc) {
                let pAux := mload(0x40)
                let p := mload(pVals)
                mstore(pAux, p)
                for { let i := 1 } lt(i, len) { i := add(i, 1) } {
                    p := mulmod(p, mload(add(pVals, mul(i, 32))), q)
                    mstore(add(pAux, mul(i, 32)), p)
                }
                for { let i := sub(len, 1) } gt(i, 0) { i := sub(i, 1) } {
                    let pV := add(pVals, mul(i, 32))
                    let v := mload(pV)
                    mstore(pV, mulmod(acc, mload(add(pAux, mul(sub(i, 1), 32))), q))
                    acc := mulmod(acc, v, q)
                }
                mstore(pVals, acc)
            }

            // STEP 6 (verify.js): shplonk.js, computeZerofiers and computeInverseDenominators, and
            // isValidInverse. The proof's inv inverts A·B: A = prod_i Z_{T_i}(y), i ≥ 1, with
            // Z_T(y) = prod_s (y^k - z_s), and B the product of every f_i's Lagrange denominators,
            // which has the closed form G·xi^E·prod_i Z_{T_i}(y) (solidity.rs, Context::new): the check
            // inv·A·B = 1 is the JS's. Then 1/A = inv·B gives each 1/Z_{T_i}(y) and, for the sets of
            // several offsets, (1/B)·G·prod_i Z_{T_i}(y) = xi^(-E) gives their CRT factors.
            function computeInversions() {
                let d
                // @generated computeInversions
            }

            // STEP 7 (verify.js, openingEvaluations; shplonk.js, computeR): r_i(y), the polynomial of
            // degree < k·t that f_i is on its roots, at y. On the roots of an offset s, X^k = z_s and
            // f_i(X) = R_s(X) = sum_j p_j(xi·w^s)·X^j, of degree < k: with one offset r_i = R_s, and
            // with several the CRT, r_i = sum_s R_s·L_s. p_j(xi·w^s) are the proof's evaluations and,
            // for Q's f, Q(xi) computed or the proof's Q_i(xi) if Q is split.
            function computeR() {
                let y := mload(pY)
                let v
                // @generated computeR
            }

            // Whether a call to the precompile 0x06 or 0x07 succeeded and returned a point: 64 bytes.
            // snarkjs checks success only; on a chain where the precompile is missing, the call
            // succeeds and returns nothing.
            function checkPointResult(success) {
                if iszero(and(success, eq(returndatasize(), 64))) {
                    fail()
                }
            }

            // G1 function to accumulate a G1 value to an address
            function g1_acc(pR, pP) {
                let mIn := mload(0x40)
                mstore(mIn, mload(pR))
                mstore(add(mIn, 32), mload(add(pR, 32)))
                mstore(add(mIn, 64), mload(pP))
                mstore(add(mIn, 96), mload(add(pP, 32)))

                let success := staticcall(sub(gas(), 2000), 6, mIn, 128, pR, 64)

                checkPointResult(success)
            }

            // G1 function to multiply a G1 value (in the calldata at pP) to value in an address
            function g1_mulAcc(pR, pP, s) {
                let success
                let mIn := mload(0x40)
                mstore(mIn, calldataload(pP))
                mstore(add(mIn, 32), calldataload(add(pP, 32)))
                mstore(add(mIn, 64), s)

                success := staticcall(sub(gas(), 2000), 7, mIn, 96, mIn, 64)

                checkPointResult(success)

                mstore(add(mIn, 64), mload(pR))
                mstore(add(mIn, 96), mload(add(pR, 32)))

                success := staticcall(sub(gas(), 2000), 6, mIn, 128, pR, 64)

                checkPointResult(success)
            }

            // G1 function to multiply a G1 value (x, y) to value in an address
            function g1_mulAccC(pR, x, y, s) {
                let success
                let mIn := mload(0x40)
                mstore(mIn, x)
                mstore(add(mIn, 32), y)
                mstore(add(mIn, 64), s)

                success := staticcall(sub(gas(), 2000), 7, mIn, 96, mIn, 64)

                checkPointResult(success)

                mstore(add(mIn, 64), mload(pR))
                mstore(add(mIn, 96), mload(add(pR, 32)))

                success := staticcall(sub(gas(), 2000), 6, mIn, 128, pR, 64)

                checkPointResult(success)
            }

            // G1 function to write s·(x, y) to an address
            function g1_mulSetC(pR, x, y, s) {
                let mIn := mload(0x40)
                mstore(mIn, x)
                mstore(add(mIn, 32), y)
                mstore(add(mIn, 64), s)

                let success := staticcall(sub(gas(), 2000), 7, mIn, 96, pR, 64)

                checkPointResult(success)
            }

            // STEP 7: shplonk.js, computeQuotients, computeF, computeE and computeJ:
            // q_0 = Z_{T_0}(y), q_i = alpha^i·Z_{T_0}(y)/Z_{T_i}(y), F = [f_0] + sum_i q_i·[f_i],
            // E = (r_0(y) + sum_i q_i·r_i(y))·[1]_1, J = q_0·[W]; the fixed [f_i] are the vkey's.
            function computeFEJ() {
                let alpha := mload(pAlpha)
                let alphaPow := 1
                let quotient
                // @generated computeFEJ

                // Compute group-encoded batch evaluation [E]_1
                g1_mulSetC(pE, G1x, G1y, e)

                // Compute the full difference [J]_1
                g1_mulSetC(pJ, calldataload(pW), calldataload(add(pW, 32)), zt0)
            }

            // STEP 7: shplonk.js, isValidPairing: e([F]_1 - [E]_1 - [J]_1 + y·[W']_1, [1]_2) ==
            // e([W']_1, [x]_2)
            function checkPairing() -> isOk {
                let mIn := mload(0x40)

                // First pairing value
                // Compute -E
                mstore(add(pE, 32), mod(sub(qf, mload(add(pE, 32))), qf))
                // Compute -J
                mstore(add(pJ, 32), mod(sub(qf, mload(add(pJ, 32))), qf))
                // F = F - E - J + y·W'
                g1_acc(pF, pE)
                g1_acc(pF, pJ)
                g1_mulAcc(pF, pWp, mload(pY))

                mstore(mIn, mload(pF))
                mstore(add(mIn, 32), mload(add(pF, 32)))

                // Second pairing value
                mstore(add(mIn, 64), G2x2)
                mstore(add(mIn, 96), G2x1)
                mstore(add(mIn, 128), G2y2)
                mstore(add(mIn, 160), G2y1)

                // Third pairing value
                // Compute -W'
                mstore(add(mIn, 192), calldataload(pWp))
                let s := calldataload(add(pWp, 32))
                s := mod(sub(qf, s), qf)
                mstore(add(mIn, 224), s)

                // Fourth pairing value
                mstore(add(mIn, 256), X2x2)
                mstore(add(mIn, 288), X2x1)
                mstore(add(mIn, 320), X2y2)
                mstore(add(mIn, 352), X2y1)

                let success := staticcall(sub(gas(), 2000), 8, mIn, 384, mIn, 0x20)

                // The precompile's answer, 32 bytes, is 1 (snarkjs takes any non-zero word).
                isOk := and(and(success, eq(returndatasize(), 0x20)), eq(mload(mIn), 1))
            }

            mstore(0x40, lastMem)

            // STEPS 1-3: the proof and the publics are well-formed
            checkInput()

            // STEP 4: the challenges
            computeChallenges()

            // STEP 5: Z_H(xi), the Zi and Q(xi)
            computeZh()
            computeZi()
            computeQ()
            // @generated checkQPiecesCall

            // STEP 6: the z_s, and inv
            computeZ()
            computeInversions()

            // STEP 7: the SHPLONK opening, with a pairing
            computeR()
            computeFEJ()
            let isValid := checkPairing()

            mstore(0, isValid)
            return(0, 0x20)
        }
    }
}
