pragma circom 2.1.0;

include "poseidon8_constants.circom";

// x^7 broken into quadratic steps (Hades S-box), matching Poseidon2's Sigma().
template Sigma8() {
    signal input in;
    signal output out;
    signal in2 <== in * in;
    signal in4 <== in2 * in2;
    signal in6 <== in4 * in2;
    out <== in6 * in;
}


template Poseidon1_8() {
    signal input in[8];
    signal output out[8];

    signal st[31][8];

    for (var i = 0; i < 8; i++) {
        st[0][i] <== in[i] + CNST8(i);
    }

    // ── 3 first full rounds (M) + transition full round (P) ──────────────────
    // Round index r in 0..3; matrix is M for r<3, P for r==3.
    component sigmaF1[4][8];
    for (var r = 0; r < 4; r++) {
        for (var t = 0; t < 8; t++) {
            sigmaF1[r][t] = Sigma8();
            sigmaF1[r][t].in <== st[r][t] + CNST8((r + 1) * 8 + t);
        }
        for (var t = 0; t < 8; t++) {
            var acc = 0;
            for (var j = 0; j < 8; j++) {
                if (r < 3) {
                    acc += M8(j, t) * sigmaF1[r][j].out;
                } else {
                    acc += P8(j, t) * sigmaF1[r][j].out;
                }
            }
            st[r + 1][t] <== acc;
        }
    }

    // ── 22 partial rounds ────────────────────────────────────────────────────
    // S stride for width 8 = 15; second slice (cols 1..7) begins at offset 7.
    component sigmaP[22];
    // sboxed[r] = pow7(st[4+r][0]) + C[5*8+r]  (the post-S-box state[0])
    signal sboxed[22];
    for (var r = 0; r < 22; r++) {
        sigmaP[r] = Sigma8();
        sigmaP[r].in <== st[4 + r][0];
        sboxed[r] <== sigmaP[r].out + CNST8(5 * 8 + r);

        // s0 = S[15r+0]*sboxed + sum_{j=1..7} S[15r+j]*st[4+r][j]
        var s0 = S8(15 * r + 0) * sboxed[r];
        for (var j = 1; j < 8; j++) {
            s0 += S8(15 * r + j) * st[4 + r][j];
        }
        st[5 + r][0] <== s0;
        // st[t] += sboxed * S[15r+7+t]  for t=1..7
        for (var t = 1; t < 8; t++) {
            st[5 + r][t] <== st[4 + r][t] + sboxed[r] * S8(15 * r + 7 + t);
        }
    }

    // ── 3 last full rounds (M) + final full round (M, last has no ARC) ───────
    // After the 22 partials the latest state is st[26]. The 4 last-full rounds
    // read st[26 + r] and write st[27 + r], producing st[30] (the permutation).
    component sigmaF2[4][8];
    for (var r = 0; r < 4; r++) {
        for (var t = 0; t < 8; t++) {
            sigmaF2[r][t] = Sigma8();
            if (r < 3) {
                sigmaF2[r][t].in <== st[26 + r][t] + CNST8(5 * 8 + 22 + r * 8 + t);
            } else {
                sigmaF2[r][t].in <== st[26 + r][t];
            }
        }
        for (var t = 0; t < 8; t++) {
            var acc = 0;
            for (var j = 0; j < 8; j++) {
                acc += M8(j, t) * sigmaF2[r][j].out;
            }
            st[27 + r][t] <== acc;
        }
    }

    for (var t = 0; t < 8; t++) {
        out[t] <== st[30][t];
    }
}
