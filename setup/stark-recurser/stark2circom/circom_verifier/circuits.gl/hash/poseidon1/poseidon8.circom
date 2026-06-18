pragma circom 2.1.0;

include "poseidon8_constants.circom";

// x^7 broken into quadratic steps (Hades S-box), matching Poseidon2's Sigma().
template Sigma7() {
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

    component sigmaF1[4][8];
    signal preF1[4][8];
    for (var r = 0; r < 4; r++) {
        for (var t = 0; t < 8; t++) {
            sigmaF1[r][t] = Sigma7();
            sigmaF1[r][t].in <== st[r][t];
            preF1[r][t] <== sigmaF1[r][t].out + CNST8((r + 1) * 8 + t);
        }
        for (var t = 0; t < 8; t++) {
            var acc = 0;
            for (var j = 0; j < 8; j++) {
                if (r < 3) {
                    acc += M8(j, t) * preF1[r][j];
                } else {
                    acc += P8(j, t) * preF1[r][j];
                }
            }
            st[r + 1][t] <== acc;
        }
    }

    // ── 22 partial rounds ────────────────────────────────────────────────────
    // S stride for width 8 = 15; second slice (cols 1..7) begins at offset 7.
    component sigmaP[22];
    signal sboxed[22];
    for (var r = 0; r < 22; r++) {
        sigmaP[r] = Sigma7();
        sigmaP[r].in <== st[4 + r][0];
        sboxed[r] <== sigmaP[r].out + CNST8(5 * 8 + r);

        var s0 = S8(15 * r + 0) * sboxed[r];
        for (var j = 1; j < 8; j++) {
            s0 += S8(15 * r + j) * st[4 + r][j];
        }
        st[5 + r][0] <== s0;
        for (var t = 1; t < 8; t++) {
            st[5 + r][t] <== st[4 + r][t] + sboxed[r] * S8(15 * r + 7 + t);
        }
    }

    component sigmaF2[4][8];
    signal preF2[4][8];
    for (var r = 0; r < 4; r++) {
        for (var t = 0; t < 8; t++) {
            sigmaF2[r][t] = Sigma7();
            sigmaF2[r][t].in <== st[26 + r][t];
            if (r < 3) {
                preF2[r][t] <== sigmaF2[r][t].out + CNST8(5 * 8 + 22 + r * 8 + t);
            } else {
                preF2[r][t] <== sigmaF2[r][t].out;
            }
        }
        for (var t = 0; t < 8; t++) {
            var acc = 0;
            for (var j = 0; j < 8; j++) {
                acc += M8(j, t) * preF2[r][j];
            }
            st[27 + r][t] <== acc;
        }
    }

    for (var t = 0; t < 8; t++) {
        out[t] <== st[30][t];
    }
}
