pragma circom 2.1.0;
pragma custom_templates;

include "poseidon_constants.circom";

// Custom gate that calculates Poseidon hash of three inputs using Neptune optimization
template custom extern_c Poseidon1_16() {
    signal input in[16];
    signal output im[12][16];
    signal output out[16];

    var st[16];
    st = in;

    var row = 0;
    var index = 0;

    im[row] <-- st;
    row++;

    for(var i=0; i < 16; i++) {
        st[i] = st[i] + CNST(i);
    }

    var newSt[16];
    for(var r = 0; r < 4; r++) {
        for(var t=0; t < 16; t++) {
            st[t] = st[t] ** 7;
            st[t] = st[t] + CNST((r + 1)*16 + t);
        }

        for(var t=0; t < 16; t++) {
            var acc = 0;
            for(var j = 0; j < 16; j++) {
                if(r < 3) {
                    acc += M(j,t) * st[j];
                } else {
                    acc += P(j,t) * st[j];
                }
            }
            newSt[t] = acc;
        }
        st = newSt;
        im[row] <-- st;
        row++;
    }

    // S stride for width 16 = 2*16 - 1 = 31 entries per partial round.
    // Second slice (column-1..15 updates) begins at offset WIDTH - 1 = 15.
    for(var r = 0; r < 22; r++) {
        im[row][index] <-- st[0];
        st[0] = st[0] ** 7;
        st[0] += CNST(80 + r);

        var s0 = 0;
        for(var j = 0; j < 16; j++) {
            s0 += S(31*r + j) * st[j];
        }
        for(var t = 1; t < 16; t++) {
            st[t] += st[0] * S(31*r + 15 + t);
        }
        st[0] = s0;

        index++;
        if(r == 10 || r == 21) {
            im[row][index] <-- 0;
            index = 0;
            row++;
            im[row] <-- st;
            row++;
        }
    }

    for(var r = 0; r < 4; r++) {
        for(var t=0; t < 16; t++) {
            st[t] = st[t] ** 7;
            if(r < 3) st[t] += CNST(102 + 16*r + t);
        }

        for(var t=0; t < 16; t++) {
            var acc = 0;
            for(var j = 0; j < 16; j++) {
                acc += M(j,t) * st[j];
            }
            newSt[t] = acc;
        }
        st = newSt;
        if(r < 3) {
            im[row] <-- st;
            row++;
        } else {
            for(var t=0; t < 16; t++) {
                out[t] <-- st[t];
            }
        }
    }
}

template custom extern_c CustPoseidon1_16() {
    signal input in[16];
    signal input key[2];
    signal output im[12][16];
    signal output out[16];

    // One-hot encoding of the 2 key bits, declared after out[16] so it lands last in the r1cs
    // signal list. Exposed so the AIR can read the input-ordering mask as a stored degree-1
    // value rather than rebuilding the degree-2 products from the key bits.
    signal output im_m[4];

    assert(key[0]*(key[0] - 1) == 0);
    assert(key[1]*(key[1] - 1) == 0);

    var initialSt[16];
    
    // Order the inputs of the Poseidon hash according to the key bit.
    if(key[0] == 0 && key[1] == 0) {
        initialSt = in;
    } else if (key[0] == 1 && key[1] == 0) {
        initialSt[0]  = in[4];
        initialSt[1]  = in[5];
        initialSt[2]  = in[6];
        initialSt[3]  = in[7];
        initialSt[4]  = in[0];
        initialSt[5]  = in[1];
        initialSt[6]  = in[2];
        initialSt[7]  = in[3];
        initialSt[8]  = in[8];
        initialSt[9]  = in[9];
        initialSt[10] = in[10];
        initialSt[11] = in[11];
        initialSt[12] = in[12];
        initialSt[13] = in[13];
        initialSt[14] = in[14];
        initialSt[15] = in[15];
    } else if (key[0] == 0 && key[1] == 1) {
        initialSt[0]  = in[4];
        initialSt[1]  = in[5];
        initialSt[2]  = in[6];
        initialSt[3]  = in[7];
        initialSt[4]  = in[8];
        initialSt[5]  = in[9];
        initialSt[6]  = in[10];
        initialSt[7]  = in[11];
        initialSt[8]  = in[0];
        initialSt[9]  = in[1];
        initialSt[10] = in[2];
        initialSt[11] = in[3];
        initialSt[12] = in[12];
        initialSt[13] = in[13];
        initialSt[14] = in[14];
        initialSt[15] = in[15];
    } else {
        initialSt[0]  = in[4];
        initialSt[1]  = in[5];
        initialSt[2]  = in[6];
        initialSt[3]  = in[7];
        initialSt[4]  = in[8];
        initialSt[5]  = in[9];
        initialSt[6]  = in[10];
        initialSt[7]  = in[11];
        initialSt[8]  = in[12];
        initialSt[9]  = in[13];
        initialSt[10] = in[14];
        initialSt[11] = in[15];
        initialSt[12] = in[0];
        initialSt[13] = in[1];
        initialSt[14] = in[2];
        initialSt[15] = in[3];
    }

    var st[16];
    st = initialSt;

    var row = 0;
    var index = 0;

    im[row] <-- st;
    row++;

    for(var i=0; i < 16; i++) {
        st[i] = st[i] + CNST(i);
    }

    var newSt[16];
    for(var r = 0; r < 4; r++) {
        for(var t=0; t < 16; t++) {
            st[t] = st[t] ** 7;
            st[t] = st[t] + CNST((r + 1)*16 + t);
        }

        for(var t=0; t < 16; t++) {
            var acc = 0;
            for(var j = 0; j < 16; j++) {
                if(r < 3) {
                    acc += M(j,t) * st[j];
                } else {
                    acc += P(j,t) * st[j];
                }
            }
            newSt[t] = acc;
        }
        st = newSt;
        im[row] <-- st;
        row++;
    }

    // S stride for width 16 = 2*16 - 1 = 31 entries per partial round.
    // Second slice (column-1..15 updates) begins at offset WIDTH - 1 = 15.
    for(var r = 0; r < 22; r++) {
        im[row][index] <-- st[0];
        st[0] = st[0] ** 7;
        st[0] += CNST(80 + r);

        var s0 = 0;
        for(var j = 0; j < 16; j++) {
            s0 += S(31*r + j) * st[j];
        }
        for(var t = 1; t < 16; t++) {
            st[t] += st[0] * S(31*r + 15 + t);
        }
        st[0] = s0;

        index++;
        if(r == 10 || r == 21) {
            im[row][index] <-- 0;
            index = 0;
            row++;
            im[row] <-- st;
            row++;
        }
    }

    for(var r = 0; r < 4; r++) {
        for(var t=0; t < 16; t++) {
            st[t] = st[t] ** 7;
            if(r < 3) st[t] += CNST(102 + 16*r + t);
        }

        for(var t=0; t < 16; t++) {
            var acc = 0;
            for(var j = 0; j < 16; j++) {
                acc += M(j,t) * st[j];
            }
            newSt[t] = acc;
        }
        st = newSt;
        if(r < 3) {
            im[row] <-- st;
            row++;
        } else {
            for(var t=0; t < 16; t++) {
                out[t] <-- st[t];
            }
        }
    }

    // One-hot the key so the AIR can read the ordering mask as a stored value.
    var mm[4];
    for (var i = 0; i < 4; i++) { mm[i] = 0; }
    mm[key[0] + 2*key[1]] = 1;
    im_m <-- mm;
}


// Calculate Poseidon Hash of 4 inputs (3 in + capacity) in GL field (each element has at most 63 bits)
// -nOuts: Number of GL field elements that are being returned as output
template Poseidon(nOuts) {
    signal input in[12];
    signal input capacity[4];
    signal output out[nOuts];

    component p = Poseidon1_16();

    // Pass the two inputs and the capacity as inputs for performing the poseidon Hash
    for (var j=0; j<12; j++) {
        p.in[j] <== in[j];
    }
    for (var j=0; j<4; j++) {
        p.in[12+j] <== capacity[j];
    }

    // Poseidon1_16 returns 16 outputs but we are only interested in returning nOuts
    for (var j=0; j<nOuts; j++) {
        out[j] <== p.out[j];
    }

    _ <== p.im;

    for (var j=nOuts; j<16; j++) {
        _ <== p.out[j];
    }
}

template CustPoseidon(arity, nOuts) {
    assert(arity == 4);
    signal input in[arity * 4];
    signal input key[2];
    signal output out[nOuts];

    component p = CustPoseidon1_16();
    p.in <== in;
    p.key <== key;

    // CustPoseidon1_16 returns 16 outputs but we are only interested in returning nOuts
    for (var j=0; j<nOuts; j++) {
        out[j] <== p.out[j];
    }

    _ <== p.im;

    for (var j=nOuts; j<arity*4; j++) {
        _ <== p.out[j];
    }
}
