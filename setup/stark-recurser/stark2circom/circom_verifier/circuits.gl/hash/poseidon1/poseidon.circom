pragma circom 2.1.0;
pragma custom_templates;

include "poseidon_constants.circom";
include "../node_children.circom";

// The Poseidon1 permutation of width 16: 4 full rounds, 22 partial (Neptune form), 4 full.
function poseidon1_16(in) {
    var st[16];
    st = in;
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
    }

    // S stride for width 16 = 2*16 - 1 = 31 entries per partial round.
    // Second slice (column-1..15 updates) begins at offset WIDTH - 1 = 15.
    for(var r = 0; r < 22; r++) {
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
    }
    return st;
}

// Custom gate: the permutation of in. Only the last round's state leaves the gate; the rest is
// recomputed when the trace is filled (see pil2-stark/src/starkpil/gate_bands.hpp).
template custom extern_c Poseidon1_16() {
    signal input in[16];
    signal output out[16];

    out <-- poseidon1_16(in);
}

// Custom gate: a Merkle node's hash, its four children ordered by key (nodeChildrenInOrder).
template custom extern_c CustPoseidon1_16() {
    signal input in[16];
    signal input key[2];
    signal output out[16];

    assert(key[0] * (key[0] - 1) == 0);
    assert(key[1] * (key[1] - 1) == 0);
    out <-- poseidon1_16(nodeChildrenInOrder(in, key));
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

    for (var j=nOuts; j<arity*4; j++) {
        _ <== p.out[j];
    }
}
