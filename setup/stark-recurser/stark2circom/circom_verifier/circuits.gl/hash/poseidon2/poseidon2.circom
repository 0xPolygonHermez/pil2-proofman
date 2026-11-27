pragma circom 2.1.0;
pragma custom_templates;

include "poseidon2_constants.circom";
include "../node_children.circom";

// (5 7 1 3) (a)
// (4 6 1 1) (b)
// (1 3 5 7) (c)   
// (1 1 4 6) (d)
function matmul_m4(a, b, c, d) {
    
    var t0 = a + b;
    var t1 = c + d;
    var t2 = 2*b + t1;
    var t3 = 2*d + t0;
    var t4 = 4*t1 + t3;
    var t5 = 4*t0 + t2;
    var t6 = t3 + t5;
    var t7 = t2 + t4;

    return [t6, t5, t7, t4];
}

function matmul_external_16(in) {
    
    var mat[16][4];
    for (var i = 0; i < 4; i++) {
        mat[i] = matmul_m4(
            in[4*i + 0],
            in[4*i + 1],
            in[4*i + 2],
            in[4*i + 3]
        );
    }

    var stored[4];
    for(var i = 0; i < 4; i++) {
        for(var j = 0; j < 4; j++) {
            stored[i] += mat[j][i];
        }
    }

    var out[16];

    for(var i = 0; i < 4; i++) {
        for(var j = 0; j < 4; j++) {
            out[4*j + i] = mat[j][i] + stored[i];
        }
    }
   
    return out;
}

// The Poseidon2 permutation of width 16 (arity 4): 4 full rounds, 22 partial, 4 full.
function poseidon2_16(in) {
    var arity = 4;
    var st[arity * 4];
    st = in;

    st = matmul_external_16(st);
    for (var r = 0; r < 4; r++) {
        for (var t = 0; t < arity * 4; t++) {
            st[t] = st[t] + CONSTANTS(arity, arity * 4 * r + t);
            st[t] = st[t] ** 7;
        }
        st = matmul_external_16(st);
    }

    for (var r = 0; r < 22; r++) {
        st[0] += CONSTANTS(arity, 4 * arity * 4 + r);
        st[0] = st[0] ** 7;

        var sum = 0;
        for (var j = 0; j < arity * 4; j++) { sum += st[j]; }
        for (var j = 0; j < arity * 4; j++) {
            st[j] = st[j] * MATRIX_DIAGONAL(arity, j);
            st[j] += sum;
        }
    }

    for (var r = 0; r < 4; r++) {
        for (var t = 0; t < arity * 4; t++) {
            st[t] = st[t] + CONSTANTS(arity, 4 * arity * 4 + 22 + arity * 4 * r + t);
            st[t] = st[t] ** 7;
        }
        st = matmul_external_16(st);
    }
    return st;
}

// Custom gate: the permutation of in. Only the last round's state leaves the gate; the rest is
// recomputed when the trace is filled (see pil2-stark/src/starkpil/gate_bands.hpp).
template custom extern_c Poseidon2_16() {
    signal input in[16];
    signal output out[16];

    out <-- poseidon2_16(in);
}

// Custom gate: a Merkle node's hash, its four children ordered by key (nodeChildrenInOrder).
template custom extern_c CustPoseidon2_16() {
    signal input in[16];
    signal input key[2];
    signal output out[16];

    assert(key[0] * (key[0] - 1) == 0);
    assert(key[1] * (key[1] - 1) == 0);
    out <-- poseidon2_16(nodeChildrenInOrder(in, key));
}

// Calculate Poseidon2 Hash of 3 inputs (2 in + capacity) in GL field (each element has at most 63 bits)
// -nOuts: Number of GL field elements that are being returned as output
template Poseidon2(arity, nOuts) {
    assert(arity == 4);
    var rate = (arity - 1) * 4;
    signal input in[rate];
    signal input capacity[4];
    signal output out[nOuts];

    component p = Poseidon2_16();

    // Pass the two inputs and the capacity as inputs for performing the poseidon Hash
    for (var j=0; j<rate; j++) {
        p.in[j] <== in[j];
    }
    for (var j=0; j<4; j++) {
        p.in[rate+j] <== capacity[j];
    }

    // Poseidon2_16 returns 16 outputs but we are only interested in returning nOuts
    for (var j=0; j<nOuts; j++) {
        out[j] <== p.out[j];
    }

    for (var j=nOuts; j<arity*4; j++) {
        _ <== p.out[j];
    }
}

// Calculate Poseidon Hash of 2 inputs in GL field (each element has at most 63 bits)
// -nOuts: Number of GL field elements that are being returned as output
template CustPoseidon2(arity, nOuts) {
    assert(arity == 4);
    signal input in[arity * 4];
    signal input key[2];
    signal output out[nOuts];

    component p = CustPoseidon2_16();
    p.in <== in;
    p.key <== key;

    // CustPoseidon2_16 returns 16 outputs but we are only interested in returning nOuts
    for (var j=0; j<nOuts; j++) {
        out[j] <== p.out[j];
    }

    
    for (var j=nOuts; j<arity*4; j++) {
        _ <== p.out[j];
    }
}
