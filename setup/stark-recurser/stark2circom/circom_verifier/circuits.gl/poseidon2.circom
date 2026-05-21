pragma circom 2.1.0;
pragma custom_templates;

include "poseidon2_constants.circom";

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

// Custom gate that calculates Poseidon hash of three inputs using Neptune optimization
template custom extern_c Poseidon16() {
    var arity = 4;
    signal input in[arity * 4];
    signal output im[12][arity * 4];
    signal output out[arity * 4];

    var st[arity * 4];
    st = in;

    st = matmul_external_16(st);

    var row = 0;
    var index = 0;
    im[row] <-- st;
    row++;

    for(var r = 0; r < 4; r++) {
        for(var t=0; t < arity * 4; t++) {
            st[t] = st[t] + CONSTANTS(arity, arity*4*r + t);
            st[t] = st[t] ** 7;
        }
        st = matmul_external_16(st);
        im[row] <-- st;
        row++;
    }

    
    for(var i = 0; i < 22; i++) {
        im[row][index] <-- st[0];
        st[0] += CONSTANTS(arity, 4*(arity*4) + i);
        st[0] = st[0] ** 7;

        var sum = 0;
        for(var j = 0; j < arity * 4; j++) {
            sum += st[j];
        }
        
        for(var j = 0; j < arity * 4; j++) {
            st[j] = st[j] * MATRIX_DIAGONAL(arity, j);
            st[j] += sum;
        }
        
        index++;
        if(i == 10 || i == 21) {
            im[row][index] <-- 0;
            index = 0;
            row++;
            im[row] <-- st;
            row++;
        }
    }

    for(var r = 0; r < 4; r++) {
        for(var t=0; t < arity * 4; t++) {
            st[t] = st[t] + CONSTANTS(arity, 4*arity*4 + 22 + arity*4*r + t);
            st[t] = st[t] ** 7;
        }

        st = matmul_external_16(st);

        if(r < 3) {
            im[row] <-- st;
            row++;
        } else {
            out <-- st;
        }
    }
}

// Custom gate that calculates Poseidon hash of two inputs using Neptune optimization
// The two inputs are sent unordered and the key that determines its position is also sent as an input
template custom extern_c CustPoseidon16() {
    var arity = 4;
    signal input in[arity * 4];
    signal input key[2];
    signal output im[12][arity * 4];
    signal output out[arity * 4];

    assert(key[0]*(key[0] - 1) == 0);
    assert(key[1]*(key[1] - 1) == 0);

    var initialSt[arity * 4];
    
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
    

    var st[arity * 4] = initialSt;
    var row = 0;
    var index = 0;

    st = matmul_external_16(st);
    im[row] <-- st;
    row++;
    for(var r = 0; r < 4; r++) {
        for(var t=0; t < arity * 4; t++) {
            st[t] = st[t] + CONSTANTS(arity, arity*4*r + t);
            st[t] = st[t] ** 7;
        }
        st = matmul_external_16(st);
        im[row] <-- st;
        row++;
    }

   

    for(var r = 0; r < 22; r++) {
        im[row][index] <-- st[0];
        st[0] += CONSTANTS(arity, 4*arity*4 + r);
        st[0] = st[0] ** 7;

        var sum = 0;
        for(var j = 0; j < arity * 4; j++) {
            sum += st[j];
        }

        for(var j = 0; j < arity * 4; j++) {
            st[j] = st[j] * MATRIX_DIAGONAL(arity, j);
            st[j] += sum;
        }

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
        for(var t=0; t < arity * 4; t++) {
            st[t] = st[t] + CONSTANTS(arity, 4*arity*4 + 22 + arity*4*r + t);
            st[t] = st[t] ** 7;
        }

        st = matmul_external_16(st);
        if(r < 3) {
            im[row] <-- st;
            row++;
        } else {
            out <-- st;
        }
    }
}

// Calculate Poseidon2 Hash of 3 inputs (2 in + capacity) in GL field (each element has at most 63 bits)
// -nOuts: Number of GL field elements that are being returned as output
template Poseidon2(arity, nOuts) {
    assert(arity == 4);
    var rate = (arity - 1) * 4;
    signal input in[rate];
    signal input capacity[4];
    signal output out[nOuts];

    component p = Poseidon16();

    // Pass the two inputs and the capacity as inputs for performing the poseidon Hash
    for (var j=0; j<rate; j++) {
        p.in[j] <== in[j];
    }
    for (var j=0; j<4; j++) {
        p.in[rate+j] <== capacity[j];
    }

    // Poseidon12 returns 12 outputs but we are only interested in returning nOuts
    for (var j=0; j<nOuts; j++) {
        out[j] <== p.out[j];
    }

    _ <== p.im;

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

    component p = CustPoseidon16();
    p.in <== in;
    p.key <== key;
    
    // Poseidon12 returns 12 outputs but we are only interested in returning nOuts
    for (var j=0; j<nOuts; j++) {
        out[j] <== p.out[j];
    }

    _ <== p.im;
    
    for (var j=nOuts; j<arity*4; j++) {
        _ <== p.out[j];
    }
}
