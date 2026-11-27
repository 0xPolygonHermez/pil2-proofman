pragma circom 2.1.0;

/*
    A multiplexer of two constant arrays by a binary selector.
*/
template MultiMux1(n) { 
    signal input c[2][n];  // Constants
    signal input {binary} s;   // Selector
    signal output out[n];

    for (var i=0; i<n; i++) {

        out[i] <== (c[1][i] - c[0][i])*s + c[0][i];

    }
}

