pragma circom 2.1.0;

/*
    Multiplexor used in recursive2
*/
template MultiMux2(n) {
    signal input c[4][n];  // Constants
    signal input {binary} s[2];   // Selector
    signal output out[n];

    signal a10[n];
    signal a1[n];
    signal a0[n];
    signal a[n];

    signal  s10;
    s10 <== s[1] * s[0];

    for (var i=0; i<n; i++) {

          a10[i] <==  ( c[3][i]-c[2][i]-c[1][i]+c[0][i] ) * s10;
           a1[i] <==  ( c[2][i]-c[0][i] ) * s[1];
           a0[i] <==  ( c[1][i]-c[0][i] ) * s[0];
            a[i] <==  ( c[0][i] );

          out[i] <==  (  a10[i] +  a1[i] +  a0[i] +  a[i] );

    }
}