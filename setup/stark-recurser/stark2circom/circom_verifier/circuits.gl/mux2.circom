pragma circom 2.1.0;

/*
    Multiplexor used in recursive2
*/
template MultiMux2(n, m) {
    signal input c[4][n][m];    // Constants
    signal input {binary} s[2]; // Selector
    signal output out[n][m];

    signal a10[n][m];
    signal a1[n][m];
    signal a0[n][m];
    signal a[n][m];

    signal s10;
    s10 <== s[1] * s[0];
    for (var i = 0; i < n; i++) {
        for (var j = 0; j < m; j++) {
            a10[i][j] <== (c[3][i][j] - c[2][i][j] - c[1][i][j] + c[0][i][j]) * s10;
            a1[i][j]  <== (             c[2][i][j]              - c[0][i][j]) * s[1];
            a0[i][j]  <== (                          c[1][i][j] - c[0][i][j]) * s[0];
            a[i][j]   <== (                                       c[0][i][j]);

            out[i][j] <==  (a10[i][j] + a1[i][j] + a0[i][j] + a[i][j]);
        }
    }
}