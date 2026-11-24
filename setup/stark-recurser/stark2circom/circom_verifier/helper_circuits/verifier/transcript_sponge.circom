pragma circom 2.1.0;
pragma custom_templates;

/*
    Fiat-Shamir transcript, a Poseidon sponge: the prover's
    Poseidon TranscriptGL in circom. `SpongePermutation`, its rate and capacity come from the family's sponge.circom
    (Goldilocks: rate 12, 4-cell capacity; BN128: one-cell capacity), included by the generated verifier.

    OPS[2k] is 0 (absorb) or 1 (squeeze), OPS[2k+1] how many words; `in` holds the absorbed words
    and `out` the squeezed ones, each in order. A squeeze reads the last permutation's outputs as a
    FIFO; an absorb drops the unread ones.
*/
template TranscriptEngine(nIn, nOut, nOps, OPS) {
    signal input in[nIn];
    signal output out[nOut];

    var R = sponge_rate();
    var C = sponge_capacity();

    // Upper bound; entries past the last permutation are never created.
    component h[nIn \ R + nOut + 1];

    var pending[R];
    var nPending = 0;
    var state[C];
    for (var i = 0; i < C; i++) { state[i] = 0; }
    var cur[R + C];
    var avail = 0;
    var nH = 0;
    var ii = 0;
    var oo = 0;
    for (var k = 0; k < nOps; k++) {
        for (var j = 0; j < OPS[2 * k + 1]; j++) {
            if (OPS[2 * k] == 0) {
                avail = 0;
                pending[nPending] = in[ii];
                ii++;
                nPending++;
            }
            if (OPS[2 * k] == 1 && avail == 0) {
                for (var i = nPending; i < R; i++) { pending[i] = 0; }
                nPending = R;
            }
            if (nPending == R) {
                h[nH] = SpongePermutation();
                for (var i = 0; i < R; i++) { h[nH].in[i] <== pending[i]; }
                for (var i = 0; i < C; i++) { h[nH].capacity[i] <== state[i]; }
                for (var i = 0; i < R + C; i++) { cur[i] = h[nH].out[i]; }
                for (var i = 0; i < C; i++) { state[i] = cur[i]; }
                nH++;
                nPending = 0;
                avail = R + C;
            }
            if (OPS[2 * k] == 1) {
                out[oo] <== cur[R + C - avail];
                avail--;
                oo++;
            }
        }
    }
}
