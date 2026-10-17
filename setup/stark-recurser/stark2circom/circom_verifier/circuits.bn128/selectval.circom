pragma circom 2.1.0;

/*
    The BN128 verifier's SelectValue, which the include path picks over circuits.gl's: the node a
    path's last levels lead to, out of the published level. The Goldilocks verifier selects through
    a custom gate (SelectValueArity2); the BN128 wrap has no such gate, so the selection is
    key*(b - a) + a, one product per word per node per level, which plonk2pil lays out as PLONK.
*/
template SelectValue(arity, nLastLevels, num_nodes_level) {
    assert(arity == 2);
    signal input values[arity**nLastLevels][4];
    signal input {binary} key[nLastLevels][1];
    signal output selected_value[4];

    if (nLastLevels == 0) {
        selected_value <== values[0];
    } else {
        var next_n = (num_nodes_level + (arity - 1)) \ arity;
        component mNext = SelectValue(arity, nLastLevels - 1, next_n);
        for (var j = 0; j < next_n; j++) {
            for (var t = 0; t < 4; t++) {
                mNext.values[j][t] <== key[0][0] * (values[2 * j + 1][t] - values[2 * j][t]) + values[2 * j][t];
            }
        }
        for (var k = next_n; k < arity**(nLastLevels - 1); k++) {
            for (var t = 0; t < 4; t++) {
                mNext.values[k][t] <== 0;
            }
        }
        signal {binary} keyTags[nLastLevels - 1][1];
        for (var b = 0; b < nLastLevels - 1; b++) {
            keyTags[b] <== key[b + 1];
        }
        mNext.key <== keyTags;
        selected_value <== mNext.selected_value;
    }
}
