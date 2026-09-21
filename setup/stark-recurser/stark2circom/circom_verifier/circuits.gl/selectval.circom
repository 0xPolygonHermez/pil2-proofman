pragma circom 2.1.0;
pragma custom_templates;

include "utils.circom";

template SelectValue(arity, nLastLevels, num_nodes_level) {
    var nBits = log2(arity);
    signal input values[arity**nLastLevels][4];
    signal input {binary}key[nLastLevels][nBits];
    signal output selected_value[4];

    if (nLastLevels == 0) {
        selected_value <== values[0];
    } else {
        var next_n = (num_nodes_level + (arity -  1)) \ arity;        
        component mNext = SelectValue(arity, nLastLevels - 1, next_n);

        component selected_values[next_n];
        component selected_values2[next_n];

        // Both arities select through a gate. Inlining `key * (b - a) + a` here
        // is a signal-by-signal product: four non-linear constraints per node
        // per level, where a gate pays none.
        if (arity == 2) {
            for (var j = 0; j < next_n; j++) {
                selected_values2[j] = SelectValueArity2();
                selected_values2[j].key <== key[0];
                for (var a = 0; a < arity; a++) {
                    selected_values2[j].values[a] <== values[arity * j + a];
                }
                mNext.values[j] <== selected_values2[j].selected_value;
            }
        } else {
            for (var j = 0; j < next_n; j++) {
                selected_values[j] = SelectValueArity4();
                selected_values[j].key <== key[0];
                for (var a = 0; a < arity; a++) {
                    selected_values[j].values[a] <== values[arity * j + a];
                }
                mNext.values[j] <== selected_values[j].selected_value;
            }
        }

        for (var k = next_n; k < arity**(nLastLevels - 1); k++) {
            for (var t = 0; t < 4; t++) {
                mNext.values[k][t] <== 0;
            }
        }

        signal {binary} keyTags[nLastLevels - 1][nBits];
        for (var b = 0; b < nLastLevels - 1; b++) {
            keyTags[b] <== key[b + 1];
        }
        mNext.key <== keyTags;

        selected_value <== mNext.selected_value;
    }
}

template custom SelectValueArity4() {
    signal input values[4][4];
    signal input key[2];
    signal output selected_value[4];

    var index = (key[0] + key[1]*2);
    selected_value <-- values[index];
}

/*
    Arity-2 twin of SelectValueArity4: pick one of two 4-element values by a
    single key bit, so the arity-2 tree needs no inline mux.
*/
template custom SelectValueArity2() {
    signal input values[2][4];
    signal input key[1];
    signal output selected_value[4];

    selected_value <-- values[key[0]];
}
