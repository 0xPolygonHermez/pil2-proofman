pragma circom 2.1.0;
pragma custom_templates;

// The tree selector of the families whose plonk air has the TreeSelector4 gate: TreeSelector
// (treeselector.circom) over it.
include "treeselector.circom";

function treeSelectorGateBits() { return 2; }

// The gate behind TreeSelector's family-neutral name.
template TreeSelectorGate() {
    signal input values[4][3];
    signal input keys[2];
    signal output out[3];

    out <== TreeSelector4()(values, keys);
}

// values[key], key LSB-first; the selection is a hint the gate's air constrains.
template custom TreeSelector4() {
    signal input values[4][3];
    signal input keys[2];

    signal output out[3];

    var root[3];

    assert(keys[0]*(keys[0] - 1) == 0);
    assert(keys[1]*(keys[1] - 1) == 0);
    
    if(keys[0] == 0 && keys[1] == 0) {
        root = values[0];
    } else if(keys[0] == 1 && keys[1] == 0) {
        root = values[1];
    } else if(keys[0] == 0 && keys[1] == 1) {
        root = values[2];
    } else {
        root = values[3];
    }

    out <-- root;
}
