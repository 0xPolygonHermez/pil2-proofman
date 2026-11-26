pragma circom 2.1.0;
pragma custom_templates;

// The tree selector of the families whose plonk air has the TreeSelector8 gate: TreeSelector
// (treeselector.circom) over it.
include "treeselector.circom";

function treeSelectorGateBits() { return 3; }

// The gate behind TreeSelector's family-neutral name.
template TreeSelectorGate() {
    signal input values[8][3];
    signal input keys[3];
    signal output out[3];

    out <== TreeSelector8()(values, keys);
}

// values[key], key LSB-first; the selection is a hint the gate's air constrains.
template custom TreeSelector8() {
    signal input values[8][3];
    signal input keys[3];

    signal output out[3];

    var root[3];

    assert(keys[0]*(keys[0] - 1) == 0);
    assert(keys[1]*(keys[1] - 1) == 0);
    assert(keys[2]*(keys[2] - 1) == 0);
    
    if(keys[0] == 0 && keys[1] == 0 && keys[2] == 0) {
        root = values[0];
    } else if(keys[0] == 1 && keys[1] == 0 && keys[2] == 0) {
        root = values[1];
    } else if(keys[0] == 0 && keys[1] == 1 && keys[2] == 0) {
        root = values[2];
    } else if(keys[0] == 1 && keys[1] == 1 && keys[2] == 0){
        root = values[3];
    } else if(keys[0] == 0 && keys[1] == 0 && keys[2] == 1) {
        root = values[4];
    } else if(keys[0] == 1 && keys[1] == 0 && keys[2] == 1) {
        root = values[5];
    } else if(keys[0] == 0 && keys[1] == 1 && keys[2] == 1){
        root = values[6];
    } else {
        root = values[7];
    }

    out <-- root;
}
