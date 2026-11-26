pragma circom 2.1.0;

/*
    values[key] for 2^nLevels values, key LSB-first. The includer (treeselector4.circom or
    treeselector8.circom) brings TreeSelectorGate, a gate over 2^g values (g = treeSelectorGateBits()):
    a tree of them takes g key bits per level, low bits first. A remainder of one bit is a linear select;
    of two bits (g = 3), one gate over four values padded with zeros.
*/
template TreeSelector(nLevels, eSize) {
    var g = treeSelectorGateBits();
    var gateSize = 1 << g;
    var n = 1 << nLevels;
    signal input values[n][eSize];
    signal input {binary} key[nLevels];
    signal output out[eSize];

    // The gates of every level but the leaves', level after level.
    var nGates = 0;
    var levelSize = n;
    for (var i = 0; i < nLevels \ g; i++) {
        levelSize = levelSize \ gateSize;
        nGates += levelSize;
    }
    component gate[nGates];

    levelSize = n;
    var levelStart = 0;
    var previousStart = 0;
    for (var i = 0; i < nLevels \ g; i++) {
        var nLevelGates = levelSize \ gateSize;
        for (var j = 0; j < nLevelGates; j++) {
            gate[levelStart + j] = TreeSelectorGate();
            for (var l = 0; l < gateSize; l++) {
                if (i == 0) {
                    gate[levelStart + j].values[l] <== values[gateSize * j + l];
                } else {
                    gate[levelStart + j].values[l] <== gate[previousStart + gateSize * j + l].out;
                }
            }
            for (var b = 0; b < g; b++) { gate[levelStart + j].keys[b] <== key[g * i + b]; }
        }
        previousStart = levelStart;
        levelStart += nLevelGates;
        levelSize = nLevelGates;
    }

    if (levelSize == 1) {
        if (nGates == 0) {
            out <== values[0];
        } else {
            out <== gate[previousStart].out;
        }
    } else if (levelSize == 2) {
        for (var k = 0; k < eSize; k++) {
            if (nGates == 0) {
                out[k] <== key[nLevels - 1] * (values[1][k] - values[0][k]) + values[0][k];
            } else {
                out[k] <== key[nLevels - 1] * (gate[previousStart + 1].out[k] - gate[previousStart].out[k]) + gate[previousStart].out[k];
            }
        }
    } else {
        component last = TreeSelectorGate();
        for (var l = 0; l < gateSize; l++) {
            if (l >= 4) {
                last.values[l] <== [0, 0, 0];
            } else if (nGates == 0) {
                last.values[l] <== values[l];
            } else {
                last.values[l] <== gate[previousStart + l].out;
            }
        }
        last.keys[0] <== key[nLevels - 2];
        last.keys[1] <== key[nLevels - 1];
        last.keys[2] <== 0;
        out <== last.out;
    }
}
