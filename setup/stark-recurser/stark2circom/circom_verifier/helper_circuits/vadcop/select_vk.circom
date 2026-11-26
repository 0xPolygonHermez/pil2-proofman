pragma circom 2.1.0;

include "iszero.circom";
include "vadcop/circuit_type.circom";

/*
    The verification key (rootC) of a child proof of airgroup `airgroupId`, by its circuit type:
    circuitTypeOf(airgroupId, 0) is the airgroup's recursive2, whose key rootCAggregated is an input (it is
    the circuit's own, unknown when it is compiled); circuitTypeOf(airgroupId, i + 1) is basic air i. When
    nullable, type 0 is an empty slot, keyed [0, 0, 0, 0]. Any other type fails.
*/
template SelectVerificationKey(nBasics, airgroupId, nullable) {
    var nTypes = nullable + 1 + nBasics;

    signal input circuitType;
    signal input rootCBasics[nBasics][4];
    signal input rootCAggregated[4];
    signal output verificationKey[4];

    // isType[i]: circuitType is type i (empty slot first when nullable, then recursive2, then the basic
    // airs); exactly one holds, and the key is the sum of the keys weighted by them.
    signal isType[nTypes];
    var nMatches = 0;
    for (var i = 0; i < nTypes; i++) {
        var expectedType = nullable == 1 && i == 0 ? 0 : circuitTypeOf(airgroupId, i - nullable);
        isType[i] <== IsZero()(expectedType - circuitType);
        nMatches += isType[i];
    }
    nMatches === 1;

    signal keys[nTypes][4];
    if (nullable == 1) { keys[0] <== [0, 0, 0, 0]; }
    keys[nullable] <== rootCAggregated;
    for (var i = 0; i < nBasics; i++) { keys[nullable + 1 + i] <== rootCBasics[i]; }

    signal accKey[nTypes][4];
    for (var i = 0; i < nTypes; i++) {
        for (var j = 0; j < 4; j++) {
            if (i == 0) {
                accKey[i][j] <== isType[i] * keys[i][j];
            } else {
                accKey[i][j] <== isType[i] * keys[i][j] + accKey[i - 1][j];
            }
        }
    }
    verificationKey <== accKey[nTypes - 1];
}
