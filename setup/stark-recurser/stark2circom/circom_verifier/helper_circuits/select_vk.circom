pragma circom 2.1.0;

include "mux1.circom";
include "mux2.circom";
include "iszero.circom";

// Given a circuit type, return the corresponding verification key
template SelectVerificationKeyNull(nBasics) {

    // 0 -> Null
    // 1 -> Aggregated
    // 2 -> Basic1 (Basic)
    // 3 -> Basic2 (Basic)
    // ...
    signal input circuitType;

    // Basic verification keys;
    signal input rootCBasics[nBasics][4];

    // Aggregated verification key. It needs to be provided as input because it is not known at compile time
    signal input rootCAggregated[4];
    
    // Verification key corresponding to the circuit type
    signal output verificationKey[4];

    // The idea is to store an array of booleans which will indicate either 0 or 1 depending if the circuit type is equal to the index
    // For instance, [0,1,0,0] would mean that the circuit type is aggregated.
    // Then, we will calculate the verification key as the sum of all the verification keys multiplied by the corresponding boolean
    signal isType[nBasics + 2];
    var isValidType = 0;
    for(var i = 0; i < nBasics + 2; i++) {
        isType[i] <== IsZero()(i - circuitType);
        isValidType += isType[i];
    }

    // Check that the type is supported
    isValidType === 1;

    signal verificationKeys[nBasics + 2][4];
    verificationKeys[0] <== [0,0,0,0];
    verificationKeys[1] <== rootCAggregated;
    for(var i = 0; i < nBasics; i++) {
        verificationKeys[i + 2] <== rootCBasics[i];
    }

    signal accVK[nBasics + 2][4];

    for(var i = 0; i < nBasics + 2; i++) {
        if(i == 0) {
            for(var j = 0; j < 4; j++) {
                accVK[i][j] <== isType[i]*verificationKeys[i][j];
            }
        } else {
            for(var j = 0; j < 4; j++) {
                accVK[i][j] <== isType[i]*verificationKeys[i][j] + accVK[i - 1][j];
            }
        }
    }

    verificationKey <== accVK[nBasics + 1];
}

// Given a circuit type, return the corresponding verification key
template SelectVerificationKey(nBasic) {

    // 0 -> Aggregated
    // 1 -> Basic1 (Basic)
    // 2 -> Basic2 (Basic)
    // ...

    signal input circuitType;
    
    // Basic verification keys;
    signal input rootCBasics[nBasic][4];

    // Aggregated verification key. It needs to be provided as input because it is not known at compile time
    signal input rootCAggregated[4];

    // Verification key corresponding to the circuit type
    signal output verificationKey[4];

    // The idea is to store an array of booleans which will indicate either 0 or 1 depending if the circuit type is equal to the index
    // For instance, [0,1,0,0] would mean that the circuit type is a aggregated.
    // Then, we will calculate the verification key as the sum of all the verification keys multiplied by the corresponding boolean
    signal isType[nBasic + 1];
    var isValidType = 0;
    for(var i = 0; i < nBasic + 1; i++) {
        isType[i] <== IsZero()(i - circuitType);
        isValidType += isType[i];
    }

    // Check that the type is supported
    isValidType === 1;

    signal verificationKeys[nBasic + 1][4];
    verificationKeys[0] <== rootCAggregated;
    for(var i = 0; i < nBasic; i++) {
        verificationKeys[i + 1] <== rootCBasics[i];
    }

    signal accVK[nBasic + 1][4];

    for(var i = 0; i < nBasic + 1; i++) {
        if(i == 0) {
            for(var j = 0; j < 4; j++) {
                accVK[i][j] <== isType[i]*verificationKeys[i][j];
            }
        } else {
            for(var j = 0; j < 4; j++) {
                accVK[i][j] <== isType[i]*verificationKeys[i][j] + accVK[i - 1][j];
            }
        }
    }

    verificationKey <== accVK[nBasic];
}