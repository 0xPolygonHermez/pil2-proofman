pragma circom 2.1.0;
pragma custom_templates;

include "bitify.circom";
include "cmul.circom";


// Two airgroup values aggregated: an Fp³ product when isProduct (as the prover aggregates them,
// global_hints.rs), a sum otherwise. isProduct is the airgroup's own type, known at compile time.
template AggregateAirgroupValues(isProduct) {
    signal input a[3];
    signal input b[3];
    signal output out[3];

    if (isProduct == 1) {
        out <== CMul()(a, b);
    } else {
        out <== [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    }
}

// What a null proof contributes: the aggregation's identity, 1 for a product and 0 for a sum.
template NullifyAirgroupValue(isProduct) {
    signal input value[3];
    signal input {binary} isNull;
    signal output out[3];

    out[0] <== value[0] + isNull * (isProduct - value[0]);
    out[1] <== (1 - isNull) * value[1];
    out[2] <== (1 - isNull) * value[2];
}

template AggregateAirgroupValuesNull(isProduct) {
    signal input a[3];
    signal input b[3];
    signal input {binary} isNullA;
    signal input {binary} isNullB;
    signal output out[3];

    signal valueA[3] <== NullifyAirgroupValue(isProduct)(a, isNullA);
    signal valueB[3] <== NullifyAirgroupValue(isProduct)(b, isNullB);
    out <== AggregateAirgroupValues(isProduct)(valueA, valueB);
}

template AggregateValues(n) {
    signal input valuesA[n];
    signal input valuesB[n];

    signal output valuesAB[n];

    for (var i = 0; i < n; i++) {
        valuesAB[i] <== valuesA[i] + valuesB[i];
    }
}

template AggregateValuesNull(n) {
    signal input valuesA[n];
    signal input valuesB[n];
    signal input {binary} isNullA; // 1 if is circuit type A is 0 (null), 0 otherwise 
    signal input {binary} isNullB; // 1 if is circuit type B is 0 (null), 0 otherwise 

    signal output valuesAB[n];

    // If circuit type A is null, then its values are zero;
    signal valuesA_nullified[n];
    for (var i = 0; i < n; i++) {
        valuesA_nullified[i] <== (1 - isNullA) * valuesA[i];
    }

    // If circuit type B is null, then its values are zero;
    signal valuesB_nullified[n];
    for (var i = 0; i < n; i++) {
        valuesB_nullified[i] <== (1 - isNullB) * valuesB[i];
    }

    for (var i = 0; i < n; i++) {
        valuesAB[i] <== valuesA_nullified[i] + valuesB_nullified[i];
    }
}

template AggregateProofsNull(n) {
    signal input nAggregatedProofs[n];
    signal input {binary} isNull[n];

    signal output totalAggregatedProofs;

    signal values[n];
    signal nPartialAggregatedProofs[n];

    values[0] <== (1 - isNull[0]) * nAggregatedProofs[0];
    LessThan20Bits()(values[0]);
    nPartialAggregatedProofs[0] <== values[0];

    for (var i = 1; i < n; i++) {
        values[i] <== (1 - isNull[i]) * nAggregatedProofs[i];
        LessThan20Bits()(values[i]);
        nPartialAggregatedProofs[i] <== nPartialAggregatedProofs[i - 1] + values[i];
        LessThan20Bits()(nPartialAggregatedProofs[i]);
    }

    totalAggregatedProofs <== nPartialAggregatedProofs[n - 1];
}

template AggregateProofs(n) {
    signal input nAggregatedProofs[n];
    signal output totalAggregatedProofs;

    signal nPartialAggregatedProofs[n];

    nPartialAggregatedProofs[0] <== nAggregatedProofs[0];
    LessThan20Bits()(nAggregatedProofs[0]);

    for (var i = 1; i < n; i++) {
        LessThan20Bits()(nAggregatedProofs[i]);
        nPartialAggregatedProofs[i] <== nPartialAggregatedProofs[i - 1] + nAggregatedProofs[i];
        LessThan20Bits()(nPartialAggregatedProofs[i]);
    }

    totalAggregatedProofs <== nPartialAggregatedProofs[n - 1];
}