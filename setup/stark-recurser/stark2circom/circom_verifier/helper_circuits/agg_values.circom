pragma circom 2.1.0;
pragma custom_templates;

include "mux1.circom";
include "bitify.circom";
include "cmul.circom";


template AggregateAirgroupValues() {
    signal input airgroupValueA[3];
    signal input airgroupValueB[3];

    signal input {binary} aggregationType; // 1 if aggregation is multiplication, 0 if aggregation is addition

    signal output airgroupValueAB[3];

    // Products are Fp³ products, as the prover aggregates them (global_hints.rs)
    signal values[2][3];
    values[0] <== [airgroupValueA[0] + airgroupValueB[0], airgroupValueA[1] + airgroupValueB[1], airgroupValueA[2] + airgroupValueB[2]];
    values[1] <== CMul()(airgroupValueA, airgroupValueB);

    // Either add or multiply the airgroupvalues according to the aggregation type and then return the result
    airgroupValueAB <== MultiMux1(3)(values, aggregationType);
}

// What a null proof contributes: the identity of the aggregation, 0 for addition and 1 for multiplication
template NullifyAirgroupValue() {
    signal input airgroupValue[3];
    signal input {binary} aggregationType;
    signal input {binary} isNull;

    signal output out[3];

    out[0] <== airgroupValue[0] + isNull * (aggregationType - airgroupValue[0]);
    out[1] <== (1 - isNull) * airgroupValue[1];
    out[2] <== (1 - isNull) * airgroupValue[2];
}

template AggregateAirgroupValuesNull() {
    signal input airgroupValueA[3];
    signal input airgroupValueB[3];
    signal input {binary} aggregationType; // 1 if aggregation is multiplication, 0 if aggregation is addition
    signal input {binary} isNullA; // 1 if is circuit type A is 0 (null), 0 otherwise 
    signal input {binary} isNullB; // 1 if is circuit type B is 0 (null), 0 otherwise 

    signal output airgroupValueAB[3];

    signal valueA[3] <== NullifyAirgroupValue()(airgroupValueA, aggregationType, isNullA);
    signal valueB[3] <== NullifyAirgroupValue()(airgroupValueB, aggregationType, isNullB);

    airgroupValueAB <== AggregateAirgroupValues()(valueA, valueB, aggregationType);
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