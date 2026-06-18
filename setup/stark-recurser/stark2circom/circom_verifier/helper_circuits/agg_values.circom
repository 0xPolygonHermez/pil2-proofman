pragma circom 2.1.0;
pragma custom_templates;

include "mux1.circom";
include "bitify.circom";


template AggregateAirgroupValues() {
    signal input airgroupValueA[3];
    signal input airgroupValueB[3];

    signal input {binary} aggregationType; // 1 if aggregation is multiplication, 0 if aggregation is addition

    signal output airgroupValueAB[3];

    signal values[2][3];
    values[0] <== [airgroupValueA[0] + airgroupValueB[0], airgroupValueA[1] + airgroupValueB[1], airgroupValueA[2] + airgroupValueB[2]];
    values[1] <== [airgroupValueA[0] * airgroupValueB[0], airgroupValueA[1] * airgroupValueB[1], airgroupValueA[2] * airgroupValueB[2]];

    // Either add or multiply the airgroupvalues according to the aggregation type and then return the result
    airgroupValueAB <== MultiMux1(3)(values, aggregationType);
}

template AggregateAirgroupValuesNull() {
    signal input airgroupValueA[3];
    signal input airgroupValueB[3];
    signal input {binary} aggregationType; // 1 if aggregation is multiplication, 0 if aggregation is addition
    signal input {binary} isNullA; // 1 if is circuit type A is 0 (null), 0 otherwise 
    signal input {binary} isNullB; // 1 if is circuit type B is 0 (null), 0 otherwise 


    signal output airgroupValueAB[3];

    // If circuit type A is null, then its airgroupvalue is zero;
    signal valueA[3] <== [ (1 - isNullA)*airgroupValueA[0], (1 - isNullA)*airgroupValueA[1], (1 - isNullA)*airgroupValueA[2] ];

    // If circuit type B is null, then its airgroupvalue is zero;
    signal valueB[3] <== [ (1 - isNullB)*airgroupValueB[0], (1 - isNullB)*airgroupValueB[1], (1 - isNullB)*airgroupValueB[2] ];

    signal values[2][3];
    values[0] <== [valueA[0] + valueB[0], valueA[1] + valueB[1], valueA[2] + valueB[2]];
    values[1] <== [valueA[0] * valueB[0], valueA[1] * valueB[1], valueA[2] * valueB[2]];

    // Either add or multiply the airgroupvalues according to the aggregation type and then return the result
    airgroupValueAB <== MultiMux1(3)(values, aggregationType);
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