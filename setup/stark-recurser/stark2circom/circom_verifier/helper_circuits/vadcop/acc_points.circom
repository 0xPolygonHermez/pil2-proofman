pragma circom 2.1.0;
pragma custom_templates;

include "mux2.circom";
include "elliptic_curve.circom";

template AccumulatePoints() {
    signal input pointA[2][5];
    signal input pointB[2][5];

    // Compute the addition of pointA and pointB
    signal output pointAB[2][5] <== AddECFp5()(pointA,pointB);
}

template AccumulatePointsNull() {
    signal input pointA[2][5];
    signal input pointB[2][5];
    signal input {binary} isNullA; // 1 if is circuit type A is 0 (null), 0 otherwise
    signal input {binary} isNullB; // 1 if is circuit type B is 0 (null), 0 otherwise

    signal output pointAB[2][5];
    
    // Compute the addition of pointA and pointB
    signal additionAB[2][5] <== AddECFp5()(pointA,pointB);

    // If a point is 𝒪, it means that the air you are trying to aggregate with does not have this stage
    // Therefore we should proceed the same way as if circuitType is null
    signal {binary} isPointAInf <== IsInfinityECFp5()(pointA);
    signal {binary} isPointBInf <== IsInfinityECFp5()(pointB);

    // If circuit type A is null || pointA is 𝒪 and circuit type B is null || pointB is 𝒪, then pointAB is 𝒪
    // If circuit type A is not null && pointA is not 𝒪 and circuit type B is null || pointB is 𝒪, then pointAB is pointA
    // If circuit type A is null || pointA is 𝒪 and circuit type B is not null && pointB is not 𝒪, then pointAB is pointB
    // If circuit type A is not null && pointA is not 𝒪 and circuit type B is not null && pointB is not 𝒪, then pointAB is additionAB
    signal {binary} s[2] <== [(1 - isNullA)*(1 - isPointAInf), (1 - isNullB)*(1 - isPointBInf)];
    pointAB <== MultiMux2(2,5)([[[0,0,0,0,0],[0,0,0,0,0]], pointA, pointB, additionAB], s);
}