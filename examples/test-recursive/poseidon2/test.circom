pragma circom 2.1.0;
pragma custom_templates;

include "test.verifier.circom";
include "mains/recursive2.circom";

// The data of the keys this circuit verifies (lib_params.rs): key_<name>(k) is key k's <name>.
function key_nBits(k) { return 17; }
function key_nBitsExt(k) { return 20; }
function key_arity(k) { return 4; }
function key_lastLevels(k) { return 2; }
function key_nQueries(k) { return 72; }
function key_batchSize(k) { return 9; }
function key_powBits(k) { return 21; }
function key_qDeg(k) { return 7; }
function key_qPolEv(k) { return 116; }
function key_nEvals(k) { return 144; }
function key_nPublics(k) { return 399; }
function key_nAirValues(k) { return 0; }
function key_nAirgroupValues(k) { return 0; }
function key_nProofValues(k) { return 0; }
function key_nStages(k) { return 2; }
function key_nChallenges(k) { return 6; }
function key_hashCommits(k) { return 1; }
function key_circuitType(k) { return 0; }
function key_nRoots(k) { return 553; }
function key_firstRootDim(k) { return 3; }
function key_verkeyInput(k) { return 1; }
function key_enableInput(k) { return 1; }
function key_nTrees(k) { return 4; }
function key_CHALLENGES_PER_STAGE(k) { return [0, 0, 2]; }
function key_AIR_VALUE_STAGE(k) { return [0]; }
function key_AIRGROUP_VALUE_STAGE(k) { return [0]; }
function key_L0_TREE_START(k) { return [0, 48, 60, 81, 128]; }
function key_ROOT_C(k) { return [0, 0, 0, 0]; }
function key_proofSize(k) { return 63057; }
function key_offAirgroupValues(k) { return -1; }
function key_offAirValues(k) { return -1; }
function key_offEvals(k) { return 12; }
function key_offFinal(k) { return 62960; }
function key_offNonce(k) { return 63056; }
function key_offBetas(k) { return -1; }
function key_offNonces(k) { return -1; }
function key_OFF_STAGE_ROOT(k) { return [0, 4, 8]; }
function key_OFF_L0_VALUES(k) { return [10804, 21236, 29076, 444]; }
function key_OFF_L0_SIBLINGS(k) { return [14260, 22100, 30588, 3828]; }
function key_OFF_L0_LAST_LEVELS(k) { return [21172, 29012, 37500, 10740]; }
function key_OFF_L0_ROOT(k) { return [0, 4, 8, -1]; }
function key_L0_ROOT_PUBLICS(k) { return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; }
function key_OFF_FOLD_ROOT(k) { return [37564, 37568, 37572, 37576, 37580]; }
function key_OFF_FOLD_VALUES(k) { return [37584, 45424, 51536, 56784, 60304]; }
function key_OFF_FOLD_SIBLINGS(k) { return [39312, 47152, 53264, 58512, 62032]; }
function key_OFF_FOLD_LAST_LEVELS(k) { return [45360, 51472, 56720, 60240, 62896]; }
function key_OFF_ANS_COEFFS(k) { return [0]; }
function key_OFF_SHAKE_COEFFS(k) { return [0]; }
function key_nOpenings(k) { return 19; }
function key_nPols(k) { return 106; }
function key_nColumnGroups(k) { return 5; }
function key_nPrefixSums(k) { return 19; }
function key_OPENING_ROOT(k) { return [11575992183625933494, 17891969694835432085, 16287335884293908063, 11970842550450932764, 12477578612481739161, 8201046199761910586, 17209560082449230913, 8992281176983196029, 2841727033376697931, 16403767884237967044, 25110052138364961, 14361513308298917157, 8892493137794983311, 13583859940902678234, 9071788333329385449, 15139302138664925958, 1, 5718075921287398682, 8167150655112846419]; }
function key_OPENING_NUM_EVALS(k) { return [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 106, 20, 1]; }
function key_EVAL_COLUMN(k) { return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 0, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 1, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 38, 39, 47, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 63]; }
function key_PREFIX_SUMS(k) { return [0, 1, 19, 2, 20, 3, 21, 4, 22, 5, 23, 6, 24, 7, 25, 8, 26, 9, 27, 10, 28, 11, 29, 12, 30, 13, 31, 14, 32, 15, 33, 16, 15, 16, 16, 17, 36, 18]; }
function key_COLUMN_GROUPS(k) { return [34, 1, 0, 125, 1, 35, 1, 1, 48, 3, 16, 84, 2, 81, 1, 3, 82, 1, 4, 83, 1, 5, 84, 1, 6, 85, 1, 7, 86, 1, 8, 87, 1, 9, 88, 1, 10, 89, 1, 11, 90, 1, 12, 91, 1, 13, 92, 1, 14, 93, 1, 15, 94, 1, 16, 95, 1, 17, 96, 1, 18, 97, 1, 19, 98, 1, 20, 99, 1, 21, 100, 1, 22, 101, 1, 23, 102, 1, 24, 103, 1, 25, 104, 1, 26, 105, 1, 27, 106, 1, 28, 107, 1, 29, 108, 1, 30, 109, 1, 31, 110, 1, 32, 111, 1, 33, 112, 1, 34, 113, 1, 35, 114, 1, 36, 115, 1, 37, 116, 1, 40, 119, 1, 41, 120, 1, 42, 121, 1, 43, 122, 1, 44, 123, 1, 45, 124, 1, 46, 126, 1, 48, 0, 1, 49, 1, 1, 50, 2, 1, 51, 3, 1, 52, 4, 1, 53, 5, 1, 54, 6, 1, 55, 7, 1, 56, 8, 1, 57, 9, 1, 58, 10, 1, 59, 11, 1, 60, 12, 1, 61, 13, 1, 62, 14, 1, 80, 32, 1, 81, 33, 1, 82, 34, 1, 83, 35, 1, 84, 36, 1, 85, 37, 1, 86, 38, 1, 87, 39, 1, 88, 40, 1, 89, 41, 1, 90, 42, 1, 91, 43, 1, 92, 44, 1, 93, 45, 1, 94, 46, 1, 95, 47, 1, 96, 51, 3, 97, 54, 3, 98, 57, 3, 99, 60, 3, 100, 63, 3, 101, 66, 3, 102, 69, 3, 103, 72, 3, 104, 75, 3, 105, 78, 3, 36, 19, 38, 117, 1, 39, 118, 1, 47, 127, 1, 64, 16, 1, 65, 17, 1, 66, 18, 1, 67, 19, 1, 68, 20, 1, 69, 21, 1, 70, 22, 1, 71, 23, 1, 72, 24, 1, 73, 25, 1, 74, 26, 1, 75, 27, 1, 76, 28, 1, 77, 29, 1, 78, 30, 1, 79, 31, 1, 37, 1, 63, 15, 1]; }
function key_OPENING_EXPONENT(k) { return [-16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2]; }
function key_nSteps(k) { return 6; }
function key_stir(k) { return 0; }
function key_stirM(k) { return 0; }
function key_stirFinalBits(k) { return 0; }
function key_STEPS(k) { return [20, 17, 14, 11, 8, 5]; }
function key_STIR_LOG_DOMAIN_SIZES(k) { return [0]; }
function key_STIR_FOLDING_FACTORS(k) { return [0]; }
function key_STIR_NUM_QUERIES(k) { return [0]; }
function key_STIR_GRINDING_BITS(k) { return [0]; }

// Key k's constraints at xi.
template ConstraintRootsAt(k) {
    signal input evals[key_nEvals(k)][3];
    signal input challenges[key_nChallenges(k)][3];
    signal input publics[nonEmpty(key_nPublics(k))];
    signal input proofvalues[nonEmpty(key_nProofValues(k))][3];
    signal input airvalues[nonEmpty(key_nAirValues(k))][3];
    signal input airgroupvalues[nonEmpty(key_nAirgroupValues(k))][3];
    signal output roots[key_nRoots(k)][3];

    roots <== ConstraintRoots0()(evals, challenges, publics, proofvalues, airvalues, airgroupvalues);
}

function vadcop_airgroupId() { return 0; }
function vadcop_airId() { return 0; }
function vadcop_nPublics() { return 8; }
function vadcop_nProofValues() { return 2; }
function vadcop_nPVMap() { return 2; }
function vadcop_PV() { return [1, 1]; }
function vadcop_W() { return 368; }
function vadcop_ec() { return 0; }
function vadcop_ecA() { return [0, 0, 0, 0, 0]; }
function vadcop_ecB() { return [0, 0, 0, 0, 0]; }
function vadcop_ecZ() { return [0, 0, 0, 0, 0]; }
function vadcop_ecC1() { return [0, 0, 0, 0, 0]; }
function vadcop_ecC2() { return [0, 0, 0, 0, 0]; }
function vadcop_multiAir() { return 1; }
function vadcop_nAgg(i) { if (i == 0) { return 2; } return 0; }
function vadcop_AGG(i) { if (i == 0) { return [0, 0]; } return [0]; }
function vadcop_nBasics(i) { if (i == 0) { return 3; } return 0; }
function vadcop_basics(i) { if (i == 0) { return [[2049806082782783542, 4300170745218565799, 4962480704197022298, 8606065544326168321], [5351164351789367245, 4932744242113048280, 8763218587036740969, 14035398193893510475], [16521124971116069568, 1822992213097185690, 15064755042065937856, 9343732855389029392]]; } return [[0, 0, 0, 0]]; }
function vadcop_rootCAgg(i) { return [0, 0, 0, 0]; }

component main {public [publics, proofValues, globalChallenge, rootCAgg]} = Recursive2(3);
