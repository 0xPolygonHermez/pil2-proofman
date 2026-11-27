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
function key_qPolEv(k) { return 119; }
function key_nEvals(k) { return 152; }
function key_nPublics(k) { return 399; }
function key_nAirValues(k) { return 0; }
function key_nAirgroupValues(k) { return 0; }
function key_nProofValues(k) { return 0; }
function key_nStages(k) { return 2; }
function key_nChallenges(k) { return 6; }
function key_hashCommits(k) { return 1; }
function key_circuitType(k) { return 0; }
function key_nRoots(k) { return 598; }
function key_firstRootDim(k) { return 3; }
function key_verkeyInput(k) { return 1; }
function key_enableInput(k) { return 1; }
function key_nTrees(k) { return 4; }
function key_CHALLENGES_PER_STAGE(k) { return [0, 0, 2]; }
function key_AIR_VALUE_STAGE(k) { return [0]; }
function key_AIRGROUP_VALUE_STAGE(k) { return [0]; }
function key_L0_TREE_START(k) { return [0, 48, 60, 81, 127]; }
function key_ROOT_C(k) { return [0, 0, 0, 0]; }
function key_proofSize(k) { return 63009; }
function key_offAirgroupValues(k) { return -1; }
function key_offAirValues(k) { return -1; }
function key_offEvals(k) { return 12; }
function key_offFinal(k) { return 62912; }
function key_offNonce(k) { return 63008; }
function key_offBetas(k) { return -1; }
function key_offNonces(k) { return -1; }
function key_OFF_STAGE_ROOT(k) { return [0, 4, 8]; }
function key_OFF_L0_VALUES(k) { return [10756, 21188, 29028, 468]; }
function key_OFF_L0_SIBLINGS(k) { return [14212, 22052, 30540, 3780]; }
function key_OFF_L0_LAST_LEVELS(k) { return [21124, 28964, 37452, 10692]; }
function key_OFF_L0_ROOT(k) { return [0, 4, 8, -1]; }
function key_L0_ROOT_PUBLICS(k) { return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; }
function key_OFF_FOLD_ROOT(k) { return [37516, 37520, 37524, 37528, 37532]; }
function key_OFF_FOLD_VALUES(k) { return [37536, 45376, 51488, 56736, 60256]; }
function key_OFF_FOLD_SIBLINGS(k) { return [39264, 47104, 53216, 58464, 61984]; }
function key_OFF_FOLD_LAST_LEVELS(k) { return [45312, 51424, 56672, 60192, 62848]; }
function key_OFF_ANS_COEFFS(k) { return [0]; }
function key_OFF_SHAKE_COEFFS(k) { return [0]; }
function key_nOpenings(k) { return 19; }
function key_nPols(k) { return 105; }
function key_nColumnGroups(k) { return 6; }
function key_nPrefixSums(k) { return 21; }
function key_OPENING_ROOT(k) { return [11575992183625933494, 17891969694835432085, 16287335884293908063, 11970842550450932764, 12477578612481739161, 8201046199761910586, 17209560082449230913, 8992281176983196029, 2841727033376697931, 16403767884237967044, 25110052138364961, 14361513308298917157, 8892493137794983311, 13583859940902678234, 9071788333329385449, 15139302138664925958, 1, 5718075921287398682, 8167150655112846419]; }
function key_OPENING_NUM_EVALS(k) { return [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 4, 105, 25, 1]; }
function key_EVAL_COLUMN(k) { return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 1, 2, 38, 39, 40, 41, 42, 43, 44, 0, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 3, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 38, 46, 47, 48, 49, 50, 51, 52, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 62]; }
function key_PREFIX_SUMS(k) { return [0, 1, 19, 2, 20, 3, 21, 4, 22, 5, 23, 6, 24, 7, 25, 8, 26, 9, 27, 10, 28, 11, 29, 12, 30, 13, 31, 14, 32, 15, 33, 16, 14, 15, 35, 16, 15, 16, 16, 17, 38, 18]; }
function key_COLUMN_GROUPS(k) { return [34, 1, 0, 124, 1, 36, 2, 1, 115, 1, 2, 116, 1, 37, 1, 3, 48, 3, 16, 76, 4, 81, 1, 5, 82, 1, 6, 83, 1, 7, 84, 1, 8, 85, 1, 9, 86, 1, 10, 87, 1, 11, 88, 1, 12, 89, 1, 13, 90, 1, 14, 91, 1, 15, 92, 1, 16, 93, 1, 17, 94, 1, 18, 95, 1, 19, 96, 1, 20, 97, 1, 21, 98, 1, 22, 99, 1, 23, 100, 1, 24, 101, 1, 25, 102, 1, 26, 103, 1, 27, 104, 1, 28, 105, 1, 29, 106, 1, 30, 107, 1, 31, 108, 1, 32, 109, 1, 33, 110, 1, 34, 111, 1, 35, 112, 1, 36, 113, 1, 37, 114, 1, 39, 118, 1, 40, 119, 1, 41, 120, 1, 42, 121, 1, 43, 122, 1, 44, 123, 1, 45, 125, 1, 53, 6, 1, 54, 7, 1, 55, 8, 1, 56, 9, 1, 57, 10, 1, 58, 11, 1, 59, 12, 1, 60, 13, 1, 61, 14, 1, 79, 32, 1, 80, 33, 1, 81, 34, 1, 82, 35, 1, 83, 36, 1, 84, 37, 1, 85, 38, 1, 86, 39, 1, 87, 40, 1, 88, 41, 1, 89, 42, 1, 90, 43, 1, 91, 44, 1, 92, 45, 1, 93, 46, 1, 94, 47, 1, 95, 51, 3, 96, 54, 3, 97, 57, 3, 98, 60, 3, 99, 63, 3, 100, 66, 3, 101, 69, 3, 102, 72, 3, 103, 75, 3, 104, 78, 3, 38, 24, 38, 117, 1, 46, 126, 1, 47, 0, 1, 48, 1, 1, 49, 2, 1, 50, 3, 1, 51, 4, 1, 52, 5, 1, 63, 16, 1, 64, 17, 1, 65, 18, 1, 66, 19, 1, 67, 20, 1, 68, 21, 1, 69, 22, 1, 70, 23, 1, 71, 24, 1, 72, 25, 1, 73, 26, 1, 74, 27, 1, 75, 28, 1, 76, 29, 1, 77, 30, 1, 78, 31, 1, 39, 1, 62, 15, 1]; }
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
function vadcop_basics(i) { if (i == 0) { return [[202556868925631864, 8253526039676467598, 973455749637265658, 17000121448691290678], [10083618062899040488, 8560748376547614721, 24964050296172539, 2383367642340288215], [14453292471017267706, 14000900561165299894, 2682987345211412535, 6512486633536822164]]; } return [[0, 0, 0, 0]]; }
function vadcop_rootCAgg(i) { return [0, 0, 0, 0]; }

component main {public [publics, proofValues, globalChallenge, rootCAgg]} = Recursive2(3);
