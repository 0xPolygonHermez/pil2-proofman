pragma circom 2.1.0;
pragma custom_templates;

include "test.verifier.circom";
include "mains/recursive1.circom";

// The data of the keys this circuit verifies (lib_params.rs): key_<name>(k) is key k's <name>.
function key_nBits(k) { return 22; }
function key_nBitsExt(k) { return 23; }
function key_arity(k) { return 2; }
function key_lastLevels(k) { return 4; }
function key_nQueries(k) { return 211; }
function key_batchSize(k) { return 27; }
function key_powBits(k) { return 24; }
function key_qDeg(k) { return 1; }
function key_qPolEv(k) { return 13; }
function key_nEvals(k) { return 23; }
function key_nPublics(k) { return 8; }
function key_nAirValues(k) { return 3; }
function key_nAirgroupValues(k) { return 2; }
function key_nProofValues(k) { return 2; }
function key_nStages(k) { return 2; }
function key_nChallenges(k) { return 6; }
function key_hashCommits(k) { return 1; }
function key_circuitType(k) { return 2; }
function key_nRoots(k) { return 19; }
function key_firstRootDim(k) { return 3; }
function key_verkeyInput(k) { return 0; }
function key_enableInput(k) { return 0; }
function key_nTrees(k) { return 5; }
function key_CHALLENGES_PER_STAGE(k) { return [0, 0, 2]; }
function key_AIR_VALUE_STAGE(k) { return [1, 1, 2]; }
function key_AIRGROUP_VALUE_STAGE(k) { return [1, 2]; }
function key_L0_TREE_START(k) { return [0, 5, 14, 17, 19, 21]; }
function key_ROOT_C(k) { return [17284145787926719296, 716892807869615064, 12286782145902210190, 5219202236615833098]; }
function key_proofSize(k) { return 163140; }
function key_offAirgroupValues(k) { return 0; }
function key_offAirValues(k) { return 6; }
function key_offEvals(k) { return 27; }
function key_offFinal(k) { return 162947; }
function key_offNonce(k) { return 163139; }
function key_offBetas(k) { return -1; }
function key_offNonces(k) { return -1; }
function key_OFF_STAGE_ROOT(k) { return [15, 19, 23]; }
function key_OFF_L0_VALUES(k) { return [33140, 50295, 68294, 96, 16618]; }
function key_OFF_L0_SIBLINGS(k) { return [34195, 52194, 68927, 518, 17040]; }
function key_OFF_L0_LAST_LEVELS(k) { return [50231, 68230, 84963, 16554, 33076]; }
function key_OFF_L0_ROOT(k) { return [15, 19, 23, -1, -2]; }
function key_L0_ROOT_PUBLICS(k) { return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 6, 7]; }
function key_OFF_FOLD_ROOT(k) { return [85027, 85031, 85035, 85039]; }
function key_OFF_FOLD_VALUES(k) { return [85043, 117179, 135811, 151067]; }
function key_OFF_FOLD_SIBLINGS(k) { return [105299, 127307, 145939, 161195]; }
function key_OFF_FOLD_LAST_LEVELS(k) { return [117115, 135747, 151003, 162883]; }
function key_OFF_ANS_COEFFS(k) { return [0]; }
function key_OFF_SHAKE_COEFFS(k) { return [0]; }
function key_nOpenings(k) { return 5; }
function key_nPols(k) { return 13; }
function key_nColumnGroups(k) { return 5; }
function key_nPrefixSums(k) { return 4; }
function key_OPENING_ROOT(k) { return [10420286214021487819, 1, 8124823329697072476, 6553637399136210105, 331116024603048646]; }
function key_OPENING_NUM_EVALS(k) { return [1, 13, 4, 3, 2]; }
function key_EVAL_COLUMN(k) { return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 10, 11, 12, 3, 4, 5, 6, 3, 5, 6, 3, 5]; }
function key_PREFIX_SUMS(k) { return [0, 1, 1, 2, 6, 3, 7, 4]; }
function key_COLUMN_GROUPS(k) { return [5, 1, 0, 5, 3, 1, 8, 1, 19, 1, 2, 20, 1, 7, 2, 1, 8, 3, 1, 9, 4, 1, 10, 8, 3, 11, 11, 3, 12, 14, 3, 8, 2, 3, 17, 1, 5, 0, 1, 6, 1, 4, 18, 1, 7, 1, 6, 1, 1]; }
function key_OPENING_EXPONENT(k) { return [-1, 0, 1, 2, 3]; }
function key_nSteps(k) { return 5; }
function key_stir(k) { return 0; }
function key_stirM(k) { return 0; }
function key_stirFinalBits(k) { return 0; }
function key_STEPS(k) { return [23, 18, 14, 10, 6]; }
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
function vadcop_nBasics(i) { return 0; }
function vadcop_basics(i) { return [[0, 0, 0, 0]]; }
function vadcop_rootCAgg(i) { return [0, 0, 0, 0]; }

component main {public [publics, proofValues, globalChallenge, rootCAgg]} = Recursive1(1);
