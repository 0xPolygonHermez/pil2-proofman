#include "zkglobals.hpp"
#include "proof2zkinStark.hpp"
#include "starks.hpp"
#include "global_constraints.hpp"
#include "gen_recursive_proof.hpp"
#include "gen_proof.hpp"
#include "logger.hpp"
#include <filesystem>
#include "setup_ctx.hpp"
#include "stark_verify.hpp"
#include "exec_file.hpp"
#include "fixed_cols.hpp"
#include "final_snark_proof.hpp"
#include "starks_api_internal.hpp"
#ifdef __USE_MPI_RMA__
#include "mpi.h"
#endif


#ifndef __USE_CUDA__
#include "verify_constraints.hpp"
#endif

#include <nlohmann/json.hpp>
using json = nlohmann::json;

using namespace CPlusPlusLogging;

ProofDoneCallback proof_done_callback = nullptr;
#ifdef __USE_MPI_RMA__
MPI_Win win;
int win_buff = -1;
#endif

void initialize_agg_readiness_tracker() {
#ifdef __USE_MPI_RMA__    
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        printf("Error: MPI not initialized when initialize_agg_readiness_tracker was called\n");
        return;
    }
    
    int rank = 0;
    int size = 1;
    
    // Note! we use MPI_COMM_WORLD directly
    int err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    err = MPI_Comm_size(MPI_COMM_WORLD, &size);
    err = MPI_Barrier(MPI_COMM_WORLD);

    if(size == 1) return;

    // Create MPI window
    // For rank 0: Create a window exposing win_buff (initialized to -1)
    // For other ranks: Create a window without exposing memory (NULL base and 0 size)
    if(rank == 0) {
        // Make sure win_buff is initialized to -1 before creating the window
        win_buff = -1;
        err = MPI_Win_create(&win_buff, sizeof(int), sizeof(int),
                        MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    } else {
        err = MPI_Win_create(NULL, 0, sizeof(int),
                            MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }
    
    if (err != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int error_string_length;
        MPI_Error_string(err, error_string, &error_string_length);
        printf("Rank %d: MPI_Win_create failed: %s\n", rank, error_string);
        return;
    }
#endif
}


void free_agg_readiness_tracker(){
#ifdef __USE_MPI_RMA__
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        printf("Error: MPI not initialized when free_agg_readiness_tracker was called\n");
        return;
    }

    int rank = 0;
    int size = 1;
    
    // Note! we use MPI_COMM_WORLD directly
    int err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    err = MPI_Comm_size(MPI_COMM_WORLD, &size);
    err = MPI_Barrier(MPI_COMM_WORLD);
    if(size == 1) return;

    // Free the window
    err = MPI_Win_free(&win);
    if (err != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int error_string_length;
        MPI_Error_string(err, error_string, &error_string_length);
        printf("Rank %d: MPI_Win_free failed: %s\n", rank, error_string);
        return;
    }
#endif
}


int agg_is_ready() {
#ifdef __USE_MPI_RMA__
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        printf("Error: MPI not initialized when agg_is_ready was called\n");
        return -1;
    }

    int rank = 0;
    int size = 1;
    
    // Note! we use MPI_COMM_WORLD directly
    int err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    err = MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(size == 1) return 0;

    // lock window on rank 0 (which contains the data we want to access)
    err = MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
    if (err != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int error_string_length;
        MPI_Error_string(err, error_string, &error_string_length);
        printf("Rank %d: MPI_Win_lock failed: %s\n", rank, error_string);
        return -1;
    }
    // get the value in the window
    int value = -1;
    err = MPI_Get(&value, 1, MPI_INT, 0, 0, 1, MPI_INT, win);
    if (err != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int error_string_length;
        MPI_Error_string(err, error_string, &error_string_length);
        printf("Rank %d: MPI_Get failed: %s\n", rank, error_string);
        MPI_Win_unlock(0, win);
        return -1;
    }
    //flush
    err = MPI_Win_flush(0, win);
    if (err != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int error_string_length;
        MPI_Error_string(err, error_string, &error_string_length);
        printf("Rank %d: MPI_Win_flush failed: %s\n", rank, error_string);
        MPI_Win_unlock(0, win);
        return -1;
    }
    // if value is -1 set my rank in the window
    if(value == -1) {
        value = rank;
        err = MPI_Put(&value, 1, MPI_INT, 0, 0, 1, MPI_INT, win);
        if (err != MPI_SUCCESS) {
            char error_string[MPI_MAX_ERROR_STRING];
            int error_string_length;
            MPI_Error_string(err, error_string, &error_string_length);
            printf("Rank %d: MPI_Put failed: %s\n", rank, error_string);
            MPI_Win_unlock(0, win);
            return -1;
        }
        //flush
        err = MPI_Win_flush(0, win);
        if (err != MPI_SUCCESS) {
            char error_string[MPI_MAX_ERROR_STRING];
            int error_string_length;
            MPI_Error_string(err, error_string, &error_string_length);
            printf("Rank %d: MPI_Win_flush failed: %s\n", rank, error_string);
            MPI_Win_unlock(0, win);
            return -1;
        }
    }
    // unlock window
    err = MPI_Win_unlock(0, win);
    if (err != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int error_string_length;
        MPI_Error_string(err, error_string, &error_string_length);
        printf("Rank %d: MPI_Win_unlock failed: %s\n", rank, error_string);
        return -1;
    }
    return value;
#endif
    return 0;
}


void reset_agg_readiness_tracker(){
#ifdef __USE_MPI_RMA__
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        printf("Error: MPI not initialized when reset_agg_readiness_tracker was called\n");
        return;
    }

    int rank = 0;
    int size = 1;
    
    // Note! we use MPI_COMM_WORLD directly
    int err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    err = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size == 1) return;

    // lock window on rank 0 (which contains the data we want to access)
    err = MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
    if (err != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int error_string_length;
        MPI_Error_string(err, error_string, &error_string_length);
        printf("Rank %d: MPI_Win_lock failed: %s\n", rank, error_string);
        return;
    }
    // set the value in the window to -1
    int value = -1;
    err = MPI_Put(&value, 1, MPI_INT, 0, 0, 1, MPI_INT, win);
    if (err != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int error_string_length;
        MPI_Error_string(err, error_string, &error_string_length);
        printf("Rank %d: MPI_Put failed: %s\n", rank, error_string);
        MPI_Win_unlock(0, win);
        return;
    }
    //flush
    err = MPI_Win_flush(0, win);
    if (err != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int error_string_length;
        MPI_Error_string(err, error_string, &error_string_length);
        printf("Rank %d: MPI_Win_flush failed: %s\n", rank, error_string);
        MPI_Win_unlock(0, win);
        return;
    }
    // unlock window
    err = MPI_Win_unlock(0, win);
    if (err != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int error_string_length;
        MPI_Error_string(err, error_string, &error_string_length);
        printf("Rank %d: MPI_Win_unlock failed: %s\n", rank, error_string);
    }
#endif
}

void save_challenges(void *pGlobalChallenge, char* globalInfoFile, char *fileDir) {

    json globalInfo;
    file2json(globalInfoFile, globalInfo);

    Goldilocks::Element *globalChallenge = (Goldilocks::Element *)pGlobalChallenge;
    
    json challengesJson = json::array();
    for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
        challengesJson[k] = Goldilocks::toString(globalChallenge[k]);
    }

    json2file(challengesJson, string(fileDir) + "/global_challenges.json");
}


void save_publics(uint64_t numPublicInputs, void *pPublicInputs, char *fileDir) {

    Goldilocks::Element* publicInputs = (Goldilocks::Element *)pPublicInputs;

    // Generate publics
    json publicStarkJson;
    for (uint64_t i = 0; i < numPublicInputs; i++)
    {
        publicStarkJson[i] = Goldilocks::toString(publicInputs[i]);
    }

    // save publics to filestarks
    json2file(publicStarkJson, string(fileDir) + "/publics.json");
}

void save_proof_values(void *pProofValues, char* globalInfoFile, char *fileDir) {
    Goldilocks::Element* proofValues = (Goldilocks::Element *)pProofValues;

    json globalInfo;
    file2json(globalInfoFile, globalInfo);

    json proofValuesJson;
    uint64_t p = 0;
    for(uint64_t i = 0; i < globalInfo["proofValuesMap"].size(); i++) {
        proofValuesJson[i] = json::array();
        if(globalInfo["proofValuesMap"][i]["stage"] == 1) {
            proofValuesJson[i][0] = Goldilocks::toString(proofValues[p++]);
            proofValuesJson[i][1] = "0";
            proofValuesJson[i][2] = "0";
        } else {
            proofValuesJson[i][0] = Goldilocks::toString(proofValues[p++]);
            proofValuesJson[i][1] = Goldilocks::toString(proofValues[p++]);
            proofValuesJson[i][2] = Goldilocks::toString(proofValues[p++]);
        }
        
    }

    json2file(proofValuesJson, string(fileDir) + "/proof_values.json");
}



// SetupCtx
// ========================================================================================

uint64_t n_hints_by_name(void *p_expression_bin, char* hintName) {
    ExpressionsBin *expressionsBin = (ExpressionsBin*)p_expression_bin;
    return expressionsBin->getNumberHintIdsByName(string(hintName));
}

void get_hint_ids_by_name(void *p_expression_bin, uint64_t* hintIds, char* hintName)
{
    ExpressionsBin *expressionsBin = (ExpressionsBin*)p_expression_bin;
    expressionsBin->getHintIdsByName(hintIds, string(hintName));
}

// StarkInfo
// ========================================================================================
void *stark_info_new(char *filename, bool recursive, bool verify_constraints, bool verify, bool gpu, bool preallocate)
{
    auto starkInfo = new StarkInfo(filename, recursive, verify_constraints, verify, gpu, preallocate);

    return starkInfo;
}

uint64_t get_proof_size(void *pStarkInfo) {
    return ((StarkInfo *)pStarkInfo)->proofSize;
}

uint64_t get_proof_pinned_size(void *pStarkInfo) {
    return ((StarkInfo *)pStarkInfo)->getPinnedProofSize();
}

void set_memory_expressions(void *pStarkInfo, uint64_t nTmp1, uint64_t nTmp3) {
    ((StarkInfo *)pStarkInfo)->setMemoryExpressions(nTmp1, nTmp3);
}

uint64_t get_const_pols_offset(void *pStarkInfo) {
    return ((StarkInfo *)pStarkInfo)->mapOffsets[std::make_pair("const", false)];
}

uint64_t get_map_total_n(void *pStarkInfo)
{
    return ((StarkInfo *)pStarkInfo)->mapTotalN;
}

uint64_t get_tree_size(void *pStarkInfo)
{
    uint64_t tree_size = MerklehashGoldilocks::getTreeNumElements((1 << ((StarkInfo *)pStarkInfo)->starkStruct.nBitsExt), 3);
    return tree_size;

}

uint64_t get_map_total_n_custom_commits_fixed(void *pStarkInfo)
{
    return ((StarkInfo *)pStarkInfo)->mapTotalNCustomCommitsFixed;
}

void stark_info_free(void *pStarkInfo)
{
    delete ((StarkInfo *)pStarkInfo);
}

// Const Pols
// ========================================================================================
bool load_const_tree(void *pStarkInfo, void *pConstTree, char *treeFilename, uint64_t constTreeSize, char* verkeyFilename) {
    ConstTree constTree;
    return constTree.loadConstTree((*(StarkInfo *)pStarkInfo), pConstTree, treeFilename, constTreeSize, verkeyFilename);
};

void load_const_pols(void *pConstPols, char *constFilename, uint64_t constSize) {
    ConstTree constTree;
    constTree.loadConstPols(pConstPols, constFilename, constSize);
};

uint64_t get_const_tree_size(void *pStarkInfo) {
    ConstTree constTree;
    if(((StarkInfo *)pStarkInfo)->starkStruct.verificationHashType == "GL") {
        return constTree.getConstTreeSizeGL(*(StarkInfo *)pStarkInfo);
    } else {
        return constTree.getConstTreeSizeBN128(*(StarkInfo *)pStarkInfo);
    }
    
};

uint64_t get_const_size(void *pStarkInfo) {
    uint64_t N = 1 << (*(StarkInfo *)pStarkInfo).starkStruct.nBits;
    return N * (*(StarkInfo *)pStarkInfo).nConstants;
}


void calculate_const_tree(void *pStarkInfo, void *pConstPolsAddress, void *pConstTreeAddress) {
    ConstTree constTree;
    if(((StarkInfo *)pStarkInfo)->starkStruct.verificationHashType == "GL") {
        constTree.calculateConstTreeGL(*(StarkInfo *)pStarkInfo, (Goldilocks::Element *)pConstPolsAddress, pConstTreeAddress);
    } else {
        constTree.calculateConstTreeBN128(*(StarkInfo *)pStarkInfo, (Goldilocks::Element *)pConstPolsAddress, pConstTreeAddress);
    }
};

void write_const_tree(void *pStarkInfo, void *pConstTreeAddress, char *treeFilename) {
    ConstTree constTree;
    if(((StarkInfo *)pStarkInfo)->starkStruct.verificationHashType == "GL") {
        constTree.writeConstTreeFileGL(*(StarkInfo *)pStarkInfo, pConstTreeAddress, treeFilename);
    } else {
        constTree.writeConstTreeFileBN128(*(StarkInfo *)pStarkInfo, pConstTreeAddress, treeFilename);
    }
};

// Expressions Bin
// ========================================================================================
void *expressions_bin_new(char* filename, bool global, bool verifier)
{
    auto expressionsBin = new ExpressionsBin(filename, global, verifier);

    return expressionsBin;
};

uint64_t get_max_n_tmp1(void *pExpressionsBin) {
    auto expressionsBin = (ExpressionsBin *)pExpressionsBin;
    return expressionsBin->maxTmp1;
};

uint64_t get_max_n_tmp3(void *pExpressionsBin){
    auto expressionsBin = (ExpressionsBin *)pExpressionsBin;
    return expressionsBin->maxTmp3;
};

uint64_t get_max_args(void *pExpressionsBin){
    auto expressionsBin = (ExpressionsBin *)pExpressionsBin;
    return expressionsBin->maxArgs;
};

uint64_t get_max_ops(void *pExpressionsBin){
    auto expressionsBin = (ExpressionsBin *)pExpressionsBin;
    return expressionsBin->maxOps;
};

void expressions_bin_free(void *pExpressionsBin)
{
    auto expressionsBin = (ExpressionsBin *)pExpressionsBin;
    delete expressionsBin;
};

// Hints
// ========================================================================================
void get_hint_field(void *pSetupCtx, void* stepsParams, void* hintFieldValues, uint64_t hintId, char* hintFieldName, void* hintOptions) 
{
    SetupCtx &setupCtx = *(SetupCtx *)pSetupCtx;
    ProverHelpers proverHelpers;

    ExpressionsPack expressionsCtx(setupCtx, &proverHelpers);

    getHintField(*(SetupCtx *)pSetupCtx, *(StepsParams *)stepsParams, expressionsCtx, (HintFieldInfo *) hintFieldValues, hintId, string(hintFieldName), *(HintFieldOptions *) hintOptions);
}

uint64_t get_hint_field_values(void *pSetupCtx, uint64_t hintId, char* hintFieldName) {
    return getHintFieldValues(*(SetupCtx *)pSetupCtx, hintId, string(hintFieldName));
}

void get_hint_field_sizes(void *pSetupCtx, void* hintFieldValues, uint64_t hintId, char* hintFieldName, void* hintOptions)
{
    getHintFieldSizes(*(SetupCtx *)pSetupCtx, (HintFieldInfo *) hintFieldValues, hintId, string(hintFieldName), *(HintFieldOptions *) hintOptions);
}

void mul_hint_fields(void *pSetupCtx, void* stepsParams, uint64_t nHints, uint64_t *hintId, char **hintFieldNameDest, char **hintFieldName1, char **hintFieldName2, void** hintOptions1, void **hintOptions2) 
{

    std::vector<std::string> hintFieldNameDests(nHints);
    std::vector<std::string> hintFieldNames1(nHints);
    std::vector<std::string> hintFieldNames2(nHints);
    std::vector<HintFieldOptions> hintOptions1Vec(nHints);
    std::vector<HintFieldOptions> hintOptions2Vec(nHints);

    for (uint64_t i = 0; i < nHints; ++i) {
        hintFieldNameDests[i] = hintFieldNameDest[i];
        hintFieldNames1[i] = hintFieldName1[i];
        hintFieldNames2[i] = hintFieldName2[i];
        hintOptions1Vec[i] = *(HintFieldOptions *)hintOptions1[i];
        hintOptions2Vec[i] = *(HintFieldOptions *)hintOptions2[i];
    }

    SetupCtx &setupCtx = *(SetupCtx *)pSetupCtx;
    ProverHelpers proverHelpers;

    ExpressionsPack expressionsCtx(setupCtx, &proverHelpers);

    return multiplyHintFields(setupCtx, *(StepsParams *)stepsParams, expressionsCtx, nHints, hintId, hintFieldNameDests.data(), hintFieldNames1.data(), hintFieldNames2.data(), hintOptions1Vec.data(), hintOptions2Vec.data());
}

void acc_hint_field(void *pSetupCtx, void* stepsParams, uint64_t hintId, char *hintFieldNameDest, char *hintFieldNameAirgroupVal, char *hintFieldName, bool add) {
    SetupCtx &setupCtx = *(SetupCtx *)pSetupCtx;
    ProverHelpers proverHelpers;

    ExpressionsPack expressionsCtx(setupCtx, &proverHelpers);
    accHintField(*(SetupCtx *)pSetupCtx, *(StepsParams *)stepsParams, expressionsCtx, hintId, string(hintFieldNameDest), string(hintFieldNameAirgroupVal), string(hintFieldName), add);
}

void acc_mul_hint_fields(void *pSetupCtx, void* stepsParams, uint64_t hintId, char *hintFieldNameDest, char *hintFieldNameAirgroupVal, char *hintFieldName1, char *hintFieldName2, void* hintOptions1, void *hintOptions2, bool add) {
    SetupCtx &setupCtx = *(SetupCtx *)pSetupCtx;
    ProverHelpers proverHelpers;

    ExpressionsPack expressionsCtx(setupCtx, &proverHelpers);
    accMulHintFields(*(SetupCtx *)pSetupCtx, *(StepsParams *)stepsParams, expressionsCtx, hintId, string(hintFieldNameDest), string(hintFieldNameAirgroupVal), string(hintFieldName1), string(hintFieldName2),*(HintFieldOptions *)hintOptions1,  *(HintFieldOptions *)hintOptions2, add);
}

uint64_t update_airgroupvalue(void *pSetupCtx, void* stepsParams, uint64_t hintId, char *hintFieldNameAirgroupVal, char *hintFieldName1, char *hintFieldName2, void* hintOptions1, void *hintOptions2, bool add) {
    return updateAirgroupValue(*(SetupCtx *)pSetupCtx, *(StepsParams *)stepsParams, hintId, string(hintFieldNameAirgroupVal), string(hintFieldName1), string(hintFieldName2),*(HintFieldOptions *)hintOptions1,  *(HintFieldOptions *)hintOptions2, add);
}

uint64_t get_hint_id(void *pSetupCtx, uint64_t hintId, char * hintFieldName) {
    return getHintId(*(SetupCtx *)pSetupCtx, hintId, string(hintFieldName));
}

uint64_t set_hint_field(void *pSetupCtx, void* params, void *values, uint64_t hintId, char * hintFieldName) 
{
    return setHintField(*(SetupCtx *)pSetupCtx,  *(StepsParams *)params, (Goldilocks::Element *)values, hintId, string(hintFieldName));
}

// Starks
// ========================================================================================

void calculate_impols_expressions(void *pSetupCtx, uint64_t step, void* stepsParams)
{
    SetupCtx &setupCtx = *(SetupCtx *)pSetupCtx;
    StepsParams &params = *(StepsParams *)stepsParams;

    ProverHelpers proverHelpers;

    ExpressionsPack expressionsCtx(setupCtx, &proverHelpers);

    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
        if(setupCtx.starkInfo.cmPolsMap[i].imPol && setupCtx.starkInfo.cmPolsMap[i].stage == step) {
            Goldilocks::Element* pAddress = setupCtx.starkInfo.cmPolsMap[i].stage == 1 ? params.trace : params.aux_trace;
            Dest destStruct(&pAddress[setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + to_string(step), false)] + setupCtx.starkInfo.cmPolsMap[i].stagePos], (1<< setupCtx.starkInfo.starkStruct.nBits), setupCtx.starkInfo.mapSectionsN["cm" + to_string(step)]);
            destStruct.addParams(setupCtx.starkInfo.cmPolsMap[i].expId, setupCtx.starkInfo.cmPolsMap[i].dim, false);
            expressionsCtx.calculateExpressions(params, destStruct, uint64_t(1 << setupCtx.starkInfo.starkStruct.nBits), false, false);
        }
    }
}

uint64_t custom_commit_size(void *pSetup, uint64_t commitId) {
    auto setupCtx = *(SetupCtx *)pSetup;

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    std::string section = setupCtx.starkInfo.customCommits[commitId].name + "0";
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];

    return (N + NExtended) * nCols + setupCtx.starkInfo.getNumNodesMT(NExtended);
}

void load_custom_commit(void *pSetup, uint64_t commitId, void *buffer, char *bufferFile)
{
    auto setupCtx = *(SetupCtx *)pSetup;

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    std::string section = setupCtx.starkInfo.customCommits[commitId].name + "0";
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[section];
    
    Goldilocks::Element *bufferGL = (Goldilocks::Element *)buffer;
    loadFileParallel(&bufferGL[setupCtx.starkInfo.mapOffsets[std::make_pair(section, false)]], bufferFile, ((N + NExtended) * nCols + setupCtx.starkInfo.getNumNodesMT(NExtended)) * sizeof(Goldilocks::Element), true, 32);
}

void write_custom_commit(void* root, uint64_t N, uint64_t NExtended, uint64_t nCols, void *buffer, char *bufferFile, bool check)
{   
    MerkleTreeGL mt(3, true, NExtended, nCols, true, true);

    NTT_Goldilocks ntt(N);
    ntt.extendPol(mt.source, (Goldilocks::Element *)buffer, NExtended, N, nCols);
    
    mt.merkelize();
    
    Goldilocks::Element *rootGL = (Goldilocks::Element *)root;
    mt.getRoot(&rootGL[0]);

    if(!check) {
        std::string buffFile = string(bufferFile);
        ofstream fw(buffFile.c_str(), std::fstream::out | std::fstream::binary);
        writeFileParallel(buffFile, root, 32, 0);
        writeFileParallel(buffFile, buffer, N * nCols * sizeof(Goldilocks::Element), 32);
        writeFileParallel(buffFile, mt.source, NExtended * nCols * sizeof(Goldilocks::Element), 32 + N * nCols * sizeof(Goldilocks::Element));
        writeFileParallel(buffFile, mt.nodes, mt.numNodes * sizeof(Goldilocks::Element), 32 + (NExtended + N) * nCols * sizeof(Goldilocks::Element));
        fw.close();
    }
}

#ifndef __USE_CUDA__
uint64_t commit_witness(uint64_t arity, uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, uint64_t instanceId, uint64_t airgroupId, uint64_t airId, void *root, void *trace, void *auxTrace, void *d_buffers, void *pSetupCtx_) {
    Goldilocks::Element *rootGL = (Goldilocks::Element *)root;
    Goldilocks::Element *traceGL = (Goldilocks::Element *)trace;
    Goldilocks::Element *auxTraceGL = (Goldilocks::Element *)auxTrace;
    uint64_t N = 1 << nBits;
    uint64_t NExtended = 1 << nBitsExt;

    MerkleTreeGL mt(arity, true, NExtended, nCols);

    NTT_Goldilocks ntt(N);
    ntt.extendPol(auxTraceGL, traceGL, NExtended, N, nCols, &auxTraceGL[NExtended * nCols + mt.numNodes]);

    mt.setSource(auxTraceGL);
    mt.setNodes(&auxTraceGL[NExtended * nCols]);
    mt.merkelize();
    mt.getRoot(rootGL);

    if (proof_done_callback != nullptr) {
        proof_done_callback(instanceId, "basic");
    }

    return 0;
}
#endif


void calculate_hash(void *pValue, void *pBuffer, uint64_t nElements, uint64_t nOutputs)
{
    TranscriptGL transcriptHash(2, true);
    transcriptHash.put((Goldilocks::Element *)pBuffer, nElements);
    transcriptHash.getState((Goldilocks::Element *)pValue, nOutputs);
}

// Transcript
// =================================================================================
void *transcript_new(uint64_t arity, bool custom)
{
    return new TranscriptGL(arity, custom);
}

void transcript_add(void *pTranscript, void *pInput, uint64_t size)
{
    auto transcript = (TranscriptGL *)pTranscript;
    auto input = (Goldilocks::Element *)pInput;

    transcript->put(input, size);
}

void transcript_free(void *pTranscript)
{
    delete (TranscriptGL *)pTranscript;
}

void get_challenge(void *pTranscript, void *pElement)
{
    TranscriptGL *transcript = (TranscriptGL *)pTranscript;
    Goldilocks::Element &challenge = *(Goldilocks::Element *)pElement;
    transcript->getField((uint64_t *)&challenge);
}


// Constraints
// =================================================================================
uint64_t get_n_constraints(void *pSetupCtx)
{
    auto setupCtx = *(SetupCtx *)pSetupCtx;
    return setupCtx.expressionsBin.constraintsInfoDebug.size();
}

void get_constraints_lines_sizes(void* pSetupCtx, uint64_t *constraintsLinesSizes)
{
    auto setupCtx = *(SetupCtx *)pSetupCtx;
    for(uint64_t i = 0; i < setupCtx.expressionsBin.constraintsInfoDebug.size(); ++i) {
        constraintsLinesSizes[i] = setupCtx.expressionsBin.constraintsInfoDebug[i].line.size();
    }
}

void get_constraints_lines(void* pSetupCtx, uint8_t **constraintsLines)
{
    auto setupCtx = *(SetupCtx *)pSetupCtx;
    for(uint64_t i = 0; i < setupCtx.expressionsBin.constraintsInfoDebug.size(); ++i) {
        std::memcpy(constraintsLines[i], setupCtx.expressionsBin.constraintsInfoDebug[i].line.data(), setupCtx.expressionsBin.constraintsInfoDebug[i].line.size());
    }
}

#ifndef __USE_CUDA__
void verify_constraints(void *pSetupCtx, void* stepsParams, void* constraintsInfo, void *d_buffers)
{
    verifyConstraints(*(SetupCtx *)pSetupCtx, *(StepsParams *)stepsParams, (ConstraintInfo *)constraintsInfo);
}
#endif

// Global Constraints
// =================================================================================
uint64_t get_n_global_constraints(void* p_globalinfo_bin)
{
    return getNumberGlobalConstraints(*(ExpressionsBin*)p_globalinfo_bin);
}

void get_global_constraints_lines_sizes(void* p_globalinfo_bin, uint64_t *constraintsLinesSizes)
{
    return getGlobalConstraintsLinesSizes(*(ExpressionsBin*)p_globalinfo_bin, constraintsLinesSizes);
}

void get_global_constraints_lines(void* p_globalinfo_bin, uint8_t **constraintsLines)
{
    return getGlobalConstraintsLines(*(ExpressionsBin*)p_globalinfo_bin, constraintsLines);
}

void verify_global_constraints(char* globalInfoFile, void* p_globalinfo_bin, void *publics, void *challenges, void *proofValues, void **airgroupValues, void *globalConstraintsInfo) {
    json globalInfo;
    file2json(globalInfoFile, globalInfo);

    verifyGlobalConstraints(globalInfo, *(ExpressionsBin*)p_globalinfo_bin, (Goldilocks::Element *)publics, (Goldilocks::Element *)challenges, (Goldilocks::Element *)proofValues, (Goldilocks::Element **)airgroupValues, (GlobalConstraintInfo *)globalConstraintsInfo);
}
 
uint64_t get_hint_field_global_constraints_values(void* p_globalinfo_bin, uint64_t hintId, char* hintFieldName) {
    return getHintFieldGlobalConstraintValues(*(ExpressionsBin*)p_globalinfo_bin, hintId, string(hintFieldName));
}

void get_hint_field_global_constraints_sizes(char* globalInfoFile, void* p_globalinfo_bin, void* hintFieldValues, uint64_t hintId, char *hintFieldName, bool print_expression)
{
    json globalInfo;
    file2json(globalInfoFile, globalInfo);

    getHintFieldGlobalConstraintSizes(globalInfo, *(ExpressionsBin*)p_globalinfo_bin, (HintFieldInfo *)hintFieldValues, hintId, string(hintFieldName), print_expression);
}


void get_hint_field_global_constraints(char* globalInfoFile, void* p_globalinfo_bin, void* hintFieldValues, void *publics, void *challenges, void *proofValues, void **airgroupValues, uint64_t hintId, char *hintFieldName, bool print_expression) 
{
    json globalInfo;
    file2json(globalInfoFile, globalInfo);

    getHintFieldGlobalConstraint(globalInfo, *(ExpressionsBin*)p_globalinfo_bin, (HintFieldInfo *)hintFieldValues, (Goldilocks::Element *)publics, (Goldilocks::Element *)challenges, (Goldilocks::Element *)proofValues, (Goldilocks::Element **)airgroupValues, hintId, string(hintFieldName), print_expression);
}

uint64_t set_hint_field_global_constraints(char* globalInfoFile, void* p_globalinfo_bin, void *proofValues, void *values, uint64_t hintId, char *hintFieldName) 
{
    json globalInfo;
    file2json(globalInfoFile, globalInfo);

    return setHintFieldGlobalConstraint(globalInfo, *(ExpressionsBin*)p_globalinfo_bin, (Goldilocks::Element *)proofValues, (Goldilocks::Element *)values, hintId, string(hintFieldName));
}

#ifndef __USE_CUDA__
// Gen proof
// =================================================================================
uint64_t gen_proof(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params_, void *globalChallenge, uint64_t* proofBuffer, char *proofFile, void *d_buffers_, bool skipRecalculation, uint64_t streamId, char *constPolsPath,  char *constTreePath)  {
    DeviceCommitBuffersCPU *d_buffers = (DeviceCommitBuffersCPU *)d_buffers_;
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx;
    StepsParams *params = (StepsParams *)params_;
    
    if (d_buffers->airgroupId != airgroupId || d_buffers->airId != airId || d_buffers->proofType != "basic") {
        uint64_t N = (1 << setupCtx->starkInfo.starkStruct.nBits);
        uint64_t sizeConstPols = N * (setupCtx->starkInfo.nConstants) * sizeof(Goldilocks::Element);
        uint64_t sizeConstTree = get_const_tree_size((void *)&setupCtx->starkInfo) * sizeof(Goldilocks::Element);
        loadFileParallel(params->pConstPolsAddress, constPolsPath, sizeConstPols);
        loadFileParallel(params->pConstPolsExtendedTreeAddress, constTreePath, sizeConstTree);
    }

    d_buffers->airgroupId = airgroupId;
    d_buffers->airId = airId;
    d_buffers->proofType = "basic";

    genProof(*(SetupCtx *)pSetupCtx, airgroupId, airId, instanceId, *(StepsParams *)params, (Goldilocks::Element *)globalChallenge, proofBuffer, string(proofFile));
    
    return 0;
}
void get_stream_proofs(void *d_buffers_){}
void get_stream_proofs_non_blocking(void *d_buffers_){}
void get_stream_id_proof(void *d_buffers_, uint64_t streamId) {}

// Recursive proof
// ================================================================================= 
void *gen_device_buffers(void *maxSizes_, uint32_t node_rank, uint32_t node_size)
{
    DeviceCommitBuffersCPU *d_buffers = new DeviceCommitBuffersCPU();
    return (void *)d_buffers;
};

uint64_t gen_device_streams(void *d_buffers_, uint64_t maxSizeProverBuffer, uint64_t maxSizeProverBufferAggregation, uint64_t maxProofSize, uint64_t maxProofsPerGPU, uint64_t maxRecursiveProofsPerGPU, uint64_t max_n_bits_ext, uint64_t maxNConstraints) { return 1; }

void reset_device_streams(void *d_buffers_) {}

uint64_t check_device_memory(uint32_t node_rank, uint32_t node_size) { return 0; }

uint64_t get_num_gpus(){ return 0;}

void free_device_buffers(void *d_buffers_) {
    DeviceCommitBuffersCPU *d_buffers = (DeviceCommitBuffersCPU *)d_buffers_;
    delete d_buffers;
}

void load_device_setup(uint64_t airgroupId, uint64_t airId, char *proofType, void *pSetupCtx_, void *d_buffers_, void *verkeyRoot_, uint64_t nStreams) {}

void load_device_const_pols(uint64_t airgroupId, uint64_t airId, uint64_t initial_offset, void *d_buffers, char *constFilename, uint64_t constSize, char *constTreeFilename, uint64_t constTreeSize, char *proofType) {}

uint64_t gen_recursive_proof(void *pSetupCtx, char* globalInfoFile, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, uint64_t* proofBuffer, char* proof_file, bool vadcop, void *d_buffers_, char *constPolsPath, char *constTreePath, char *proofType) {
    json globalInfo;
    file2json(globalInfoFile, globalInfo);

    DeviceCommitBuffersCPU *d_buffers = (DeviceCommitBuffersCPU *)d_buffers_;
    SetupCtx *setupCtx = (SetupCtx *)pSetupCtx;

    if (d_buffers->airgroupId != airgroupId || d_buffers->airId != airId || d_buffers->proofType != string(proofType)) {
        uint64_t N = (1 << setupCtx->starkInfo.starkStruct.nBits);
        uint64_t sizeConstPols = N * (setupCtx->starkInfo.nConstants) * sizeof(Goldilocks::Element);
        uint64_t sizeConstTree = get_const_tree_size((void *)&setupCtx->starkInfo) * sizeof(Goldilocks::Element);
        loadFileParallel(pConstPols, constPolsPath, sizeConstPols);
        loadFileParallel(pConstTree, constTreePath, sizeConstTree);
    }

    d_buffers->airgroupId = airgroupId;
    d_buffers->airId = airId;
    d_buffers->proofType = string(proofType);


    Goldilocks::Element evals[setupCtx->starkInfo.evMap.size() * FIELD_EXTENSION];
    Goldilocks::Element challenges[setupCtx->starkInfo.challengesMap.size() * FIELD_EXTENSION];
    Goldilocks::Element airgroupValues[FIELD_EXTENSION];

    StepsParams params = {
        .trace = (Goldilocks::Element *)witness,
        .aux_trace = (Goldilocks::Element *)aux_trace,
        .publicInputs = (Goldilocks::Element *)pPublicInputs,
        .proofValues = nullptr,
        .challenges = challenges,
        .airgroupValues = airgroupValues,
        .evals = evals,
        .xDivXSub = nullptr,
        .pConstPolsAddress = (Goldilocks::Element *)pConstPols,
        .pConstPolsExtendedTreeAddress = (Goldilocks::Element *)pConstTree,
        .pCustomCommitsFixed = nullptr,
    };

    genProof(*setupCtx, airgroupId, airId, instanceId, params, nullptr, proofBuffer, string(proof_file), true);
    
    return 0;
}

#endif

void launch_callback(uint64_t instanceId, char *proofType) {
    if (proof_done_callback != nullptr) {
        proof_done_callback(instanceId, proofType);
    }
}

void add_publics_aggregation(void *pProof, uint64_t offset, void *pPublics, uint64_t nPublicsAggregation) {
    uint64_t *proof = (uint64_t *)pProof;
    Goldilocks::Element *publics = (Goldilocks::Element *)pPublics;

    for(uint64_t i = 0; i < nPublicsAggregation; i++) {
        proof[offset + i] = Goldilocks::toU64(publics[i]);
    }
}


void *gen_recursive_proof_final(void *pSetupCtx, char* globalInfoFile, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, char* proof_file) {
    json globalInfo;
    file2json(globalInfoFile, globalInfo);

    return genRecursiveProofBN128(*(SetupCtx *)pSetupCtx, globalInfo, airgroupId, airId, instanceId, (Goldilocks::Element *)witness, (Goldilocks::Element *)aux_trace, (Goldilocks::Element *)pConstPols, (Goldilocks::Element *)pConstTree, (Goldilocks::Element *)pPublicInputs, nullptr, string(proof_file));
}

void read_exec_file(uint64_t *exec_data, char *exec_file, uint64_t nCommitedPols) {
    readExecFile(exec_data, string(exec_file), nCommitedPols);
}

void get_committed_pols(void *circomWitness, uint64_t* execData, void *witness, void* pPublics, uint64_t sizeWitness, uint64_t N, uint64_t nPublics, uint64_t nCommitedPols) {
    getCommitedPols((Goldilocks::Element *)circomWitness, execData, (Goldilocks::Element *)witness, (Goldilocks::Element *)pPublics, sizeWitness, N, nPublics, nCommitedPols);
}

void gen_final_snark_proof(void *circomWitnessFinal, char* zkeyFile, char* outputDir) {
    genFinalSnarkProof(circomWitnessFinal, string(zkeyFile), string(outputDir));
}

void setLogLevel(uint64_t level) {
    LogLevel new_level;
    switch(level) {
        case 0:
            new_level = DISABLE_LOG;
            break;
        case 1:
        case 2:
        case 3:
            new_level = LOG_LEVEL_INFO;
            break;
        case 4:
            new_level = LOG_LEVEL_DEBUG;
            break;
        case 5:
            new_level = LOG_LEVEL_TRACE;
            break;
        default:
            cerr << "Invalid log level: " << level << endl;
            return;
    }

    Logger::getInstance(LOG_TYPE::CONSOLE)->updateLogLevel((LOG_LEVEL)new_level);
}


// Stark Verify
// =================================================================================
bool stark_verify(uint64_t* proof, void *pStarkInfo, void *pExpressionsBin, char *verkeyFile, void *pPublics, void *pProofValues, void *pChallenges) {
    Goldilocks::Element *challenges = (Goldilocks::Element *)pChallenges;
    bool vadcop = challenges == nullptr ? false : true;
    StarkInfo starkInfo = *(StarkInfo *)pStarkInfo;
    json jProof = pointer2json(proof, starkInfo);
    return starkVerify<Goldilocks::Element>(jProof, *(StarkInfo *)pStarkInfo, *(ExpressionsBin *)pExpressionsBin, string(verkeyFile), (Goldilocks::Element *)pPublics, (Goldilocks::Element *)pProofValues, vadcop, (Goldilocks::Element *)pChallenges);
}

bool stark_verify_bn128(void* jProof, void *pStarkInfo, void *pExpressionsBin, char *verkeyFile, void *pPublics) {
    return starkVerify<RawFr::Element>(*(nlohmann::json*) jProof, *(StarkInfo *)pStarkInfo, *(ExpressionsBin *)pExpressionsBin, string(verkeyFile), (Goldilocks::Element *)pPublics, nullptr, false, nullptr);

}

bool stark_verify_from_file(char* proofFile, void *pStarkInfo, void *pExpressionsBin, char *verkeyFile, void *pPublics, void *pProofValues, void *pChallenges) {
    Goldilocks::Element *challenges = (Goldilocks::Element *)pChallenges;
    bool vadcop = challenges == nullptr ? false : true;
    StarkInfo starkInfo = *((StarkInfo *)pStarkInfo);
    json jProof;
    file2json(proofFile, jProof);
    if (starkInfo.starkStruct.verificationHashType == "GL") {
        return starkVerify<Goldilocks::Element>(jProof, *(StarkInfo *)pStarkInfo, *(ExpressionsBin *)pExpressionsBin, string(verkeyFile), (Goldilocks::Element *)pPublics, (Goldilocks::Element *)pProofValues, vadcop, (Goldilocks::Element *)pChallenges);
    } else {
        return starkVerify<RawFr::Element>(jProof, *(StarkInfo *)pStarkInfo, *(ExpressionsBin *)pExpressionsBin, string(verkeyFile), (Goldilocks::Element *)pPublics, (Goldilocks::Element *)pProofValues, vadcop, (Goldilocks::Element *)pChallenges);
    }
}


// Fixed cols
// =================================================================================
void write_fixed_cols_bin(char* binFile, char* airgroupName, char* airName, uint64_t N, uint64_t nFixedPols, void* fixedPolsInfo) {
    writeFixedColsBin(string(binFile), string(airgroupName), string(airName), N, nFixedPols, (FixedPolsInfo *)fixedPolsInfo);
}

uint64_t get_omp_max_threads(){
    return omp_get_max_threads();
}

void set_omp_num_threads(uint64_t num_threads){
    omp_set_num_threads(num_threads);
}

uint64_t goldilocks_add_ffi(const uint64_t *in1, const uint64_t *in2)
{
    const auto &i1 = *reinterpret_cast<const Goldilocks::Element *>(in1);
    const auto &i2 = *reinterpret_cast<const Goldilocks::Element *>(in2);

    return Goldilocks::add(i1, i2).fe;
}

void goldilocks_add_assign_ffi(uint64_t *result, const uint64_t *in1, const uint64_t *in2)
{
    auto &res = *reinterpret_cast<Goldilocks::Element *>(result);
    const auto &i1 = *reinterpret_cast<const Goldilocks::Element *>(in1);
    const auto &i2 = *reinterpret_cast<const Goldilocks::Element *>(in2);

    Goldilocks::add(res, i1, i2);
}

uint64_t goldilocks_sub_ffi(const uint64_t *in1, const uint64_t *in2)
{
    const auto &i1 = *reinterpret_cast<const Goldilocks::Element *>(in1);
    const auto &i2 = *reinterpret_cast<const Goldilocks::Element *>(in2);

    return Goldilocks::sub(i1, i2).fe;
}

void goldilocks_sub_assign_ffi(uint64_t *result, const uint64_t *in1, const uint64_t *in2)
{
    auto &res = *reinterpret_cast<Goldilocks::Element *>(result);
    const auto &i1 = *reinterpret_cast<const Goldilocks::Element *>(in1);
    const auto &i2 = *reinterpret_cast<const Goldilocks::Element *>(in2);

    Goldilocks::sub(res, i1, i2);
}

uint64_t goldilocks_mul_ffi(const uint64_t *in1, const uint64_t *in2)
{
    const auto &i1 = *reinterpret_cast<const Goldilocks::Element *>(in1);
    const auto &i2 = *reinterpret_cast<const Goldilocks::Element *>(in2);

    return Goldilocks::mul(i1, i2).fe;
}

void goldilocks_mul_assign_ffi(uint64_t *result, const uint64_t *in1, const uint64_t *in2)
{
    auto &res = *reinterpret_cast<Goldilocks::Element *>(result);
    const auto &i1 = *reinterpret_cast<const Goldilocks::Element *>(in1);
    const auto &i2 = *reinterpret_cast<const Goldilocks::Element *>(in2);

    Goldilocks::mul(res, i1, i2);
}

uint64_t goldilocks_div_ffi(const uint64_t *in1, const uint64_t *in2)
{
    const auto &i1 = *reinterpret_cast<const Goldilocks::Element *>(in1);
    const auto &i2 = *reinterpret_cast<const Goldilocks::Element *>(in2);

    return Goldilocks::div(i1, i2).fe;
}

void goldilocks_div_assign_ffi(uint64_t *result, const uint64_t *in1, const uint64_t *in2)
{
    auto &res = *reinterpret_cast<Goldilocks::Element *>(result);
    const auto &i1 = *reinterpret_cast<const Goldilocks::Element *>(in1);
    const auto &i2 = *reinterpret_cast<const Goldilocks::Element *>(in2);

    Goldilocks::div(res, i1, i2);
}

uint64_t goldilocks_neg_ffi(const uint64_t *in1) {
    const auto &i1 = *reinterpret_cast<const Goldilocks::Element *>(in1);

    return Goldilocks::neg(i1).fe;
}

uint64_t goldilocks_inv_ffi(const uint64_t *in1) {
    const auto &i1 = *reinterpret_cast<const Goldilocks::Element *>(in1);

    return Goldilocks::inv(i1).fe;
}

void register_proof_done_callback(ProofDoneCallback cb) {
    proof_done_callback = cb;
}

