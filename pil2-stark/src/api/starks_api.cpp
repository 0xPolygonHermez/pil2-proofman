#include "zkglobals.hpp"
#include "proof2zkinStark.hpp"
#include "starks.hpp"
#include "verify_constraints.hpp"
#include "hints.hpp"
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

#include <nlohmann/json.hpp>
using json = nlohmann::json;

using namespace CPlusPlusLogging;

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


void save_publics(unsigned long numPublicInputs, void *pPublicInputs, char *fileDir) {

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
void *stark_info_new(char *filename, bool verify)
{
    auto starkInfo = new StarkInfo(filename, verify);

    return starkInfo;
}

uint64_t get_proof_size(void *pStarkInfo) {
    StarkInfo *starkInfo = (StarkInfo *)pStarkInfo;
    return starkInfo->proofSize;
}

uint64_t get_map_total_n(void *pStarkInfo, bool recursive)
{
    StarkInfo *starkInfo = (StarkInfo *)pStarkInfo;
    if(recursive) {
        starkInfo->addMemoryRecursive();
    }
    return starkInfo->mapTotalN;
}

uint64_t get_map_total_n_custom_commits_fixed(void *pStarkInfo)
{
    StarkInfo *starkInfo = (StarkInfo *)pStarkInfo;
    return starkInfo->mapTotalNCustomCommitsFixed;
}

void stark_info_free(void *pStarkInfo)
{
    auto starkInfo = (StarkInfo *)pStarkInfo;
    delete starkInfo;
}

// Const Pols
// ========================================================================================
bool load_const_tree(void *pStarkInfo, void *pConstTree, char *treeFilename, uint64_t constTreeSize, char* verkeyFilename) {
    ConstTree constTree;
    auto starkInfo = *(StarkInfo *)pStarkInfo;
    return constTree.loadConstTree(starkInfo, pConstTree, treeFilename, constTreeSize, verkeyFilename);
};

void load_const_pols(void *pConstPols, char *constFilename, uint64_t constSize) {
    ConstTree constTree;
    constTree.loadConstPols(pConstPols, constFilename, constSize);
};

uint64_t get_const_tree_size(void *pStarkInfo) {
    ConstTree constTree;
    auto starkInfo = *(StarkInfo *)pStarkInfo;
    if(starkInfo.starkStruct.verificationHashType == "GL") {
        return constTree.getConstTreeSizeGL(starkInfo);
    } else {
        return constTree.getConstTreeSizeBN128(starkInfo);
    }
    
};

uint64_t get_const_size(void *pStarkInfo) {
    auto starkInfo = *(StarkInfo *)pStarkInfo;
    uint64_t N = 1 << starkInfo.starkStruct.nBits;
    return N * starkInfo.nConstants;
}


void calculate_const_tree(void *pStarkInfo, void *pConstPolsAddress, void *pConstTreeAddress) {
    ConstTree constTree;
    auto starkInfo = *(StarkInfo *)pStarkInfo;
    if(starkInfo.starkStruct.verificationHashType == "GL") {
        constTree.calculateConstTreeGL(*(StarkInfo *)pStarkInfo, (Goldilocks::Element *)pConstPolsAddress, pConstTreeAddress);
    } else {
        constTree.calculateConstTreeBN128(*(StarkInfo *)pStarkInfo, (Goldilocks::Element *)pConstPolsAddress, pConstTreeAddress);
    }
};

void write_const_tree(void *pStarkInfo, void *pConstTreeAddress, char *treeFilename) {
    ConstTree constTree;
    auto starkInfo = *(StarkInfo *)pStarkInfo;
    if(starkInfo.starkStruct.verificationHashType == "GL") {
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
void expressions_bin_free(void *pExpressionsBin)
{
    auto expressionsBin = (ExpressionsBin *)pExpressionsBin;
    delete expressionsBin;
};

// Hints
// ========================================================================================
void get_hint_field(void *pSetupCtx, void* stepsParams, void* hintFieldValues, uint64_t hintId, char* hintFieldName, void* hintOptions) 
{
    getHintField(*(SetupCtx *)pSetupCtx, *(StepsParams *)stepsParams, (HintFieldInfo *) hintFieldValues, hintId, string(hintFieldName), *(HintFieldOptions *) hintOptions);
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

    return multiplyHintFields(*(SetupCtx *)pSetupCtx, *(StepsParams *)stepsParams, nHints, hintId, hintFieldNameDests.data(), hintFieldNames1.data(), hintFieldNames2.data(), hintOptions1Vec.data(), hintOptions2Vec.data());
}

void acc_hint_field(void *pSetupCtx, void* stepsParams, void *pBuffHelper, uint64_t hintId, char *hintFieldNameDest, char *hintFieldNameAirgroupVal, char *hintFieldName, bool add) {
    accHintField(*(SetupCtx *)pSetupCtx, *(StepsParams *)stepsParams, (Goldilocks::Element *)pBuffHelper, hintId, string(hintFieldNameDest), string(hintFieldNameAirgroupVal), string(hintFieldName), add);
}

void acc_mul_hint_fields(void *pSetupCtx, void* stepsParams, void *pBuffHelper, uint64_t hintId, char *hintFieldNameDest, char *hintFieldNameAirgroupVal, char *hintFieldName1, char *hintFieldName2, void* hintOptions1, void *hintOptions2, bool add) {
    accMulHintFields(*(SetupCtx *)pSetupCtx, *(StepsParams *)stepsParams, (Goldilocks::Element *)pBuffHelper, hintId, string(hintFieldNameDest), string(hintFieldNameAirgroupVal), string(hintFieldName1), string(hintFieldName2),*(HintFieldOptions *)hintOptions1,  *(HintFieldOptions *)hintOptions2, add);
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

    std::vector<Dest> dests;
    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); i++) {
        if(setupCtx.starkInfo.cmPolsMap[i].imPol && setupCtx.starkInfo.cmPolsMap[i].stage == step) {
            Goldilocks::Element* pAddress = setupCtx.starkInfo.cmPolsMap[i].stage == 1 ? params.trace : params.aux_trace;
            Dest destStruct(&pAddress[setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + to_string(step), false)] + setupCtx.starkInfo.cmPolsMap[i].stagePos], (1<< setupCtx.starkInfo.starkStruct.nBits), setupCtx.starkInfo.mapSectionsN["cm" + to_string(step)]);
            destStruct.addParams(setupCtx.expressionsBin.expressionsInfo[setupCtx.starkInfo.cmPolsMap[i].expId], false);
            
            dests.push_back(destStruct);
        }
    }

    if(dests.size() == 0) return;

    ProverHelpers proverHelpers;

#ifdef __AVX512__
    ExpressionsAvx512 expressionsCtx(setupCtx, proverHelpers);
#elif defined(__AVX2__)
    ExpressionsAvx expressionsCtx(setupCtx, proverHelpers);
#else
    ExpressionsPack expressionsCtx(setupCtx, proverHelpers);
#endif

    expressionsCtx.calculateExpressions(params, setupCtx.expressionsBin.expressionsBinArgsExpressions, dests, uint64_t(1 << setupCtx.starkInfo.starkStruct.nBits), false);
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

void commit_witness(uint64_t arity, uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, void *root, void *trace, void *auxTrace) {
    Goldilocks::Element *rootGL = (Goldilocks::Element *)root;
    Goldilocks::Element *traceGL = (Goldilocks::Element *)trace;
    Goldilocks::Element *auxTraceGL = (Goldilocks::Element *)auxTrace;
    uint64_t N = 1 << nBits;
    uint64_t NExtended = 1 << nBitsExt;

    NTT_Goldilocks ntt(N);
    ntt.extendPol(auxTraceGL, traceGL, NExtended, N, nCols);

    MerkleTreeGL mt(arity, true, NExtended, nCols);
    mt.setSource(auxTraceGL);
    mt.setNodes(&auxTraceGL[NExtended * nCols]);
    mt.merkelize();
    mt.getRoot(rootGL);
}

void calculate_hash(void *pValue, void *pBuffer, uint64_t nElements)
{
    TranscriptGL transcriptHash(2, true);
    transcriptHash.put((Goldilocks::Element *)pBuffer, nElements);
    transcriptHash.getState((Goldilocks::Element *)pValue);

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

void transcript_add_polinomial(void *pTranscript, void *pPolinomial)
{
    auto transcript = (TranscriptGL *)pTranscript;
    auto pol = (Polinomial *)pPolinomial;

    for (uint64_t i = 0; i < pol->degree(); i++)
    {
        transcript->put(pol->operator[](i), pol->dim());
    }
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

void verify_constraints(void *pSetupCtx, void* stepsParams, void* constraintsInfo)
{
    verifyConstraints(*(SetupCtx *)pSetupCtx, *(StepsParams *)stepsParams, (ConstraintInfo *)constraintsInfo);
}

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

// Gen proof
// =================================================================================
void gen_proof(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params, void *globalChallenge, void* pBuffHelper, uint64_t* proofBuffer, char *proofFile) {
    genProof(*(SetupCtx *)pSetupCtx, airgroupId, airId, instanceId, *(StepsParams *)params, (Goldilocks::Element *)globalChallenge, (Goldilocks::Element *)pBuffHelper, proofBuffer, string(proofFile));
}

// Recursive proof
// ================================================================================= 
void gen_recursive_proof(void *pSetupCtx, char* globalInfoFile, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, uint64_t* proofBuffer, char* proof_file, bool vadcop) {
    json globalInfo;
    file2json(globalInfoFile, globalInfo);

    genRecursiveProof<Goldilocks::Element>(*(SetupCtx *)pSetupCtx, globalInfo, airgroupId, airId, instanceId, (Goldilocks::Element *)witness,  (Goldilocks::Element *)aux_trace, (Goldilocks::Element *)pConstPols, (Goldilocks::Element *)pConstTree, (Goldilocks::Element *)pPublicInputs, proofBuffer, string(proof_file), vadcop);
}

void *gen_recursive_proof_final(void *pSetupCtx, char* globalInfoFile, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, char* proof_file) {
    json globalInfo;
    file2json(globalInfoFile, globalInfo);

    return genRecursiveProof<RawFr::Element>(*(SetupCtx *)pSetupCtx, globalInfo, airgroupId,  airId, instanceId, (Goldilocks::Element *)witness, (Goldilocks::Element *)aux_trace, (Goldilocks::Element *)pConstPols, (Goldilocks::Element *)pConstTree, (Goldilocks::Element *)pPublicInputs, nullptr, string(proof_file), false);
}

void get_committed_pols(void *circomWitness, char* execFile, void *witness, void* pPublics, uint64_t sizeWitness, uint64_t N, uint64_t nPublics, uint64_t nCommitedPols) {
    getCommitedPols((Goldilocks::Element *)circomWitness, string(execFile), (Goldilocks::Element *)witness, (Goldilocks::Element *)pPublics, sizeWitness, N, nPublics, nCommitedPols);
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