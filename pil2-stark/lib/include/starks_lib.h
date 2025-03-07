#ifndef LIB_API_H
#define LIB_API_H
#include <stdint.h>

    // Save Proof
    // ========================================================================================
    void save_challenges(void *pChallenges, char* globalInfoFile, char *fileDir);
    void save_publics(unsigned long numPublicInputs, void *pPublicInputs, char *fileDir);
    void save_proof_values(void *pProofValues, char* globalInfoFile, char *fileDir);


    // SetupCtx
    // ========================================================================================
    uint64_t n_hints_by_name(void *p_expression_bin, char* hintName);
    void get_hint_ids_by_name(void *p_expression_bin, uint64_t* hintIds, char* hintName);

    // Stark Info
    // ========================================================================================
    void *stark_info_new(char* filename, bool verifier);
    uint64_t get_buffer_size_contribution_air(void *pStarkInfo);
    uint64_t get_proof_size(void *pStarkInfo);
    uint64_t get_map_total_n(void *pStarkInfo, bool recursive);
    uint64_t get_map_total_n_custom_commits_fixed(void *pStarkInfo);

    void stark_info_free(void *pStarkInfo);

    // Const Pols
    // ========================================================================================
    bool load_const_tree(void *pStarkInfo, void *pConstTree, char *treeFilename, uint64_t constTreeSize, char* verkeyFilename);
    void load_const_pols(void *pConstPols, char *constFilename, uint64_t constSize);
    uint64_t get_const_tree_size(void *pStarkInfo);
    uint64_t get_const_size(void *pStarkInfo);
    void calculate_const_tree(void *pStarkInfo, void *pConstPolsAddress, void *pConstTree);
    void write_const_tree(void *pStarkInfo, void *pConstTreeAddress, char *treeFilename);

    // Expressions Bin
    // ========================================================================================
    void *expressions_bin_new(char* filename, bool global, bool verifier);
    void expressions_bin_free(void *pExpressionsBin);

    // Hints
    // ========================================================================================
    void get_hint_field(void *pSetupCtx, void* stepsParams, void* hintFieldValues, uint64_t hintId, char* hintFieldName, void* hintOptions);
    uint64_t get_hint_field_values(void *pSetupCtx, uint64_t hintId, char* hintFieldName);
    void get_hint_field_sizes(void *pSetupCtx, void* hintFieldValues, uint64_t hintId, char* hintFieldName, void* hintOptions);
    void mul_hint_fields(void *pSetupCtx, void* stepsParams, uint64_t nHints, uint64_t *hintId, char **hintFieldNameDest, char **hintFieldName1, char **hintFieldName2, void** hintOptions1, void **hintOptions2); 
    void acc_hint_field(void *pSetupCtx, void* stepsParams, void *pBuffHelper, uint64_t hintId, char *hintFieldNameDest, char *hintFieldNameAirgroupVal, char *hintFieldName, bool add);
    void acc_mul_hint_fields(void *pSetupCtx, void* stepsParams, void *pBuffHelper, uint64_t hintId, char *hintFieldNameDest, char *hintFieldNameAirgroupVal, char *hintFieldName1, char *hintFieldName2,  void* hintOptions1, void *hintOptions2, bool add);
    uint64_t update_airgroupvalue(void *pSetupCtx, void* stepsParams, uint64_t hintId, char *hintFieldNameAirgroupVal, char *hintFieldName1, char *hintFieldName2, void* hintOptions1, void *hintOptions2, bool add);
    uint64_t set_hint_field(void *pSetupCtx, void* stepsParams, void *values, uint64_t hintId, char* hintFieldName);
    uint64_t get_hint_id(void *pSetupCtx, uint64_t hintId, char * hintFieldName);

    // Starks
    // ========================================================================================
    void calculate_impols_expressions(void *pSetupCtx, uint64_t step, void* stepsParams);
    
    void load_custom_commit(void *pSetup, uint64_t commitId, void *buffer, char *customCommitFile);
    void write_custom_commit(void* root, uint64_t N, uint64_t NExtended, uint64_t nCols, void *buffer, char *bufferFile, bool check);

    void commit_witness(uint64_t arity, uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, void *root, void *trace, void *auxTrace);
    void calculate_hash(void *pValue, void *pBuffer, uint64_t nElements);

    // Transcript
    // =================================================================================
    void *transcript_new(uint64_t arity, bool custom);
    void transcript_add(void *pTranscript, void *pInput, uint64_t size);
    void transcript_add_polinomial(void *pTranscript, void *pPolinomial);
    void transcript_free(void *pTranscript);
    void get_challenge(void *pTranscript, void *pElement);

    // Constraints
    // =================================================================================
    uint64_t get_n_constraints(void *pSetupCtx);
    void get_constraints_lines_sizes(void* pSetupCtx, uint64_t *constraintsLinesSizes);
    void get_constraints_lines(void* pSetupCtx, uint8_t **constraintsLines);
    void verify_constraints(void *pSetupCtx, void* stepsParams, void* constraintsInfo);

    // Global constraints
    // =================================================================================
    uint64_t get_n_global_constraints(void* p_globalinfo_bin);
    void get_global_constraints_lines_sizes(void* p_globalinfo_bin, uint64_t *constraintsLinesSizes);
    void get_global_constraints_lines(void* p_globalinfo_bin, uint8_t **constraintsLines);
    void verify_global_constraints(char* globalInfoFile, void *globalBin, void *publics, void* challenges, void *proofValues, void **airgroupValues, void* globalConstraintsInfo);
    uint64_t get_hint_field_global_constraints_values(void* p_globalinfo_bin, uint64_t hintId, char* hintFieldName);
    void get_hint_field_global_constraints_sizes(char* globalInfoFile, void* p_globalinfo_bin, void* hintFieldValues, uint64_t hintId, char *hintFieldName, bool print_expression);
    void get_hint_field_global_constraints(char* globalInfoFile, void* p_globalinfo_bin, void* hintFieldValues, void *publics, void *challenges, void *proofValues, void **airgroupValues, uint64_t hintId, char *hintFieldName, bool print_expression);
    uint64_t set_hint_field_global_constraints(char* globalInfoFile, void* p_globalinfo_bin, void *proofValues, void *values, uint64_t hintId, char *hintFieldName);
    
    // Gen proof && Recursive Proof
    // =================================================================================
    void gen_proof(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params, void *globalChallenge, void* pBuffHelper, uint64_t* proofBuffer, char *proofFile);
    void gen_recursive_proof(void *pSetupCtx, char* globalInfoFile, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, uint64_t* proofBuffer, char *proof_file, bool vadcop);
    void get_committed_pols(void *circomWitness, char* execFile, void *witness, void* pPublics, uint64_t sizeWitness, uint64_t N, uint64_t nPublics, uint64_t nCols);
    void *gen_recursive_proof_final(void *pSetupCtx, char* globalInfoFile, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, char* proof_file);

    // Final proof
    // =================================================================================
    void gen_final_snark_proof(void *circomWitnessFinal, char* zkeyFile, char* outputDir);

    // Util calls
    // =================================================================================
    void setLogLevel(uint64_t level);

    // Stark Verify
    // =================================================================================
    bool stark_verify(uint64_t* jProof, void *pStarkInfo, void *pExpressionsBin, char *verkey, void *pPublics, void *pProofValues, void *challenges);
    bool stark_verify_bn128(void* jProof, void *pStarkInfo, void *pExpressionsBin, char *verkey, void *pPublics);
    bool stark_verify_from_file(char *proof, void *pStarkInfo, void *pExpressionsBin, char *verkey, void *pPublics, void *pProofValues, void *challenges);

    // Fixed cols
    // =================================================================================
    void write_fixed_cols_bin(char* binFile, char* airgroupName, char* airName, uint64_t N, uint64_t nFixedPols, void* fixedPolsInfo);
    
    // OMP
    // =================================================================================
    uint64_t get_omp_max_threads();
    void set_omp_num_threads(uint64_t num_threads);
    
#endif