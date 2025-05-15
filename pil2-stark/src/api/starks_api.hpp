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
    void *stark_info_new(char* filename, bool recursive, bool verify_constraints, bool verify, bool gpu, bool preallocate);
    uint64_t get_proof_size(void *pStarkInfo);
    void set_memory_expressions(void *pStarkInfo, uint64_t nTmp1, uint64_t nTmp3);
    uint64_t get_map_total_n(void *pStarkInfo);
    uint64_t get_const_pols_offset(void *pStarkInfo);
    uint64_t get_map_total_n_custom_commits_fixed(void *pStarkInfo);
    uint64_t get_tree_size(void *pStarkInfo);
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
    uint64_t get_max_n_tmp1(void *pExpressionsBin);
    uint64_t get_max_n_tmp3(void *pExpressionsBin);
    uint64_t get_max_args(void *pExpressionsBin);
    uint64_t get_max_ops(void *pExpressionsBin);
    void expressions_bin_free(void *pExpressionsBin);

    // Hints
    // ========================================================================================
    void get_hint_field(void *pSetupCtx, void* stepsParams, void* hintFieldValues, uint64_t hintId, char* hintFieldName, void* hintOptions);
    uint64_t get_hint_field_values(void *pSetupCtx, uint64_t hintId, char* hintFieldName);
    void get_hint_field_sizes(void *pSetupCtx, void* hintFieldValues, uint64_t hintId, char* hintFieldName, void* hintOptions);
    void mul_hint_fields(void *pSetupCtx, void* stepsParams, uint64_t nHints, uint64_t *hintId, char **hintFieldNameDest, char **hintFieldName1, char **hintFieldName2, void** hintOptions1, void **hintOptions2); 
    void acc_hint_field(void *pSetupCtx, void* stepsParams, uint64_t hintId, char *hintFieldNameDest, char *hintFieldNameAirgroupVal, char *hintFieldName, bool add);
    void acc_mul_hint_fields(void *pSetupCtx, void* stepsParams, uint64_t hintId, char *hintFieldNameDest, char *hintFieldNameAirgroupVal, char *hintFieldName1, char *hintFieldName2,  void* hintOptions1, void *hintOptions2, bool add);
    uint64_t update_airgroupvalue(void *pSetupCtx, void* stepsParams, uint64_t hintId, char *hintFieldNameAirgroupVal, char *hintFieldName1, char *hintFieldName2, void* hintOptions1, void *hintOptions2, bool add);
    uint64_t set_hint_field(void *pSetupCtx, void* stepsParams, void *values, uint64_t hintId, char* hintFieldName);
    uint64_t get_hint_id(void *pSetupCtx, uint64_t hintId, char * hintFieldName);

    // Starks
    // ========================================================================================
    void calculate_impols_expressions(void *pSetupCtx, uint64_t step, void* stepsParams);
    
    uint64_t custom_commit_size(void *pSetup, uint64_t commitId);
    void load_custom_commit(void *pSetup, uint64_t commitId, void *buffer, char *customCommitFile);
    void write_custom_commit(void* root, uint64_t N, uint64_t NExtended, uint64_t nCols, void *buffer, char *bufferFile, bool check);

    void commit_witness(uint64_t arity, uint64_t nBits, uint64_t nBitsExt, uint64_t nCols, void *root, void *trace, void *auxTrace, void *d_buffers);
    void get_stream_roots(void *d_buffers);
    void calculate_hash(void *pValue, void *pBuffer, uint64_t nElements, uint64_t nOutputs);

    // Transcript
    // =================================================================================
    void *transcript_new(uint64_t arity, bool custom);
    void transcript_add(void *pTranscript, void *pInput, uint64_t size);
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
    void gen_proof(void *pSetupCtx, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void *params, void *globalChallenge, uint64_t* proofBuffer, char *proofFile, void *d_buffers, bool loadConstants, char *constPolsPath,  char *constTreePath);
    void gen_recursive_proof(void *pSetupCtx, char* globalInfoFile, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, uint64_t* proofBuffer, char *proof_file, bool vadcop, void *d_buffers, bool loadConstants, char *constPolsPath, char *constTreePath, char *proofType);
    void get_committed_pols(void *circomWitness, char* execFile, void *witness, void* pPublics, uint64_t sizeWitness, uint64_t N, uint64_t nPublics, uint64_t nCols);
    void *gen_recursive_proof_final(void *pSetupCtx, char* globalInfoFile, uint64_t airgroupId, uint64_t airId, uint64_t instanceId, void* witness, void* aux_trace, void *pConstPols, void *pConstTree, void* pPublicInputs, char* proof_file);
    void get_stream_proofs(void *d_buffers_);
    void get_stream_proofs_non_blocking(void *d_buffers_);
    void add_publics_aggregation(void *pProof, uint64_t offset, void *pPublics, uint64_t nPublicsAggregation);
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
    
    // GPU calls
    // =================================================================================
    void *gen_device_buffers(void *maxSizes);
    void free_device_buffers(void *d_buffers);
    void set_device_mpi(uint32_t mpi_node_rank);
    void set_device(uint32_t gpu_id);
    void load_device_const_pols(uint64_t airgroupId, uint64_t airId, uint64_t initial_offset, void *d_buffers, char *constFilename, uint64_t constSize, char *constTreeFilename, uint64_t constTreeSize, char* proofType);
    void gen_device_streams(void *d_buffers, uint64_t maxSizeTrace, uint64_t maxSizeContribution, uint64_t maxSizeThread,  uint64_t maxSizeConst, uint64_t maxSizeConstTree, uint64_t maxSizeTraceAggregation, uint64_t maxSizeProverBufferAggregation, uint64_t maxSizeConstAggregation, uint64_t maxSizeConstTreeAggregation, uint64_t maxProofSize, uint64_t maxProofPerGPU);
    uint64_t check_device_memory();
    
    typedef void (*ProofDoneCallback)(uint64_t instanceId, const char* proofType);
    
    void register_proof_done_callback(ProofDoneCallback cb);
#endif