// FFI bindings for the pil2-stark C++ library
// These functions use C linkage (extern "C") so they have consistent names across platforms

#[allow(dead_code)]
extern "C" {
    // Save Proof
    // ========================================================================================
    pub fn save_challenges(
        pChallenges: *mut ::std::os::raw::c_void,
        globalInfoFile: *mut ::std::os::raw::c_char,
        fileDir: *mut ::std::os::raw::c_char,
    );
    
    pub fn save_publics(
        numPublicInputs: u64,
        pPublicInputs: *mut ::std::os::raw::c_void,
        fileDir: *mut ::std::os::raw::c_char,
    );
    
    pub fn save_proof_values(
        pProofValues: *mut ::std::os::raw::c_void,
        globalInfoFile: *mut ::std::os::raw::c_char,
        fileDir: *mut ::std::os::raw::c_char,
    );

    // SetupCtx
    // ========================================================================================
    pub fn n_hints_by_name(
        p_expression_bin: *mut ::std::os::raw::c_void, 
        hintName: *mut ::std::os::raw::c_char
    ) -> u64;
    
    pub fn get_hint_ids_by_name(
        p_expression_bin: *mut ::std::os::raw::c_void,
        hintIds: *mut u64,
        hintName: *mut ::std::os::raw::c_char,
    );

    // Stark Info
    // ========================================================================================
    pub fn stark_info_new(
        filename: *mut ::std::os::raw::c_char,
        recursive: bool,
        verify_constraints: bool,
        verify: bool,
        gpu: bool,
        preallocate: bool,
    ) -> *mut ::std::os::raw::c_void;
    
    pub fn get_proof_size(pStarkInfo: *mut ::std::os::raw::c_void) -> u64;

    pub fn get_proof_pinned_size(pStarkInfo: *mut ::std::os::raw::c_void) -> u64;
    
    pub fn set_memory_expressions(pStarkInfo: *mut ::std::os::raw::c_void, nTmp1: u64, nTmp3: u64);
    
    pub fn get_map_total_n(pStarkInfo: *mut ::std::os::raw::c_void) -> u64;
    
    pub fn get_const_pols_offset(pStarkInfo: *mut ::std::os::raw::c_void) -> u64;
    
    pub fn get_map_total_n_custom_commits_fixed(pStarkInfo: *mut ::std::os::raw::c_void) -> u64;
    
    pub fn get_tree_size(pStarkInfo: *mut ::std::os::raw::c_void) -> u64;
    
    pub fn stark_info_free(pStarkInfo: *mut ::std::os::raw::c_void);

    // Const Pols
    // ========================================================================================
    pub fn load_const_tree(
        pStarkInfo: *mut ::std::os::raw::c_void,
        pConstTree: *mut ::std::os::raw::c_void,
        treeFilename: *mut ::std::os::raw::c_char,
        constTreeSize: u64,
        verkeyFilename: *mut ::std::os::raw::c_char,
    ) -> bool;
    
    pub fn load_const_pols(
        pConstPols: *mut ::std::os::raw::c_void,
        constFilename: *mut ::std::os::raw::c_char,
        constSize: u64,
    );
    
    pub fn get_const_tree_size(pStarkInfo: *mut ::std::os::raw::c_void) -> u64;
    
    pub fn get_const_size(pStarkInfo: *mut ::std::os::raw::c_void) -> u64;
    
    pub fn calculate_const_tree(
        pStarkInfo: *mut ::std::os::raw::c_void,
        pConstPolsAddress: *mut ::std::os::raw::c_void,
        pConstTree: *mut ::std::os::raw::c_void,
    );
    
    pub fn write_const_tree(
        pStarkInfo: *mut ::std::os::raw::c_void,
        pConstTreeAddress: *mut ::std::os::raw::c_void,
        treeFilename: *mut ::std::os::raw::c_char,
    );

    // Expressions Bin
    // ========================================================================================
    pub fn expressions_bin_new(
        filename: *mut ::std::os::raw::c_char,
        global: bool,
        verifier: bool,
    ) -> *mut ::std::os::raw::c_void;
    
    pub fn get_max_n_tmp1(pExpressionsBin: *mut ::std::os::raw::c_void) -> u64;
    
    pub fn get_max_n_tmp3(pExpressionsBin: *mut ::std::os::raw::c_void) -> u64;
    
    pub fn get_max_args(pExpressionsBin: *mut ::std::os::raw::c_void) -> u64;
    
    pub fn get_max_ops(pExpressionsBin: *mut ::std::os::raw::c_void) -> u64;
    
    pub fn expressions_bin_free(pExpressionsBin: *mut ::std::os::raw::c_void);

    // Hints
    // ========================================================================================
    pub fn get_hint_field(
        pSetupCtx: *mut ::std::os::raw::c_void,
        stepsParams: *mut ::std::os::raw::c_void,
        hintFieldValues: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldName: *mut ::std::os::raw::c_char,
        hintOptions: *mut ::std::os::raw::c_void,
    );
    
    pub fn get_hint_field_values(
        pSetupCtx: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldName: *mut ::std::os::raw::c_char,
    ) -> u64;
    
    pub fn get_hint_field_sizes(
        pSetupCtx: *mut ::std::os::raw::c_void,
        hintFieldValues: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldName: *mut ::std::os::raw::c_char,
        hintOptions: *mut ::std::os::raw::c_void,
    );
    
    pub fn mul_hint_fields(
        pSetupCtx: *mut ::std::os::raw::c_void,
        stepsParams: *mut ::std::os::raw::c_void,
        nHints: u64,
        hintId: *mut u64,
        hintFieldNameDest: *mut *mut ::std::os::raw::c_char,
        hintFieldName1: *mut *mut ::std::os::raw::c_char,
        hintFieldName2: *mut *mut ::std::os::raw::c_char,
        hintOptions1: *mut *mut ::std::os::raw::c_void,
        hintOptions2: *mut *mut ::std::os::raw::c_void,
    );
    
    pub fn acc_hint_field(
        pSetupCtx: *mut ::std::os::raw::c_void,
        stepsParams: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldNameDest: *mut ::std::os::raw::c_char,
        hintFieldNameAirgroupVal: *mut ::std::os::raw::c_char,
        hintFieldName: *mut ::std::os::raw::c_char,
        add: bool,
    );
    
    pub fn acc_mul_hint_fields(
        pSetupCtx: *mut ::std::os::raw::c_void,
        stepsParams: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldNameDest: *mut ::std::os::raw::c_char,
        hintFieldNameAirgroupVal: *mut ::std::os::raw::c_char,
        hintFieldName1: *mut ::std::os::raw::c_char,
        hintFieldName2: *mut ::std::os::raw::c_char,
        hintOptions1: *mut ::std::os::raw::c_void,
        hintOptions2: *mut ::std::os::raw::c_void,
        add: bool,
    );
    
    pub fn update_airgroupvalue(
        pSetupCtx: *mut ::std::os::raw::c_void,
        stepsParams: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldNameAirgroupVal: *mut ::std::os::raw::c_char,
        hintFieldName1: *mut ::std::os::raw::c_char,
        hintFieldName2: *mut ::std::os::raw::c_char,
        hintOptions1: *mut ::std::os::raw::c_void,
        hintOptions2: *mut ::std::os::raw::c_void,
        add: bool,
    ) -> u64;
    
    pub fn set_hint_field(
        pSetupCtx: *mut ::std::os::raw::c_void,
        stepsParams: *mut ::std::os::raw::c_void,
        values: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldName: *mut ::std::os::raw::c_char,
    ) -> u64;
    
    pub fn get_hint_id(
        pSetupCtx: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldName: *mut ::std::os::raw::c_char,
    ) -> u64;

    // Calculate impols expressions
    // ========================================================================================
    pub fn calculate_impols_expressions(
        pSetupCtx: *mut ::std::os::raw::c_void,
        step: u64,
        stepsParams: *mut ::std::os::raw::c_void,
    );

    // Custom Commits
    // ========================================================================================
    pub fn custom_commit_size(pSetup: *mut ::std::os::raw::c_void, commitId: u64) -> u64;
    
    pub fn load_custom_commit(
        pSetup: *mut ::std::os::raw::c_void,
        commitId: u64,
        buffer: *mut ::std::os::raw::c_void,
        customCommitFile: *mut ::std::os::raw::c_char,
    );
    
    pub fn write_custom_commit(
        root: *mut ::std::os::raw::c_void,
        N: u64,
        NExtended: u64,
        nCols: u64,
        buffer: *mut ::std::os::raw::c_void,
        bufferFile: *mut ::std::os::raw::c_char,
        check: bool,
    );

    // Witness Commit
    // ========================================================================================
    pub fn commit_witness(
        arity: u64,
        nBits: u64,
        nBitsExt: u64,
        nCols: u64,
        instanceId: u64,
        airgroupId: u64,
        airId: u64,
        root: *mut ::std::os::raw::c_void,
        trace: *mut ::std::os::raw::c_void,
        auxTrace: *mut ::std::os::raw::c_void,
        d_buffers: *mut ::std::os::raw::c_void,
        pSetupCtx_: *mut ::std::os::raw::c_void,
    ) -> u64;

    // Hash & Transcript
    // ========================================================================================
    pub fn calculate_hash(
        pValue: *mut ::std::os::raw::c_void,
        pBuffer: *mut ::std::os::raw::c_void,
        nElements: u64,
        nOutputs: u64,
    );
    
    pub fn transcript_new(arity: u64, custom: bool) -> *mut ::std::os::raw::c_void;
    
    pub fn transcript_add(pTranscript: *mut ::std::os::raw::c_void, pInput: *mut ::std::os::raw::c_void, size: u64);
    
    pub fn transcript_free(pTranscript: *mut ::std::os::raw::c_void);
    
    pub fn get_challenge(pTranscript: *mut ::std::os::raw::c_void, pElement: *mut ::std::os::raw::c_void);

    // Constraints Verification
    // ========================================================================================
    pub fn get_n_constraints(pSetupCtx: *mut ::std::os::raw::c_void) -> u64;
    
    pub fn get_constraints_lines_sizes(pSetupCtx: *mut ::std::os::raw::c_void, constraintsLinesSizes: *mut u64);
    
    pub fn get_constraints_lines(pSetupCtx: *mut ::std::os::raw::c_void, constraintsLines: *mut *mut u8);
    
    pub fn verify_constraints(
        pSetupCtx: *mut ::std::os::raw::c_void,
        stepsParams: *mut ::std::os::raw::c_void,
        constraintsInfo: *mut ::std::os::raw::c_void,
    );

    // Global Constraints
    // ========================================================================================
    pub fn get_n_global_constraints(p_globalinfo_bin: *mut ::std::os::raw::c_void) -> u64;
    
    pub fn get_global_constraints_lines_sizes(
        p_globalinfo_bin: *mut ::std::os::raw::c_void,
        constraintsLinesSizes: *mut u64,
    );
    
    pub fn get_global_constraints_lines(p_globalinfo_bin: *mut ::std::os::raw::c_void, constraintsLines: *mut *mut u8);
    
    pub fn verify_global_constraints(
        globalInfoFile: *mut ::std::os::raw::c_char,
        globalBin: *mut ::std::os::raw::c_void,
        publics: *mut ::std::os::raw::c_void,
        challenges: *mut ::std::os::raw::c_void,
        proofValues: *mut ::std::os::raw::c_void,
        airgroupValues: *mut *mut ::std::os::raw::c_void,
        globalConstraintsInfo: *mut ::std::os::raw::c_void,
    );
    
    pub fn get_hint_field_global_constraints_values(
        p_globalinfo_bin: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldName: *mut ::std::os::raw::c_char,
    ) -> u64;
    
    pub fn get_hint_field_global_constraints_sizes(
        globalInfoFile: *mut ::std::os::raw::c_char,
        p_globalinfo_bin: *mut ::std::os::raw::c_void,
        hintFieldValues: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldName: *mut ::std::os::raw::c_char,
        print_expression: bool,
    );
    
    pub fn get_hint_field_global_constraints(
        globalInfoFile: *mut ::std::os::raw::c_char,
        p_globalinfo_bin: *mut ::std::os::raw::c_void,
        hintFieldValues: *mut ::std::os::raw::c_void,
        publics: *mut ::std::os::raw::c_void,
        challenges: *mut ::std::os::raw::c_void,
        proofValues: *mut ::std::os::raw::c_void,
        airgroupValues: *mut *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldName: *mut ::std::os::raw::c_char,
        print_expression: bool,
    );
    
    pub fn set_hint_field_global_constraints(
        globalInfoFile: *mut ::std::os::raw::c_char,
        p_globalinfo_bin: *mut ::std::os::raw::c_void,
        proofValues: *mut ::std::os::raw::c_void,
        values: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldName: *mut ::std::os::raw::c_char,
    ) -> u64;

    // Proof Generation
    // ========================================================================================
    pub fn gen_proof(
        pSetupCtx: *mut ::std::os::raw::c_void,
        airgroupId: u64,
        airId: u64,
        instanceId: u64,
        params: *mut ::std::os::raw::c_void,
        globalChallenge: *mut ::std::os::raw::c_void,
        proofBuffer: *mut u64,
        proofFile: *mut ::std::os::raw::c_char,
        d_buffers: *mut ::std::os::raw::c_void,
        skipRecalculation: bool,
        streamId: u64,
        constPolsPath: *mut ::std::os::raw::c_char,
        constTreePath: *mut ::std::os::raw::c_char,
    ) -> u64;
    
    pub fn gen_recursive_proof(
        pSetupCtx: *mut ::std::os::raw::c_void,
        globalInfoFile: *mut ::std::os::raw::c_char,
        airgroupId: u64,
        airId: u64,
        instanceId: u64,
        witness: *mut ::std::os::raw::c_void,
        aux_trace: *mut ::std::os::raw::c_void,
        pPublicInputs: *mut ::std::os::raw::c_void,
        proofBuffer: *mut u64,
        proof_file: *mut ::std::os::raw::c_char,
        vadcop: bool,
        d_buffers: *mut ::std::os::raw::c_void,
        constPolsPath: *mut ::std::os::raw::c_char,
        constTreePath: *mut ::std::os::raw::c_char,
        proofType: *mut ::std::os::raw::c_char,
    ) -> u64;
    
    pub fn read_exec_file(exec_data: *mut u64, exec_file: *mut ::std::os::raw::c_char, nCommitedPols: u64);
    
    pub fn get_committed_pols(
        circomWitness: *mut ::std::os::raw::c_void,
        execData: *mut u64,
        witness: *mut ::std::os::raw::c_void,
        pPublics: *mut ::std::os::raw::c_void,
        sizeWitness: u64,
        N: u64,
        nPublics: u64,
        nCols: u64,
    );
    
    pub fn gen_recursive_proof_final(
        pSetupCtx: *mut ::std::os::raw::c_void,
        globalInfoFile: *mut ::std::os::raw::c_char,
        airgroupId: u64,
        airId: u64,
        instanceId: u64,
        witness: *mut ::std::os::raw::c_void,
        aux_trace: *mut ::std::os::raw::c_void,
        pConstPols: *mut ::std::os::raw::c_void,
        pConstTree: *mut ::std::os::raw::c_void,
        pPublicInputs: *mut ::std::os::raw::c_void,
        proof_file: *mut ::std::os::raw::c_char,
    ) -> *mut ::std::os::raw::c_void;

    // Stream Management
    // ========================================================================================
    pub fn get_stream_proofs(d_buffers_: *mut ::std::os::raw::c_void);
    
    pub fn get_stream_proofs_non_blocking(d_buffers_: *mut ::std::os::raw::c_void);
    
    pub fn get_stream_id_proof(d_buffers_: *mut ::std::os::raw::c_void, streamId: u64);

    // Aggregation
    // ========================================================================================
    pub fn add_publics_aggregation(
        pProof: *mut ::std::os::raw::c_void,
        offset: u64,
        pPublics: *mut ::std::os::raw::c_void,
        nPublicsAggregation: u64,
    );
    
    pub fn gen_final_snark_proof(
        circomWitnessFinal: *mut ::std::os::raw::c_void,
        zkeyFile: *mut ::std::os::raw::c_char,
        outputDir: *mut ::std::os::raw::c_char,
    );

    // Utilities
    // ========================================================================================
    pub fn setLogLevel(level: u64);

    // Verification
    // ========================================================================================
    pub fn stark_verify(
        jProof: *mut u64,
        pStarkInfo: *mut ::std::os::raw::c_void,
        pExpressionsBin: *mut ::std::os::raw::c_void,
        verkey: *mut ::std::os::raw::c_char,
        pPublics: *mut ::std::os::raw::c_void,
        pProofValues: *mut ::std::os::raw::c_void,
        challenges: *mut ::std::os::raw::c_void,
    ) -> bool;
    
    pub fn stark_verify_bn128(
        jProof: *mut ::std::os::raw::c_void,
        pStarkInfo: *mut ::std::os::raw::c_void,
        pExpressionsBin: *mut ::std::os::raw::c_void,
        verkey: *mut ::std::os::raw::c_char,
        pPublics: *mut ::std::os::raw::c_void,
    ) -> bool;
    
    pub fn stark_verify_from_file(
        proof: *mut ::std::os::raw::c_char,
        pStarkInfo: *mut ::std::os::raw::c_void,
        pExpressionsBin: *mut ::std::os::raw::c_void,
        verkey: *mut ::std::os::raw::c_char,
        pPublics: *mut ::std::os::raw::c_void,
        pProofValues: *mut ::std::os::raw::c_void,
        challenges: *mut ::std::os::raw::c_void,
    ) -> bool;

    // Fixed Columns
    // ========================================================================================
    pub fn write_fixed_cols_bin(
        binFile: *mut ::std::os::raw::c_char,
        airgroupName: *mut ::std::os::raw::c_char,
        airName: *mut ::std::os::raw::c_char,
        N: u64,
        nFixedPols: u64,
        fixedPolsInfo: *mut ::std::os::raw::c_void,
    );

    // OpenMP
    // ========================================================================================
    pub fn get_omp_max_threads() -> u64;
    
    pub fn set_omp_num_threads(num_threads: u64);

    // GPU/Device Management
    // ========================================================================================
    pub fn gen_device_buffers(
        maxSizes_: *mut ::std::os::raw::c_void,
        node_rank: u32,
        node_size: u32,
    ) -> *mut ::std::os::raw::c_void;
    
    pub fn free_device_buffers(d_buffers: *mut ::std::os::raw::c_void);
    
    pub fn load_device_const_pols(
        airgroupId: u64,
        airId: u64,
        initial_offset: u64,
        d_buffers: *mut ::std::os::raw::c_void,
        constFilename: *mut ::std::os::raw::c_char,
        constSize: u64,
        constTreeFilename: *mut ::std::os::raw::c_char,
        constTreeSize: u64,
        proofType: *mut ::std::os::raw::c_char,
    );
    
    pub fn load_device_setup(
        airgroupId: u64,
        airId: u64,
        proofType: *mut ::std::os::raw::c_char,
        pSetupCtx_: *mut ::std::os::raw::c_void,
        d_buffers_: *mut ::std::os::raw::c_void,
        verkeyRoot_: *mut ::std::os::raw::c_void,
        n_streams: u64,
    );
    
    pub fn gen_device_streams(
        d_buffers_: *mut ::std::os::raw::c_void,
        maxSizeProverBuffer: u64,
        maxSizeProverBufferAggregation: u64,
        maxProofSize: u64,
        maxProofsPerGPU: u64,
        maxRecursiveProofsPerGPU: u64,
        max_n_bits_ext: u64,
    ) -> u64;
    
    pub fn reset_device_streams(
        d_buffers_: *mut ::std::os::raw::c_void,
    );
    
    pub fn check_device_memory(
        node_rank: u32,
        node_size: u32,
    ) -> u64;
    
    pub fn get_num_gpus() -> u64;

    // Callback Management
    // ========================================================================================
    pub fn register_proof_done_callback(cb: ProofDoneCallback);
    
    pub fn launch_callback(instanceId: u64, proofType: *mut ::std::os::raw::c_char);
}

// Type definitions
pub type ProofDoneCallback =
    ::std::option::Option<unsafe extern "C" fn(instanceId: u64, proofType: *const ::std::os::raw::c_char)>;
