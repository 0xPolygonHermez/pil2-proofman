extern "C" {
    #[link_name = "\u{1}_Z15save_challengesPvPcS0_"]
    pub fn save_challenges(
        pChallenges: *mut ::std::os::raw::c_void,
        globalInfoFile: *mut ::std::os::raw::c_char,
        fileDir: *mut ::std::os::raw::c_char,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z12save_publicsmPvPc"]
    pub fn save_publics(
        numPublicInputs: ::std::os::raw::c_ulong,
        pPublicInputs: *mut ::std::os::raw::c_void,
        fileDir: *mut ::std::os::raw::c_char,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z17save_proof_valuesPvPcS0_"]
    pub fn save_proof_values(
        pProofValues: *mut ::std::os::raw::c_void,
        globalInfoFile: *mut ::std::os::raw::c_char,
        fileDir: *mut ::std::os::raw::c_char,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z15n_hints_by_namePvPc"]
    pub fn n_hints_by_name(p_expression_bin: *mut ::std::os::raw::c_void, hintName: *mut ::std::os::raw::c_char)
        -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z20get_hint_ids_by_namePvPmPc"]
    pub fn get_hint_ids_by_name(
        p_expression_bin: *mut ::std::os::raw::c_void,
        hintIds: *mut u64,
        hintName: *mut ::std::os::raw::c_char,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z14stark_info_newPcb"]
    pub fn stark_info_new(filename: *mut ::std::os::raw::c_char, verifier: bool) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z14get_proof_sizePv"]
    pub fn get_proof_size(pStarkInfo: *mut ::std::os::raw::c_void) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z15get_map_total_nPv"]
    pub fn get_map_total_n(pStarkInfo: *mut ::std::os::raw::c_void) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z36get_map_total_n_custom_commits_fixedPv"]
    pub fn get_map_total_n_custom_commits_fixed(pStarkInfo: *mut ::std::os::raw::c_void) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z15stark_info_freePv"]
    pub fn stark_info_free(pStarkInfo: *mut ::std::os::raw::c_void);
}
extern "C" {
    #[link_name = "\u{1}_Z15load_const_treePvS_PcmS0_"]
    pub fn load_const_tree(
        pStarkInfo: *mut ::std::os::raw::c_void,
        pConstTree: *mut ::std::os::raw::c_void,
        treeFilename: *mut ::std::os::raw::c_char,
        constTreeSize: u64,
        verkeyFilename: *mut ::std::os::raw::c_char,
    ) -> bool;
}
extern "C" {
    #[link_name = "\u{1}_Z15load_const_polsPvPcm"]
    pub fn load_const_pols(
        pConstPols: *mut ::std::os::raw::c_void,
        constFilename: *mut ::std::os::raw::c_char,
        constSize: u64,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z19get_const_tree_sizePv"]
    pub fn get_const_tree_size(pStarkInfo: *mut ::std::os::raw::c_void) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z14get_const_sizePv"]
    pub fn get_const_size(pStarkInfo: *mut ::std::os::raw::c_void) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z20calculate_const_treePvS_S_"]
    pub fn calculate_const_tree(
        pStarkInfo: *mut ::std::os::raw::c_void,
        pConstPolsAddress: *mut ::std::os::raw::c_void,
        pConstTree: *mut ::std::os::raw::c_void,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z16write_const_treePvS_Pc"]
    pub fn write_const_tree(
        pStarkInfo: *mut ::std::os::raw::c_void,
        pConstTreeAddress: *mut ::std::os::raw::c_void,
        treeFilename: *mut ::std::os::raw::c_char,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z19expressions_bin_newPcbb"]
    pub fn expressions_bin_new(
        filename: *mut ::std::os::raw::c_char,
        global: bool,
        verifier: bool,
    ) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z20expressions_bin_freePv"]
    pub fn expressions_bin_free(pExpressionsBin: *mut ::std::os::raw::c_void);
}
extern "C" {
    #[link_name = "\u{1}_Z14get_hint_fieldPvS_S_mPcS_"]
    pub fn get_hint_field(
        pSetupCtx: *mut ::std::os::raw::c_void,
        stepsParams: *mut ::std::os::raw::c_void,
        hintFieldValues: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldName: *mut ::std::os::raw::c_char,
        hintOptions: *mut ::std::os::raw::c_void,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z21get_hint_field_valuesPvmPc"]
    pub fn get_hint_field_values(
        pSetupCtx: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldName: *mut ::std::os::raw::c_char,
    ) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z20get_hint_field_sizesPvS_mPcS_"]
    pub fn get_hint_field_sizes(
        pSetupCtx: *mut ::std::os::raw::c_void,
        hintFieldValues: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldName: *mut ::std::os::raw::c_char,
        hintOptions: *mut ::std::os::raw::c_void,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z15mul_hint_fieldsPvS_mPmPPcS2_S2_PS_S3_"]
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
}
extern "C" {
    #[link_name = "\u{1}_Z14acc_hint_fieldPvS_mPcS0_S0_b"]
    pub fn acc_hint_field(
        pSetupCtx: *mut ::std::os::raw::c_void,
        stepsParams: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldNameDest: *mut ::std::os::raw::c_char,
        hintFieldNameAirgroupVal: *mut ::std::os::raw::c_char,
        hintFieldName: *mut ::std::os::raw::c_char,
        add: bool,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z19acc_mul_hint_fieldsPvS_mPcS0_S0_S0_S_S_b"]
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
}
extern "C" {
    #[link_name = "\u{1}_Z20update_airgroupvaluePvS_mPcS0_S0_S_S_b"]
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
}
extern "C" {
    #[link_name = "\u{1}_Z14set_hint_fieldPvS_S_mPc"]
    pub fn set_hint_field(
        pSetupCtx: *mut ::std::os::raw::c_void,
        stepsParams: *mut ::std::os::raw::c_void,
        values: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldName: *mut ::std::os::raw::c_char,
    ) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z11get_hint_idPvmPc"]
    pub fn get_hint_id(
        pSetupCtx: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldName: *mut ::std::os::raw::c_char,
    ) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z28calculate_impols_expressionsPvmS_"]
    pub fn calculate_impols_expressions(
        pSetupCtx: *mut ::std::os::raw::c_void,
        step: u64,
        stepsParams: *mut ::std::os::raw::c_void,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z18load_custom_commitPvmS_Pc"]
    pub fn load_custom_commit(
        pSetup: *mut ::std::os::raw::c_void,
        commitId: u64,
        buffer: *mut ::std::os::raw::c_void,
        customCommitFile: *mut ::std::os::raw::c_char,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z19write_custom_commitPvmmmS_Pcb"]
    pub fn write_custom_commit(
        root: *mut ::std::os::raw::c_void,
        N: u64,
        NExtended: u64,
        nCols: u64,
        buffer: *mut ::std::os::raw::c_void,
        bufferFile: *mut ::std::os::raw::c_char,
        check: bool,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z14commit_witnessmmmmPvS_S_"]
    pub fn commit_witness(
        arity: u64,
        nBits: u64,
        nBitsExt: u64,
        nCols: u64,
        root: *mut ::std::os::raw::c_void,
        trace: *mut ::std::os::raw::c_void,
        auxTrace: *mut ::std::os::raw::c_void,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z14calculate_hashPvS_m"]
    pub fn calculate_hash(pValue: *mut ::std::os::raw::c_void, pBuffer: *mut ::std::os::raw::c_void, nElements: u64);
}
extern "C" {
    #[link_name = "\u{1}_Z14transcript_newmb"]
    pub fn transcript_new(arity: u64, custom: bool) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z14transcript_addPvS_m"]
    pub fn transcript_add(pTranscript: *mut ::std::os::raw::c_void, pInput: *mut ::std::os::raw::c_void, size: u64);
}
extern "C" {
    #[link_name = "\u{1}_Z25transcript_add_polinomialPvS_"]
    pub fn transcript_add_polinomial(
        pTranscript: *mut ::std::os::raw::c_void,
        pPolinomial: *mut ::std::os::raw::c_void,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z15transcript_freePv"]
    pub fn transcript_free(pTranscript: *mut ::std::os::raw::c_void);
}
extern "C" {
    #[link_name = "\u{1}_Z13get_challengePvS_"]
    pub fn get_challenge(pTranscript: *mut ::std::os::raw::c_void, pElement: *mut ::std::os::raw::c_void);
}
extern "C" {
    #[link_name = "\u{1}_Z17get_n_constraintsPv"]
    pub fn get_n_constraints(pSetupCtx: *mut ::std::os::raw::c_void) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z27get_constraints_lines_sizesPvPm"]
    pub fn get_constraints_lines_sizes(pSetupCtx: *mut ::std::os::raw::c_void, constraintsLinesSizes: *mut u64);
}
extern "C" {
    #[link_name = "\u{1}_Z21get_constraints_linesPvPPh"]
    pub fn get_constraints_lines(pSetupCtx: *mut ::std::os::raw::c_void, constraintsLines: *mut *mut u8);
}
extern "C" {
    #[link_name = "\u{1}_Z18verify_constraintsPvS_S_"]
    pub fn verify_constraints(
        pSetupCtx: *mut ::std::os::raw::c_void,
        stepsParams: *mut ::std::os::raw::c_void,
        constraintsInfo: *mut ::std::os::raw::c_void,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z24get_n_global_constraintsPv"]
    pub fn get_n_global_constraints(p_globalinfo_bin: *mut ::std::os::raw::c_void) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z34get_global_constraints_lines_sizesPvPm"]
    pub fn get_global_constraints_lines_sizes(
        p_globalinfo_bin: *mut ::std::os::raw::c_void,
        constraintsLinesSizes: *mut u64,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z28get_global_constraints_linesPvPPh"]
    pub fn get_global_constraints_lines(p_globalinfo_bin: *mut ::std::os::raw::c_void, constraintsLines: *mut *mut u8);
}
extern "C" {
    #[link_name = "\u{1}_Z25verify_global_constraintsPcPvS0_S0_S0_PS0_S0_"]
    pub fn verify_global_constraints(
        globalInfoFile: *mut ::std::os::raw::c_char,
        globalBin: *mut ::std::os::raw::c_void,
        publics: *mut ::std::os::raw::c_void,
        challenges: *mut ::std::os::raw::c_void,
        proofValues: *mut ::std::os::raw::c_void,
        airgroupValues: *mut *mut ::std::os::raw::c_void,
        globalConstraintsInfo: *mut ::std::os::raw::c_void,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z40get_hint_field_global_constraints_valuesPvmPc"]
    pub fn get_hint_field_global_constraints_values(
        p_globalinfo_bin: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldName: *mut ::std::os::raw::c_char,
    ) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z39get_hint_field_global_constraints_sizesPcPvS0_mS_b"]
    pub fn get_hint_field_global_constraints_sizes(
        globalInfoFile: *mut ::std::os::raw::c_char,
        p_globalinfo_bin: *mut ::std::os::raw::c_void,
        hintFieldValues: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldName: *mut ::std::os::raw::c_char,
        print_expression: bool,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z33get_hint_field_global_constraintsPcPvS0_S0_S0_S0_PS0_mS_b"]
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
}
extern "C" {
    #[link_name = "\u{1}_Z33set_hint_field_global_constraintsPcPvS0_S0_mS_"]
    pub fn set_hint_field_global_constraints(
        globalInfoFile: *mut ::std::os::raw::c_char,
        p_globalinfo_bin: *mut ::std::os::raw::c_void,
        proofValues: *mut ::std::os::raw::c_void,
        values: *mut ::std::os::raw::c_void,
        hintId: u64,
        hintFieldName: *mut ::std::os::raw::c_char,
    ) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z9gen_proofPvmmmS_S_PmPc"]
    pub fn gen_proof(
        pSetupCtx: *mut ::std::os::raw::c_void,
        airgroupId: u64,
        airId: u64,
        instanceId: u64,
        params: *mut ::std::os::raw::c_void,
        globalChallenge: *mut ::std::os::raw::c_void,
        proofBuffer: *mut u64,
        proofFile: *mut ::std::os::raw::c_char,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z19gen_recursive_proofPvPcmmmS_S_S_S_S_PmS0_b"]
    pub fn gen_recursive_proof(
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
        proofBuffer: *mut u64,
        proof_file: *mut ::std::os::raw::c_char,
        vadcop: bool,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z18get_committed_polsPvPcS_S_mmmm"]
    pub fn get_committed_pols(
        circomWitness: *mut ::std::os::raw::c_void,
        execFile: *mut ::std::os::raw::c_char,
        witness: *mut ::std::os::raw::c_void,
        pPublics: *mut ::std::os::raw::c_void,
        sizeWitness: u64,
        N: u64,
        nPublics: u64,
        nCols: u64,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z25gen_recursive_proof_finalPvPcmmmS_S_S_S_S_S0_"]
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
}
extern "C" {
    #[link_name = "\u{1}_Z21gen_final_snark_proofPvPcS0_"]
    pub fn gen_final_snark_proof(
        circomWitnessFinal: *mut ::std::os::raw::c_void,
        zkeyFile: *mut ::std::os::raw::c_char,
        outputDir: *mut ::std::os::raw::c_char,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z11setLogLevelm"]
    pub fn setLogLevel(level: u64);
}
extern "C" {
    #[link_name = "\u{1}_Z12stark_verifyPmPvS0_PcS0_S0_S0_"]
    pub fn stark_verify(
        jProof: *mut u64,
        pStarkInfo: *mut ::std::os::raw::c_void,
        pExpressionsBin: *mut ::std::os::raw::c_void,
        verkey: *mut ::std::os::raw::c_char,
        pPublics: *mut ::std::os::raw::c_void,
        pProofValues: *mut ::std::os::raw::c_void,
        challenges: *mut ::std::os::raw::c_void,
    ) -> bool;
}
extern "C" {
    #[link_name = "\u{1}_Z18stark_verify_bn128PvS_S_PcS_"]
    pub fn stark_verify_bn128(
        jProof: *mut ::std::os::raw::c_void,
        pStarkInfo: *mut ::std::os::raw::c_void,
        pExpressionsBin: *mut ::std::os::raw::c_void,
        verkey: *mut ::std::os::raw::c_char,
        pPublics: *mut ::std::os::raw::c_void,
    ) -> bool;
}
extern "C" {
    #[link_name = "\u{1}_Z22stark_verify_from_filePcPvS0_S_S0_S0_S0_"]
    pub fn stark_verify_from_file(
        proof: *mut ::std::os::raw::c_char,
        pStarkInfo: *mut ::std::os::raw::c_void,
        pExpressionsBin: *mut ::std::os::raw::c_void,
        verkey: *mut ::std::os::raw::c_char,
        pPublics: *mut ::std::os::raw::c_void,
        pProofValues: *mut ::std::os::raw::c_void,
        challenges: *mut ::std::os::raw::c_void,
    ) -> bool;
}
extern "C" {
    #[link_name = "\u{1}_Z20write_fixed_cols_binPcS_S_mmPv"]
    pub fn write_fixed_cols_bin(
        binFile: *mut ::std::os::raw::c_char,
        airgroupName: *mut ::std::os::raw::c_char,
        airName: *mut ::std::os::raw::c_char,
        N: u64,
        nFixedPols: u64,
        fixedPolsInfo: *mut ::std::os::raw::c_void,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z19get_omp_max_threadsv"]
    pub fn get_omp_max_threads() -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z19set_omp_num_threadsm"]
    pub fn set_omp_num_threads(num_threads: u64);
}
