extern "C" {
    #[link_name = "\u{1}_Z10save_proofPvS_mS_PcS0_"]
    pub fn save_proof(
        pStarkInfo: *mut ::std::os::raw::c_void,
        pFriProof: *mut ::std::os::raw::c_void,
        numPublicInputs: ::std::os::raw::c_ulong,
        pPublicInputs: *mut ::std::os::raw::c_void,
        publicsOutputFile: *mut ::std::os::raw::c_char,
        filePrefix: *mut ::std::os::raw::c_char,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z15zkevm_steps_newv"]
    pub fn zkevm_steps_new() -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z16zkevm_steps_freePv"]
    pub fn zkevm_steps_free(pZkevmSteps: *mut ::std::os::raw::c_void);
}
extern "C" {
    #[link_name = "\u{1}_Z14c12a_steps_newv"]
    pub fn c12a_steps_new() -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z15c12a_steps_freePv"]
    pub fn c12a_steps_free(pC12aSteps: *mut ::std::os::raw::c_void);
}
extern "C" {
    #[link_name = "\u{1}_Z20recursive1_steps_newv"]
    pub fn recursive1_steps_new() -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z21recursive1_steps_freePv"]
    pub fn recursive1_steps_free(pRecursive1Steps: *mut ::std::os::raw::c_void);
}
extern "C" {
    #[link_name = "\u{1}_Z20recursive2_steps_newv"]
    pub fn recursive2_steps_new() -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z21recursive2_steps_freePv"]
    pub fn recursive2_steps_free(Recursive2Steps: *mut ::std::os::raw::c_void);
}
extern "C" {
    #[link_name = "\u{1}_Z17generic_steps_newv"]
    pub fn generic_steps_new() -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z18generic_steps_freePv"]
    pub fn generic_steps_free(pGenericSteps: *mut ::std::os::raw::c_void);
}
extern "C" {
    #[link_name = "\u{1}_Z13fri_proof_newPv"]
    pub fn fri_proof_new(pStarks: *mut ::std::os::raw::c_void) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z18fri_proof_get_rootPvmm"]
    pub fn fri_proof_get_root(
        pFriProof: *mut ::std::os::raw::c_void,
        root_index: u64,
        root_subindex: u64,
    ) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z23fri_proof_get_tree_rootPvmm"]
    pub fn fri_proof_get_tree_root(
        pFriProof: *mut ::std::os::raw::c_void,
        tree_index: u64,
        root_index: u64,
    ) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z14fri_proof_freePv"]
    pub fn fri_proof_free(pFriProof: *mut ::std::os::raw::c_void);
}
extern "C" {
    #[link_name = "\u{1}_Z10config_newPc"]
    pub fn config_new(filename: *mut ::std::os::raw::c_char) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z11config_freePv"]
    pub fn config_free(pConfig: *mut ::std::os::raw::c_void);
}
extern "C" {
    #[link_name = "\u{1}_Z13starkinfo_newPc"]
    pub fn starkinfo_new(filename: *mut ::std::os::raw::c_char) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z13get_mapTotalNPv"]
    pub fn get_mapTotalN(pStarkInfo: *mut ::std::os::raw::c_void) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z14set_mapOffsetsPvS_"]
    pub fn set_mapOffsets(
        pStarkInfo: *mut ::std::os::raw::c_void,
        pChelpers: *mut ::std::os::raw::c_void,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z15get_map_offsetsPvPcb"]
    pub fn get_map_offsets(
        pStarkInfo: *mut ::std::os::raw::c_void,
        stage: *mut ::std::os::raw::c_char,
        flag: bool,
    ) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z18get_map_sections_nPvPc"]
    pub fn get_map_sections_n(
        pStarkInfo: *mut ::std::os::raw::c_void,
        stage: *mut ::std::os::raw::c_char,
    ) -> u64;
}
extern "C" {
    #[link_name = "\u{1}_Z14starkinfo_freePv"]
    pub fn starkinfo_free(pStarkInfo: *mut ::std::os::raw::c_void);
}
extern "C" {
    #[link_name = "\u{1}_Z10starks_newPvPcbS0_S_S_S_"]
    pub fn starks_new(
        pConfig: *mut ::std::os::raw::c_void,
        constPols: *mut ::std::os::raw::c_char,
        mapConstPolsFile: bool,
        constantsTree: *mut ::std::os::raw::c_char,
        starkInfo: *mut ::std::os::raw::c_void,
        cHelpers: *mut ::std::os::raw::c_void,
        pAddress: *mut ::std::os::raw::c_void,
    ) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z14get_stark_infoPv"]
    pub fn get_stark_info(pStarks: *mut ::std::os::raw::c_void) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z11starks_freePv"]
    pub fn starks_free(pStarks: *mut ::std::os::raw::c_void);
}
extern "C" {
    #[link_name = "\u{1}_Z12chelpers_newPc"]
    pub fn chelpers_new(cHelpers: *mut ::std::os::raw::c_char) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z13chelpers_freePv"]
    pub fn chelpers_free(pChelpers: *mut ::std::os::raw::c_void);
}
extern "C" {
    #[link_name = "\u{1}_Z10init_hintsv"]
    pub fn init_hints();
}
extern "C" {
    #[link_name = "\u{1}_Z16steps_params_newPvS_S_S_S_"]
    pub fn steps_params_new(
        pStarks: *mut ::std::os::raw::c_void,
        pChallenges: *mut ::std::os::raw::c_void,
        pSubproofValues: *mut ::std::os::raw::c_void,
        pEvals: *mut ::std::os::raw::c_void,
        pPublicInputs: *mut ::std::os::raw::c_void,
    ) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z22get_steps_params_fieldPvPc"]
    pub fn get_steps_params_field(
        pStepsParams: *mut ::std::os::raw::c_void,
        name: *mut ::std::os::raw::c_char,
    ) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z17steps_params_freePv"]
    pub fn steps_params_free(pStepsParams: *mut ::std::os::raw::c_void);
}
extern "C" {
    #[link_name = "\u{1}_Z20extend_and_merkelizePvmS_S_"]
    pub fn extend_and_merkelize(
        pStarks: *mut ::std::os::raw::c_void,
        step: u64,
        pParams: *mut ::std::os::raw::c_void,
        proof: *mut ::std::os::raw::c_void,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z16treesGL_get_rootPvmS_"]
    pub fn treesGL_get_root(
        pStarks: *mut ::std::os::raw::c_void,
        index: u64,
        root: *mut ::std::os::raw::c_void,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z13compute_stagePvjmS_S_S_S_"]
    pub fn compute_stage(
        pStarks: *mut ::std::os::raw::c_void,
        elementType: u32,
        step: u64,
        pParams: *mut ::std::os::raw::c_void,
        pProof: *mut ::std::os::raw::c_void,
        pTranscript: *mut ::std::os::raw::c_void,
        pChelpersSteps: *mut ::std::os::raw::c_void,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z13compute_evalsPvS_S_"]
    pub fn compute_evals(
        pStarks: *mut ::std::os::raw::c_void,
        pParams: *mut ::std::os::raw::c_void,
        pProof: *mut ::std::os::raw::c_void,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z15compute_fri_polPvmS_S_"]
    pub fn compute_fri_pol(
        pStarks: *mut ::std::os::raw::c_void,
        step: u64,
        pParams: *mut ::std::os::raw::c_void,
        cHelpersSteps: *mut ::std::os::raw::c_void,
    ) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z19compute_fri_foldingPvS_S_mS_"]
    pub fn compute_fri_folding(
        pStarks: *mut ::std::os::raw::c_void,
        pProof: *mut ::std::os::raw::c_void,
        pFriPol: *mut ::std::os::raw::c_void,
        step: u64,
        pChallenge: *mut ::std::os::raw::c_void,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z19compute_fri_queriesPvS_Pm"]
    pub fn compute_fri_queries(
        pStarks: *mut ::std::os::raw::c_void,
        pProof: *mut ::std::os::raw::c_void,
        friQueries: *mut u64,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z14get_proof_rootPvmm"]
    pub fn get_proof_root(
        pProof: *mut ::std::os::raw::c_void,
        stage_id: u64,
        index: u64,
    ) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z18get_vector_pointerPvPc"]
    pub fn get_vector_pointer(
        pStarks: *mut ::std::os::raw::c_void,
        name: *mut ::std::os::raw::c_char,
    ) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z13resize_vectorPvmb"]
    pub fn resize_vector(pVector: *mut ::std::os::raw::c_void, newSize: u64, value: bool);
}
extern "C" {
    #[link_name = "\u{1}_Z21set_bool_vector_valuePvmb"]
    pub fn set_bool_vector_value(pVector: *mut ::std::os::raw::c_void, index: u64, value: bool);
}
extern "C" {
    #[link_name = "\u{1}_Z24clean_symbols_calculatedPv"]
    pub fn clean_symbols_calculated(pStarks: *mut ::std::os::raw::c_void);
}
extern "C" {
    #[link_name = "\u{1}_Z21set_symbol_calculatedPvjm"]
    pub fn set_symbol_calculated(pStarks: *mut ::std::os::raw::c_void, operand: u32, id: u64);
}
extern "C" {
    #[link_name = "\u{1}_Z14calculate_hashPvS_S_m"]
    pub fn calculate_hash(
        pStarks: *mut ::std::os::raw::c_void,
        pHhash: *mut ::std::os::raw::c_void,
        pBuffer: *mut ::std::os::raw::c_void,
        nElements: u64,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z22commit_pols_starks_newPvmm"]
    pub fn commit_pols_starks_new(
        pAddress: *mut ::std::os::raw::c_void,
        degree: u64,
        nCommitedPols: u64,
    ) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z23commit_pols_starks_freePv"]
    pub fn commit_pols_starks_free(pCommitPolsStarks: *mut ::std::os::raw::c_void);
}
extern "C" {
    #[link_name = "\u{1}_Z24circom_get_commited_polsPvPcS0_S_mm"]
    pub fn circom_get_commited_pols(
        pCommitPolsStarks: *mut ::std::os::raw::c_void,
        zkevmVerifier: *mut ::std::os::raw::c_char,
        execFile: *mut ::std::os::raw::c_char,
        zkin: *mut ::std::os::raw::c_void,
        N: u64,
        nCols: u64,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z35circom_recursive1_get_commited_polsPvPcS0_S_mm"]
    pub fn circom_recursive1_get_commited_pols(
        pCommitPolsStarks: *mut ::std::os::raw::c_void,
        zkevmVerifier: *mut ::std::os::raw::c_char,
        execFile: *mut ::std::os::raw::c_char,
        zkin: *mut ::std::os::raw::c_void,
        N: u64,
        nCols: u64,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z8zkin_newPvS_mS_mS_"]
    pub fn zkin_new(
        pStarkInfo: *mut ::std::os::raw::c_void,
        pFriProof: *mut ::std::os::raw::c_void,
        numPublicInputs: ::std::os::raw::c_ulong,
        pPublicInputs: *mut ::std::os::raw::c_void,
        numRootC: ::std::os::raw::c_ulong,
        pRootC: *mut ::std::os::raw::c_void,
    ) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z14transcript_newjmb"]
    pub fn transcript_new(
        elementType: u32,
        arity: u64,
        custom: bool,
    ) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z14transcript_addPvS_m"]
    pub fn transcript_add(
        pTranscript: *mut ::std::os::raw::c_void,
        pInput: *mut ::std::os::raw::c_void,
        size: u64,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z25transcript_add_polinomialPvS_"]
    pub fn transcript_add_polinomial(
        pTranscript: *mut ::std::os::raw::c_void,
        pPolinomial: *mut ::std::os::raw::c_void,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z15transcript_freePvj"]
    pub fn transcript_free(pTranscript: *mut ::std::os::raw::c_void, elementType: u32);
}
extern "C" {
    #[link_name = "\u{1}_Z13get_challengePvS_S_"]
    pub fn get_challenge(
        pStarks: *mut ::std::os::raw::c_void,
        pTranscript: *mut ::std::os::raw::c_void,
        pElement: *mut ::std::os::raw::c_void,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z16get_permutationsPvPmmm"]
    pub fn get_permutations(
        pTranscript: *mut ::std::os::raw::c_void,
        res: *mut u64,
        n: u64,
        nBits: u64,
    );
}
extern "C" {
    #[link_name = "\u{1}_Z14polinomial_newmmPc"]
    pub fn polinomial_new(
        degree: u64,
        dim: u64,
        name: *mut ::std::os::raw::c_char,
    ) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z24polinomial_get_p_elementPvm"]
    pub fn polinomial_get_p_element(
        pPolinomial: *mut ::std::os::raw::c_void,
        index: u64,
    ) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    #[link_name = "\u{1}_Z15polinomial_freePv"]
    pub fn polinomial_free(pPolinomial: *mut ::std::os::raw::c_void);
}
extern "C" {
    #[link_name = "\u{1}_Z22goldilocks_linear_hashPvS_"]
    pub fn goldilocks_linear_hash(
        pInput: *mut ::std::os::raw::c_void,
        pOutput: *mut ::std::os::raw::c_void,
    );
}