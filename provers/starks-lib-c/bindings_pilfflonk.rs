// FFI bindings for the pilfflonk C API (pil2-stark/src/api/pilfflonk_api.hpp), written by hand:
// keep them in sync with the header.

// Status codes (`enum pilfflonk_status`).
pub const PILFFLONK_OK: ::std::os::raw::c_int = 0;
pub const PILFFLONK_ERR_INVALID_ARGUMENT: ::std::os::raw::c_int = 1;
pub const PILFFLONK_ERR_NON_CANONICAL: ::std::os::raw::c_int = 2;
pub const PILFFLONK_ERR_INTERNAL: ::std::os::raw::c_int = 3;
pub const PILFFLONK_ERR_INVALID_POINT: ::std::os::raw::c_int = 4;
pub const PILFFLONK_ERR_IO: ::std::os::raw::c_int = 5;
pub const PILFFLONK_ERR_FORMAT: ::std::os::raw::c_int = 6;
pub const PILFFLONK_ERR_UNSATISFIED: ::std::os::raw::c_int = 7;

// What `pilfflonk_transcript_absorb` reads (`enum pilfflonk_transcript_kind`).
pub const PILFFLONK_TRANSCRIPT_FR: u32 = 0;
pub const PILFFLONK_TRANSCRIPT_G1: u32 = 1;

extern "C" {
    pub fn pilfflonk_last_error() -> *const ::std::os::raw::c_char;

    pub fn pilfflonk_last_status() -> ::std::os::raw::c_int;

    pub fn pilfflonk_fr_check_canonical(scalar: *const u8) -> ::std::os::raw::c_int;

    pub fn pilfflonk_keccak256(data: *const u8, len: u64, out: *mut u8) -> ::std::os::raw::c_int;

    pub fn pilfflonk_transcript_new() -> *mut ::std::os::raw::c_void;

    pub fn pilfflonk_transcript_free(transcript: *mut ::std::os::raw::c_void);

    pub fn pilfflonk_transcript_absorb(
        transcript: *mut ::std::os::raw::c_void,
        data: *const u8,
        n: u64,
        kind: u32,
    ) -> ::std::os::raw::c_int;

    pub fn pilfflonk_transcript_squeeze(transcript: *mut ::std::os::raw::c_void, out: *mut u8)
        -> ::std::os::raw::c_int;

    pub fn pilfflonk_srs_from_ptau(
        ptau_path: *const ::std::os::raw::c_char,
        n_g1: u64,
        srs_path: *const ::std::os::raw::c_char,
    ) -> ::std::os::raw::c_int;

    pub fn pilfflonk_srs_load(srs_path: *const ::std::os::raw::c_char) -> *mut ::std::os::raw::c_void;

    pub fn pilfflonk_srs_free(srs: *mut ::std::os::raw::c_void);

    pub fn pilfflonk_srs_g2(srs: *const ::std::os::raw::c_void, i: u64, out_g2: *mut u8) -> ::std::os::raw::c_int;

    pub fn pilfflonk_commit_fixed(
        srs: *const ::std::os::raw::c_void,
        n_bits: u64,
        k: u64,
        evals: *const u8,
        out_g1: *mut u8,
    ) -> ::std::os::raw::c_int;

    pub fn pilfflonk_ctx_new(proving_key_dir: *const ::std::os::raw::c_char) -> *mut ::std::os::raw::c_void;

    pub fn pilfflonk_ctx_free(ctx: *mut ::std::os::raw::c_void);

    pub fn pilfflonk_ctx_n_bits_ext(
        ctx: *const ::std::os::raw::c_void,
        airgroup_id: u64,
        air_id: u64,
        out: *mut u64,
    ) -> ::std::os::raw::c_int;

    pub fn pilfflonk_ctx_srs_g2(ctx: *const ::std::os::raw::c_void, i: u64, out_g2: *mut u8) -> ::std::os::raw::c_int;

    pub fn pilfflonk_ctx_fixed_commitments(
        ctx: *const ::std::os::raw::c_void,
        airgroup_id: u64,
        air_id: u64,
        out_g1: *mut u8,
        n: u64,
    ) -> ::std::os::raw::c_int;

    pub fn pilfflonk_instance_new(
        ctx: *const ::std::os::raw::c_void,
        airgroup_id: u64,
        air_id: u64,
        stage1: *const u8,
        stage1_len: u64,
        air_values: *const u8,
        n_air_values: u64,
        publics: *const u8,
        n_publics: u64,
        proof_values: *const u8,
        n_proof_values: u64,
        insecure_blinding_seed: *const u8,
    ) -> *mut ::std::os::raw::c_void;

    pub fn pilfflonk_instance_free(instance: *mut ::std::os::raw::c_void);

    pub fn pilfflonk_commit_stage(
        instance: *mut ::std::os::raw::c_void,
        stage: u32,
        challenges: *const u8,
        n_challenges: u64,
        out_g1: *mut u8,
        n_out: u64,
    ) -> ::std::os::raw::c_int;

    pub fn pilfflonk_commit_q(
        instance: *mut ::std::os::raw::c_void,
        challenges: *const u8,
        n_challenges: u64,
        out_g1: *mut u8,
        n_out: u64,
    ) -> ::std::os::raw::c_int;

    pub fn pilfflonk_instance_set_q_part_bits(
        instance: *mut ::std::os::raw::c_void,
        part_bits: u64,
    ) -> ::std::os::raw::c_int;

    pub fn pilfflonk_instance_column(
        instance: *const ::std::os::raw::c_void,
        stage: u32,
        stage_pos: u64,
        out: *mut u8,
        n: u64,
    ) -> ::std::os::raw::c_int;

    pub fn pilfflonk_opening_new(
        instances: *const *const ::std::os::raw::c_void,
        n_instances: u64,
        xi_seed: *const u8,
    ) -> *mut ::std::os::raw::c_void;

    pub fn pilfflonk_opening_free(opening: *mut ::std::os::raw::c_void);

    pub fn pilfflonk_opening_n_evaluations(opening: *const ::std::os::raw::c_void) -> u64;

    pub fn pilfflonk_opening_evaluations(
        opening: *const ::std::os::raw::c_void,
        n: u64,
        out: *mut u8,
    ) -> ::std::os::raw::c_int;

    pub fn pilfflonk_opening_q(
        opening: *const ::std::os::raw::c_void,
        instance: u64,
        out: *mut u8,
    ) -> ::std::os::raw::c_int;

    pub fn pilfflonk_opening_open(
        opening: *const ::std::os::raw::c_void,
        transcript: *mut ::std::os::raw::c_void,
        out_w: *mut u8,
        out_wp: *mut u8,
        out_inv: *mut u8,
        out_inv_zh: *mut u8,
    ) -> ::std::os::raw::c_int;

    pub fn pilfflonk_ctx_n_constraints(
        ctx: *const ::std::os::raw::c_void,
        airgroup_id: u64,
        air_id: u64,
        out: *mut u64,
    ) -> ::std::os::raw::c_int;

    pub fn pilfflonk_ctx_constraint(
        ctx: *const ::std::os::raw::c_void,
        airgroup_id: u64,
        air_id: u64,
        index: u64,
        stage: *mut u64,
        first_row: *mut u64,
        last_row: *mut u64,
        im_pol: *mut u32,
        line_len: *mut u64,
    ) -> ::std::os::raw::c_int;

    pub fn pilfflonk_ctx_constraint_line(
        ctx: *const ::std::os::raw::c_void,
        airgroup_id: u64,
        air_id: u64,
        index: u64,
        out: *mut u8,
        n: u64,
    ) -> ::std::os::raw::c_int;

    pub fn pilfflonk_check(
        instance: *mut ::std::os::raw::c_void,
        challenges: *const u8,
        n_challenges: u64,
        max_rows: u64,
        n_constraints: u64,
        out_n_failed: *mut u64,
        out_rows: *mut u64,
        out_values: *mut u8,
    ) -> ::std::os::raw::c_int;

    pub fn pilfflonk_check_column(
        instance: *mut ::std::os::raw::c_void,
        challenges: *const u8,
        n_challenges: u64,
        stage: u32,
        stage_pos: u64,
        out: *mut u8,
        n: u64,
    ) -> ::std::os::raw::c_int;
}
