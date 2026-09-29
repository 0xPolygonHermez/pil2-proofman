// FFI bindings for the pilfflonk C API (pil2-stark/src/api/pilfflonk_api.hpp), written by hand:
// keep them in sync with the header.

// Status codes (`enum pilfflonk_status`).
pub const PILFFLONK_OK: ::std::os::raw::c_int = 0;
pub const PILFFLONK_ERR_INVALID_ARGUMENT: ::std::os::raw::c_int = 1;
pub const PILFFLONK_ERR_NON_CANONICAL: ::std::os::raw::c_int = 2;
pub const PILFFLONK_ERR_INTERNAL: ::std::os::raw::c_int = 3;
pub const PILFFLONK_ERR_INVALID_POINT: ::std::os::raw::c_int = 4;

// What `pilfflonk_transcript_absorb` reads (`enum pilfflonk_transcript_kind`).
pub const PILFFLONK_TRANSCRIPT_FR: u32 = 0;
pub const PILFFLONK_TRANSCRIPT_G1: u32 = 1;

extern "C" {
    pub fn pilfflonk_last_error() -> *const ::std::os::raw::c_char;

    pub fn pilfflonk_fr_check_canonical(scalar: *const u8) -> ::std::os::raw::c_int;

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
}
