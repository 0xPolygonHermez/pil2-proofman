use std::ffi::c_void;

use proofman_starks_lib_c::{
    transcript_add_c, transcript_add_polinomial_c, transcript_free_c, transcript_new_c, get_challenge_c
};

pub struct FFITranscript {
    pub p_transcript: *mut c_void,
}

impl FFITranscript {
    /// Creates a new transcript struct
    /// element_type: 0 for BN128, 1 for Goldilocks
    pub fn new(arity: u64, custom: bool) -> Self {
        let p_transcript = transcript_new_c(arity, custom);

        Self { p_transcript }
    }

    pub fn add_elements(&self, input: *mut u8, size: usize) {
        transcript_add_c(self.p_transcript, input, size as u64);
    }

    pub fn add_polinomial(&self, p_polinomial: *mut c_void) {
        transcript_add_polinomial_c(self.p_transcript, p_polinomial);
    }

    pub fn get_challenge(&self, p_element: *mut c_void) {
        get_challenge_c(self.p_transcript, p_element);
    }

    /// Frees the memory of the transcript
    pub fn free(&self) {
        transcript_free_c(self.p_transcript);
    }
}
