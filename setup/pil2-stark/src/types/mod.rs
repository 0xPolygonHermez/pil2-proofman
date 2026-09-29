pub mod output;
pub use pil_info::types::pilout_info;
pub mod security;
pub mod stark_info;
pub mod stark_struct;

/// The STARK's extension degree (Goldilocks' cubic extension): the dimension of its challenges,
/// evaluations, FRI values and quotient. It is the `ext_dim` of `pil_info::FieldCfg::goldilocks()`,
/// kept as a constant here because the STARK writers and estimates are Goldilocks-only.
pub const FIELD_EXTENSION: usize = 3;

#[cfg(test)]
mod tests {
    use super::FIELD_EXTENSION;

    #[test]
    fn field_extension_is_goldilocks_ext_dim() {
        assert_eq!(FIELD_EXTENSION, pil_info::FieldCfg::goldilocks().ext_dim());
    }
}
