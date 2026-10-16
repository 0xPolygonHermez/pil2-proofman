pub mod output;
pub use pil_info::types::pilout_info;
pub mod security;
pub mod stark_info;
pub mod stark_struct;

use anyhow::{bail, Result};

/// The STARK's extension degree (Goldilocks' cubic extension): the dimension of its challenges,
/// evaluations, FRI values and quotient. It is the `ext_dim` of `pil_info::FieldCfg::goldilocks()`,
/// kept as a constant here because the STARK writers and estimates are Goldilocks-only.
pub const FIELD_EXTENSION: usize = 3;

/// Goldilocks, `p = 2^64 − 2^32 + 1`: the `modulus` of `pil_info::FieldCfg::goldilocks()`. The
/// STARK files store each value as a canonical element, a `u64` below it.
pub const GOLDILOCKS_MODULUS: u64 = 0xFFFF_FFFF_0000_0001;

/// A number of the passes, narrowed to the `u64` the STARK files store: the passes keep numbers as
/// decimal strings of any width (a `0x` prefix makes it hexadecimal, as a JSON file may hold it).
/// A number that is not a canonical Goldilocks element, one below p, is an error: it is neither
/// truncated nor reduced.
pub fn goldilocks_u64(value: &str) -> Result<u64> {
    let parsed = match value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => value.parse::<u64>(),
    };
    match parsed {
        Ok(v) if v < GOLDILOCKS_MODULUS => Ok(v),
        _ => bail!("the number {value} is not a Goldilocks element (an integer below p = {GOLDILOCKS_MODULUS})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_extension_is_goldilocks_ext_dim() {
        assert_eq!(FIELD_EXTENSION, pil_info::FieldCfg::goldilocks().ext_dim());
    }

    #[test]
    fn goldilocks_modulus_is_the_passes_one() {
        assert_eq!(GOLDILOCKS_MODULUS.to_string(), pil_info::FieldCfg::goldilocks().modulus().to_string());
    }

    #[test]
    fn a_goldilocks_element_is_narrowed_as_is() {
        assert_eq!(goldilocks_u64("0").unwrap(), 0);
        assert_eq!(goldilocks_u64("18446744069414584320").unwrap(), GOLDILOCKS_MODULUS - 1);
        assert_eq!(goldilocks_u64("0x10").unwrap(), 16);
        assert_eq!(goldilocks_u64("0XFFFFFFFF00000000").unwrap(), GOLDILOCKS_MODULUS - 1);
    }

    /// What `parse().unwrap_or(0)` turned into 0, or let through non-canonical, is refused.
    #[test]
    fn a_number_that_is_not_a_goldilocks_element_is_refused() {
        let p = GOLDILOCKS_MODULUS.to_string();
        let wide = "21888242871839275222246405745257275088548364400416034343698204186575808495616";
        for value in [p.as_str(), "18446744073709551615", "18446744073709551616", wide, "0x1ffffffffffffffff", "-1", ""]
        {
            let err = goldilocks_u64(value).unwrap_err().to_string();
            assert_eq!(
                err,
                format!("the number {value} is not a Goldilocks element (an integer below p = {GOLDILOCKS_MODULUS})")
            );
        }
    }
}
