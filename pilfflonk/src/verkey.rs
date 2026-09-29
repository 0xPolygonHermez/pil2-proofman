//! `<air>.verkey.json` (A.6): the commitments of the fixed `f_i` of an AIR.

use serde::{Deserialize, Serialize};

use crate::error::PilfflonkResult;
use crate::field::G1Affine;
use crate::json::JsonFile;

/// The commitments `[f_i(τ)]₁` of the fixed `f_i` of an AIR, in the order of its layout (the
/// fixed `f_i` are its first entries): `[["x", "y"], …]`. The vkey holds the same points, as
/// `f<i>` (A.6). Where the STARK has the Merkle root of `.const`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AirVerkey(pub Vec<G1Affine>);

impl JsonFile for AirVerkey {
    fn validate(&self) -> PilfflonkResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::FqBytes;

    #[test]
    fn it_is_an_array_of_points() {
        let verkey = AirVerkey(vec![
            G1Affine { x: FqBytes::from_u64(1), y: FqBytes::from_u64(2) },
            G1Affine { x: FqBytes::from_u64(3), y: FqBytes::from_u64(4) },
        ]);
        let text = verkey.to_json_string().unwrap();
        assert_eq!(text, "[\n [\n  \"1\",\n  \"2\"\n ],\n [\n  \"3\",\n  \"4\"\n ]\n]");
        assert_eq!(AirVerkey::from_json_str(&text).unwrap(), verkey);
        assert!(AirVerkey::from_json_str(r#"[[1, 2]]"#).is_err());
    }
}
