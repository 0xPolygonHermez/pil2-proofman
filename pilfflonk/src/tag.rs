//! The fields of the files that hold a constant: each is a type with one value, written as that
//! constant and read only from it, so that a file of another backend, field or protocol is
//! refused as it is parsed.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

macro_rules! string_tag {
    ($(#[$doc:meta])* $name:ident = $value:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub struct $name;

        impl $name {
            pub const VALUE: &'static str = $value;
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(Self::VALUE)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let s = String::deserialize(deserializer)?;
                if s == Self::VALUE {
                    Ok($name)
                } else {
                    Err(D::Error::custom(format!("expected {:?}, found {:?}", Self::VALUE, s)))
                }
            }
        }
    };
}

string_tag!(
    /// `"backend"` of the globalInfo.
    Backend = "pilfflonk"
);

string_tag!(
    /// `"field"` of the globalInfo.
    Field = "bn254"
);

string_tag!(
    /// `"modulus"` of the globalInfo: `r`, in decimal.
    Modulus = crate::field::BN254_R
);

string_tag!(
    /// `"transcript"` of the globalInfo (pilfflonk/docs/protocol.md#transcript).
    Transcript = "keccak256"
);

string_tag!(
    /// `"protocol"` of the vkey and of the JSON view of the proof.
    Protocol = "pilfflonk"
);

string_tag!(
    /// `"curve"` of the vkey and of the JSON view of the proof, snarkjs's name of BN254.
    Curve = "bn128"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tag_is_its_constant_and_nothing_else() {
        assert_eq!(serde_json::to_string(&Backend).unwrap(), r#""pilfflonk""#);
        assert_eq!(serde_json::from_str::<Backend>(r#""pilfflonk""#).unwrap(), Backend);
        let err = serde_json::from_str::<Backend>(r#""stark""#).unwrap_err();
        assert!(err.to_string().contains(r#"expected "pilfflonk", found "stark""#), "{err}");
        assert!(serde_json::from_str::<Curve>("1").is_err());
        assert_eq!(serde_json::to_string(&Modulus).unwrap(), format!("{:?}", crate::field::BN254_R));
    }
}
