//! BN254 values as pilfflonk's files carry them: scalars of `Fr`, affine points of G1 and G2 and
//! the vkey digest.
//!
//! In memory a scalar or a coordinate is its canonical 32 bytes, little-endian, as the C API
//! passes it (pilfflonk/docs/README.md#c-api). In JSON it is a decimal string
//! (pilfflonk/docs/formats.md#json-encoding), written without sign, spaces or leading zeros, and
//! read only in that form: a file this crate accepts is one it could have written, so a value has a
//! single spelling and the digest of the vkey (pilfflonk/docs/formats.md#digest) is the same
//! whether it is computed over the file or over the value read from it.
//!
//! Only the ranges are checked (`< r`, `< q`): whether a point is on its curve is for the
//! verifier to check on its input (pilfflonk/docs/verifier.md#steps, steps 1–3), and for the C++
//! that computes the points.

use std::fmt;
use std::sync::OnceLock;

use num_bigint::BigUint;
use proofman_fields::Bn254;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{invalid, PilfflonkResult};

/// `r`: the order of BN254's G1, the modulus of `Fr`, in decimal. It is `"modulus"` in the
/// globalInfo (pilfflonk/docs/formats.md#globalinfo).
pub const BN254_R: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495617";

/// `q`: the modulus of BN254's base field `Fq`, where the coordinates of the points live.
pub const BN254_Q: &str = "21888242871839275222246405745257275088696311157297823662689037894645226208583";

/// Bytes of a scalar or a coordinate.
pub const FIELD_BYTES: usize = 32;

/// Bytes of an affine G1 point `x‖y`.
pub const G1_BYTES: usize = 2 * FIELD_BYTES;

/// Bytes of an affine G2 point `x.c0‖x.c1‖y.c0‖y.c1` at the C API.
pub const G2_BYTES: usize = 4 * FIELD_BYTES;

fn modulus(cell: &'static OnceLock<BigUint>, decimal: &str) -> &'static BigUint {
    cell.get_or_init(|| BigUint::parse_bytes(decimal.as_bytes(), 10).unwrap_or_default())
}

pub(crate) fn r() -> &'static BigUint {
    static R: OnceLock<BigUint> = OnceLock::new();
    modulus(&R, BN254_R)
}

fn q() -> &'static BigUint {
    static Q: OnceLock<BigUint> = OnceLock::new();
    modulus(&Q, BN254_Q)
}

/// The integer a decimal string spells, if it is in canonical form: digits only, and no leading
/// zero unless the number is 0.
pub(crate) fn parse_canonical_decimal(s: &str) -> Option<BigUint> {
    let well_formed = !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) && (s == "0" || !s.starts_with('0'));
    if well_formed {
        BigUint::parse_bytes(s.as_bytes(), 10)
    } else {
        None
    }
}

fn to_le_array(value: &BigUint) -> Option<[u8; FIELD_BYTES]> {
    let digits = value.to_bytes_le();
    if digits.len() > FIELD_BYTES {
        return None;
    }
    let mut bytes = [0u8; FIELD_BYTES];
    bytes[..digits.len()].copy_from_slice(&digits);
    Some(bytes)
}

fn reversed(mut bytes: [u8; FIELD_BYTES]) -> [u8; FIELD_BYTES] {
    bytes.reverse();
    bytes
}

macro_rules! field_element {
    ($(#[$doc:meta])* $name:ident, $modulus:ident, $modulus_name:literal) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
        pub struct $name([u8; FIELD_BYTES]);

        impl $name {
            pub const ZERO: Self = Self([0; FIELD_BYTES]);

            /// From its canonical little-endian bytes, the C API's form
            /// (pilfflonk/docs/README.md#c-api).
            pub fn from_le_bytes(bytes: [u8; FIELD_BYTES]) -> PilfflonkResult<Self> {
                if BigUint::from_bytes_le(&bytes) < *$modulus() {
                    Ok(Self(bytes))
                } else {
                    invalid!(concat!("a ", stringify!($name), " must be below ", $modulus_name, ", and these bytes are not"))
                }
            }

            /// From its canonical big-endian bytes, the form of the proof bytes
            /// (pilfflonk/docs/formats.md#proof).
            pub fn from_be_bytes(bytes: [u8; FIELD_BYTES]) -> PilfflonkResult<Self> {
                Self::from_le_bytes(reversed(bytes))
            }

            /// From a canonical decimal string (see the module).
            pub fn from_decimal(s: &str) -> PilfflonkResult<Self> {
                match parse_canonical_decimal(s).filter(|v| v < $modulus()).and_then(|v| to_le_array(&v)) {
                    Some(bytes) => Ok(Self(bytes)),
                    None => invalid!(
                        concat!("{:?} is not a ", stringify!($name), ": a decimal number below ", $modulus_name,
                            ", without sign or leading zeros"),
                        s
                    ),
                }
            }

            pub fn from_u64(value: u64) -> Self {
                let mut bytes = [0u8; FIELD_BYTES];
                bytes[..8].copy_from_slice(&value.to_le_bytes());
                Self(bytes)
            }

            pub fn to_le_bytes(&self) -> [u8; FIELD_BYTES] {
                self.0
            }

            pub fn to_be_bytes(&self) -> [u8; FIELD_BYTES] {
                reversed(self.0)
            }

            pub fn to_decimal(&self) -> String {
                BigUint::from_bytes_le(&self.0).to_str_radix(10)
            }

            pub fn is_zero(&self) -> bool {
                self.0 == [0; FIELD_BYTES]
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.to_decimal())
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({})"), self.to_decimal())
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(&self.to_decimal())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let s = String::deserialize(deserializer)?;
                Self::from_decimal(&s).map_err(D::Error::custom)
            }
        }
    };
}

field_element!(
    /// A canonical element of `Fr` (`< r`): a column value, an evaluation, a public, a challenge.
    /// The C API's `FrBytes` (pilfflonk/docs/README.md#c-api).
    FrBytes,
    r,
    "r"
);

/// A `Bn254` is an element of `Fr` in the type a witness is computed in
/// (pilfflonk/docs/README.md#witness). It is always below `r`, as an `FrBytes` is, so the
/// conversions cannot fail: they only change the representation.
impl From<Bn254> for FrBytes {
    fn from(value: Bn254) -> Self {
        Self(value.to_le_bytes())
    }
}

impl From<FrBytes> for Bn254 {
    fn from(value: FrBytes) -> Self {
        // Below r, so it is a Bn254.
        Bn254::from_le_bytes(value.0).unwrap_or_default()
    }
}

field_element!(
    /// A canonical element of `Fq` (`< q`): a coordinate of a point.
    FqBytes,
    q,
    "q"
);

/// An affine point of G1, `(x, y)`. The point at infinity is `(0, 0)`, ffiasm's affine form of it
/// and what the C API writes for it (`pilfflonk_commit_fixed`). JSON: `["x", "y"]`, in the verkey
/// and the vkey (pilfflonk/docs/formats.md#verkey).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct G1Affine {
    pub x: FqBytes,
    pub y: FqBytes,
}

impl G1Affine {
    pub const INFINITY: Self = Self { x: FqBytes::ZERO, y: FqBytes::ZERO };

    pub fn is_infinity(&self) -> bool {
        self.x.is_zero() && self.y.is_zero()
    }

    /// `x‖y`, each coordinate 32 bytes big-endian: the transcript's and the proof's encoding
    /// (pilfflonk/docs/protocol.md#transcript, pilfflonk/docs/formats.md#proof).
    pub fn to_be_bytes(&self) -> [u8; G1_BYTES] {
        let mut bytes = [0u8; G1_BYTES];
        bytes[..FIELD_BYTES].copy_from_slice(&self.x.to_be_bytes());
        bytes[FIELD_BYTES..].copy_from_slice(&self.y.to_be_bytes());
        bytes
    }

    pub fn from_be_bytes(bytes: &[u8; G1_BYTES]) -> PilfflonkResult<Self> {
        let (x, y) = split_coordinates(bytes);
        Ok(Self { x: FqBytes::from_be_bytes(x)?, y: FqBytes::from_be_bytes(y)? })
    }

    /// `x‖y`, each coordinate 32 bytes little-endian: the C API's encoding
    /// (pilfflonk/docs/README.md#c-api).
    pub fn to_le_bytes(&self) -> [u8; G1_BYTES] {
        let mut bytes = [0u8; G1_BYTES];
        bytes[..FIELD_BYTES].copy_from_slice(&self.x.to_le_bytes());
        bytes[FIELD_BYTES..].copy_from_slice(&self.y.to_le_bytes());
        bytes
    }

    pub fn from_le_bytes(bytes: &[u8; G1_BYTES]) -> PilfflonkResult<Self> {
        let (x, y) = split_coordinates(bytes);
        Ok(Self { x: FqBytes::from_le_bytes(x)?, y: FqBytes::from_le_bytes(y)? })
    }
}

fn split_coordinates(bytes: &[u8; G1_BYTES]) -> ([u8; FIELD_BYTES], [u8; FIELD_BYTES]) {
    let mut x = [0u8; FIELD_BYTES];
    let mut y = [0u8; FIELD_BYTES];
    x.copy_from_slice(&bytes[..FIELD_BYTES]);
    y.copy_from_slice(&bytes[FIELD_BYTES..]);
    (x, y)
}

impl Serialize for G1Affine {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.x, &self.y).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for G1Affine {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (x, y) = <(FqBytes, FqBytes)>::deserialize(deserializer)?;
        Ok(Self { x, y })
    }
}

/// An affine point of G2, `(x, y)` with each coordinate in `Fq2 = Fq[u]/(u² + 1)` as `[c0, c1]`
/// for `c0 + c1·u`: ffiasm's and snarkjs's order. JSON: `[["x.c0", "x.c1"], ["y.c0", "y.c1"]]`
/// (the vkey's `X_2`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct G2Affine {
    pub x: [FqBytes; 2],
    pub y: [FqBytes; 2],
}

/// `[1]₂`, the generator of G2, as `x.c0, x.c1, y.c0, y.c1` in decimal: the `[1]₂` of every BN254
/// library and every ptau, and the `[τ]₂` of a ptau with `τ = 1`.
pub const G2_GENERATOR: [&str; 4] = [
    "10857046999023057135944570762232829481370756359578518086990519993285655852781",
    "11559732032986387107991004021392285783925812861821192530917403151452391805634",
    "8495653923123431417604973247489272438418190587263600148770280649306958101930",
    "4082367875863433681332203403145435568316851327593401208105741076214120093531",
];

impl G2Affine {
    /// `[1]₂` ([`G2_GENERATOR`]).
    pub fn generator() -> PilfflonkResult<Self> {
        let [x_c0, x_c1, y_c0, y_c1] = G2_GENERATOR;
        Ok(Self {
            x: [FqBytes::from_decimal(x_c0)?, FqBytes::from_decimal(x_c1)?],
            y: [FqBytes::from_decimal(y_c0)?, FqBytes::from_decimal(y_c1)?],
        })
    }

    /// `x.c0‖x.c1‖y.c0‖y.c1`, each coordinate 32 bytes little-endian: the C API's encoding, which
    /// `pilfflonk_g2_check` reads.
    pub fn to_le_bytes(&self) -> [u8; G2_BYTES] {
        let mut bytes = [0u8; G2_BYTES];
        for (chunk, coordinate) in bytes.chunks_exact_mut(FIELD_BYTES).zip([self.x[0], self.x[1], self.y[0], self.y[1]])
        {
            chunk.copy_from_slice(&coordinate.to_le_bytes());
        }
        bytes
    }

    /// `x.c0‖x.c1‖y.c0‖y.c1`, each coordinate 32 bytes little-endian: the C API's encoding
    /// (`pilfflonk_srs_g2`). Refuses a coordinate that is not below `q`.
    pub fn from_le_bytes(bytes: &[u8; G2_BYTES]) -> PilfflonkResult<Self> {
        let mut coordinates = [FqBytes::ZERO; 4];
        for (coordinate, chunk) in coordinates.iter_mut().zip(bytes.chunks_exact(FIELD_BYTES)) {
            let mut le = [0u8; FIELD_BYTES];
            le.copy_from_slice(chunk);
            *coordinate = FqBytes::from_le_bytes(le)?;
        }
        let [x_c0, x_c1, y_c0, y_c1] = coordinates;
        Ok(Self { x: [x_c0, x_c1], y: [y_c0, y_c1] })
    }
}

impl Serialize for G2Affine {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.x, &self.y).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for G2Affine {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (x, y) = <([FqBytes; 2], [FqBytes; 2])>::deserialize(deserializer)?;
        Ok(Self { x, y })
    }
}

/// The Keccak-256 digest of the vkey (pilfflonk/docs/formats.md#digest). JSON: `"0x"` and 64
/// lowercase hexadecimal digits, the 32 bytes in the order Keccak outputs them.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Digest(pub [u8; 32]);

impl Digest {
    /// `digest mod r`, what the transcript absorbs (pilfflonk/docs/protocol.md#transcript, step 1):
    /// the 32 bytes read as a big-endian integer, as the hexadecimal string spells it.
    pub fn to_fr(&self) -> FrBytes {
        let reduced = BigUint::from_bytes_be(&self.0) % r();
        // Below r, so it fits in 32 bytes.
        FrBytes(to_le_array(&reduced).unwrap_or_default())
    }

    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(66);
        s.push_str("0x");
        for byte in self.0 {
            s.push_str(&format!("{byte:02x}"));
        }
        s
    }

    pub fn from_hex(s: &str) -> PilfflonkResult<Self> {
        let digits = s.strip_prefix("0x").map(str::as_bytes).unwrap_or_default();
        let well_formed = digits.len() == 64 && digits.iter().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
        if !well_formed {
            return invalid!("{s:?} is not a digest: \"0x\" and 64 lowercase hexadecimal digits");
        }
        let mut bytes = [0u8; 32];
        for (byte, pair) in bytes.iter_mut().zip(digits.chunks(2)) {
            *byte = (hex_value(pair[0]) << 4) | hex_value(pair[1]);
        }
        Ok(Self(bytes))
    }
}

fn hex_value(digit: u8) -> u8 {
    match digit {
        b'0'..=b'9' => digit - b'0',
        _ => digit - b'a' + 10,
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Digest({})", self.to_hex())
    }
}

impl Serialize for Digest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Self::from_hex(&s).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const R_HEX: &str = "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001";
    const Q_HEX: &str = "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47";

    #[test]
    fn the_moduli_are_bn254s() {
        // Written out independently in hexadecimal, as pil2-stark/test/pilfflonk has them.
        assert_eq!(*r(), BigUint::parse_bytes(R_HEX.as_bytes(), 16).unwrap());
        assert_eq!(*q(), BigUint::parse_bytes(Q_HEX.as_bytes(), 16).unwrap());
    }

    #[test]
    fn a_scalar_is_below_r() {
        let r_minus_1 = (r() - 1u32).to_str_radix(10);
        assert_eq!(FrBytes::from_decimal(&r_minus_1).unwrap().to_decimal(), r_minus_1);
        assert!(FrBytes::from_decimal(BN254_R).is_err());
        // Between r and q: a coordinate, not a scalar.
        let r_plus_1 = (r() + 1u32).to_str_radix(10);
        assert!(FrBytes::from_decimal(&r_plus_1).is_err());
        assert!(FqBytes::from_decimal(&r_plus_1).is_ok());
        assert!(FqBytes::from_decimal(BN254_Q).is_err());

        let mut r_le = [0u8; 32];
        let digits = r().to_bytes_le();
        r_le[..digits.len()].copy_from_slice(&digits);
        assert!(FrBytes::from_le_bytes(r_le).is_err());
        assert!(FrBytes::from_be_bytes(reversed(r_le)).is_err());
        assert!(FrBytes::from_le_bytes([0xff; 32]).is_err());
    }

    #[test]
    fn a_bn254_is_an_fr_bytes() {
        use proofman_fields::{Field, QuotientMap};

        let r_minus_1 = (r() - 1u32).to_str_radix(10);
        let values = [
            Bn254::ZERO,
            Bn254::ONE,
            Bn254::NEG_ONE,
            Bn254::GENERATOR,
            Bn254::from_int(-2),
            Bn254::W[28],
            Bn254::W[28].inverse(),
        ];
        for x in values {
            let bytes = FrBytes::from(x);
            assert_eq!(bytes.to_le_bytes(), x.to_le_bytes());
            assert_eq!(Bn254::from(bytes), x);
            assert_eq!(bytes.to_decimal(), x.to_string());
            // The two JSON forms are the same.
            assert_eq!(serde_json::to_string(&bytes).unwrap(), serde_json::to_string(&x).unwrap());
        }
        assert_eq!(FrBytes::from(Bn254::NEG_ONE).to_decimal(), r_minus_1);
        assert_eq!(Bn254::from(FrBytes::from_decimal(&r_minus_1).unwrap()), Bn254::NEG_ONE);
        assert_eq!(Bn254::from(FrBytes::from_u64(u64::MAX)), Bn254::from_int(u64::MAX));
        assert_eq!(Bn254::from(Digest([0xff; 32]).to_fr()).to_string(), Digest([0xff; 32]).to_fr().to_decimal());
    }

    #[test]
    fn only_the_canonical_decimal_spelling_is_read() {
        for s in ["", "01", "00", "+1", "-1", " 1", "1 ", "1e3", "0x1", "1.0", "١"] {
            assert!(FrBytes::from_decimal(s).is_err(), "{s:?} must be refused");
        }
        assert_eq!(FrBytes::from_decimal("0").unwrap(), FrBytes::ZERO);
        assert_eq!(FrBytes::from_decimal("258").unwrap(), FrBytes::from_u64(258));
        assert!(serde_json::from_str::<FrBytes>("258").is_err(), "a JSON number is not a scalar");
    }

    #[test]
    fn bytes_are_little_endian_in_memory_and_big_endian_in_the_proof() {
        let v = FrBytes::from_u64(0x0102);
        let mut le = [0u8; 32];
        le[0] = 0x02;
        le[1] = 0x01;
        assert_eq!(v.to_le_bytes(), le);
        assert_eq!(v.to_be_bytes(), reversed(le));
        assert_eq!(FrBytes::from_be_bytes(v.to_be_bytes()).unwrap(), v);

        let p = G1Affine { x: FqBytes::from_u64(1), y: FqBytes::from_u64(2) };
        let be = p.to_be_bytes();
        assert_eq!((be[31], be[63]), (1, 2));
        assert_eq!(G1Affine::from_be_bytes(&be).unwrap(), p);
        let le = p.to_le_bytes();
        assert_eq!((le[0], le[32]), (1, 2));
        assert_eq!(G1Affine::from_le_bytes(&le).unwrap(), p);
    }

    #[test]
    fn points_are_arrays_of_decimal_strings() {
        let p = G1Affine { x: FqBytes::from_u64(1), y: FqBytes::from_u64(2) };
        assert_eq!(serde_json::to_string(&p).unwrap(), r#"["1","2"]"#);
        assert_eq!(serde_json::from_str::<G1Affine>(r#"["1","2"]"#).unwrap(), p);
        assert!(serde_json::from_str::<G1Affine>(r#"["1","2","1"]"#).is_err());
        assert!(G1Affine::INFINITY.is_infinity() && !p.is_infinity());

        let g2 = G2Affine { x: [FqBytes::from_u64(1), FqBytes::from_u64(2)], y: [FqBytes::from_u64(3), FqBytes::ZERO] };
        let json = serde_json::to_string(&g2).unwrap();
        assert_eq!(json, r#"[["1","2"],["3","0"]]"#);
        assert_eq!(serde_json::from_str::<G2Affine>(&json).unwrap(), g2);
        assert_eq!(G2Affine::from_le_bytes(&g2.to_le_bytes()).unwrap(), g2);
        let generator = G2Affine::generator().unwrap();
        assert_eq!(G2Affine::from_le_bytes(&generator.to_le_bytes()).unwrap(), generator);
        assert_eq!(generator.x[0].to_decimal(), G2_GENERATOR[0]);
    }

    #[test]
    fn the_digest_is_0x_and_64_lowercase_hex_digits() {
        let mut bytes = [0u8; 32];
        bytes[0] = 0x2b;
        bytes[31] = 0x01;
        let d = Digest(bytes);
        let hex = d.to_hex();
        assert_eq!(hex, format!("0x2b{}01", "0".repeat(60)));
        assert!(Digest::from_hex(&hex.replace("0x2b", "0x2B")).is_err());
        assert_eq!(Digest::from_hex(&hex).unwrap(), d);
        assert!(Digest::from_hex(&hex[2..]).is_err());
        assert!(Digest::from_hex(&hex[..65]).is_err());

        // digest mod r: the all-ones digest is above r.
        let big = Digest([0xff; 32]);
        let expected = (BigUint::from_bytes_be(&[0xff; 32]) % r()).to_str_radix(10);
        assert_eq!(big.to_fr().to_decimal(), expected);
        assert_eq!(d.to_fr().to_be_bytes(), bytes, "a digest below r is itself");
    }
}
