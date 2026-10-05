//! `Bn128`, the scalar field `Fr` of BN128: the integers modulo
//! `r = 21888242871839275222246405745257275088548364400416034343698204186575808495617`, the order
//! of the curve's groups. Not its base field `Fq`, where the coordinates of the points live. It is
//! named after its curve, as `Goldilocks` is after its prime.
//!
//! It is for computing a pilfflonk witness in Rust (pilfflonk/docs/README.md#witness): the BN128
//! arithmetic of the prover is ffiasm's, in C++, and this type is not used there.
//!
//! An element is kept in Montgomery form, `a·2^256 mod r`, in four 64-bit limbs, little-endian, and
//! always below `r`: ffiasm's representation of an element of `Fr` (`RawFr::Element`). Being always
//! reduced, the limbs are unique, so equality and hashing work on them directly; order works on the
//! canonical value.
//!
//! The canonical value leaves the type in two forms:
//! - 32 bytes little-endian (`to_le_bytes`, `from_le_bytes`): pilfflonk's witness and `.const`
//!   encoding (pilfflonk/docs/formats.md#witness-directory, pilfflonk/docs/formats.md#fixed-columns);
//! - a decimal string, without sign, spaces or leading zeros (`Display`, `from_decimal`):
//!   pilfflonk's JSON encoding (pilfflonk/docs/formats.md#json-encoding), and the one serde uses.
//!   The `Field` trait only asks for `Serialize` and `DeserializeOwned`: `Goldilocks` is a JSON
//!   number, but a JSON number cannot hold 254 bits in most readers. Only the canonical spelling is
//!   read back, as with `FrBytes`.

use core::cmp::Ordering;
use core::fmt;
use core::fmt::{Debug, Display, Formatter};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num_bigint::BigUint;
use serde::de::{Error, Unexpected, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{quotient_map_small_int, Field, PrimeField, QuotientMap};

/// An element of `Fr`, in Montgomery form (see the module).
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
pub struct Bn128([u64; 4]);

impl Bn128 {
    /// `r`.
    const MODULUS: [u64; 4] = [0x43e1f593f0000001, 0x2833e84879b97091, 0xb85045b68181585d, 0x30644e72e131a029];

    /// `r − 2`, the exponent of Fermat's inverse.
    const MODULUS_MINUS_2: [u64; 4] = [0x43e1f593efffffff, 0x2833e84879b97091, 0xb85045b68181585d, 0x30644e72e131a029];

    /// `−r⁻¹ mod 2^64`.
    const INV: u64 = 0xc2e1f593efffffff;

    /// `2^512 mod r`, which takes a value into Montgomery form.
    const R2: [u64; 4] = [0x1bb8e645ae216da7, 0x53fe3ab1e35c59e3, 0x8c49833d53bb8085, 0x0216d0b17f4e44a5];

    /// The digits of `r`: no element has more.
    const MAX_DECIMAL_DIGITS: usize = 77;

    /// `r − 1 = 2^28 · odd`.
    pub const TWO_ADICITY: usize = 28;

    /// `W[i] = 5^((r − 1)/2^i)`, a primitive `2^i`-th root of unity: the std's `Bn128_Gen[i]`
    /// (`pil2-components/lib/std/pil/bn128.pil`), and the root of ffjavascript's `Fr.w[i]` and of
    /// ffiasm's FFT.
    pub const W: [Self; Self::TWO_ADICITY + 1] = [
        Self::constant("1"),
        Self::constant("21888242871839275222246405745257275088548364400416034343698204186575808495616"),
        Self::constant("21888242871839275217838484774961031246007050428528088939761107053157389710902"),
        Self::constant("19540430494807482326159819597004422086093766032135589407132600596362845576832"),
        Self::constant("14940766826517323942636479241147756311199852622225275649687664389641784935947"),
        Self::constant("4419234939496763621076330863786513495701855246241724391626358375488475697872"),
        Self::constant("9088801421649573101014283686030284801466796108869023335878462724291607593530"),
        Self::constant("10359452186428527605436343203440067497552205259388878191021578220384701716497"),
        Self::constant("3478517300119284901893091970156912948790432420133812234316178878452092729974"),
        Self::constant("6837567842312086091520287814181175430087169027974246751610506942214842701774"),
        Self::constant("3161067157621608152362653341354432744960400845131437947728257924963983317266"),
        Self::constant("1120550406532664055539694724667294622065367841900378087843176726913374367458"),
        Self::constant("4158865282786404163413953114870269622875596290766033564087307867933865333818"),
        Self::constant("197302210312744933010843010704445784068657690384188106020011018676818793232"),
        Self::constant("20619701001583904760601357484951574588621083236087856586626117568842480512645"),
        Self::constant("20402931748843538985151001264530049874871572933694634836567070693966133783803"),
        Self::constant("421743594562400382753388642386256516545992082196004333756405989743524594615"),
        Self::constant("12650941915662020058015862023665998998969191525479888727406889100124684769509"),
        Self::constant("11699596668367776675346610687704220591435078791727316319397053191800576917728"),
        Self::constant("15549849457946371566896172786938980432421851627449396898353380550861104573629"),
        Self::constant("17220337697351015657950521176323262483320249231368149235373741788599650842711"),
        Self::constant("13536764371732269273912573961853310557438878140379554347802702086337840854307"),
        Self::constant("12143866164239048021030917283424216263377309185099704096317235600302831912062"),
        Self::constant("934650972362265999028062457054462628285482693704334323590406443310927365533"),
        Self::constant("5709868443893258075976348696661355716898495876243883251619397131511003808859"),
        Self::constant("19200870435978225707111062059747084165650991997241425080699860725083300967194"),
        Self::constant("7419588552507395652481651088034484897579724952953562618697845598160172257810"),
        Self::constant("2082940218526944230311718225077035922214683169814847712455127909555749686340"),
        Self::constant("19103219067921713944291392827692070036145651957329286315305642004821462161904"),
    ];

    /// The element with canonical value `limbs`, which must be below `r`.
    const fn from_canonical_limbs_unchecked(limbs: [u64; 4]) -> Self {
        Self(mont_mul(&limbs, &Self::R2))
    }

    /// The element with canonical value `limbs`, if it is below `r`.
    const fn from_canonical_limbs(limbs: [u64; 4]) -> Option<Self> {
        if is_below(&limbs, &Self::MODULUS) {
            Some(Self::from_canonical_limbs_unchecked(limbs))
        } else {
            None
        }
    }

    /// The canonical value, in limbs.
    #[inline]
    const fn canonical_limbs(&self) -> [u64; 4] {
        mont_mul(&self.0, &[1, 0, 0, 0])
    }

    /// A constant from its decimal value. Only for constant items: the compiler evaluates it, and a
    /// wrong literal fails the build.
    const fn constant(decimal: &str) -> Self {
        match Self::from_decimal(decimal) {
            Some(value) => value,
            None => panic!("not a canonical decimal element of Fr"),
        }
    }

    /// From its canonical 32 bytes, little-endian. `None` if the value is not below `r`.
    pub fn from_le_bytes(bytes: [u8; 32]) -> Option<Self> {
        let mut limbs = [0u64; 4];
        for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks_exact(8)) {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            *limb = u64::from_le_bytes(word);
        }
        Self::from_canonical_limbs(limbs)
    }

    /// The canonical value, 32 bytes little-endian.
    pub fn to_le_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for (chunk, limb) in bytes.chunks_exact_mut(8).zip(self.canonical_limbs()) {
            chunk.copy_from_slice(&limb.to_le_bytes());
        }
        bytes
    }

    /// From a decimal string in canonical form: digits only, no leading zero unless the number is
    /// 0, and below `r`. `None` otherwise.
    pub const fn from_decimal(s: &str) -> Option<Self> {
        let digits = s.as_bytes();
        if digits.is_empty() || digits.len() > Self::MAX_DECIMAL_DIGITS || (digits[0] == b'0' && digits.len() > 1) {
            return None;
        }
        let mut limbs = [0u64; 4];
        let mut i = 0;
        while i < digits.len() {
            if !digits[i].is_ascii_digit() {
                return None;
            }
            // limbs = 10·limbs + digit. It cannot overflow: 77 digits are below 2^256.
            let mut carry = (digits[i] - b'0') as u64;
            let mut j = 0;
            while j < 4 {
                (limbs[j], carry) = mac(carry, limbs[j], 10, 0);
                j += 1;
            }
            i += 1;
        }
        Self::from_canonical_limbs(limbs)
    }

    /// Writes the canonical value in decimal at the end of `buf`, and returns those digits.
    fn write_decimal<'a>(&self, buf: &'a mut [u8; Self::MAX_DECIMAL_DIGITS]) -> &'a str {
        const TEN_19: u64 = 10_000_000_000_000_000_000;
        let mut value = self.canonical_limbs();
        let mut start = buf.len();
        loop {
            // (value, chunk) = (value / 10^19, value mod 10^19).
            let mut rem = 0u64;
            for limb in value.iter_mut().rev() {
                let cur = (u128::from(rem) << 64) | u128::from(*limb);
                *limb = (cur / u128::from(TEN_19)) as u64;
                rem = (cur % u128::from(TEN_19)) as u64;
            }
            let last = value == [0; 4];
            // A chunk has 19 digits, except the most significant one, which has no leading zeros.
            for _ in 0..19 {
                start -= 1;
                buf[start] = b'0' + (rem % 10) as u8;
                rem /= 10;
                if last && rem == 0 {
                    break;
                }
            }
            if last {
                break;
            }
        }
        // ASCII digits only.
        core::str::from_utf8(&buf[start..]).unwrap_or_default()
    }

    /// `self^exponent`, with `exponent` in 64-bit limbs, little-endian.
    pub fn exp_u256(&self, exponent: [u64; 4]) -> Self {
        // Fixed windows of 4 bits, from the most significant.
        let mut table = [Self::ONE; 16];
        let mut power = Self::ONE;
        for entry in table.iter_mut() {
            *entry = power;
            power *= *self;
        }
        let mut result = Self::ONE;
        for limb in exponent.iter().rev() {
            for window in (0..16).rev() {
                result = result.exp_power_of_2(4);
                result *= table[((limb >> (4 * window)) & 0xf) as usize];
            }
        }
        result
    }
}

/// `a + b·c + carry` as (low, high) words.
#[inline(always)]
const fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let t = a as u128 + (b as u128) * (c as u128) + carry as u128;
    (t as u64, (t >> 64) as u64)
}

/// `a + b + carry` as (sum, carry).
#[inline(always)]
const fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let t = a as u128 + b as u128 + carry as u128;
    (t as u64, (t >> 64) as u64)
}

/// `a − b − borrow` as (difference, borrow), with `borrow` 0 or 1.
#[inline(always)]
const fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let (d, b1) = a.overflowing_sub(b);
    let (d, b2) = d.overflowing_sub(borrow);
    (d, (b1 | b2) as u64)
}

/// `a < b`.
#[inline(always)]
const fn is_below(a: &[u64; 4], b: &[u64; 4]) -> bool {
    let (_, borrow) = sbb(a[0], b[0], 0);
    let (_, borrow) = sbb(a[1], b[1], borrow);
    let (_, borrow) = sbb(a[2], b[2], borrow);
    let (_, borrow) = sbb(a[3], b[3], borrow);
    borrow == 1
}

/// `a mod r`, for `a < 2r`.
#[inline(always)]
const fn reduce_once(a: [u64; 4]) -> [u64; 4] {
    let m = &Bn128::MODULUS;
    let (d0, borrow) = sbb(a[0], m[0], 0);
    let (d1, borrow) = sbb(a[1], m[1], borrow);
    let (d2, borrow) = sbb(a[2], m[2], borrow);
    let (d3, borrow) = sbb(a[3], m[3], borrow);
    if borrow == 1 {
        a
    } else {
        [d0, d1, d2, d3]
    }
}

/// `a + b mod r`. The sum is below `2r < 2^255`, so it fits in four limbs.
#[inline(always)]
const fn add_mod(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let (s0, carry) = adc(a[0], b[0], 0);
    let (s1, carry) = adc(a[1], b[1], carry);
    let (s2, carry) = adc(a[2], b[2], carry);
    let (s3, _) = adc(a[3], b[3], carry);
    reduce_once([s0, s1, s2, s3])
}

/// `a − b mod r`.
#[inline(always)]
const fn sub_mod(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let (d0, borrow) = sbb(a[0], b[0], 0);
    let (d1, borrow) = sbb(a[1], b[1], borrow);
    let (d2, borrow) = sbb(a[2], b[2], borrow);
    let (d3, borrow) = sbb(a[3], b[3], borrow);
    // On a borrow, add r back.
    let mask = 0u64.wrapping_sub(borrow);
    let m = &Bn128::MODULUS;
    let (d0, carry) = adc(d0, m[0] & mask, 0);
    let (d1, carry) = adc(d1, m[1] & mask, carry);
    let (d2, carry) = adc(d2, m[2] & mask, carry);
    let (d3, _) = adc(d3, m[3] & mask, carry);
    [d0, d1, d2, d3]
}

/// `a·b·2^−256 mod r`, Montgomery multiplication (CIOS), for `a < 2^256` and `b < r`.
///
/// The rounds leave `(a·b + k·r)/2^256` for some `k < 2^256`, which is below `2r < 2^255`: it fits
/// in four limbs, and one subtraction reduces it. Within the rounds, `t` may need a fifth limb.
#[inline(always)]
const fn mont_mul(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let m = &Bn128::MODULUS;
    let mut t = [0u64; 6];
    let mut i = 0;
    while i < 4 {
        // t += a·b[i]
        let mut carry = 0;
        let mut j = 0;
        while j < 4 {
            (t[j], carry) = mac(t[j], a[j], b[i], carry);
            j += 1;
        }
        (t[4], t[5]) = adc(t[4], carry, 0);

        // t = (t + k·r) / 2^64, with k such that the division is exact.
        let k = t[0].wrapping_mul(Bn128::INV);
        let (_, mut carry) = mac(t[0], k, m[0], 0);
        j = 1;
        while j < 4 {
            (t[j - 1], carry) = mac(t[j], k, m[j], carry);
            j += 1;
        }
        (t[3], carry) = adc(t[4], carry, 0);
        t[4] = t[5] + carry;
        i += 1;
    }
    reduce_once([t[0], t[1], t[2], t[3]])
}

impl Ord for Bn128 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.canonical_limbs().iter().rev().cmp(other.canonical_limbs().iter().rev())
    }
}

impl PartialOrd for Bn128 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for Bn128 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut buf = [0u8; Self::MAX_DECIMAL_DIGITS];
        f.pad_integral(true, "", self.write_decimal(&mut buf))
    }
}

impl Debug for Bn128 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl Serialize for Bn128 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut buf = [0u8; Self::MAX_DECIMAL_DIGITS];
        serializer.serialize_str(self.write_decimal(&mut buf))
    }
}

impl<'de> Deserialize<'de> for Bn128 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct DecimalVisitor;

        impl Visitor<'_> for DecimalVisitor {
            type Value = Bn128;

            fn expecting(&self, f: &mut Formatter<'_>) -> fmt::Result {
                f.write_str("a decimal string below r, without sign or leading zeros")
            }

            fn visit_str<E: Error>(self, s: &str) -> Result<Bn128, E> {
                Bn128::from_decimal(s).ok_or_else(|| E::invalid_value(Unexpected::Str(s), &self))
            }
        }

        deserializer.deserialize_str(DecimalVisitor)
    }
}

quotient_map_small_int!(Bn128, u128, [u8, u16, u32, u64]);
quotient_map_small_int!(Bn128, i128, [i8, i16, i32, i64]);

impl QuotientMap<u128> for Bn128 {
    /// Every `u128` is below `r`, and so canonical.
    #[inline]
    fn from_int(int: u128) -> Self {
        Self::from_canonical_limbs_unchecked([int as u64, (int >> 64) as u64, 0, 0])
    }

    #[inline]
    fn from_canonical_checked(int: u128) -> Option<Self> {
        Some(Self::from_int(int))
    }

    #[inline(always)]
    unsafe fn from_canonical_unchecked(int: u128) -> Self {
        Self::from_int(int)
    }
}

impl QuotientMap<i128> for Bn128 {
    /// A negative `x` is `r − |x|`. Every `i128` is within `(r − 1)/2` of 0, and so canonical.
    #[inline]
    fn from_int(int: i128) -> Self {
        let magnitude = Self::from_int(int.unsigned_abs());
        if int < 0 {
            -magnitude
        } else {
            magnitude
        }
    }

    #[inline]
    fn from_canonical_checked(int: i128) -> Option<Self> {
        Some(Self::from_int(int))
    }

    #[inline(always)]
    unsafe fn from_canonical_unchecked(int: i128) -> Self {
        Self::from_int(int)
    }
}

impl Field for Bn128 {
    const ZERO: Self = Self([0; 4]);
    const ONE: Self = Self::from_canonical_limbs_unchecked([1, 0, 0, 0]);
    const TWO: Self = Self::from_canonical_limbs_unchecked([2, 0, 0, 0]);
    const NEG_ONE: Self = Self::from_canonical_limbs_unchecked([
        Self::MODULUS[0] - 1,
        Self::MODULUS[1],
        Self::MODULUS[2],
        Self::MODULUS[3],
    ]);
    /// 5, the smallest quadratic non-residue: the generator of ffjavascript and ffiasm, and
    /// pilfflonk's coset shift (pilfflonk/docs/protocol.md#extended-coset).
    const GENERATOR: Self = Self::from_canonical_limbs_unchecked([5, 0, 0, 0]);

    /// By Fermat: `self^(r − 2)`.
    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        Some(self.exp_u256(Self::MODULUS_MINUS_2))
    }
}

impl PrimeField for Bn128 {
    fn as_canonical_biguint(&self) -> BigUint {
        BigUint::from_bytes_le(&self.to_le_bytes())
    }
}

impl Add for Bn128 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(add_mod(&self.0, &rhs.0))
    }
}

impl AddAssign for Bn128 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Bn128 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(sub_mod(&self.0, &rhs.0))
    }
}

impl SubAssign for Bn128 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for Bn128 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self(sub_mod(&[0; 4], &self.0))
    }
}

impl Mul for Bn128 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(mont_mul(&self.0, &rhs.0))
    }
}

impl MulAssign for Bn128 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Bn128 {
    type Output = Self;

    /// Panics if `rhs` is 0, as `Field::inverse`.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

impl DivAssign for Bn128 {
    /// Panics if `rhs` is 0, as `Field::inverse`.
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::format;
    use alloc::string::ToString;
    use alloc::vec::Vec;
    use core::hint::black_box;
    use std::time::Instant;

    use rand::Rng;

    use super::*;

    const R_DECIMAL: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495617";

    fn big(decimal: &str) -> BigUint {
        BigUint::parse_bytes(decimal.as_bytes(), 10).unwrap()
    }

    fn r() -> BigUint {
        big(R_DECIMAL)
    }

    fn from_limbs(limbs: [u64; 4]) -> BigUint {
        let mut bytes = [0u8; 32];
        for (chunk, limb) in bytes.chunks_exact_mut(8).zip(limbs) {
            chunk.copy_from_slice(&limb.to_le_bytes());
        }
        BigUint::from_bytes_le(&bytes)
    }

    fn to_limbs(value: &BigUint) -> [u64; 4] {
        assert!(value.bits() <= 256);
        let mut limbs = [0u64; 4];
        for (limb, digit) in limbs.iter_mut().zip(value.to_u64_digits()) {
            *limb = digit;
        }
        limbs
    }

    /// The value of an element. `bytes_are_canonical_little_endian` checks the bytes against
    /// `num-bigint`.
    fn value(x: Bn128) -> BigUint {
        BigUint::from_bytes_le(&x.to_le_bytes())
    }

    /// The element of a value below `r`.
    fn element(v: &BigUint) -> Bn128 {
        let mut bytes = [0u8; 32];
        let digits = v.to_bytes_le();
        bytes[..digits.len()].copy_from_slice(&digits);
        Bn128::from_le_bytes(bytes).unwrap()
    }

    /// The edge values (0, 1, r − 1, (r ± 1)/2, around the limb boundaries) and then `n` random
    /// values below `r`.
    fn samples<R: Rng>(rng: &mut R, n: usize) -> Vec<BigUint> {
        let r = r();
        let mut values: Vec<BigUint> = [0u32, 1, 2, 3, 5].into_iter().map(BigUint::from).collect();
        values.extend([&r - 1u32, &r - 2u32, (&r - 1u32) >> 1, (&r + 1u32) >> 1]);
        for bits in [64, 128, 192, 253] {
            let power = BigUint::from(1u32) << bits;
            values.extend([&power - 1u32, power]);
        }
        let edges = values.len();
        while values.len() < edges + n {
            let mut bytes = [0u8; 32];
            for chunk in bytes.chunks_exact_mut(8) {
                chunk.copy_from_slice(&rng.next_u64().to_le_bytes());
            }
            bytes[31] &= 0x3f;
            let v = BigUint::from_bytes_le(&bytes);
            if v < r {
                values.push(v);
            }
        }
        values
    }

    const N_EDGES: usize = 17;

    #[test]
    fn the_constants_are_bn128s() {
        let r = r();
        assert_eq!(from_limbs(Bn128::MODULUS), r);
        assert_eq!(from_limbs(Bn128::MODULUS_MINUS_2), &r - 2u32);
        assert_eq!((BigUint::from(Bn128::INV) * &r + 1u32) % (BigUint::from(1u32) << 64), BigUint::from(0u32));
        assert_eq!(from_limbs(Bn128::R2), (BigUint::from(1u32) << 512) % &r);
        assert_eq!(Bn128::MAX_DECIMAL_DIGITS, R_DECIMAL.len());
        // The Montgomery form is ffiasm's: 1 is 2^256 mod r.
        assert_eq!(from_limbs(Bn128::ONE.0), (BigUint::from(1u32) << 256) % &r);

        assert_eq!(value(Bn128::ZERO), BigUint::from(0u32));
        assert_eq!(value(Bn128::ONE), BigUint::from(1u32));
        assert_eq!(value(Bn128::TWO), BigUint::from(2u32));
        assert_eq!(value(Bn128::NEG_ONE), &r - 1u32);
        assert_eq!(value(Bn128::GENERATOR), BigUint::from(5u32));
        assert_eq!(Bn128::default(), Bn128::ZERO);
        assert_eq!(Bn128::ONE + Bn128::NEG_ONE, Bn128::ZERO);
        assert_eq!(Bn128::ONE + Bn128::ONE, Bn128::TWO);
        assert!(Bn128::ZERO.is_zero() && Bn128::ONE.is_one() && !Bn128::NEG_ONE.is_one());
        assert_eq!(Bn128::ONE.as_canonical_biguint(), BigUint::from(1u32));
    }

    #[test]
    fn arithmetic_matches_num_bigint() {
        let r = r();
        let zero = BigUint::from(0u32);
        let mut rng = rand::rng();
        let values = samples(&mut rng, 1000);
        let elements: Vec<Bn128> = values.iter().map(element).collect();

        let check_pair = |(a, x): (&BigUint, Bn128), (b, y): (&BigUint, Bn128)| {
            assert_eq!(value(x + y), (a + b) % &r, "{a} + {b}");
            assert_eq!(value(x - y), (a + &r - b) % &r, "{a} - {b}");
            assert_eq!(value(x * y), a * b % &r, "{a} · {b}");
            assert_eq!(x.cmp(&y), a.cmp(b), "{a} <=> {b}");
            let (mut sum, mut difference, mut product) = (x, x, x);
            sum += y;
            difference -= y;
            product *= y;
            assert_eq!((sum, difference, product), (x + y, x - y, x * y));
            if *b != zero {
                let quotient = x / y;
                assert_eq!(value(quotient), a * b.modpow(&(&r - 2u32), &r) % &r, "{a} / {b}");
                assert_eq!(quotient * y, x);
                let mut q = x;
                q /= y;
                assert_eq!(q, quotient);
            }
        };

        // Every pair of edge values, and then consecutive random values.
        for i in 0..N_EDGES {
            for j in 0..N_EDGES {
                check_pair((&values[i], elements[i]), (&values[j], elements[j]));
            }
        }
        for i in N_EDGES..values.len() - 1 {
            check_pair((&values[i], elements[i]), (&values[i + 1], elements[i + 1]));
        }

        for (a, &x) in values.iter().zip(&elements) {
            assert_eq!(value(-x), (&r - a) % &r, "−{a}");
            assert_eq!(value(x.square()), a * a % &r, "{a}^2");
            assert_eq!(value(x.double()), (a << 1u32) % &r, "2·{a}");
            match x.try_inverse() {
                None => assert_eq!(*a, zero),
                Some(inverse) => {
                    assert_eq!(value(inverse), a.modpow(&(&r - 2u32), &r), "1/{a}");
                    assert_eq!(x * inverse, Bn128::ONE);
                    assert_eq!(x.inverse(), inverse);
                }
            }
            let e = rng.next_u64();
            assert_eq!(value(x.exp_u64(e)), a.modpow(&BigUint::from(e), &r), "{a}^{e}");
            let e = [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64()];
            assert_eq!(value(x.exp_u256(e)), a.modpow(&from_limbs(e), &r), "{a}^{}", from_limbs(e));
        }
        assert_eq!(Bn128::GENERATOR.exp_u256([0; 4]), Bn128::ONE);
        assert_eq!(Bn128::ZERO.exp_u256([0; 4]), Bn128::ONE, "0^0 = 1, as exp_u64");
        assert_eq!(Bn128::ZERO.exp_u64(0), Bn128::ONE);
    }

    #[test]
    #[should_panic(expected = "Tried to invert zero")]
    fn dividing_by_zero_panics() {
        let _ = Bn128::ONE / Bn128::ZERO;
    }

    #[test]
    fn integers_map_to_their_residues() {
        let r = r();
        let signed = |v: i128| {
            if v < 0 {
                &r - BigUint::from(v.unsigned_abs())
            } else {
                BigUint::from(v.unsigned_abs())
            }
        };
        let mut rng = rand::rng();
        let mut unsigned: Vec<u128> = [0, 1, 2, u128::from(u64::MAX), u128::from(u64::MAX) + 1, u128::MAX].into();
        unsigned.extend((0..1000).map(|_| (u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64())));
        for u in unsigned {
            assert_eq!(value(Bn128::from_int(u)), BigUint::from(u), "{u}");
            assert_eq!(value(Bn128::from_int(u as u64)), BigUint::from(u as u64));
            assert_eq!(value(Bn128::from_int(u as usize)), BigUint::from(u as usize));
            assert_eq!(value(Bn128::from_int(u as u32)), BigUint::from(u as u32));
            assert_eq!(value(Bn128::from_int(u as u16)), BigUint::from(u as u16));
            assert_eq!(value(Bn128::from_int(u as u8)), BigUint::from(u as u8));
            assert_eq!(Bn128::from_canonical_checked(u), Some(Bn128::from_int(u)));
            assert_eq!(unsafe { Bn128::from_canonical_unchecked(u) }, Bn128::from_int(u));

            // The same bits as signed integers: a negative x is r − |x|.
            let i = u as i128;
            assert_eq!(value(Bn128::from_int(i)), signed(i), "{i}");
            assert_eq!(value(Bn128::from_int(i as i64)), signed(i128::from(i as i64)));
            assert_eq!(value(Bn128::from_int(i as isize)), signed(i as isize as i128));
            assert_eq!(value(Bn128::from_int(i as i32)), signed(i128::from(i as i32)));
            assert_eq!(value(Bn128::from_int(i as i16)), signed(i128::from(i as i16)));
            assert_eq!(value(Bn128::from_int(i as i8)), signed(i128::from(i as i8)));
            assert_eq!(Bn128::from_canonical_checked(i), Some(Bn128::from_int(i)));
            assert_eq!(Bn128::from_canonical_checked(i as i64), Some(Bn128::from_int(i as i64)));
            assert_eq!(unsafe { Bn128::from_canonical_unchecked(i) }, Bn128::from_int(i));
            assert_eq!(Bn128::from_int(i) + Bn128::from_int(i.wrapping_neg()), Bn128::ZERO);
        }
        assert_eq!(Bn128::from_int(-1), Bn128::NEG_ONE);
        assert_eq!(Bn128::from_int(-5i8), -Bn128::GENERATOR);
        assert_eq!(value(Bn128::from_int(i128::MIN)), &r - (BigUint::from(1u32) << 127));
        assert_eq!(value(Bn128::from_int(i64::MIN)), &r - (BigUint::from(1u32) << 63));
        assert_eq!(Bn128::from_int(true as u8), Bn128::from_bool(true));
    }

    #[test]
    fn bytes_are_canonical_little_endian() {
        let r = r();
        let bytes_of = |v: &BigUint| {
            let mut bytes = [0u8; 32];
            let digits = v.to_bytes_le();
            bytes[..digits.len()].copy_from_slice(&digits);
            bytes
        };
        let mut rng = rand::rng();
        for v in samples(&mut rng, 1000) {
            let bytes = bytes_of(&v);
            let x = Bn128::from_le_bytes(bytes).unwrap();
            assert_eq!(x.to_le_bytes(), bytes);
            assert_eq!(x.as_canonical_biguint(), v);
            assert_eq!(x, Bn128::from_decimal(&v.to_str_radix(10)).unwrap(), "the bytes and the digits of {v} agree");
        }

        let mut le = [0u8; 32];
        (le[0], le[1]) = (0x02, 0x01);
        assert_eq!(Bn128::from_int(0x0102u32).to_le_bytes(), le);
        assert_eq!(Bn128::from_le_bytes([0; 32]), Some(Bn128::ZERO));
        assert_eq!(Bn128::from_le_bytes(bytes_of(&(&r - 1u32))), Some(Bn128::NEG_ONE));
        for refused in [r.clone(), &r + 1u32, BigUint::from(1u32) << 254, (BigUint::from(1u32) << 256) - 1u32] {
            assert_eq!(Bn128::from_le_bytes(bytes_of(&refused)), None, "{refused} is not below r");
        }
        assert_eq!(Bn128::from_le_bytes([0xff; 32]), None, "2^256 − 1 is refused");
    }

    #[test]
    fn decimals_are_canonical() {
        let r = r();
        for s in ["", "01", "00", "+1", "-1", " 1", "1 ", "1e3", "0x1", "1.0", "١", R_DECIMAL] {
            assert_eq!(Bn128::from_decimal(s), None, "{s:?} must be refused");
        }
        let above_r = [&r + 1u32, (BigUint::from(1u32) << 256) - 1u32, BigUint::from(10u32).pow(77)];
        for v in above_r {
            assert_eq!(Bn128::from_decimal(&v.to_str_radix(10)), None, "{v} is not below r");
        }
        assert_eq!(Bn128::from_decimal("0"), Some(Bn128::ZERO));
        assert_eq!(Bn128::from_decimal("258"), Some(Bn128::from_int(258u32)));
        assert_eq!(Bn128::from_decimal(&(&r - 1u32).to_str_radix(10)), Some(Bn128::NEG_ONE));

        let mut rng = rand::rng();
        for v in samples(&mut rng, 1000) {
            let x = element(&v);
            let decimal = v.to_str_radix(10);
            assert_eq!(x.to_string(), decimal);
            assert_eq!(format!("{x:?}"), decimal);
            assert_eq!(Bn128::from_decimal(&decimal), Some(x));
        }
        // Formatted as an integer.
        assert_eq!(
            format!("{:>5}|{:<5}|{:05}", Bn128::from_int(42), Bn128::from_int(42), Bn128::from_int(42)),
            "   42|42   |00042"
        );
        assert_eq!(Bn128::ZERO.to_string(), "0");
        assert_eq!(Bn128::from_int(10_000_000_000_000_000_000u128).to_string(), "10000000000000000000");
    }

    #[test]
    fn serde_is_a_canonical_decimal_string() {
        let mut rng = rand::rng();
        let values = samples(&mut rng, 1000);
        let elements: Vec<Bn128> = values.iter().map(element).collect();
        for (v, x) in values.iter().zip(&elements) {
            let json = serde_json::to_string(x).unwrap();
            assert_eq!(json, format!("\"{v}\""));
            assert_eq!(serde_json::from_str::<Bn128>(&json).unwrap(), *x);
        }
        let json = serde_json::to_string(&elements).unwrap();
        assert_eq!(serde_json::from_str::<Vec<Bn128>>(&json).unwrap(), elements);

        assert!(serde_json::from_str::<Bn128>("5").is_err(), "a JSON number is not a Bn128");
        assert!(serde_json::from_str::<Bn128>("\"05\"").is_err());
        assert!(serde_json::from_str::<Bn128>(&format!("\"{R_DECIMAL}\"")).is_err());
        assert!(serde_json::from_str::<Bn128>("\"-1\"").is_err());
    }

    #[test]
    fn order_is_the_canonical_values() {
        let mut rng = rand::rng();
        let values = samples(&mut rng, 200);
        let mut elements: Vec<Bn128> = values.iter().map(element).collect();
        elements.sort();
        let mut sorted = values;
        sorted.sort();
        assert_eq!(elements.iter().map(|&x| value(x)).collect::<Vec<_>>(), sorted);
        assert!(Bn128::ZERO < Bn128::ONE && Bn128::ONE < Bn128::TWO && Bn128::TWO < Bn128::NEG_ONE);
    }

    #[test]
    fn the_roots_of_unity_are_5_to_the_r_minus_1_over_2_to_the_i() {
        let r = r();
        let r_minus_1 = &r - 1u32;
        assert_eq!(r_minus_1.trailing_zeros(), Some(Bn128::TWO_ADICITY as u64));
        let five = BigUint::from(5u32);
        for (i, &w) in Bn128::W.iter().enumerate() {
            let exponent = &r_minus_1 >> i;
            assert_eq!(value(w), five.modpow(&exponent, &r), "W[{i}] = 5^((r − 1)/2^{i})");
            assert_eq!(w, Bn128::GENERATOR.exp_u256(to_limbs(&exponent)));
            // Of order exactly 2^i.
            assert_eq!(w.exp_power_of_2(i), Bn128::ONE, "W[{i}]^(2^{i}) = 1");
            if i > 0 {
                assert_eq!(w.exp_power_of_2(i - 1), Bn128::NEG_ONE, "W[{i}]^(2^{}) = −1", i - 1);
                assert_eq!(w.square(), Bn128::W[i - 1]);
            }
        }
        // ffjavascript 0.3.1's `Fr.w[8]` and `Fr.w[28]` for BN128, as `pilfflonk/src/oracle/fr.rs`
        // pins them.
        assert_eq!(
            Bn128::W[8].to_string(),
            "3478517300119284901893091970156912948790432420133812234316178878452092729974"
        );
        assert_eq!(
            Bn128::W[28].to_string(),
            "19103219067921713944291392827692070036145651957329286315305642004821462161904"
        );
        // 5 is not a square (Euler's criterion): 5^((r − 1)/2) = r − 1.
        assert_eq!(Bn128::GENERATOR.exp_u256(to_limbs(&(&r_minus_1 >> 1u32))), Bn128::NEG_ONE);
        assert_eq!(Bn128::from_int(4u8).exp_u256(to_limbs(&(&r_minus_1 >> 1u32))), Bn128::ONE);
    }

    /// Values and the results of the operations on them, from ffiasm's `RawFr`, the prover's `Fr`
    /// (`pil2-stark/src/bn128/src/ffiasm`, `fr.asm` with `__USE_ASSEMBLY__`), and computed again in
    /// Python: the two agree. `mont_a` is ffiasm's raw Montgomery form of `a`. For `k` = 0…3, `a`,
    /// `b` and `e` are the SHA-256, read big-endian, of `pilfflonk M38a a<k>`, `… b<k>` and
    /// `… e<k>`, `a` and `b` mod `r`; then two rows of edge values.
    struct Vector {
        a: &'static str,
        b: &'static str,
        e: &'static str,
        mont_a: [u64; 4],
        add: &'static str,
        sub: &'static str,
        neg: &'static str,
        mul: &'static str,
        square: &'static str,
        inv: &'static str,
        div: &'static str,
        exp: &'static str,
    }

    const FFIASM: [Vector; 6] = [
        Vector {
            a: "17147972627989245849441464127981672595321389725206737327899879144674926091749",
            b: "5386898837096491630958867201678313178511443420684052161099861674026447628501",
            e: "91390435813284853653644713319041949595749315393862654058276747930801275399482",
            mont_a: [0x578463f81fec061f, 0x9d09250339f96604, 0x44dc72811e0477b8, 0x165c93775573d3e5],
            add: "646628593246462258153925584402710685284468745474755145301536632125565224633",
            sub: "11761073790892754218482596926303359416809946304522685166800017470648478463248",
            neg: "4740270243850029372804941617275602493226974675209297015798325041900882403868",
            mul: "20241615000466785276763606764707323383069147455902412384910811862603364190198",
            square: "14171249343729302838012044596634345371292253386659166522750841092489432880171",
            inv: "20347045568233696952218564168533661772598873940678203326169460296653882568623",
            div: "8915636529527988227553755537150207667004597197495065697859981045267024968347",
            exp: "7650231442643680002583340048028869188570571460716731138456472283005430969418",
        },
        Vector {
            a: "18140807975167841009530929838577203293105645323292383046091477191316279118506",
            b: "11687365215786175813862320693650118337027370766127145723155112818589597415931",
            e: "87127015234230581967694341538831787006777876835582138897797717640613450066936",
            mont_a: [0x9b30b61b4beea41e, 0x14e221788dd649bb, 0xd1a2c5a96fa5e89a, 0x2d3a4212d6b605e4],
            add: "7939930319114741601146844786970046541584651689003494425548385823330068038820",
            sub: "6453442759381665195668609144927084956078274557165237322936364372726681702575",
            neg: "3747434896671434212715475906680071795442719077123651297606726995259529377111",
            mul: "21020496616491612133955129414810311418529463532098154068215795007423023671761",
            square: "16449346545384530165648549396530111243986391674205601642936517778163411767254",
            inv: "1809407509307718491440398603283248564612157905587739715857714735160430158464",
            div: "21177631621307333861498115511290873826152158709295198902090324321368150040817",
            exp: "20776626358743816779430980806524406644650178124638234762189608109092026130792",
        },
        Vector {
            a: "4240631938138875219396876379120885722529726527559485186274835663223223049315",
            b: "16516494287572109519811712801043014165198416643783652305923799214983591330197",
            e: "22074850482077297537677019236460601198072122157222155323494304533684437107650",
            mont_a: [0x0a986bc08cfa5a03, 0x59c9f88e44983210, 0x6817aef116c58214, 0x1c084343cd7059a9],
            add: "20757126225710984739208589180163899887728143171343137492198634878206814379512",
            sub: "9612380522406040921831569323335146645879674284191867224049240634815440214735",
            neg: "17647610933700400002849529366136389366018637872856549157423368523352585446302",
            mul: "3073702752984261581251127895518177586820500613689391607499951921915678166861",
            square: "20453222605233640388680754954441384795485517475731942216193326277744733973590",
            inv: "16162642847054760139885827046635760259648806359236027862129258494484159471992",
            div: "16128050348294430923014344270454063465587991407393150505505934728914105349063",
            exp: "15646096388962501189482259129210671893429174024816198944864788628377034923874",
        },
        Vector {
            a: "21221894442211278610934057226965601558257841108070841890570346091294992654936",
            b: "13392660614083539631982965996815403523375910518365302776459634832819724269333",
            e: "12979858080735652686467186684965591474950821086922939478634436348317392425492",
            mont_a: [0x22e5d1d900c9e04e, 0xe8f7921337f23cb9, 0x0036ff8d59afea21, 0x0112de3e9b2ec007],
            add: "12726312184455543020670617478523729993085387226020110323331776737538908428652",
            sub: "7829233828127738978951091230150198034881930589705539114110711258475268385603",
            neg: "666348429627996611312348518291673530290523292345192453127858095280815840681",
            mul: "17495879674760966637385873901788946922968035704233270290623389141293532730247",
            square: "14700246922340540320975721352831827395963193105915417598703161187357538586490",
            inv: "17591244019166269280405999617887492371362733772884263071473926446084294821719",
            div: "7351671759286715212489540027802773435630090333543223675537031787070341013857",
            exp: "14849188920960097827246848716885737826711885029583353933756794720678193327995",
        },
        Vector {
            a: "21888242871839275222246405745257275088548364400416034343698204186575808495616",
            b: "21888242871839275222246405745257275088548364400416034343698204186575808495615",
            e: "21888242871839275222246405745257275088548364400416034343698204186575808495616",
            mont_a: [0x974bc177a0000006, 0xf13771b2da58a367, 0x51e1a2470908122e, 0x2259d6b14729c0fa],
            add: "21888242871839275222246405745257275088548364400416034343698204186575808495614",
            sub: "1",
            neg: "1",
            mul: "2",
            square: "1",
            inv: "21888242871839275222246405745257275088548364400416034343698204186575808495616",
            div: "10944121435919637611123202872628637544274182200208017171849102093287904247809",
            exp: "1",
        },
        Vector {
            a: "1",
            b: "5",
            e: "10944121435919637611123202872628637544274182200208017171849102093287904247808",
            mont_a: [0xac96341c4ffffffb, 0x36fc76959f60cd29, 0x666ea36f7879462e, 0x0e0a77c19a07df2f],
            add: "6",
            sub: "21888242871839275222246405745257275088548364400416034343698204186575808495613",
            neg: "21888242871839275222246405745257275088548364400416034343698204186575808495616",
            mul: "5",
            square: "1",
            inv: "1",
            div: "8755297148735710088898562298102910035419345760166413737479281674630323398247",
            exp: "1",
        },
    ];

    #[test]
    fn matches_ffiasm() {
        for v in &FFIASM {
            let (a, b) = (Bn128::from_decimal(v.a).unwrap(), Bn128::from_decimal(v.b).unwrap());
            assert_eq!(a.0, v.mont_a, "the Montgomery form of {} is ffiasm's", v.a);
            assert_eq!((a + b).to_string(), v.add);
            assert_eq!((a - b).to_string(), v.sub);
            assert_eq!((-a).to_string(), v.neg);
            assert_eq!((a * b).to_string(), v.mul);
            assert_eq!(a.square().to_string(), v.square);
            assert_eq!(a.inverse().to_string(), v.inv);
            assert_eq!((a / b).to_string(), v.div);
            assert_eq!(a.exp_u256(to_limbs(&big(v.e))).to_string(), v.exp);
        }
    }

    /// A rough measure of throughput, in a dependent chain. In release:
    /// `cargo test --release -p proofman-fields -- --ignored --nocapture bn128_timing`.
    #[test]
    #[ignore = "timing"]
    fn bn128_timing() {
        let mut rng = rand::rng();
        let values = samples(&mut rng, 2);
        let (x0, y) = (element(&values[N_EDGES]), element(&values[N_EDGES + 1]));

        const MULS: u32 = 1 << 22;
        let start = Instant::now();
        let mut x = x0;
        for _ in 0..MULS {
            x = black_box(x) * y;
        }
        let mul_ns = start.elapsed().as_secs_f64() * 1e9 / f64::from(MULS);
        black_box(x);

        const INVERSES: u32 = 1 << 12;
        let start = Instant::now();
        let mut x = x0;
        for _ in 0..INVERSES {
            // Never 0: x and y are random, and so not 0.
            x = black_box(x).inverse() * y;
        }
        let inverse_ns = start.elapsed().as_secs_f64() * 1e9 / f64::from(INVERSES);
        black_box(x);

        std::println!(
            "Bn128: mul {mul_ns:.1} ns ({:.1} M/s), inverse {:.2} µs ({:.0} k/s)",
            1e3 / mul_ns,
            inverse_ns / 1e3,
            1e6 / inverse_ns
        );
    }
}
