// WARNING: This file has been autogenerated from the PILOUT file.
// Manual modifications are not recommended and may be overwritten.
use proofman_common::WitnessPilout;

pub const PILOUT_HASH: &[u8] = b"FibonacciSq-hash";

//AIRGROUP CONSTANTS

pub const FIBONACCI_SQUARE_AIRGROUP_ID: usize = 0;

pub const MODULE_AIRGROUP_ID: usize = 1;

pub const U_8_AIR_AIRGROUP_ID: usize = 2;

//AIR CONSTANTS

pub const FIBONACCI_SQUARE_AIR_IDS: &[usize] = &[0];

pub const MODULE_AIR_IDS: &[usize] = &[0];

pub const U_8_AIR_AIR_IDS: &[usize] = &[0];

pub struct Pilout;

impl Pilout {
    pub fn pilout() -> WitnessPilout {
        let mut pilout = WitnessPilout::new("FibonacciSq", 2, PILOUT_HASH.to_vec());

        let air_group = pilout.add_air_group(Some("FibonacciSquare"));
        air_group.add_air(Some("FibonacciSquare"), 1024);

        let air_group = pilout.add_air_group(Some("Module"));
        air_group.add_air(Some("Module"), 1024);

        let air_group = pilout.add_air_group(Some("U8Air"));
        air_group.add_air(Some("U8Air"), 256);

        pilout
    }
}