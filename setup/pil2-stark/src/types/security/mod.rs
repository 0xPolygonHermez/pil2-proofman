pub mod regimes;
pub mod pcs;

pub fn goldilocks_safe_extension_field_size() -> f64 {
    let p = ((1u128 << 64) - (1 << 32) + 1) as f64;
    p * p * p
}
