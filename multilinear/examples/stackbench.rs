//! Prover cost of stacking in the WHIR PCS: C columns of 2^n committed and opened as today
//! (δ-batched Φ) vs one stacked column of C·2^n. Same code paths, same params.
//!   cargo run --release -p proofman-multilinear --example stackbench -- <n> <C>...
use std::time::Instant;

use proofman_fields::{Field, Goldilocks, PrimeField64};
use proofman_multilinear::{eq_evals, MlParams, MlPcs, MlTranscript, Whir};

type Ext = proofman_fields::CubicExtensionField<Goldilocks>;

fn col(len: usize, seed: u64) -> Vec<Goldilocks> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            Goldilocks::new(x % Goldilocks::ORDER_U64)
        })
        .collect()
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let n = args[0];
    let params = MlParams { log_blowup: 1, n_queries: 64, ..MlParams::default() };
    println!("n=2^{n} blowup 2^{} fold k={:?}", params.log_blowup, params.whir_fold_schedule);
    for &c in &args[1..] {
        let lc = c.trailing_zeros() as usize;
        assert!(c.is_power_of_two());
        let cols: Vec<Vec<Goldilocks>> = (0..c).map(|i| col(1 << n, i as u64 + 1)).collect();
        let refs: Vec<&[Goldilocks]> = cols.iter().map(|v| v.as_slice()).collect();
        let point: Vec<Ext> = (0..n + lc).map(|i| Ext::from_base(Goldilocks::new(1000 + i as u64))).collect();

        // --- Today: C columns, δ-batched Φ of 2^n.
        let t = Instant::now();
        let com = Whir::commit(&refs, &params);
        let t_commit = ms(t);
        let t = Instant::now();
        let delta = Ext::from_base(Goldilocks::new(7));
        let mut phi = vec![Ext::ZERO; 1 << n];
        let mut d = Ext::ONE;
        for cv in &cols {
            for (o, &v) in phi.iter_mut().zip(cv.iter()) {
                *o += d * Ext::from_base(v);
            }
            d *= delta;
        }
        let t_batch = ms(t);
        let t = Instant::now();
        let mut tr = MlTranscript::new(params.hash);
        let _ = Whir::open(&params, &mut tr, phi, &point[..n], &[&com]);
        let t_open = ms(t);
        drop(com);

        // --- Stacked: one column of C·2^n (column index = high variables).
        let stacked: Vec<Goldilocks> = cols.concat();
        drop(cols);
        let t = Instant::now();
        let scom = Whir::commit(&[stacked.as_slice()], &params);
        let s_commit = ms(t);
        let t = Instant::now();
        let sphi: Vec<Ext> = stacked.iter().map(|&v| Ext::from_base(v)).collect();
        let s_batch = ms(t);
        let t = Instant::now();
        let mut tr = MlTranscript::new(params.hash);
        let _ = Whir::open(&params, &mut tr, sphi, &point, &[&scom]);
        let s_open = ms(t);
        let _ = eq_evals(&point[..1]);

        // --- No stacking, first fold of 1 bit (verifier opens 2 values per column).
        let p1 = MlParams { whir_fold_schedule: vec![1, 4, 4, 4], ..params.clone() };
        let cols1: Vec<Vec<Goldilocks>> = (0..c).map(|i| stacked[i << n..(i + 1) << n].to_vec()).collect();
        let refs1: Vec<&[Goldilocks]> = cols1.iter().map(|v| v.as_slice()).collect();
        let t = Instant::now();
        let com1 = Whir::commit(&refs1, &p1);
        let k_commit = ms(t);
        let mut phi1 = vec![Ext::ZERO; 1 << n];
        let mut d = Ext::ONE;
        let t = Instant::now();
        for cv in &cols1 {
            for (o, &v) in phi1.iter_mut().zip(cv.iter()) {
                *o += d * Ext::from_base(v);
            }
            d *= delta;
        }
        let k_batch = ms(t);
        let t = Instant::now();
        let mut tr = MlTranscript::new(p1.hash);
        let _ = Whir::open(&p1, &mut tr, phi1, &point[..n], &[&com1]);
        let k_open = ms(t);
        let k1 = k_commit + k_batch + k_open;
        println!("C={c:>4} | k1=1, no stacking: commit {k_commit:8.0} batch {k_batch:6.0} open {k_open:7.0} = {k1:8.0} ms");

        let base = t_commit + t_batch + t_open;
        let st = s_commit + s_batch + s_open;
        println!(
            "C={c:>4} | today: commit {t_commit:8.0} batch {t_batch:6.0} open {t_open:7.0} = {base:8.0} ms | \
             stacked: commit {s_commit:8.0} lift {s_batch:6.0} open {s_open:7.0} = {st:8.0} ms | ratio {:.2}",
            st / base
        );
    }
}
