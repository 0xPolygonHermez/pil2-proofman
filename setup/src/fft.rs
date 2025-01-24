use crate::f3g::F3g;
use num_bigint::BigUint;
use num_traits::{One, Zero};
use std::cmp;

#[derive(Debug)]
pub struct FFT {
    pub field: F3g,
    pub w: Vec<BigUint>,
    pub wi: Vec<BigUint>,
    pub roots: Vec<Vec<BigUint>>,
}

impl FFT {
    pub fn new(field: F3g, w0: Option<BigUint>) -> Self {
        let mut fft = FFT {
            field: field.clone(),
            w: vec![BigUint::zero(); (field.s + 1) as usize],
            wi: vec![BigUint::zero(); (field.s + 1) as usize],
            roots: vec![vec![]; (field.s + 1) as usize],
        };

        let mut nqr = field.one.clone();
        while field.exp(&nqr, &field.half) == field.one {
            nqr = field.add(&nqr, &field.one);
        }

        fft.w[field.s as usize] = field.exp(&nqr, &field.t);
        if let Some(w0) = w0 {
            fft.w[field.s as usize] = w0;
        }
        fft.wi[field.s as usize] = field.inv(&fft.w[field.s as usize]);

        for n in (0..field.s as usize).rev() {
            fft.w[n] = field.square(&fft.w[n + 1]);
            fft.wi[n] = field.square(&fft.wi[n + 1]);
        }

        fft.set_roots(cmp::min(field.s as usize, 15));
        fft
    }

    fn set_roots(&mut self, n: usize) {
        for i in (0..=n).rev() {
            if self.roots[i].is_empty() {
                let mut r = self.field.one.clone();
                let nroots = 1 << i;
                self.roots[i] = vec![BigUint::zero(); nroots];
                for j in 0..nroots {
                    self.roots[i][j] = r.clone();
                    r = self.field.mul(&r, &self.w[i]);
                }
            }
        }
    }

    pub fn fft(&mut self, p: &[BigUint]) -> Result<Vec<BigUint>, String> {
        if p.is_empty() {
            return Ok(vec![]);
        }
        if !p.len().is_power_of_two() {
            return Err("Input size must be a power of 2".to_string());
        }

        let bits = (p.len() as f64).log2().ceil() as usize;
        self.set_roots(bits);

        let n = 1 << bits;
        let mut buff = vec![BigUint::zero(); n];
        for (i, val) in p.iter().enumerate() {
            let r = FFT::bit_reverse(i, bits);
            buff[r] = val.clone();
        }

        for s in 1..=bits {
            let m = 1 << s;
            let mdiv2 = m >> 1;
            let winc = &self.roots[s][1];
            for k in (0..n).step_by(m) {
                let mut w = self.field.one.clone();
                for j in 0..mdiv2 {
                    let t = self.field.mul(&w, &buff[k + j + mdiv2]);
                    let u = buff[k + j].clone();
                    buff[k + j] = self.field.add(&u, &t);
                    buff[k + j + mdiv2] = self.field.sub(&u, &t);
                    w = self.field.mul(&w, winc);
                }
            }
        }

        Ok(buff)
    }

    pub fn ifft(&mut self, p: &[BigUint]) -> Result<Vec<BigUint>, String> {
        let q = self.fft(p)?;
        let n = p.len();
        let n_inv = self.field.inv(&BigUint::from(n as u64));
        let mut res = vec![BigUint::zero(); q.len()];
        for i in 0..n {
            res[(n - i) % n] = self.field.mul(&q[i], &n_inv);
        }
        Ok(res)
    }

    fn bit_reverse(mut x: usize, n_bits: usize) -> usize {
        let mut result = 0;
        for _ in 0..n_bits {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        result
    }
}
