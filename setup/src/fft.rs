use num_bigint::BigUint;
use num_traits::{One, Zero};
use std::collections::HashMap;

use crate::f3g::F3g;

pub struct FFT {
    pub w: Vec<BigUint>,                     // Roots of unity
    pub wi: Vec<BigUint>,                    // Inverse roots of unity
    pub roots: HashMap<usize, Vec<BigUint>>, // Cached roots for each level
    pub field: F3g,                          // Field arithmetic
}

impl FFT {
    pub fn new(field: F3g, w0: Option<BigUint>) -> Self {
        let mut w = vec![BigUint::zero(); field.s as usize + 1];
        let mut wi = vec![BigUint::zero(); field.s as usize + 1];

        let mut nqr = field.one.clone();
        while field.exp(&nqr, &field.half) == field.one {
            nqr = field.add(&nqr, &field.one);
        }

        let rem = field.t.clone();
        let s = field.s as usize;

        w[s] = field.exp(&nqr, &rem);
        if let Some(ref w0) = w0 {
            w[s] = w0.clone();
        }
        wi[s] = field.inv(&w[s]);

        for n in (0..s).rev() {
            w[n] = field.square(&w[n + 1]);
            wi[n] = field.square(&wi[n + 1]);
        }

        let mut fft = FFT { w, wi, roots: HashMap::new(), field };
        fft.set_roots(s.min(15));
        fft
    }

    fn set_roots(&mut self, n: usize) {
        for i in (0..=n).rev() {
            if self.roots.contains_key(&i) {
                continue;
            }

            let mut r = self.field.one.clone();
            let nroots = 1 << i;
            let mut roots_i = Vec::with_capacity(nroots);
            for _ in 0..nroots {
                roots_i.push(r.clone());
                r = self.field.mul(&r, &self.w[i]);
            }
            self.roots.insert(i, roots_i);
        }
    }

    pub fn fft(&mut self, p: &[BigUint]) -> Vec<BigUint> {
        let n = p.len();
        if n <= 1 {
            return p.to_vec();
        }

        let bits = (n as f64).log2().ceil() as usize;
        self.set_roots(bits);

        if n != (1 << bits) {
            panic!("Size must be a power of 2");
        }

        let mut buff = vec![BigUint::zero(); n];
        for (i, &val) in p.iter().enumerate() {
            let r = Self::rev(i, bits);
            buff[r] = val.clone();
        }

        for s in 1..=bits {
            let m = 1 << s;
            let mdiv2 = m >> 1;
            let winc = &self.roots[&s][1];
            for k in (0..n).step_by(m) {
                let mut w = BigUint::one();
                for j in 0..mdiv2 {
                    let t = self.field.mul(&w, &buff[k + j + mdiv2]);
                    let u = buff[k + j].clone();
                    buff[k + j] = self.field.add(&u, &t);
                    buff[k + j + mdiv2] = self.field.sub(&u, &t);
                    w = self.field.mul(&w, winc);
                }
            }
        }

        buff
    }

    pub fn ifft(&mut self, p: &[BigUint]) -> Vec<BigUint> {
        let n = p.len();
        let q = self.fft(p);
        let n_inv = self.field.inv(&BigUint::from(n));

        let mut res = vec![BigUint::zero(); n];
        for i in 0..n {
            res[(n - i) % n] = self.field.mul(&q[i], &n_inv);
        }

        res
    }

    fn rev(x: usize, n_bits: usize) -> usize {
        let mut x = x as u32;
        x = (x >> 1) & 0x55555555 | (x & 0x55555555) << 1;
        x = (x >> 2) & 0x33333333 | (x & 0x33333333) << 2;
        x = (x >> 4) & 0x0F0F0F0F | (x & 0x0F0F0F0F) << 4;
        x = (x >> 8) & 0x00FF00FF | (x & 0x00FF00FF) << 8;
        x = (x >> 16) | (x << 16);
        (x as usize) >> (32 - n_bits)
    }
}
