use std::sync::{Arc, RwLock};

use proofman_common::{AirInstance, BufferPool, FromTrace, ProofCtx, ProofmanResult, SetupCtx};
use proofman_witness::WitnessComponent;
use proofman_fields::PrimeField64;

use crate::pil_helpers::Sha2Trace;

use super::{
    sha2_constants::{
        CARRY_ROWS, CHR_ROWS, CHUNK_BITS, CLOCKS, CLOCKS_LOAD_INPUT, CLOCKS_LOAD_STATE, LANES, NCHUNK, NUM_ROUNDS,
        P3_ROWS, RANGE_ROWS, RC, SLOT_BITS,
    },
    sha2_helpers::{
        big_sigma0, big_sigma1, bit, ch, compress, maj, random_sha2_input, self_test, slice, small_sigma0, small_sigma1,
    },
};

/// Tag of each parity/maj lookup, in the order Σ0, Σ1, maj, σ0, σ1 (1 selects maj).
const P3_TAGS: [usize; 5] = [0, 0, 1, 0, 0];

pub struct Sha2Air {
    instance_ids: RwLock<Vec<usize>>,
}

impl Sha2Air {
    pub fn new() -> Arc<Self> {
        self_test();
        Arc::new(Self { instance_ids: RwLock::new(Vec::new()) })
    }

    /// Fill the whole trace and wrap it as an air instance.
    #[allow(clippy::needless_range_loop)]
    fn compute_witness_inner<F: PrimeField64>(
        &self,
        buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<AirInstance<F>> {
        let mut trace = Sha2Trace::<F>::new_from_vec_zeroes(buffer_pool.take_buffer())?;
        let n = trace.num_rows();
        let num_ops = n / CLOCKS;
        let used = num_ops * CLOCKS;

        tracing::debug!(
            "··· Creating SHA2 instance with {} inputs ({} cycles x {} lanes) [{} / {} rows used {:.2}%]",
            num_ops * LANES,
            num_ops,
            LANES,
            used,
            n,
            used as f64 / n as f64 * 100.0
        );

        // 1] The words every row holds, lane by lane: the load-state rows [d,c,b,a] / [h,g,f,e], then
        //    round t on row 4+t with (a_t, e_t, W_t). The trailing rows no cycle covers stay zero.
        let mut wa = vec![[0u32; LANES]; n];
        let mut we = vec![[0u32; LANES]; n];
        let mut ww = vec![[0u32; LANES]; n];
        for op in 0..num_ops {
            let base = op * CLOCKS;
            for k in 0..LANES {
                let (state, block) = random_sha2_input((op * LANES + k) as u64);
                let (av, ev, w, _) = compress(&state, &block);
                for i in 0..CLOCKS_LOAD_STATE {
                    wa[base + i][k] = state[3 - i];
                    we[base + i][k] = state[7 - i];
                }
                for t in 0..NUM_ROUNDS {
                    wa[base + CLOCKS_LOAD_STATE + t][k] = av[t];
                    we[base + CLOCKS_LOAD_STATE + t][k] = ev[t];
                    ww[base + CLOCKS_LOAD_STATE + t][k] = w[t];
                }
            }
        }

        // 2] Every row, reading the rows before it cyclically, as the constraints do: the cells, the
        //    table outputs, the carries and the lookup multiplicities. The lookups run on every row.
        let mut m_p3 = vec![0u64; P3_ROWS];
        let mut m_chr = vec![0u64; CHR_ROWS];
        let mut m_range = vec![0u64; RANGE_ROWS];
        let mut m_carry = vec![0u64; CARRY_ROWS];
        let back = |r: usize, j: usize| (r + n - j) % n;
        let chunk = |x: u32, h: usize| ((x >> (CHUNK_BITS * h)) & 0xff) as u64;

        for r in 0..n {
            let (a1, a2, a3, a4) = (wa[back(r, 1)], wa[back(r, 2)], wa[back(r, 3)], wa[back(r, 4)]);
            let (e1, e2, e3, e4) = (we[back(r, 1)], we[back(r, 2)], we[back(r, 3)], we[back(r, 4)]);
            let (w0, w2, w7, w15, w16) = (ww[r], ww[back(r, 2)], ww[back(r, 7)], ww[back(r, 15)], ww[back(r, 16)]);
            let f_bs0: [u32; LANES] = core::array::from_fn(|k| big_sigma0(a1[k]));
            let f_bs1: [u32; LANES] = core::array::from_fn(|k| big_sigma1(e1[k]));
            let f_mj: [u32; LANES] = core::array::from_fn(|k| maj(a1[k], a2[k], a3[k]));
            let f_ss0: [u32; LANES] = core::array::from_fn(|k| small_sigma0(w15[k]));
            let f_ss1: [u32; LANES] = core::array::from_fn(|k| small_sigma1(w2[k]));
            let f_ch: [u32; LANES] = core::array::from_fn(|k| ch(e1[k], e2[k], e3[k]));

            // The 3-bit sum each parity/maj lookup keys on, per lane and position
            let sum3 = |x: &[u32; LANES], k: usize, p: usize, rot: [usize; 3]| -> u64 {
                rot.iter().map(|&o| bit(x[k], (p + o) % 32)).sum()
            };
            let shr = |x: u32, p: usize, s: usize| if p + s < 32 { bit(x, p + s) } else { 0 };
            let key_sum = |f: usize, k: usize, p: usize| -> u64 {
                match f {
                    0 => sum3(&a1, k, p, [2, 13, 22]),
                    1 => sum3(&e1, k, p, [6, 11, 25]),
                    2 => bit(a1[k], p) + bit(a2[k], p) + bit(a3[k], p),
                    3 => bit(w15[k], (p + 7) % 32) + bit(w15[k], (p + 18) % 32) + shr(w15[k], p, 3),
                    _ => bit(w2[k], (p + 17) % 32) + bit(w2[k], (p + 19) % 32) + shr(w2[k], p, 10),
                }
            };
            let outs_of = [&f_bs0, &f_bs1, &f_mj, &f_ss0, &f_ss1];

            let row = &mut trace[r];
            for i in 0..32 {
                row.sa[i] = F::from_u64(slice(&wa[r], i));
                row.se[i] = F::from_u64(slice(&we[r], i));
                row.sw[i] = F::from_u64(slice(&w0, i));
            }

            for g in 0..16 {
                let s = CHUNK_BITS * (g / 4) + 2 * (g % 4);
                let mut outs = [0u64; 5];
                for f in 0..5 {
                    let (mut i0, mut i1, mut out) = (0usize, 0usize, 0u64);
                    for k in 0..LANES {
                        i0 += (key_sum(f, k, s) as usize) << (2 * k);
                        i1 += (key_sum(f, k, s + 1) as usize) << (2 * k);
                        out += (bit(outs_of[f][k], s) + 2 * bit(outs_of[f][k], s + 1)) << (SLOT_BITS * k);
                    }
                    m_p3[(P3_TAGS[f] << 20) + (i1 << 10) + i0] += 1;
                    outs[f] = out;
                }
                row.bs0[g] = F::from_u64(outs[0]);
                row.bs1[g] = F::from_u64(outs[1]);
                row.mj[g] = F::from_u64(outs[2]);
                row.ss0[g] = F::from_u64(outs[3]);
                row.ss1[g] = F::from_u64(outs[4]);
            }

            // ch over u = e + 2f + 4g, carrying the range check of this row's a cell
            for p in 0..32 {
                let (mut iu, mut ir) = (0usize, 0usize);
                for k in 0..LANES {
                    let u = bit(e1[k], p) + 2 * bit(e2[k], p) + 4 * bit(e3[k], p);
                    iu += (u as usize) << (3 * k);
                    ir += (bit(wa[r][k], p) as usize) << k;
                }
                m_chr[(ir << 15) + iu] += 1;
                row.chv[p] = F::from_u64(slice(&f_ch, p));
            }

            // e and w range checks, four cells per lookup
            for i in (0..32).step_by(4) {
                for word in [&we[r], &w0] {
                    let idx: usize = (0..4)
                        .map(|j| (0..LANES).map(|k| (bit(word[k], i + j) as usize) << k).sum::<usize>() << (5 * j))
                        .sum();
                    m_range[idx] += 1;
                }
            }

            // Carries: the additions bind only on round rows (a, e) and schedule rows (w)
            let clock = if r < used { Some(r % CLOCKS) } else { None };
            let mix = matches!(clock, Some(c) if c >= CLOCKS_LOAD_STATE);
            let wcomp = matches!(clock, Some(c) if c >= CLOCKS_LOAD_STATE + CLOCKS_LOAD_INPUT);
            let kw = match clock {
                Some(c) if mix => RC[c - CLOCKS_LOAD_STATE],
                _ => 0,
            };
            let (mut ca, mut ce, mut cw) = ([[0u64; LANES]; NCHUNK], [[0u64; LANES]; NCHUNK], [[0u64; LANES]; NCHUNK]);
            for k in 0..LANES {
                let (mut cin_a, mut cin_e, mut cin_w) = (0u64, 0u64, 0u64);
                for h in 0..NCHUNK {
                    if mix {
                        let t1 =
                            chunk(e4[k], h) + chunk(f_bs1[k], h) + chunk(f_ch[k], h) + chunk(kw, h) + chunk(w0[k], h);
                        let sum_a = t1 + chunk(f_bs0[k], h) + chunk(f_mj[k], h) + cin_a;
                        let sum_e = chunk(a4[k], h) + t1 + cin_e;
                        debug_assert_eq!(sum_a & 0xff, chunk(wa[r][k], h));
                        debug_assert_eq!(sum_e & 0xff, chunk(we[r][k], h));
                        cin_a = sum_a >> CHUNK_BITS;
                        cin_e = sum_e >> CHUNK_BITS;
                        ca[h][k] = cin_a;
                        ce[h][k] = cin_e;
                    }
                    if wcomp {
                        let sum_w =
                            chunk(f_ss1[k], h) + chunk(w7[k], h) + chunk(f_ss0[k], h) + chunk(w16[k], h) + cin_w;
                        debug_assert_eq!(sum_w & 0xff, chunk(w0[k], h));
                        cin_w = sum_w >> CHUNK_BITS;
                        cw[h][k] = cin_w;
                    }
                }
            }
            for h in 0..NCHUNK {
                for (col, c) in [(&mut row.ca[h], &ca[h]), (&mut row.ce[h], &ce[h]), (&mut row.cw[h], &cw[h])] {
                    *col = F::from_u64((0..LANES).map(|k| c[k] << (SLOT_BITS * k)).sum());
                    m_carry[(0..LANES).rev().fold(0usize, |acc, k| acc * 7 + c[k] as usize)] += 1;
                }
            }
        }

        // 3] Multiplicities: table row R lives in block R / n at row R % n
        for local in 0..n {
            let row = &mut trace[local];
            for (b, m) in row.sha2_mul_p3.iter_mut().enumerate() {
                *m = F::from_u64(m_p3.get(b * n + local).copied().unwrap_or(0));
            }
            for (b, m) in row.sha2_mul_chr.iter_mut().enumerate() {
                *m = F::from_u64(m_chr.get(b * n + local).copied().unwrap_or(0));
            }
            for (b, m) in row.sha2_mul_range.iter_mut().enumerate() {
                *m = F::from_u64(m_range.get(b * n + local).copied().unwrap_or(0));
            }
            for (b, m) in row.sha2_mul_carry.iter_mut().enumerate() {
                *m = F::from_u64(m_carry.get(b * n + local).copied().unwrap_or(0));
            }
        }

        Ok(AirInstance::new_from_trace(FromTrace::new(&mut trace)))
    }
}

impl<F: PrimeField64> WitnessComponent<F> for Sha2Air {
    fn execute(
        &self,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        global_ids: &RwLock<Vec<usize>>,
    ) -> ProofmanResult<()> {
        let global_id = pctx.add_instance(Sha2Trace::<F>::AIRGROUP_ID, Sha2Trace::<F>::AIR_ID)?;
        *self.instance_ids.write().unwrap() = vec![global_id];
        global_ids.write().unwrap().push(global_id);
        Ok(())
    }

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        if stage != 1 {
            return Ok(());
        }

        let air_instance = self.compute_witness_inner::<F>(buffer_pool)?;
        pctx.add_air_instance(air_instance, instance_ids[0]);
        Ok(())
    }
}
