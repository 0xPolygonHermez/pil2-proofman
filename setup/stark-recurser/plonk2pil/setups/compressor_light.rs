//! Compressor Light setup — direct port of compressor_light.setup.js.
//! 22 committed pols, 22 S cols, 16 C polys, 14 rows/Poseidon, 2 CMul/row, 2 rows/FFT4.
//! Key differences: no halfRows, offset field on partialRows, maxUsed=7 for regular plonk rows.

use super::{gen_pil_str, PilTemplateParams};
use super::super::r1cs::to_plonk::{ckey, filter_fft4_gate_uses, filter_gate_uses, get_custom_gates_info, r1cs2plonk};
use super::super::r1cs::types::{PlonkOptions, R1csFile, SetupResult};
use super::super::utils::{build_fixed_pols, build_s_polynomials, log2, mulp};
use std::collections::HashMap;

const COMMITTED_POLS: usize = 22;
const N_COLS: usize = 22;
const N_C_POLYS: usize = 16;
const POSEIDON_ROWS: usize = 14;
const CMUL_PER_ROW: usize = 2;

// (row, offset, n_used, max_used)
type PR = (usize, usize, usize, usize);

pub fn light_compressor(r1cs: &R1csFile, options: &PlonkOptions) -> SetupResult {
    let (plonk_constraints, plonk_additions) = r1cs2plonk(r1cs);
    tracing::info!("Number of plonk constraints: {}", plonk_constraints.len());

    let mut cgi = get_custom_gates_info(r1cs);
    let n_poseidon = cgi.n_poseidon12 + cgi.n_cust_poseidon12;
    let n_cmul_rows = cgi.n_cmul.div_ceil(CMUL_PER_ROW);
    let n_poseidon_rows = n_poseidon * POSEIDON_ROWS;
    let n_fft4_rows = cgi.n_fft4 * 2; // 2 rows per FFT4 gate
    let n_ev_pol4_rows = cgi.n_ev_pol4;
    let n_tree_sel4_rows = cgi.n_tree_selector4;
    let n_sel_val1_rows = cgi.n_select_val1;

    // twoExtraConstraints = 12 rows per Poseidon gate (r+1 through r+12)
    let two_count = 12 * n_poseidon;

    cgi.n_plonk_rows = {
        let mut partial: HashMap<String, (usize, usize)> = HashMap::new();
        let mut two = two_count;
        let mut rows = 0usize;
        for c in &plonk_constraints {
            let k = ckey(c);
            let remove = if let Some(pr) = partial.get_mut(&k) {
                pr.0 += 1;
                pr.0 == pr.1
            } else {
                false
            };
            if remove {
                partial.remove(&k);
                continue;
            }
            if partial.contains_key(&k) {
                continue;
            }
            // halfRows is always empty in this variant
            if two > 0 {
                two -= 1;
                partial.insert(k, (1, 2));
            } else {
                partial.insert(k.clone(), (1, 7));
                rows += 1;
            }
        }
        rows
    };

    let n_used = cgi.n_plonk_rows
        + n_cmul_rows
        + n_poseidon_rows
        + n_fft4_rows
        + n_ev_pol4_rows
        + n_tree_sel4_rows
        + n_sel_val1_rows;

    let n_bits = if n_used <= 1 { 1 } else { log2((n_used - 1) as u32) as usize + 1 };
    let n = 1usize << n_bits;
    let n_publics = r1cs.header.n_outputs + r1cs.header.n_pub_inputs;
    let max_degree = options.max_constraint_degree.unwrap_or(8);
    let airgroup_name = options.airgroup_name.clone().unwrap_or_else(|| format!("Compressor{:x}", n_used as u64));

    let pil_str = gen_pil_str(&PilTemplateParams {
        template_file: "compressor_light",
        template_name: "CompressorLight",
        namespace_name: &airgroup_name,
        n_bits,
        n_publics,
        max_constraint_degree: max_degree,
        n_plonk_rows: cgi.n_plonk_rows,
        n_poseidon_compressor: cgi.n_cust_poseidon12,
        n_poseidon_sponge: cgi.n_poseidon12,
        n_cmul_rows,
        n_ev_pol4: cgi.n_ev_pol4,
        n_fft4: cgi.n_fft4,
        n_tree_selector4: cgi.n_tree_selector4,
        n_select_val1: cgi.n_select_val1,
    });

    tracing::info!("NUsed: {}, nBits: {}, N: {}", n_used, n_bits, n);

    let mut s_map: Vec<Vec<u32>> = (0..COMMITTED_POLS).map(|_| vec![0u32; n]).collect();
    // 16 C polynomials (only C[0..4] are set in non-Poseidon rows)
    let mut cv: Vec<Vec<u64>> = (0..N_C_POLYS).map(|_| vec![0u64; n]).collect();

    let mut two_extra: Vec<usize> = Vec::new();

    let poseidon_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.poseidon12_id);
    let poseidon_cust_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.cust_poseidon12_id);
    let cmul_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.cmul_id);
    let fft4_uses = filter_fft4_gate_uses(&r1cs.custom_gates_uses, &cgi.fft4_parameters);
    let ev_pol4_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.ev_pol4_id);
    let tree_sel4_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.tree_selector4_id);
    let sel_val1_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.select_val1_id);

    let mut r = 0usize;

    // ── Poseidon sponge (14 rows: input,r0,r1,r2,r3,r4,im1,r15,im2,r26,r27,r28,r29,out) ─
    tracing::info!("Processing {} poseidon gates...", poseidon_uses.len());
    for cgu in &poseidon_uses {
        assert_eq!(cgu.signals.len(), 14 * 16);
        let s = &cgu.signals;
        let (input, round0, round1, round2, round3, round4) =
            (&s[0..16], &s[16..32], &s[32..48], &s[48..64], &s[64..80], &s[80..96]);
        let (im1, round15, im2) = (&s[96..112], &s[112..128], &s[128..144]);
        let (round26, round27, round28, round29, output) =
            (&s[144..160], &s[160..176], &s[176..192], &s[192..208], &s[208..224]);
        for k in 0..16 {
            s_map[k][r] = input[k] as u32;
            s_map[k][r + 1] = round0[k] as u32;
            s_map[k][r + 2] = round1[k] as u32;
            s_map[k][r + 3] = round2[k] as u32;
            s_map[k][r + 4] = round3[k] as u32;
            s_map[k][r + 5] = round4[k] as u32;
            s_map[k][r + 7] = round15[k] as u32;
            s_map[k][r + 9] = round26[k] as u32;
            s_map[k][r + 10] = round27[k] as u32;
            s_map[k][r + 11] = round28[k] as u32;
            s_map[k][r + 12] = round29[k] as u32;
            s_map[k][r + 13] = output[k] as u32;
        }
        for k in 0..11 {
            s_map[k][r + 6] = im1[k] as u32;
            s_map[k][r + 8] = im2[k] as u32;
        }
        // Rows r+1 through r+12: extra plonk slots (C[0..4] set when constraint placed)
        for off in 1..=12 {
            two_extra.push(r + off);
        }
        r += 14;
    }
    assert_eq!(r, 14 * poseidon_uses.len());

    // ── Poseidon custom / compressor (14 rows) ────────────────────────────────
    tracing::info!("Processing {} poseidon custom gates...", poseidon_cust_uses.len());
    for cgu in &poseidon_cust_uses {
        assert_eq!(cgu.signals.len(), 14 * 16 + 2);
        let s = &cgu.signals;
        let (input, fb, sb) = (&s[0..16], s[16], s[17]);
        let (round0, round1, round2, round3, round4) = (&s[18..34], &s[34..50], &s[50..66], &s[66..82], &s[82..98]);
        let (im1, round15, im2) = (&s[98..114], &s[114..130], &s[130..146]);
        let (round26, round27, round28, round29, output) =
            (&s[146..162], &s[162..178], &s[178..194], &s[194..210], &s[210..226]);
        for k in 0..16 {
            s_map[k][r] = input[k] as u32;
            s_map[k][r + 1] = round0[k] as u32;
            s_map[k][r + 2] = round1[k] as u32;
            s_map[k][r + 3] = round2[k] as u32;
            s_map[k][r + 4] = round3[k] as u32;
            s_map[k][r + 5] = round4[k] as u32;
            s_map[k][r + 7] = round15[k] as u32;
            s_map[k][r + 9] = round26[k] as u32;
            s_map[k][r + 10] = round27[k] as u32;
            s_map[k][r + 11] = round28[k] as u32;
            s_map[k][r + 12] = round29[k] as u32;
            s_map[k][r + 13] = output[k] as u32;
        }
        s_map[16][r] = fb as u32;
        s_map[17][r] = sb as u32;
        for k in 0..11 {
            s_map[k][r + 6] = im1[k] as u32;
            s_map[k][r + 8] = im2[k] as u32;
        }
        for off in 1..=12 {
            two_extra.push(r + off);
        }
        r += 14;
    }
    assert_eq!(r, 14 * poseidon_uses.len() + 14 * poseidon_cust_uses.len());

    // ── CMul (2/row) ──────────────────────────────────────────────────────────
    tracing::info!("Processing {} cmul gates...", cmul_uses.len());
    let mut cmul_row: i64 = -1;
    let mut cmul_used = 0usize;
    for cgu in &cmul_uses {
        assert_eq!(cgu.signals.len(), 9);
        if cmul_row >= 0 {
            let row = cmul_row as usize;
            for (i, item) in s_map[9 * cmul_used..].iter_mut().enumerate().take(9) {
                item[row] = cgu.signals[i] as u32;
            }
            cmul_used += 1;
            if cmul_used == CMUL_PER_ROW {
                cmul_row = -1;
                cmul_used = 0;
            }
        } else {
            for (i, item) in s_map.iter_mut().enumerate().take(9) {
                item[r] = cgu.signals[i] as u32;
            }
            for item in cv.iter_mut().take(5) {
                item[r] = 0;
            }
            cmul_row = r as i64;
            cmul_used = 1;
            r += 1;
        }
    }
    assert_eq!(r, 14 * poseidon_uses.len() + 14 * poseidon_cust_uses.len() + n_cmul_rows);

    // ── EvPol4 ────────────────────────────────────────────────────────────────
    tracing::info!("Processing {} evPol4 gates...", ev_pol4_uses.len());
    for cgu in &ev_pol4_uses {
        for (i, item) in s_map.iter_mut().enumerate().take(21) {
            item[r] = cgu.signals[i] as u32;
        }
        for item in cv.iter_mut().take(5) {
            item[r] = 0;
        }
        r += 1;
    }

    // ── FFT4 (2 rows per gate) ────────────────────────────────────────────────
    tracing::info!("Processing {} fft4 gates...", fft4_uses.len());
    for cgu in &fft4_uses {
        // signals[0..11] → row r, signals[12..23] → row r+1
        for (i, row) in s_map.iter_mut().enumerate().take(12) {
            row[r] = cgu.signals[i] as u32;
            row[r + 1] = cgu.signals[i + 12] as u32;
        }
        let p = cgi.fft4_parameters.get(&cgu.id).expect("FFT4 params");
        let (fft_type, scale, first_w, inc_w) = (p[3], p[2], p[0], p[1]);
        let fw2 = mulp(first_w, first_w);
        if fft_type == 4 {
            cv[0][r] = scale; // C[0][r]
            cv[1][r] = mulp(scale, fw2); // C[1][r]
            cv[2][r] = mulp(scale, first_w); // C[2][r]
            cv[3][r] = mulp(mulp(scale, first_w), fw2); // C[3][r]
            cv[4][r] = mulp(mulp(scale, first_w), inc_w); // C[4][r]
            cv[0][r + 1] = mulp(mulp(mulp(scale, first_w), fw2), inc_w); // C[0][r+1]
            cv[1][r + 1] = 0;
            cv[2][r + 1] = 0;
            cv[3][r + 1] = 0; // C[1..3][r+1]
        } else if fft_type == 2 {
            for item in cv.iter_mut().take(5) {
                item[r] = 0;
            } // C[0..4][r] = 0
            cv[0][r + 1] = 0; // C[0][r+1]
            cv[1][r + 1] = scale; // C[1][r+1]
            cv[2][r + 1] = mulp(scale, first_w); // C[2][r+1]
            cv[3][r + 1] = mulp(mulp(scale, first_w), inc_w); // C[3][r+1]
        } else {
            panic!("Invalid FFT4 type: {}", fft_type);
        }
        r += 2;
    }

    // ── TreeSelector4 ─────────────────────────────────────────────────────────
    tracing::info!("Processing {} treeSelector4 gates...", tree_sel4_uses.len());
    for cgu in &tree_sel4_uses {
        assert_eq!(cgu.signals.len(), 17);
        for (i, item) in s_map.iter_mut().enumerate().take(17) {
            item[r] = cgu.signals[i] as u32;
        }
        for item in cv.iter_mut().take(5) {
            item[r] = 0;
        }
        r += 1;
    }

    // ── SelectVal1 ────────────────────────────────────────────────────────────
    tracing::info!("Processing {} selectVal1 gates...", sel_val1_uses.len());
    for cgu in &sel_val1_uses {
        assert_eq!(cgu.signals.len(), 22);
        for (i, item) in s_map.iter_mut().enumerate().take(22) {
            item[r] = cgu.signals[i] as u32;
        }
        for item in cv.iter_mut().take(5) {
            item[r] = 0;
        }
        r += 1;
    }

    assert_eq!(r, n_used - cgi.n_plonk_rows, "pre-plonk row count mismatch");

    // ── Plonk constraints (no halfRows, offset-based partial rows) ────────────
    tracing::info!("Placing {} plonk constraints...", plonk_constraints.len());
    let mut partial: HashMap<String, PR> = HashMap::new();

    for (idx, c) in plonk_constraints.iter().enumerate() {
        if idx % 10_000 == 0 {
            tracing::debug!("constraint {}/{}", idx, plonk_constraints.len());
        }
        let k = ckey(c);

        let in_partial = partial.contains_key(&k);
        if in_partial {
            let pr = partial.get_mut(&k).unwrap();
            let col = pr.1 + pr.2 * 3; // offset + nUsed*3
            s_map[col][pr.0] = c[0] as u32;
            s_map[col + 1][pr.0] = c[1] as u32;
            s_map[col + 2][pr.0] = c[2] as u32;
            pr.2 += 1;
            if pr.2 == pr.3 {
                partial.remove(&k);
            }
        } else if !two_extra.is_empty() {
            let row = two_extra.remove(0);
            cv[0][row] = c[3];
            cv[1][row] = c[4];
            cv[2][row] = c[5];
            cv[3][row] = c[6];
            cv[4][row] = c[7];
            // Pre-fill both slots at offset 16 (cols 16-21)
            for i in 0..2 {
                s_map[16 + 3 * i][row] = c[0] as u32;
                s_map[16 + 3 * i + 1][row] = c[1] as u32;
                s_map[16 + 3 * i + 2][row] = c[2] as u32;
            }
            partial.insert(k, (row, 16, 1, 2));
        } else {
            cv[0][r] = c[3];
            cv[1][r] = c[4];
            cv[2][r] = c[5];
            cv[3][r] = c[6];
            cv[4][r] = c[7];
            // Pre-fill all 7 slots at offset 0 (cols 0-20)
            for i in 0..7 {
                s_map[3 * i][r] = c[0] as u32;
                s_map[3 * i + 1][r] = c[1] as u32;
                s_map[3 * i + 2][r] = c[2] as u32;
            }
            partial.insert(k, (r, 0, 1, 7));
            r += 1;
        }
    }
    assert_eq!(r, n_used, "row count mismatch: {} != {}", r, n_used);

    // ── S polynomials ─────────────────────────────────────────────────────────
    let sv = build_s_polynomials(N_COLS, n, n_bits, r, &s_map);
    let fixed_pols = build_fixed_pols(&airgroup_name, &cv, &sv);

    SetupResult {
        fixed_pols,
        pil_str,
        n_bits,
        n_used,
        s_map,
        plonk_additions,
        airgroup_name: airgroup_name.clone(),
        air_name: airgroup_name,
    }
}
