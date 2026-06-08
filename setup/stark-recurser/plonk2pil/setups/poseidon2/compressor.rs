//! Compressor setup — direct port of compressor_setup.js.
//! 52 committed pols, 36 S cols, 10 rows/Poseidon, 4 CMul/row.

use super::super::super::r1cs::to_plonk::{
    ckey, filter_fft4_gate_uses, filter_gate_uses, get_custom_gates_info, r1cs2plonk,
};
use super::super::super::r1cs::types::{PlonkOptions, R1csFile, SetupResult};
use super::super::super::utils::{build_fixed_pols, build_s_polynomials, log2, mulp};
use super::{gen_pil_str, PilTemplateParams};
use proofman_common::hash_family::GateRole;
use std::collections::HashMap;

const COMMITTED_POLS: usize = 52;
const N_COLS: usize = 36;
const POSEIDON_ROWS: usize = 10;
const COL_P: usize = 36;
const CMUL_PER_ROW: usize = 4;

fn rand_hex() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    format!("{:x}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64)
}

// (row, n_used, max_used)
type PR = (usize, usize, usize);

pub fn compressor(r1cs: &R1csFile, options: &PlonkOptions) -> SetupResult {
    let (plonk_constraints, plonk_additions) = r1cs2plonk(r1cs);
    tracing::info!("Number of plonk constraints: {}", plonk_constraints.len());

    let mut cgi = get_custom_gates_info(r1cs);
    let n_poseidon_sponge = cgi.n(GateRole::PoseidonSponge);
    let n_poseidon_compression = cgi.n(GateRole::PoseidonCompression);
    let n_poseidon = n_poseidon_sponge + n_poseidon_compression;
    let n_cmul_rows = cgi.n(GateRole::CMul).div_ceil(CMUL_PER_ROW);
    let n_poseidon_rows = n_poseidon * POSEIDON_ROWS;
    let n_fft4_rows = cgi.n(GateRole::Fft4);
    let n_ev_pol4_rows = cgi.n(GateRole::EvPol4);
    let n_tree_sel4_rows = cgi.n(GateRole::TreeSelector);
    let n_sel_val1_rows = cgi.n(GateRole::SelectVal1);

    // Row-count tiers (used to pre-compute n_plonk_rows)
    let twelve_count = n_poseidon * 8;
    let six_count = n_poseidon + n_tree_sel4_rows;
    let five_count = n_ev_pol4_rows;
    let four_count = n_poseidon + n_sel_val1_rows;

    cgi.n_plonk_rows = {
        let mut partial: HashMap<String, (usize, usize)> = HashMap::new(); // (n_used, max_used)
        let mut half: Vec<(usize, usize)> = Vec::new();
        let (mut twelve, mut six, mut five, mut four) = (twelve_count, six_count, five_count, four_count);
        let mut rows = 0usize;
        for c in &plonk_constraints {
            let k = ckey(c);
            if let Some(pr) = partial.get_mut(&k) {
                pr.0 += 1;
                if pr.0 == pr.1 {
                    partial.remove(&k);
                }
            } else if !half.is_empty() {
                let mut pr = half.remove(0);
                pr.0 += 1;
                partial.insert(k, pr);
            } else if twelve > 0 {
                twelve -= 1;
                partial.insert(k, (1, 6));
                half.push((6, 12));
            } else if six > 0 {
                six -= 1;
                partial.insert(k, (7, 12));
            } else if five > 0 {
                five -= 1;
                partial.insert(k, (8, 12));
            } else if four > 0 {
                four -= 1;
                partial.insert(k, (9, 12));
            } else {
                partial.insert(k.clone(), (1, 6));
                half.push((6, 12));
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
    let airgroup_name = options.airgroup_name.clone().unwrap_or_else(|| format!("Compressor{}", rand_hex()));

    let pil_str = gen_pil_str(&PilTemplateParams {
        template_file: "poseidon2/compressor",
        template_name: "Compressor",
        namespace_name: &airgroup_name,
        n_bits,
        n_publics,
        max_constraint_degree: 5,
        n_plonk_rows: cgi.n_plonk_rows,
        n_poseidon_compressor: n_poseidon_compression,
        n_poseidon_sponge,
        n_cmul_rows,
        n_ev_pol4: cgi.n(GateRole::EvPol4),
        n_fft4: cgi.n(GateRole::Fft4),
        n_tree_selector4: cgi.n(GateRole::TreeSelector),
        n_select_val1: cgi.n(GateRole::SelectVal1),
    });

    tracing::info!("NUsed: {}, nBits: {}, N: {}", n_used, n_bits, n);

    let mut s_map: Vec<Vec<u32>> = (0..COMMITTED_POLS).map(|_| vec![0u32; n]).collect();
    let mut cv: Vec<Vec<u64>> = (0..10).map(|_| vec![0u64; n]).collect();

    // Extra-constraint row queues
    let mut twelve_extra: Vec<usize> = Vec::new();
    let mut six_extra: Vec<usize> = Vec::new();
    let mut five_extra: Vec<usize> = Vec::new();
    let mut four_extra: Vec<usize> = Vec::new();

    let poseidon_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::PoseidonSponge));
    let poseidon_cust_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::PoseidonCompression));
    let cmul_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::CMul));
    let fft4_uses = filter_fft4_gate_uses(&r1cs.custom_gates_uses, &cgi.fft4_parameters);
    let ev_pol4_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::EvPol4));
    let tree_sel4_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::TreeSelector));
    let sel_val1_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::SelectVal1));

    let mut r = 0usize;

    // ── Poseidon sponge (10 rows) ─────────────────────────────────────────────
    tracing::info!("Processing {} poseidon gates...", poseidon_uses.len());
    for cgu in &poseidon_uses {
        assert_eq!(cgu.signals.len(), 14 * 16);
        let s = &cgu.signals;
        let (input, round0, round1, round2, round3, round4) =
            (&s[0..16], &s[16..32], &s[32..48], &s[48..64], &s[64..80], &s[80..96]);
        let (im1, _r15, im2) = (&s[96..112], &s[112..128], &s[128..144]);
        let (round26, round27, round28, round29, output) =
            (&s[144..160], &s[160..176], &s[176..192], &s[192..208], &s[208..224]);
        for i in 0..16 {
            s_map[i][r] = input[i] as u32;
            s_map[i + COL_P][r] = round0[i] as u32;
            s_map[i + COL_P][r + 1] = round1[i] as u32;
            s_map[i + COL_P][r + 2] = round2[i] as u32;
            s_map[i + COL_P][r + 3] = round3[i] as u32;
            s_map[i + COL_P][r + 4] = round4[i] as u32;
            s_map[i + COL_P][r + 6] = round26[i] as u32;
            s_map[i + COL_P][r + 7] = round27[i] as u32;
            s_map[i + COL_P][r + 8] = round28[i] as u32;
            s_map[i + COL_P][r + 9] = round29[i] as u32;
            s_map[i][r + 9] = output[i] as u32;
        }
        for i in 0..11 {
            s_map[i + COL_P][r + 5] = im1[i] as u32;
            if i < 5 {
                s_map[i + 47][r + 5] = im2[i] as u32;
            } else {
                s_map[(i - 5) + 18][r] = im2[i] as u32;
            }
        }
        for off in 0..10 {
            for item in cv.iter_mut() {
                item[r + off] = 0;
            }
        }
        four_extra.push(r);
        for off in 1..=8 {
            twelve_extra.push(r + off);
        }
        six_extra.push(r + 9);
        r += 10;
    }
    assert_eq!(r, 10 * poseidon_uses.len());

    // ── Poseidon custom / compressor (10 rows) ────────────────────────────────
    tracing::info!("Processing {} poseidon custom gates...", poseidon_cust_uses.len());
    for cgu in &poseidon_cust_uses {
        assert_eq!(cgu.signals.len(), 14 * 16 + 2);
        let s = &cgu.signals;
        let (input, fb, sb) = (&s[0..16], s[16], s[17]);
        let (round0, round1, round2, round3, round4) = (&s[18..34], &s[34..50], &s[50..66], &s[66..82], &s[82..98]);
        let (im1, _r15, im2) = (&s[98..114], &s[114..130], &s[130..146]);
        let (round26, round27, round28, round29, output) =
            (&s[146..162], &s[162..178], &s[178..194], &s[194..210], &s[210..226]);
        for i in 0..16 {
            s_map[i][r] = input[i] as u32;
            s_map[i + COL_P][r] = round0[i] as u32;
            s_map[i + COL_P][r + 1] = round1[i] as u32;
            s_map[i + COL_P][r + 2] = round2[i] as u32;
            s_map[i + COL_P][r + 3] = round3[i] as u32;
            s_map[i + COL_P][r + 4] = round4[i] as u32;
            s_map[i + COL_P][r + 6] = round26[i] as u32;
            s_map[i + COL_P][r + 7] = round27[i] as u32;
            s_map[i + COL_P][r + 8] = round28[i] as u32;
            s_map[i + COL_P][r + 9] = round29[i] as u32;
            s_map[i][r + 9] = output[i] as u32;
        }
        s_map[16][r] = fb as u32;
        s_map[17][r] = sb as u32;
        for i in 0..11 {
            s_map[i + COL_P][r + 5] = im1[i] as u32;
            if i < 5 {
                s_map[i + 47][r + 5] = im2[i] as u32;
            } else {
                s_map[(i - 5) + 18][r] = im2[i] as u32;
            }
        }
        for off in 0..10 {
            for item in cv.iter_mut() {
                item[r + off] = 0;
            }
        }
        four_extra.push(r);
        for off in 1..=8 {
            twelve_extra.push(r + off);
        }
        six_extra.push(r + 9);
        r += 10;
    }
    assert_eq!(r, 10 * poseidon_uses.len() + 10 * poseidon_cust_uses.len());

    // ── CMul (4/row) ──────────────────────────────────────────────────────────
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
            for item in cv.iter_mut() {
                item[r] = 0;
            }
            cmul_row = r as i64;
            cmul_used = 1;
            r += 1;
        }
    }
    assert_eq!(r, 10 * poseidon_uses.len() + 10 * poseidon_cust_uses.len() + n_cmul_rows);

    // ── EvPol4 ────────────────────────────────────────────────────────────────
    tracing::info!("Processing {} evPol4 gates...", ev_pol4_uses.len());
    for cgu in &ev_pol4_uses {
        for (i, item) in s_map.iter_mut().enumerate().take(21) {
            item[r] = cgu.signals[i] as u32;
        }
        for item in cv.iter_mut() {
            item[r] = 0;
        }
        five_extra.push(r);
        r += 1;
    }

    // ── FFT4 (1 row) ──────────────────────────────────────────────────────────
    tracing::info!("Processing {} fft4 gates...", fft4_uses.len());
    for cgu in &fft4_uses {
        for (i, item) in s_map.iter_mut().enumerate().take(24) {
            item[r] = cgu.signals[i] as u32;
        }
        let p = cgi.fft4_parameters.get(&cgu.id).expect("FFT4 params");
        let (fft_type, scale, first_w, inc_w) = (p[3], p[2], p[0], p[1]);
        let fw2 = mulp(first_w, first_w);
        if fft_type == 4 {
            cv[0][r] = scale;
            cv[1][r] = mulp(scale, fw2);
            cv[2][r] = mulp(scale, first_w);
            cv[3][r] = mulp(mulp(scale, first_w), fw2);
            cv[4][r] = mulp(mulp(scale, first_w), inc_w);
            cv[5][r] = mulp(mulp(mulp(scale, first_w), fw2), inc_w);
            for item in cv.iter_mut().skip(6) {
                item[r] = 0;
            }
        } else if fft_type == 2 {
            for item in cv.iter_mut().take(6) {
                item[r] = 0;
            }
            cv[6][r] = scale;
            cv[7][r] = mulp(scale, first_w);
            cv[8][r] = mulp(mulp(scale, first_w), inc_w);
            cv[9][r] = 0;
        } else {
            panic!("Invalid FFT4 type: {}", fft_type);
        }
        r += 1;
    }

    // ── TreeSelector4 ─────────────────────────────────────────────────────────
    tracing::info!("Processing {} treeSelector4 gates...", tree_sel4_uses.len());
    for cgu in &tree_sel4_uses {
        assert_eq!(cgu.signals.len(), 17);
        for (i, item) in s_map.iter_mut().enumerate().take(17) {
            item[r] = cgu.signals[i] as u32;
        }
        for item in cv.iter_mut() {
            item[r] = 0;
        }
        six_extra.push(r);
        r += 1;
    }

    // ── SelectVal1 ────────────────────────────────────────────────────────────
    tracing::info!("Processing {} selectVal1 gates...", sel_val1_uses.len());
    for cgu in &sel_val1_uses {
        assert_eq!(cgu.signals.len(), 22);
        for (i, item) in s_map.iter_mut().enumerate().take(22) {
            item[r] = cgu.signals[i] as u32;
        }
        for item in cv.iter_mut() {
            item[r] = 0;
        }
        four_extra.push(r);
        r += 1;
    }

    // ── Plonk constraints ─────────────────────────────────────────────────────
    tracing::info!("Placing {} plonk constraints...", plonk_constraints.len());
    let mut partial: HashMap<String, PR> = HashMap::new(); // (row, n_used, max_used)
    let mut half: Vec<PR> = Vec::new();

    for (idx, c) in plonk_constraints.iter().enumerate() {
        if idx % 10_000 == 0 {
            tracing::debug!("constraint {}/{}", idx, plonk_constraints.len());
        }
        let k = ckey(c);

        if let Some(pr) = partial.get_mut(&k) {
            let n = pr.1;
            s_map[n * 3][pr.0] = c[0] as u32;
            s_map[n * 3 + 1][pr.0] = c[1] as u32;
            s_map[n * 3 + 2][pr.0] = c[2] as u32;
            pr.1 += 1;
            if pr.1 == pr.2 {
                partial.remove(&k);
            }
        } else if !half.is_empty() {
            let mut pr = half.remove(0);
            cv[5][pr.0] = c[3];
            cv[6][pr.0] = c[4];
            cv[7][pr.0] = c[5];
            cv[8][pr.0] = c[6];
            cv[9][pr.0] = c[7];
            for i in pr.1..pr.2 {
                s_map[3 * i][pr.0] = c[0] as u32;
                s_map[3 * i + 1][pr.0] = c[1] as u32;
                s_map[3 * i + 2][pr.0] = c[2] as u32;
            }
            pr.1 += 1;
            partial.insert(k, pr);
        } else if !twelve_extra.is_empty() {
            let row = twelve_extra.remove(0);
            cv[0][row] = c[3];
            cv[1][row] = c[4];
            cv[2][row] = c[5];
            cv[3][row] = c[6];
            cv[4][row] = c[7];
            for i in 0..6 {
                s_map[3 * i][row] = c[0] as u32;
                s_map[3 * i + 1][row] = c[1] as u32;
                s_map[3 * i + 2][row] = c[2] as u32;
            }
            partial.insert(k.clone(), (row, 1, 6));
            half.push((row, 6, 12));
        } else if !six_extra.is_empty() {
            let row = six_extra.remove(0);
            cv[5][row] = c[3];
            cv[6][row] = c[4];
            cv[7][row] = c[5];
            cv[8][row] = c[6];
            cv[9][row] = c[7];
            for i in 6..12 {
                s_map[3 * i][row] = c[0] as u32;
                s_map[3 * i + 1][row] = c[1] as u32;
                s_map[3 * i + 2][row] = c[2] as u32;
            }
            partial.insert(k, (row, 7, 12));
        } else if !five_extra.is_empty() {
            let row = five_extra.remove(0);
            cv[5][row] = c[3];
            cv[6][row] = c[4];
            cv[7][row] = c[5];
            cv[8][row] = c[6];
            cv[9][row] = c[7];
            for i in 7..12 {
                s_map[3 * i][row] = c[0] as u32;
                s_map[3 * i + 1][row] = c[1] as u32;
                s_map[3 * i + 2][row] = c[2] as u32;
            }
            partial.insert(k, (row, 8, 12));
        } else if !four_extra.is_empty() {
            let row = four_extra.remove(0);
            cv[5][row] = c[3];
            cv[6][row] = c[4];
            cv[7][row] = c[5];
            cv[8][row] = c[6];
            cv[9][row] = c[7];
            for i in 8..12 {
                s_map[3 * i][row] = c[0] as u32;
                s_map[3 * i + 1][row] = c[1] as u32;
                s_map[3 * i + 2][row] = c[2] as u32;
            }
            partial.insert(k, (row, 9, 12));
        } else {
            cv[0][r] = c[3];
            cv[1][r] = c[4];
            cv[2][r] = c[5];
            cv[3][r] = c[6];
            cv[4][r] = c[7];
            for i in 0..6 {
                s_map[3 * i][r] = c[0] as u32;
                s_map[3 * i + 1][r] = c[1] as u32;
                s_map[3 * i + 2][r] = c[2] as u32;
            }
            partial.insert(k.clone(), (r, 1, 6));
            half.push((r, 6, 12));
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
