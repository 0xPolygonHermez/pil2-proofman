//! Compressor setup.
//! 52 committed pols, 36 S cols, 9 rows/Poseidon1, 4 CMul/row.

use super::super::super::r1cs::to_plonk::{
    ckey, filter_fft4_gate_uses, filter_gate_uses, get_custom_gates_info, r1cs2plonk,
};
use super::super::super::r1cs::types::{PlonkOptions, R1csFile, SetupResult};
use super::super::super::utils::{build_fixed_pols, build_s_polynomials, log2, mulp};
use super::{gen_pil_str, PilTemplateParams};
use proofman_common::hash_family::GateRole;
use std::collections::HashMap;

const COMMITTED_POLS: usize = 52;
const N_COLS: usize = 36; // S connection columns
const POSEIDON_ROWS: usize = 9;
const COL_P: usize = 36; // first Poseidon chain column offset (width-16 state slot: cols 36..51)
const CMUL_PER_ROW: usize = 4;
const POSEIDON_WIDTH: usize = 16;

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
    let n_poseidon1_compression = cgi.n(GateRole::PoseidonCompression);
    let n_poseidon1_sponge = cgi.n(GateRole::PoseidonSponge);
    let n_total_poseidon = n_poseidon1_compression + n_poseidon1_sponge;
    let n_cmul_rows = cgi.n(GateRole::CMul).div_ceil(CMUL_PER_ROW);
    let n_poseidon_rows = n_total_poseidon * POSEIDON_ROWS;
    let n_fft4_rows = cgi.n(GateRole::Fft4);
    let n_ev_pol4_rows = cgi.n(GateRole::EvPol4);
    let n_tree_sel8_rows = cgi.n(GateRole::TreeSelector);
    let n_sel_val1_rows = cgi.n(GateRole::SelectVal1);

    // Per-gate row tiers for plonk piggyback. With CHECK_PLONK = PLONK + R1 + R2 +
    // PR_PRIME + PR + R26 + R27 + R28 (7 selectors after dropping R4 storage row),
    // seven rows per Poseidon gate have all 12 plonk gates active → twelve tier.
    // INIT picks up gates 8..11 via POSEIDON1_INIT (four tier); FINAL picks up gates
    // 6..11 via POSEIDON1_FINAL (six tier). TreeSelector8 consumes plonk-band cells
    // a[0..29] — no piggyback at TreeSel rows.
    let twelve_count = n_total_poseidon * 7; // R1, R2, R3/PR', PR, R26, R27, R28
    let six_count = n_total_poseidon; // FINAL rows only (TreeSelector8 no longer piggybacks)
    let five_count = n_ev_pol4_rows;
    let four_count = n_total_poseidon + n_sel_val1_rows; // INIT rows + SelectVal1
    let _ = n_tree_sel8_rows;

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
        + n_tree_sel8_rows
        + n_sel_val1_rows;

    let n_bits = if n_used <= 1 { 1 } else { log2((n_used - 1) as u32) as usize + 1 };
    let n = 1usize << n_bits;
    let n_publics = r1cs.header.n_outputs + r1cs.header.n_pub_inputs;
    let airgroup_name = options.airgroup_name.clone().unwrap_or_else(|| format!("Compressor{}", rand_hex()));

    let pil_str = gen_pil_str(&PilTemplateParams {
        template_file: "poseidon1/compressor",
        template_name: "Compressor",
        namespace_name: &airgroup_name,
        n_bits,
        n_publics,
        max_constraint_degree: 5,
        n_plonk_rows: cgi.n_plonk_rows,
        n_poseidon1_compression,
        n_poseidon1_sponge,
        n_cmul_rows,
        n_ev_pol4: cgi.n(GateRole::EvPol4),
        n_fft4: cgi.n(GateRole::Fft4),
        n_tree_selector8: cgi.n(GateRole::TreeSelector),
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

    // CustPoseidon1 (compression) gates come first, then Poseidon1 (sponge) gates,
    // matching the fixed-col patterns in compressor.pil.
    let cust_poseidon1_uses = if n_poseidon1_compression > 0 {
        filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::PoseidonCompression))
    } else {
        Vec::new()
    };
    let poseidon1_uses = if n_poseidon1_sponge > 0 {
        filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::PoseidonSponge))
    } else {
        Vec::new()
    };
    let cmul_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::CMul));
    let fft4_uses = filter_fft4_gate_uses(&r1cs.custom_gates_uses, &cgi.fft4_parameters);
    let ev_pol4_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::EvPol4));
    let tree_sel8_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::TreeSelector));
    let sel_val1_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::SelectVal1));

    let mut r = 0usize;

    // ── Poseidon1 — 9 rows per gate (compression then sponge) ────────────────
    // CustPoseidon1_16 (compression) signal layout: in[16] + key[2] + im[11][16] + out[16] = 210.
    // Poseidon1_16   (sponge)      signal layout: in[16]         + im[11][16] + out[16] = 208.
    //
    // R4 (= circom im[3], round-3 P-matrix output) is NOT stored — the PIL computes
    // it from R3 via the P-matrix expression and feeds it as the partial chain's
    // degree-7 input. State expressions stay at degree 7; anchor checks at degree 8.
    //
    // Witness layout per gate (chain slot = a[36..51], width 16):
    //   row 0 (INIT):    a[0..15]=input, a[16..17]=key (Cust only), a[18..23]=anchors[16..21] (6 overflow),
    //                    a[36..51]=input(dup, R0 — permuted for Cust)
    //   row 1 (R1):                                                  a[36..51]=R1
    //   row 2 (R2):                                                  a[36..51]=R2
    //   row 3 (R3/PR'):                                              a[36..51]=R3
    //   row 4 (PR):                                                  a[36..51]=anchors[0..15] (chain slot FULL)
    //   row 5 (R26):                                                 a[36..51]=R26 (= circom im[7])
    //   row 6 (R27):                                                 a[36..51]=R27
    //   row 7 (R28):                                                 a[36..51]=R28
    //   row 8 (FINAL):   a[0..15]=output                             a[36..51]=R29
    let process_poseidon1 = |s: &[u64],
                             is_compression: bool,
                             s_map: &mut [Vec<u32>],
                             cv: &mut [Vec<u64>],
                             twelve_extra: &mut Vec<usize>,
                             six_extra: &mut Vec<usize>,
                             four_extra: &mut Vec<usize>,
                             r: usize| {
        let key_off = if is_compression { 2 } else { 0 };
        let expected = POSEIDON_WIDTH + key_off + 12 * POSEIDON_WIDTH + POSEIDON_WIDTH;
        assert_eq!(s.len(), expected, "unexpected Poseidon1 signal count");

        let input = &s[0..POSEIDON_WIDTH];
        let key = if is_compression { Some(&s[POSEIDON_WIDTH..POSEIDON_WIDTH + 2]) } else { None };
        let im_base = POSEIDON_WIDTH + key_off;
        let r0 = &s[im_base..im_base + POSEIDON_WIDTH]; // im[0]
        let r1 = &s[im_base + POSEIDON_WIDTH..im_base + 2 * POSEIDON_WIDTH]; // im[1]
        let r2 = &s[im_base + 2 * POSEIDON_WIDTH..im_base + 3 * POSEIDON_WIDTH]; // im[2]
        let r3 = &s[im_base + 3 * POSEIDON_WIDTH..im_base + 4 * POSEIDON_WIDTH]; // im[3]
                                                                                 // im[4] = R4 (post-P state) — not stored in the 9-row variant
        let im1 = &s[im_base + 5 * POSEIDON_WIDTH..im_base + 6 * POSEIDON_WIDTH]; // im[5]: h1 anchors[0..10]
                                                                                  // im[6] = midState (intermediate, not used with single-chain layout)
        let im2 = &s[im_base + 7 * POSEIDON_WIDTH..im_base + 8 * POSEIDON_WIDTH]; // im[7]: h2 anchors[0..10]
        let r26 = &s[im_base + 8 * POSEIDON_WIDTH..im_base + 9 * POSEIDON_WIDTH]; // im[8]
        let r27 = &s[im_base + 9 * POSEIDON_WIDTH..im_base + 10 * POSEIDON_WIDTH]; // im[9]
        let r28 = &s[im_base + 10 * POSEIDON_WIDTH..im_base + 11 * POSEIDON_WIDTH]; // im[10]
        let r29 = &s[im_base + 11 * POSEIDON_WIDTH..im_base + 12 * POSEIDON_WIDTH]; // im[11]
        let output = &s[im_base + 12 * POSEIDON_WIDTH..im_base + 13 * POSEIDON_WIDTH];

        for i in 0..POSEIDON_WIDTH {
            s_map[i][r] = input[i] as u32;
            s_map[i + COL_P][r] = r0[i] as u32; // row 0 chain slot = R0 (= circom im[0], permuted input signal)
            s_map[i + COL_P][r + 1] = r1[i] as u32;
            s_map[i + COL_P][r + 2] = r2[i] as u32;
            s_map[i + COL_P][r + 3] = r3[i] as u32;
            // row 4 chain slot is anchors (filled below) — R4 NOT stored.
            s_map[i + COL_P][r + 5] = r26[i] as u32;
            s_map[i + COL_P][r + 6] = r27[i] as u32;
            s_map[i + COL_P][r + 7] = r28[i] as u32;
            s_map[i + COL_P][r + 8] = r29[i] as u32;
            s_map[i][r + 8] = output[i] as u32;
        }

        // Partial-chain anchors (single 22-round chain, viewed as one flat array):
        //   anchors[0..15]  → row 4 chain slot a[36..51] (fully fills chain slot)
        //   anchors[16..21] → row 0 cols 18..23 (overflow, plonk band)
        // Source: anchors[0..10] = im1 (h1[0..10]); anchors[11..21] = im2 (h2[0..10]).
        // First 5 of im2 sit at row 4 chain-slot tail (cols 47..51); last 6 of im2 spill to row 0.
        for i in 0..11 {
            s_map[i + COL_P][r + 4] = im1[i] as u32; // anchors[0..10] = im_h1
        }
        for i in 0..5 {
            s_map[i + 11 + COL_P][r + 4] = im2[i] as u32; // anchors[11..15] = im_h2[0..4]
        }
        for i in 0..6 {
            s_map[i + 18][r] = im2[i + 5] as u32; // anchors[16..21] = im_h2[5..10]
        }

        // Key bits at INIT row cols 16..17 (compression only). At INIT, plonk gate 5
        // (cols 15..17) doesn't fire (gate-5 selector = CHECK_PLONK, which excludes
        // INIT), so cells 16, 17 are free for non-plonk witness. This frees the PR row
        // plonk band so PR can join CHECK_PLONK and become a twelve-tier piggyback row.
        if let Some(k) = key {
            s_map[16][r] = k[0] as u32;
            s_map[17][r] = k[1] as u32;
        }

        for off in 0..POSEIDON_ROWS {
            for item in cv.iter_mut() {
                item[r + off] = 0;
            }
        }

        // Plonk piggyback queues. CHECK_PLONK fires at R1, R2, PR_PRIME, PR, R26,
        // R27, R28 → twelve tier (all plonk gates active). INIT picks up gates 8..11 via
        // POSEIDON1_INIT (four tier); FINAL picks up gates 6..11 via POSEIDON1_FINAL
        // (six tier).
        four_extra.push(r); // INIT row
        twelve_extra.push(r + 1); // R1
        twelve_extra.push(r + 2); // R2
        twelve_extra.push(r + 3); // R3 / PR'
        twelve_extra.push(r + 4); // PR (chain slot holds anchors, plonk band fully free)
        twelve_extra.push(r + 5); // R26
        twelve_extra.push(r + 6); // R27
        twelve_extra.push(r + 7); // R28
        six_extra.push(r + 8); // FINAL row
    };

    tracing::info!("Processing {} CustPoseidon1 (compression) gates...", cust_poseidon1_uses.len());
    for cgu in &cust_poseidon1_uses {
        process_poseidon1(
            &cgu.signals,
            true, // is_compression
            &mut s_map,
            &mut cv,
            &mut twelve_extra,
            &mut six_extra,
            &mut four_extra,
            r,
        );
        r += POSEIDON_ROWS;
    }
    assert_eq!(r, POSEIDON_ROWS * cust_poseidon1_uses.len());

    tracing::info!("Processing {} Poseidon1 (sponge) gates...", poseidon1_uses.len());
    for cgu in &poseidon1_uses {
        process_poseidon1(
            &cgu.signals,
            false, // is_compression
            &mut s_map,
            &mut cv,
            &mut twelve_extra,
            &mut six_extra,
            &mut four_extra,
            r,
        );
        r += POSEIDON_ROWS;
    }
    assert_eq!(r, POSEIDON_ROWS * (cust_poseidon1_uses.len() + poseidon1_uses.len()));

    // We don't use five_extra in the Poseidon section; suppress unused warning.
    let _ = &five_extra;

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
    assert_eq!(r, n_poseidon_rows + n_cmul_rows);

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

    // ── TreeSelector8 ─────────────────────────────────────────────────────────
    // TreeSelector8 signal layout: values[8][3] + keys[3] + out[3] = 30 signals.
    // Occupies plonk-band cells a[0..29] → no plonk piggyback at TreeSel rows.
    tracing::info!("Processing {} treeSelector8 gates...", tree_sel8_uses.len());
    for cgu in &tree_sel8_uses {
        assert_eq!(cgu.signals.len(), 30);
        for (i, item) in s_map.iter_mut().enumerate().take(30) {
            item[r] = cgu.signals[i] as u32;
        }
        for item in cv.iter_mut() {
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
    let mut pure_plonk_rows: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut plonk_in_pure: usize = 0;
    let mut plonk_in_custom: usize = 0;

    for (idx, c) in plonk_constraints.iter().enumerate() {
        if idx % 10_000 == 0 {
            tracing::debug!("constraint {}/{}", idx, plonk_constraints.len());
        }
        let k = ckey(c);

        if let Some(pr) = partial.get_mut(&k) {
            let n = pr.1;
            let row = pr.0;
            s_map[n * 3][row] = c[0] as u32;
            s_map[n * 3 + 1][row] = c[1] as u32;
            s_map[n * 3 + 2][row] = c[2] as u32;
            pr.1 += 1;
            if pr.1 == pr.2 {
                partial.remove(&k);
            }
            if pure_plonk_rows.contains(&row) {
                plonk_in_pure += 1;
            } else {
                plonk_in_custom += 1;
            }
        } else if !half.is_empty() {
            let mut pr = half.remove(0);
            let row = pr.0;
            cv[5][row] = c[3];
            cv[6][row] = c[4];
            cv[7][row] = c[5];
            cv[8][row] = c[6];
            cv[9][row] = c[7];
            for i in pr.1..pr.2 {
                s_map[3 * i][row] = c[0] as u32;
                s_map[3 * i + 1][row] = c[1] as u32;
                s_map[3 * i + 2][row] = c[2] as u32;
            }
            pr.1 += 1;
            partial.insert(k, pr);
            if pure_plonk_rows.contains(&row) {
                plonk_in_pure += 1;
            } else {
                plonk_in_custom += 1;
            }
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
            plonk_in_custom += 1;
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
            plonk_in_custom += 1;
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
            plonk_in_custom += 1;
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
            plonk_in_custom += 1;
        } else {
            pure_plonk_rows.insert(r);
            plonk_in_pure += 1;
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

    tracing::info!(
        "Plonk placement: {} constraints in {} pure plonk rows, {} constraints piggybacked on custom-gate rows",
        plonk_in_pure,
        pure_plonk_rows.len(),
        plonk_in_custom,
    );

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
