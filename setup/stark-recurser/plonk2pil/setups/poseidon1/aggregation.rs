//! Aggregation setup.
//! 48 committed pols, 24 S cols, 5 rows/Poseidon1, 2 CMul/row.
//! Chain slots relocated low (a[16..31], a[32..47]) so the gate fits a 48-col band.
//! Plonk band is a[0..15] on Poseidon rows (a[16..47] = chains); a[0..23] elsewhere.

use crate::plonk2pil::r1cs::to_plonk::{ckey, filter_fft4_gate_uses, filter_gate_uses, get_custom_gates_info};
use crate::plonk2pil::r1cs::types::{PlonkOptions, R1csFile, SetupResult};
use crate::plonk2pil::utils::{build_fixed_pols, build_s_polynomials, log2, mulp};
use crate::plonk2pil::merge_copies::{apply_remap_to_s_map, r1cs2plonk_merged, verify_merge_soundness};
use super::{gen_pil_str, PilTemplateParams};
use proofman_common::hash_family::GateRole;
use std::collections::HashMap;

const COMMITTED_POLS: usize = 48;
const N_COLS: usize = 24;
const POSEIDON_ROWS: usize = 5;
const COL_P1: usize = 16; // chain 1 slot (width-16: cols 16..31)
const COL_P2: usize = 32; // chain 2 slot (width-16: cols 32..47)
const CMUL_PER_ROW: usize = 2;
const TREESEL_ROWS: usize = 2; // TreeSelector8 now spans 2 rows (a[0..14] + a[0..14]')
const POSEIDON_WIDTH: usize = 16;

fn rand_hex() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    format!("{:x}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64)
}

type PR = (usize, usize, usize); // (row, n_used, max_used)

pub fn aggregation_compressor(r1cs: &R1csFile, options: &PlonkOptions) -> SetupResult {
    let (plonk_constraints, plonk_additions, copy_merge) = r1cs2plonk_merged(r1cs, options.merge_copies);
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

    // Plonk piggyback tiers (a[48] band). 8 gates: q0 = gates 0,1 (a[0..5]); q1 = gates 2..7
    // (a[6..23]). A row = one q0 constraint (gates 0,1) + one q1 constraint (its q1 gates).
    //   PR', FINAL' : q0 (gates 0,1) + q1 gates 2,3,4  (a[0..14]) → "five" rows.
    //   PR          : q0 (gates 0,1) + q1 gate 2       (a[0..8], a[9..14]=anchors) → "three" rows.
    //   INIT/FINAL  : S0 = I/O → no plonk.
    //   cmul row    : q1 gates 6,7 only (a[18..23]; cmul uses a[0..17]).
    //   evpol row   : q1 gate 7 only  (a[21..23]; evpol uses a[0..20]).
    //   TreeSelector8 (2 rows, a[0..14]) / FFT4 (a[0..23]) / SelectVal1 (a[0..21]) → no piggyback.
    // Slot layout: q0 half always (1,2); q1 half end varies. cmul/evpol slots have no q0.
    let n_tree_sel8_rows = n_tree_sel8_rows * TREESEL_ROWS;
    let five_count = n_total_poseidon * 2; // PR' + FINAL'
    let three_count = n_total_poseidon; // PR
    let cmul_plonk_count = n_cmul_rows; // gates 6,7 on cmul rows
    let ev_plonk_count = n_ev_pol4_rows; // gate 7 on evpol rows

    cgi.n_plonk_rows = {
        let mut partial: HashMap<String, (usize, usize)> = HashMap::new(); // (next, end)
        let mut half: Vec<(usize, usize)> = Vec::new(); // open q1 halves (next, end)
        let (mut five, mut three) = (five_count, three_count);
        let (mut cmul_pl, mut ev_pl) = (cmul_plonk_count, ev_plonk_count);
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
                if pr.0 == pr.1 {
                    // 1-gate q1 half already exhausted; nothing to track.
                } else {
                    partial.insert(k, pr);
                }
            } else if five > 0 {
                five -= 1;
                partial.insert(k, (1, 2)); // q0 gates 0,1
                half.push((2, 5)); // q1 gates 2,3,4
            } else if three > 0 {
                three -= 1;
                partial.insert(k, (1, 2)); // q0 gates 0,1
                half.push((2, 3)); // q1 gate 2 only
            } else if cmul_pl > 0 {
                cmul_pl -= 1;
                partial.insert(k, (7, 8)); // opener took gate 6; gate 7 can coalesce
            } else if ev_pl > 0 {
                ev_pl -= 1; // q1 gate 7 only (single gate, no partial to track)
            } else {
                partial.insert(k.clone(), (1, 2)); // q0 gates 0,1
                half.push((2, 8)); // q1 gates 2..7
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
    let max_degree = options.max_constraint_degree.unwrap_or(8);
    let airgroup_name = options.airgroup_name.clone().unwrap_or_else(|| format!("Compressor{}", rand_hex()));

    let pil_str = gen_pil_str(&PilTemplateParams {
        template_file: "poseidon1/aggregator",
        template_name: "Aggregator",
        namespace_name: &airgroup_name,
        n_bits,
        n_publics,
        max_constraint_degree: max_degree,
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

    let mut five_extra: Vec<usize> = Vec::new(); // PR', FINAL' rows (q0 gates 0,1 + q1 gates 2,3,4)
    let mut three_extra: Vec<usize> = Vec::new(); // PR row (q0 gates 0,1 + q1 gate 2)
    let mut cmul_extra: Vec<usize> = Vec::new(); // cmul rows (q1 gates 6,7)
    let mut ev_extra: Vec<usize> = Vec::new(); // evpol rows (q1 gate 7)

    // CustPoseidon1 (compression) gates come first, then Poseidon1 (sponge) gates,
    // matching the fixed-col patterns in aggregator.pil.
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

    // ── Poseidon1 — 5 rows per gate (compression then sponge) ────────────────
    // CustPoseidon1_16 (compression) signal layout: in[16] + key[2] + im[11][16] + out[16] = 210.
    // Poseidon1_16   (sponge)      signal layout: in[16]         + im[11][16] + out[16] = 208.
    //
    // im[] sub-layout (set by the circom Poseidon1_16 / CustPoseidon1_16 template):
    //   im[0]=R0, im[1]=R1, im[2]=R2, im[3]=R3, im[4]=R4 (post-P transition state),
    //   im[5]=h1 anchors (11 used + 1 pad), im[6]=midState (intermediate, unused),
    //   im[7]=h2 anchors (11 used + pad), im[8]=R26, im[9]=R27, im[10]=R28, im[11]=R29
    //
    // Witness layout per gate (5 rows; chain1=a[16..31], chain2=a[32..47]):
    //   row 0 (INIT):    a[0..15]=input,                       a[16..31]=R0,  a[32..47]=R1
    //   row 1 (PR'):     a[15]=key0,                           a[16..31]=R2,  a[32..47]=R3
    //   row 2 (PR):      a[9..14]=anchors[16..21], a[15]=key1,  a[16..31]=R4,  a[32..47]=anchors[0..15]
    //   row 3 (FINAL'):                                        a[16..31]=R26, a[32..47]=R27
    //   row 4 (FINAL):   a[0..15]=output,                      a[16..31]=R28, a[32..47]=R29
    //
    // The 22 lane-0 partial anchors are circom im[5][0..10] (rounds 0..10) and
    // im[7][0..10] (rounds 11..21): anchors[0..15] → chain-2 a[32..47] @ PR,
    // anchors[16..21] → a[9..14] @ PR (same row).
    let process_poseidon1 = |s: &[u64],
                             is_compression: bool,
                             s_map: &mut [Vec<u32>],
                             cv: &mut [Vec<u64>],
                             five_extra: &mut Vec<usize>,
                             three_extra: &mut Vec<usize>,
                             r: usize| {
        let key_off = if is_compression { 2 } else { 0 };
        let expected = 16 + key_off + 12 * POSEIDON_WIDTH + POSEIDON_WIDTH;
        assert_eq!(s.len(), expected, "unexpected Poseidon1 signal count");

        let input = &s[0..POSEIDON_WIDTH];
        let key = if is_compression { Some(&s[POSEIDON_WIDTH..POSEIDON_WIDTH + 2]) } else { None };
        let im_base = POSEIDON_WIDTH + key_off;
        let r0 = &s[im_base..im_base + POSEIDON_WIDTH]; // im[0]
        let r1 = &s[im_base + POSEIDON_WIDTH..im_base + 2 * POSEIDON_WIDTH]; // im[1]
        let r2 = &s[im_base + 2 * POSEIDON_WIDTH..im_base + 3 * POSEIDON_WIDTH]; // im[2]
        let r3 = &s[im_base + 3 * POSEIDON_WIDTH..im_base + 4 * POSEIDON_WIDTH]; // im[3]
        let r4 = &s[im_base + 4 * POSEIDON_WIDTH..im_base + 5 * POSEIDON_WIDTH]; // im[4]: post-P transition
        let anchors_h1 = &s[im_base + 5 * POSEIDON_WIDTH..im_base + 6 * POSEIDON_WIDTH]; // im[5]: anchors rounds 0..10
                                                                                         // im[6] = midState (intermediate, not stored)
        let anchors_h2 = &s[im_base + 7 * POSEIDON_WIDTH..im_base + 8 * POSEIDON_WIDTH]; // im[7]: anchors rounds 11..21
        let r26 = &s[im_base + 8 * POSEIDON_WIDTH..im_base + 9 * POSEIDON_WIDTH]; // im[8]
        let r27 = &s[im_base + 9 * POSEIDON_WIDTH..im_base + 10 * POSEIDON_WIDTH]; // im[9]
        let r28 = &s[im_base + 10 * POSEIDON_WIDTH..im_base + 11 * POSEIDON_WIDTH]; // im[10]
        let r29 = &s[im_base + 11 * POSEIDON_WIDTH..im_base + 12 * POSEIDON_WIDTH]; // im[11]
        let output = &s[im_base + 12 * POSEIDON_WIDTH..im_base + 13 * POSEIDON_WIDTH];

        for i in 0..POSEIDON_WIDTH {
            s_map[i][r] = input[i] as u32;
            s_map[i + COL_P1][r] = r0[i] as u32; // row 0 chain 1 = R0 (= circom im[0], permuted input signal)
            s_map[i + COL_P2][r] = r1[i] as u32; // row 0 chain 2 = R1
            s_map[i + COL_P1][r + 1] = r2[i] as u32; // row 1 chain 1 = R2
            s_map[i + COL_P2][r + 1] = r3[i] as u32; // row 1 chain 2 = R3
            s_map[i + COL_P1][r + 2] = r4[i] as u32; // row 2 chain 1 = R4 (stored transition)
            s_map[i + COL_P1][r + 3] = r26[i] as u32; // row 3 chain 1 = R26
            s_map[i + COL_P2][r + 3] = r27[i] as u32; // row 3 chain 2 = R27
            s_map[i + COL_P1][r + 4] = r28[i] as u32; // row 4 chain 1 = R28
            s_map[i + COL_P2][r + 4] = r29[i] as u32; // row 4 chain 2 = R29
            s_map[i][r + 4] = output[i] as u32; // row 4 a[0..15] = output
        }

        // 22 lane-0 partial anchors: rounds 0..10 = anchors_h1[0..10], rounds 11..21 =
        // anchors_h2[0..10]. anchors[0..15] → chain-2 a[32..47] @ PR (row r+2); the 6
        // overflow anchors[16..21] → a[9..14] on the SAME PR row (read unprimed by the PIL).
        let anchor = |round: usize| -> u64 {
            if round <= 10 {
                anchors_h1[round]
            } else {
                anchors_h2[round - 11]
            }
        };
        for round in 0..16 {
            s_map[round + COL_P2][r + 2] = anchor(round) as u32; // anchors[0..15] → chain-2 @ PR
        }
        for round in 16..22 {
            s_map[(round - 16) + 9][r + 2] = anchor(round) as u32; // anchors[16..21] → a[9..14] @ PR
        }

        // Key bits (compression only) relocated to a[15]: key0 on PR' (row r+1), key1 on
        // PR (row r+2). a[15] is the spare cell after 5 plonk gates (a[0..14]) on those rows.
        // Can't use a[16..17] (chain-1 R0 at INIT) as the old layout did.
        if let Some(k) = key {
            s_map[15][r + 1] = k[0] as u32;
            s_map[15][r + 2] = k[1] as u32;
        }

        for off in 0..POSEIDON_ROWS {
            for item in cv.iter_mut() {
                item[r + off] = 0;
            }
        }

        // Plonk piggyback queues (a[0..15] band on Poseidon rows). PR' and FINAL' expose
        // gates 0..4 (a[0..14]) → five_extra. PR exposes gates 0..2 (a[0..8]; a[9..14]=anchors,
        // a[15]=key1) → three_extra. INIT/FINAL host input/output → no plonk.
        five_extra.push(r + 1); // PR'
        three_extra.push(r + 2); // PR
        five_extra.push(r + 3); // FINAL'
    };

    tracing::info!("Processing {} CustPoseidon1 (compression) gates...", cust_poseidon1_uses.len());
    for cgu in &cust_poseidon1_uses {
        process_poseidon1(
            &cgu.signals,
            true, // is_compression
            &mut s_map,
            &mut cv,
            &mut five_extra,
            &mut three_extra,
            r,
        );
        r += POSEIDON_ROWS;
    }

    tracing::info!("Processing {} Poseidon1 (sponge) gates...", poseidon1_uses.len());
    for cgu in &poseidon1_uses {
        process_poseidon1(
            &cgu.signals,
            false, // is_compression
            &mut s_map,
            &mut cv,
            &mut five_extra,
            &mut three_extra,
            r,
        );
        r += POSEIDON_ROWS;
    }
    assert_eq!(r, n_poseidon_rows);

    // ── CMul (2/row) ──────────────────────────────────────────────────────────
    // 2 cmuls per row on a[0..8], a[9..17]; a[18..23] free → plonk gates 6,7 (cmul_extra).
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
            cmul_extra.push(r); // a[18..23] free → plonk gates 6,7
            cmul_row = r as i64;
            cmul_used = 1;
            r += 1;
        }
    }
    assert_eq!(r, n_poseidon_rows + n_cmul_rows);

    // ── EvPol4 ────────────────────────────────────────────────────────────────
    // EvPol4 uses a[0..20]; a[21..23] free → plonk gate 7 (ev_extra).
    tracing::info!("Processing {} evPol4 gates...", ev_pol4_uses.len());
    for cgu in &ev_pol4_uses {
        for (i, item) in s_map.iter_mut().enumerate().take(21) {
            item[r] = cgu.signals[i] as u32;
        }
        for item in cv.iter_mut() {
            item[r] = 0;
        }
        ev_extra.push(r);
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

    // ── TreeSelector8 (2 rows) ──────────────────────────────────────────────────
    // 30 signals split 24 + 6: the 8 vals[8][3] (signals 0..23) on the gate row a[0..23];
    // key[3]+res[3] (signals 24..29) on the next row a[0..5]. Packing the gate row full
    // minimizes the copy openings leaking to the second row. No plonk piggyback.
    tracing::info!("Processing {} treeSelector8 gates...", tree_sel8_uses.len());
    for cgu in &tree_sel8_uses {
        assert_eq!(cgu.signals.len(), 30);
        for (i, item) in s_map.iter_mut().enumerate().take(24) {
            item[r] = cgu.signals[i] as u32; // vals → a[0..23] @ row r
        }
        for (i, item) in s_map.iter_mut().enumerate().take(6) {
            item[r + 1] = cgu.signals[24 + i] as u32; // key+res → a[0..5] @ row r+1
        }
        for item in cv.iter_mut() {
            item[r] = 0;
            item[r + 1] = 0;
        }
        r += TREESEL_ROWS;
    }

    // ── SelectVal1 ────────────────────────────────────────────────────────────
    // Uses a[0..21]; only a[22..23] free (not a full 3-cell gate) → no plonk piggyback.
    tracing::info!("Processing {} selectVal1 gates...", sel_val1_uses.len());
    for cgu in &sel_val1_uses {
        assert_eq!(cgu.signals.len(), 22);
        for (i, item) in s_map.iter_mut().enumerate().take(22) {
            item[r] = cgu.signals[i] as u32;
        }
        for item in cv.iter_mut() {
            item[r] = 0;
        }
        r += 1;
    }

    // ── Plonk constraints ─────────────────────────────────────────────────────
    tracing::info!("Placing {} plonk constraints...", plonk_constraints.len());
    let mut partial: HashMap<String, PR> = HashMap::new();
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
            // Opener took gate pr.1; only keep the slot open if a later gate can coalesce.
            pr.1 += 1;
            if pr.1 < pr.2 {
                partial.insert(k, pr);
            }
            if pure_plonk_rows.contains(&row) {
                plonk_in_pure += 1;
            } else {
                plonk_in_custom += 1;
            }
        } else if !five_extra.is_empty() {
            // PR' / FINAL': q0 gates 0,1 (dup) now; q1 gates 2,3,4 queued.
            let row = five_extra.remove(0);
            cv[0][row] = c[3];
            cv[1][row] = c[4];
            cv[2][row] = c[5];
            cv[3][row] = c[6];
            cv[4][row] = c[7];
            s_map[0][row] = c[0] as u32;
            s_map[1][row] = c[1] as u32;
            s_map[2][row] = c[2] as u32;
            s_map[3][row] = c[0] as u32;
            s_map[4][row] = c[1] as u32;
            s_map[5][row] = c[2] as u32;
            partial.insert(k.clone(), (row, 1, 2));
            half.push((row, 2, 5));
            plonk_in_custom += 1;
        } else if !three_extra.is_empty() {
            // PR: q0 gates 0,1 (dup) now; q1 gate 2 queued.
            let row = three_extra.remove(0);
            cv[0][row] = c[3];
            cv[1][row] = c[4];
            cv[2][row] = c[5];
            cv[3][row] = c[6];
            cv[4][row] = c[7];
            s_map[0][row] = c[0] as u32;
            s_map[1][row] = c[1] as u32;
            s_map[2][row] = c[2] as u32;
            s_map[3][row] = c[0] as u32;
            s_map[4][row] = c[1] as u32;
            s_map[5][row] = c[2] as u32;
            partial.insert(k.clone(), (row, 1, 2));
            half.push((row, 2, 3));
            plonk_in_custom += 1;
        } else if !cmul_extra.is_empty() {
            // cmul row: q1 gates 6,7 (no q0). Dup across gates 6,7 (cells 18..23).
            let row = cmul_extra.remove(0);
            cv[5][row] = c[3];
            cv[6][row] = c[4];
            cv[7][row] = c[5];
            cv[8][row] = c[6];
            cv[9][row] = c[7];
            for i in 6..8 {
                s_map[3 * i][row] = c[0] as u32;
                s_map[3 * i + 1][row] = c[1] as u32;
                s_map[3 * i + 2][row] = c[2] as u32;
            }
            partial.insert(k, (row, 7, 8));
            plonk_in_custom += 1;
        } else if !ev_extra.is_empty() {
            // evpol row: q1 gate 7 only (single gate, no partial to track).
            let row = ev_extra.remove(0);
            cv[5][row] = c[3];
            cv[6][row] = c[4];
            cv[7][row] = c[5];
            cv[8][row] = c[6];
            cv[9][row] = c[7];
            s_map[21][row] = c[0] as u32;
            s_map[22][row] = c[1] as u32;
            s_map[23][row] = c[2] as u32;
            plonk_in_custom += 1;
        } else {
            // Pure-plonk row: q0 gates 0,1 (dup) now; q1 gates 2..7 queued.
            pure_plonk_rows.insert(r);
            plonk_in_pure += 1;
            cv[0][r] = c[3];
            cv[1][r] = c[4];
            cv[2][r] = c[5];
            cv[3][r] = c[6];
            cv[4][r] = c[7];
            s_map[0][r] = c[0] as u32;
            s_map[1][r] = c[1] as u32;
            s_map[2][r] = c[2] as u32;
            s_map[3][r] = c[0] as u32;
            s_map[4][r] = c[1] as u32;
            s_map[5][r] = c[2] as u32;
            partial.insert(k.clone(), (r, 1, 2));
            half.push((r, 2, 8));
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

    // Apply copy-merge remap to every placed cell (incl. custom-gate I/O) so the
    // connection argument ties merged signals — the soundness-critical sweep,
    // then assert each merged equality is actually re-enforced in-band.
    apply_remap_to_s_map(&mut s_map, &copy_merge.remap);
    verify_merge_soundness(&s_map, &copy_merge.merged_reps, N_COLS);
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
