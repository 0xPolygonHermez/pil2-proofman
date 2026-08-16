//! Aggregation setup — direct port of aggregation_setup.js.
//! 48 committed pols, 24 S cols, 5 rows/Poseidon, 2 CMul/row.
//! Chain slots relocated low (a[16..31], a[32..47]) so the gate fits a 48-col band.
//! Plonk band is a[0..15] on Poseidon rows (a[16..47] = chains); a[0..23] elsewhere.

use super::{gen_pil_str, PilTemplateParams};
use crate::plonk2pil::r1cs::to_plonk::{ckey, filter_fft4_gate_uses, filter_gate_uses, get_custom_gates_info};
use crate::plonk2pil::r1cs::types::{PlonkOptions, R1csFile, SetupResult};
use crate::plonk2pil::utils::{build_fixed_pols, build_s_polynomials, log2, mulp};
use crate::plonk2pil::merge_copies::{apply_remap_to_s_map, r1cs2plonk_merged, verify_merge_soundness};
use proofman_common::hash_family::GateRole;
use std::collections::HashMap;

const COMMITTED_POLS: usize = 48;
const N_COLS: usize = 24;
const POSEIDON_ROWS: usize = 5;
const COL_P1: usize = 16;
const COL_P2: usize = 32;
const CMUL_PER_ROW: usize = 2;

fn rand_hex() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    format!("{:x}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64)
}

type PR = (usize, usize, usize); // (row, n_used, max_used)

pub fn aggregation_compressor(r1cs: &R1csFile, options: &PlonkOptions) -> SetupResult {
    let (plonk_constraints, plonk_additions, copy_merge) = r1cs2plonk_merged(r1cs, options.merge_copies);
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

    // Plonk piggyback tiers (a[48] band). 8 gates: q0 = gates 0,1 (a[0..5]); q1 = gates 2..7
    // (a[6..23]). A row = one q0 constraint (gates 0,1) + one q1 constraint (its q1 gates).
    //   PR', FINAL' : q0 (gates 0,1) + q1 gates 2,3,4  (a[0..14]) → "five" rows.
    //   PR          : q0 (gates 0,1) + q1 gate 2       (a[0..8]; a[9..14]=anchors) → "three" rows.
    //   cmul / tree : q1 gates 6,7   (a[18..23]).
    //   evpol       : q1 gate 7      (a[21..23]).
    //   INIT/FINAL host input/output → no plonk; FFT4/SelectVal1 fill the band → no plonk.
    let five_count = n_poseidon * 2; // PR' + FINAL'
    let three_count = n_poseidon; // PR
    let cmul_plonk_count = n_cmul_rows + n_tree_sel4_rows; // gates 6,7 on cmul + tree rows
    let ev_plonk_count = n_ev_pol4_rows; // gate 7 on evpol rows

    cgi.n_plonk_rows = {
        let mut partial: HashMap<String, (usize, usize)> = HashMap::new();
        let mut half: Vec<(usize, usize)> = Vec::new();
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
                if pr.0 < pr.1 {
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
                ev_pl -= 1; // q1 gate 7 only (single gate)
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
        + n_tree_sel4_rows
        + n_sel_val1_rows;

    let n_bits = if n_used <= 1 { 1 } else { log2((n_used - 1) as u32) as usize + 1 };
    let n = 1usize << n_bits;
    let n_publics = r1cs.header.n_outputs + r1cs.header.n_pub_inputs;
    let max_degree = options.max_constraint_degree.unwrap_or(8);
    let airgroup_name = options.airgroup_name.clone().unwrap_or_else(|| format!("Compressor{}", rand_hex()));

    let pil_str = gen_pil_str(&PilTemplateParams {
        template_file: "poseidon2/aggregator",
        template_name: "Aggregator",
        namespace_name: &airgroup_name,
        n_bits,
        n_publics,
        max_constraint_degree: max_degree,
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

    let mut five_extra: Vec<usize> = Vec::new(); // PR', FINAL' rows (q0 gates 0,1 + q1 gates 2,3,4)
    let mut three_extra: Vec<usize> = Vec::new(); // PR row (q0 gates 0,1 + q1 gate 2)
    let mut cmul_extra: Vec<usize> = Vec::new(); // cmul + tree rows (q1 gates 6,7)
    let mut ev_extra: Vec<usize> = Vec::new(); // evpol rows (q1 gate 7)

    let poseidon_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::PoseidonSponge));
    let poseidon_cust_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::PoseidonCompression));
    let cmul_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::CMul));
    let fft4_uses = filter_fft4_gate_uses(&r1cs.custom_gates_uses, &cgi.fft4_parameters);
    let ev_pol4_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::EvPol4));
    let tree_sel4_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::TreeSelector));
    let sel_val1_uses = filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::SelectVal1));

    let mut r = 0usize;

    // ── Poseidon sponge (5 rows) ──────────────────────────────────────────────
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
            s_map[i + COL_P1][r] = round0[i] as u32;
            s_map[i + COL_P2][r] = round1[i] as u32;
            s_map[i + COL_P1][r + 1] = round2[i] as u32;
            s_map[i + COL_P2][r + 1] = round3[i] as u32;
            s_map[i + COL_P1][r + 2] = round4[i] as u32;
            s_map[i + COL_P1][r + 3] = round26[i] as u32;
            s_map[i + COL_P2][r + 3] = round27[i] as u32;
            s_map[i + COL_P1][r + 4] = round28[i] as u32;
            s_map[i + COL_P2][r + 4] = round29[i] as u32;
            s_map[i][r + 4] = output[i] as u32;
        }
        // anchors[0..15] → chain-F2 a[32..47] @ PR; anchors[16..21] → a[9..14] @ PR (same row).
        for i in 0..11 {
            s_map[i + COL_P2][r + 2] = im1[i] as u32; // anchors[0..10] → a[32..42]
            if i < 5 {
                s_map[i + COL_P2 + 11][r + 2] = im2[i] as u32; // anchors[11..15] → a[43..47]
            } else {
                s_map[(i - 5) + 9][r + 2] = im2[i] as u32; // anchors[16..21] → a[9..14] @ PR
            }
        }
        for off in 0..5 {
            for item in cv.iter_mut() {
                item[r + off] = 0;
            }
        }
        // q0/q1 piggyback: PR' + FINAL' expose gates 0..4 (five); PR exposes gates 0..2 (three).
        five_extra.push(r + 1); // PR'
        three_extra.push(r + 2); // PR
        five_extra.push(r + 3); // FINAL'
        r += 5;
    }
    assert_eq!(r, 5 * poseidon_uses.len());

    // ── Poseidon custom / compressor (5 rows) ────────────────────────────────
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
            s_map[i + COL_P1][r] = round0[i] as u32;
            s_map[i + COL_P2][r] = round1[i] as u32;
            s_map[i + COL_P1][r + 1] = round2[i] as u32;
            s_map[i + COL_P2][r + 1] = round3[i] as u32;
            s_map[i + COL_P1][r + 2] = round4[i] as u32;
            s_map[i + COL_P1][r + 3] = round26[i] as u32;
            s_map[i + COL_P2][r + 3] = round27[i] as u32;
            s_map[i + COL_P1][r + 4] = round28[i] as u32;
            s_map[i + COL_P2][r + 4] = round29[i] as u32;
            s_map[i][r + 4] = output[i] as u32;
        }
        // Key bits (fb=key0, sb=key1) relocated to a[15]: key0 on PR' (r+1), key1 on PR (r+2).
        s_map[15][r + 1] = fb as u32;
        s_map[15][r + 2] = sb as u32;
        // anchors[0..15] → chain-F2 a[32..47] @ PR; anchors[16..21] → a[9..14] @ PR (same row).
        for i in 0..11 {
            s_map[i + COL_P2][r + 2] = im1[i] as u32; // anchors[0..10] → a[32..42]
            if i < 5 {
                s_map[i + COL_P2 + 11][r + 2] = im2[i] as u32; // anchors[11..15] → a[43..47]
            } else {
                s_map[(i - 5) + 9][r + 2] = im2[i] as u32; // anchors[16..21] → a[9..14] @ PR
            }
        }
        for off in 0..5 {
            for item in cv.iter_mut() {
                item[r + off] = 0;
            }
        }
        five_extra.push(r + 1); // PR'
        three_extra.push(r + 2); // PR
        five_extra.push(r + 3); // FINAL'
        r += 5;
    }
    assert_eq!(r, 5 * poseidon_uses.len() + 5 * poseidon_cust_uses.len());

    // ── CMul (3/row) ──────────────────────────────────────────────────────────
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
    assert_eq!(r, 5 * poseidon_uses.len() + 5 * poseidon_cust_uses.len() + n_cmul_rows);

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

    // ── TreeSelector4 (1 row) ───────────────────────────────────────────────────
    // 17 signals → a[0..16]; a[18..23] free → plonk gates 6,7 (cmul_extra tier).
    tracing::info!("Processing {} treeSelector4 gates...", tree_sel4_uses.len());
    for cgu in &tree_sel4_uses {
        assert_eq!(cgu.signals.len(), 17);
        for (i, item) in s_map.iter_mut().enumerate().take(17) {
            item[r] = cgu.signals[i] as u32;
        }
        for item in cv.iter_mut() {
            item[r] = 0;
        }
        cmul_extra.push(r); // gates 6,7 (a[18..23]) free
        r += 1;
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
            // Opener took gate pr.1; keep the slot open only if a later gate can coalesce.
            pr.1 += 1;
            if pr.1 < pr.2 {
                partial.insert(k, pr);
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
        } else if !cmul_extra.is_empty() {
            // cmul / tree row: q1 gates 6,7 (no q0). Dup across gates 6,7 (cells 18..23).
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
        } else if !ev_extra.is_empty() {
            // evpol row: q1 gate 7 only (single gate, no partial).
            let row = ev_extra.remove(0);
            cv[5][row] = c[3];
            cv[6][row] = c[4];
            cv[7][row] = c[5];
            cv[8][row] = c[6];
            cv[9][row] = c[7];
            s_map[21][row] = c[0] as u32;
            s_map[22][row] = c[1] as u32;
            s_map[23][row] = c[2] as u32;
        } else {
            // New pure-plonk row: q0 gates 0,1 (dup) now; q1 gates 2..7 queued.
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

    // ── S polynomials ─────────────────────────────────────────────────────────
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
