use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, RwLock,
    },
};
use rayon::prelude::*;

use proofman_fields::PrimeField64;

use proofman_witness::WitnessComponent;
use proofman_common::{
    register_host_buffer, unregister_host_buffer, AirInstance, BufferPool, ProofCtx, ProofmanError, ProofmanResult,
    SetupCtx, TraceInfo, Setup,
};
use proofman_hints::{get_hint_ids_by_name, HintFieldOptions};

use crate::{
    get_global_hint_field_constant_a_as, get_hint_field_constant_a_as, get_hint_field_constant_a_as_string,
    get_hint_field_constant_as, RCMultiplicity,
};

pub struct StdVirtualTable<F: PrimeField64> {
    _phantom: std::marker::PhantomData<F>,
    pub global_id_by_uid: HashMap<usize, usize>,   // uid -> global_id
    pub indices_by_global_id: Vec<(usize, usize)>, // global_id -> (air_idx, uid_idx)
    pub virtual_table_airs: Option<Vec<Arc<VirtualTableAir<F>>>>,
}
pub struct VirtualTableAir<F: PrimeField64> {
    airgroup_id: usize,
    air_id: usize,
    shift: u64,
    mask: u64,
    num_rows: usize,
    num_cols: usize,
    table_ids: Vec<(usize, u64)>, // (table_id, acc_height)
    // Parallel to `table_ids`: prover-counted tables drop increments here (counts come via `ProofCtx::prover_counts`).
    prover_owned: std::sync::OnceLock<Vec<bool>>,
    // Flat col-major: idx = col * num_rows + row. Single allocation.
    multiplicities: Vec<AtomicU64>,
    table_instance_id: AtomicU64,
    calculated: AtomicBool,
    shared_tables: bool,
    // Persistent trace buffer slot. Pre-allocated in `StdVirtualTable::new`; taken in
    // `calculate_witness` and refilled by `ProofCtx::free_instance_traces`.
    trace_buffer: Arc<Mutex<Option<Vec<F>>>>,
    // Pinned-registration page base for `trace_buffer`; unpinned it takes the staging H2D path.
    trace_buffer_pinned: Option<usize>,
}

impl<F: PrimeField64> Drop for VirtualTableAir<F> {
    fn drop(&mut self) {
        // Runs before the field drops, so the pages are still live here.
        if let Some(base) = self.trace_buffer_pinned {
            unregister_host_buffer(base);
        }
    }
}

/// Geometry of one virtual-table air, as the prover needs it registered.
pub struct VtLayout {
    pub airgroup_id: u64,
    pub air_id: u64,
    pub num_rows: u64,
    pub num_cols: u64,
    pub table_ids: Vec<u64>,
    pub acc_bases: Vec<u64>,
}

/// Tables the prover counts itself. TEMPORARY: suppresses `inc_virtual_row` calls zisk still makes
/// for claimed tables. Must be read lazily: the host sets it AFTER the virtual table airs are built,
/// so reading it at construction would double-count.
static PROVER_OWNED_TABLES: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();

pub fn set_prover_owned_tables(tables: Vec<u64>) {
    let _ = PROVER_OWNED_TABLES.set(tables);
}

fn prover_owned_tables() -> &'static [u64] {
    PROVER_OWNED_TABLES.get().map(|v| v.as_slice()).unwrap_or(&[])
}

/// Counts from `fit_virtual_table_maps`, for the caller's summary (which also knows the
/// range-claimed tables).
pub struct VtFitSummary {
    /// Virtual tables examined (non-zero height), across every virtual-table air.
    pub considered: usize,
    /// Total bytes across every exact map (GPU-resident per device).
    pub exact_bytes: u64,
    pub elapsed_ms: u128,
    /// Considered but not fitted: range tables (`collect_prover_owned_ranges`) or std-owned ones.
    pub unclaimed_ids: Vec<u64>,
}

/// One fitted row map: an exact-match map `(key columns, table, slots)` for `table_id`. Slots is
/// explicit: the packed layout has a header, so it cannot be derived from the table length.
pub struct VtFittedMap {
    pub table_id: u64,
    pub map: (usize, Vec<u64>, u64),
}

/// Marks a map whose keys are packed into one word. Must match MUL_MAP_PACKED on the C side.
const MUL_MAP_PACKED: u64 = 0x5041_434B_4544_4B56;

/// Must match `MUL_MAP_MAX_PROBE` in `multiplicity_decoders.hpp`. The self-checks use the same bound,
/// or a longer chain would verify here but miss silently at runtime.
const MUL_MAP_MAX_PROBE: usize = 4096;

// Fitting a table's row map (tuple -> row) from its own fixed columns (COL_* tuple, UID_* table id).
// A fit is accepted only if it reproduces EVERY entry; otherwise the table must be std-owned.

/// Exact-match map. Keys are the lookup's whole tuple (a prefix that
/// separates the table's rows need not be how lookups address it).
///
/// Duplicate tuples are allowed: the lookup constrains only the SUM of their multiplicities, so
/// all counts go to the first.
fn fit_exact_map(samples: &[(Vec<u64>, u64)], width: usize) -> Option<(usize, Vec<u64>, u64)> {
    // Constant trailing columns are group padding (a COL_* group is as wide as its widest table),
    // not part of this table's key.
    let mut nkey = width;
    while nkey > 1 {
        let c = nkey - 1;
        let first = samples[0].0[c];
        if samples.iter().all(|(t, _)| t[c] == first) {
            nkey -= 1;
        } else {
            break;
        }
    }
    let mut slots = 1usize;
    while slots * 4 < samples.len() * 5 {
        slots <<= 1;
    }

    // Pack the key into one word when the table's own values fit in <= 63 bits (so a packed key can
    // never equal the empty marker); otherwise keep the verbatim layout. The decoder range-checks
    // each column against its stored width, so a wider lookup value misses instead of aliasing.
    let widths: Vec<u32> = (0..nkey)
        .map(|c| {
            let m = samples.iter().map(|(t, _)| t[c]).max().unwrap_or(0);
            64 - m.leading_zeros()
        })
        .collect();
    if widths.iter().sum::<u32>() <= 63 {
        let mut shift = Vec::with_capacity(nkey);
        let mut at = 0u32;
        for w in widths.iter() {
            shift.push(at);
            at += w;
        }
        let pack = |t: &[u64]| -> u64 { (0..nkey).fold(0u64, |acc, c| acc | (t[c] << shift[c])) };
        // [MAGIC][(shift << 32) | width per key column][slots of (packed key, row)]. The shape lives
        // in the header so the scatter job needs no extra field (register pressure).
        let head = 1 + nkey;
        let mut kv = vec![u64::MAX; head + slots * 2];
        kv[0] = MUL_MAP_PACKED;
        for c in 0..nkey {
            kv[1 + c] = ((shift[c] as u64) << 32) | widths[c] as u64;
        }
        let mut collided = false;
        for (t, r) in samples.iter() {
            let k = pack(&t[..nkey]);
            let mut i = (hash_key(&[k]) as usize) & (slots - 1);
            loop {
                let slot = head + i * 2;
                if kv[slot] == u64::MAX {
                    kv[slot] = k;
                    kv[slot + 1] = *r;
                    break;
                }
                if kv[slot] == k {
                    collided = true; // two entries share this key
                    break;
                }
                i = (i + 1) & (slots - 1);
            }
        }
        // Probe every entry back with the decoder's algorithm (duplicated on the C side): a mismatch
        // there looks like a hang, here it fails loudly.
        for (t, r) in samples.iter() {
            let k = pack(&t[..nkey]);
            let mut h = k.wrapping_add(0x9E37_79B9_7F4A_7C15);
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h ^= h >> 32;
            let mut i = (h as usize) & (slots - 1);
            let mut found = None;
            for _ in 0..slots.min(MUL_MAP_MAX_PROBE) {
                let slot = head + i * 2;
                if kv[slot] == u64::MAX {
                    break;
                }
                if kv[slot] == k {
                    found = Some(kv[slot + 1]);
                    break;
                }
                i = (i + 1) & (slots - 1);
            }
            let ok = match found {
                Some(got) => got == *r || collided,
                None => false,
            };
            if !ok {
                tracing::error!(
                    "packed map does not probe back: key {:?} packs to {k:#x}, expected row {r}, \
                     got {found:?} -- keeping the verbatim layout for this table",
                    &t[..nkey]
                );
                return fit_exact_map_verbatim(samples, nkey, slots);
            }
        }
        return Some((nkey, kv, slots as u64));
    }

    fit_exact_map_verbatim(samples, nkey, slots)
}

/// Verbatim-key map, one word per column: used when keys do not pack or the packed map fails to
/// probe back. Last resort, so it is self-checked too.
fn fit_exact_map_verbatim(samples: &[(Vec<u64>, u64)], nkey: usize, slots: usize) -> Option<(usize, Vec<u64>, u64)> {
    let stride = nkey + 1;
    let mut kv = vec![u64::MAX; slots * stride];
    let mut collided = false;
    for (t, r) in samples.iter() {
        let mut i = (hash_key(&t[..nkey]) as usize) & (slots - 1);
        loop {
            let slot = i * stride;
            if kv[slot..slot + nkey].iter().all(|v| *v == u64::MAX) {
                kv[slot..slot + nkey].copy_from_slice(&t[..nkey]);
                kv[slot + nkey] = *r;
                break;
            }
            if kv[slot..slot + nkey] == t[..nkey] {
                collided = true; // two entries share this key
                break;
            }
            i = (i + 1) & (slots - 1);
        }
    }
    if !verify_exact_map_verbatim(samples, &kv, nkey, slots, collided) {
        return None;
    }
    Some((nkey, kv, slots as u64))
}

/// Probe every sample back through the verbatim map as the GPU decoder does (bounded by
/// `MUL_MAP_MAX_PROBE`). On failure the table is left unfitted.
fn verify_exact_map_verbatim(
    samples: &[(Vec<u64>, u64)],
    kv: &[u64],
    nkey: usize,
    slots: usize,
    collided: bool,
) -> bool {
    let stride = nkey + 1;
    for (t, r) in samples.iter() {
        let mut i = (hash_key(&t[..nkey]) as usize) & (slots - 1);
        let mut found = None;
        for _ in 0..slots.min(MUL_MAP_MAX_PROBE) {
            let slot = i * stride;
            if kv[slot..slot + nkey].iter().all(|v| *v == u64::MAX) {
                break;
            }
            if kv[slot..slot + nkey] == t[..nkey] {
                found = Some(kv[slot + nkey]);
                break;
            }
            i = (i + 1) & (slots - 1);
        }
        let ok = match found {
            Some(got) => got == *r || collided,
            None => false,
        };
        if !ok {
            tracing::error!(
                "verbatim map does not probe back: key {:?}, expected row {r}, got {found:?} -- \
                 table left unfitted",
                &t[..nkey]
            );
            return false;
        }
    }
    true
}

/// Must match mulResolveRow on the C side.
fn hash_key(key: &[u64]) -> u64 {
    let mut h: u64 = 0;
    for k in key {
        h ^= k.wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_add(h << 6).wrapping_add(h >> 2);
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 32;
    }
    h
}

/// How many elements each table's lookups supply. The key width comes from the assumes side, not
/// the table's (padded) COL_* groups.
fn lookup_arity<F: PrimeField64>(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>) -> HashMap<u64, usize> {
    let mut arity: HashMap<u64, usize> = HashMap::new();
    for (airgroup_id, airs) in pctx.global_info.airs.iter().enumerate() {
        for air_id in 0..airs.len() {
            let Ok(setup) = sctx.get_setup(airgroup_id, air_id) else { continue };
            let hints = get_hint_ids_by_name(setup.p_setup.p_expressions_bin, "gsum_debug_data");
            for h in hints {
                let o = HintFieldOptions::default();
                let Ok(ty) = get_hint_field_constant_as::<u64, F>(
                    pctx,
                    setup,
                    airgroup_id,
                    air_id,
                    h as usize,
                    "type_piop",
                    o.clone(),
                ) else {
                    continue;
                };
                if ty != 0 {
                    continue;
                } // assumes side only
                let Ok(ids) = get_hint_field_constant_a_as::<u64, F>(
                    pctx,
                    setup,
                    airgroup_id,
                    air_id,
                    h as usize,
                    "opids",
                    o.clone(),
                ) else {
                    continue;
                };
                let Ok(len) = get_hint_field_constant_as::<u64, F>(
                    pctx,
                    setup,
                    airgroup_id,
                    air_id,
                    h as usize,
                    "len_expressions",
                    o,
                ) else {
                    continue;
                };
                for id in ids {
                    // On disagreement keep the smallest, so the planner catches the mismatch.
                    let e = arity.entry(id).or_insert(len as usize);
                    if (len as usize) < *e {
                        *e = len as usize;
                    }
                }
            }
        }
    }
    arity
}

/// The column each tuple element really reads, per proves-side `gsum_debug_data` hint of `tid`
/// (its `name_exprs`, aliases included).
///
/// `COL_*` names alone are not enough: `simplify_virtual_fixed` drops constant columns and
/// deduplicates identical ones across groups, leaving holes in a group's `j` sequence.
fn proves_side_tuple_names<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    setup: &Setup<F>,
    airgroup_id: usize,
    air_id: usize,
    tid: u64,
) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    for h in get_hint_ids_by_name(setup.p_setup.p_expressions_bin, "gsum_debug_data") {
        let o = HintFieldOptions::default();
        let Ok(ty) =
            get_hint_field_constant_as::<u64, F>(pctx, setup, airgroup_id, air_id, h as usize, "type_piop", o.clone())
        else {
            continue;
        };
        if ty != 1 {
            continue;
        } // proves side: the table's own rows
        let Ok(ids) =
            get_hint_field_constant_a_as::<u64, F>(pctx, setup, airgroup_id, air_id, h as usize, "opids", o.clone())
        else {
            continue;
        };
        if !ids.contains(&tid) {
            continue;
        }
        let Ok(names) =
            get_hint_field_constant_a_as_string::<F>(pctx, setup, airgroup_id, air_id, h as usize, "name_exprs", o)
        else {
            continue;
        };
        if !names.is_empty() {
            out.push(names);
        }
    }
    out
}

/// Recover an exact row map for every virtual table from its `COL_*`/`UID_*` fixed columns.
///
/// Tables that do not fit are skipped; the caller requires them to be std-owned. Range
/// tables (no `COL_*` group) are claimed via `collect_prover_owned_ranges` instead.
pub fn fit_virtual_table_maps<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    layouts: &[VtLayout],
) -> ProofmanResult<(Vec<VtFittedMap>, VtFitSummary)> {
    use std::io::{BufReader, Read};

    let arity = lookup_arity(pctx, sctx);
    let mut out = Vec::new();
    // Tallied for the caller's summary; per-table logs below are debug/trace only.
    let __t0 = std::time::Instant::now();
    let mut considered = 0usize;
    let mut considered_ids: Vec<u64> = Vec::new();
    let mut exact_bytes: u64 = 0;
    for l in layouts {
        let (airgroup_id, air_id) = (l.airgroup_id as usize, l.air_id as usize);
        let setup = sctx.get_setup(airgroup_id, air_id)?;
        let acc_heights = &l.acc_bases;
        let num_muls = l.num_cols as usize;
        let num_rows = l.num_rows as usize;
        let n_const = setup.stark_info.n_constants as usize;
        let Some(pol_map) = setup.stark_info.const_pols_map.as_ref() else { continue };

        // COL_<g>_<base>_<k> and UID_<g>: the group is the tuple, the uid says which table a row
        // belongs to.
        let mut groups: std::collections::BTreeMap<u64, (Vec<(u64, usize)>, Option<usize>)> = Default::default();
        for (idx, pm) in pol_map.iter().enumerate() {
            let name = pm.name.as_str();
            if let Some(rest) = name.strip_prefix("COL_") {
                let parts: Vec<&str> = rest.split('_').collect();
                if parts.len() == 3 {
                    if let (Ok(g), Ok(k)) = (parts[0].parse::<u64>(), parts[2].parse::<u64>()) {
                        groups.entry(g).or_default().0.push((k, idx));
                    }
                }
            } else if let Some(rest) = name.strip_prefix("UID_") {
                if let Ok(g) = rest.parse::<u64>() {
                    groups.entry(g).or_default().1 = Some(idx);
                }
            }
        }
        for v in groups.values_mut() {
            v.0.sort_unstable();
        }
        let name_to_idx: HashMap<&str, usize> =
            pol_map.iter().enumerate().map(|(i, pm)| (pm.name.as_str(), i)).collect();
        if groups.is_empty() {
            continue;
        }

        // The .const is raw row-major u64 with an n_constants stride.
        let path = setup.const_pols_path.replace(".const_gpu", ".const");
        let Ok(file) = std::fs::File::open(&path) else {
            tracing::debug!("virtual table fit: cannot open {path}; skipping");
            continue;
        };
        // A group with no UID column belongs wholly to one table.
        let uid_idx: Vec<(u64, usize)> = groups.iter().filter_map(|(g, v)| v.1.map(|u| (*g, u))).collect();
        let mut uids: std::collections::BTreeMap<u64, Vec<u64>> =
            uid_idx.iter().map(|(g, _)| (*g, Vec::with_capacity(num_rows))).collect();
        // Read the whole .const once; every table of this air indexes into it.
        let row_bytes = n_const * 8;
        let mut const_buf = vec![0u8; num_rows * row_bytes];
        let complete_rows = {
            let mut rd = BufReader::with_capacity(1 << 22, &file);
            let mut filled = 0usize;
            loop {
                match rd.read(&mut const_buf[filled..]) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => filled += n,
                }
            }
            filled / row_bytes
        };
        for r in 0..complete_rows {
            let row = &const_buf[r * row_bytes..(r + 1) * row_bytes];
            for (g, c) in uid_idx.iter() {
                let b = &row[c * 8..c * 8 + 8];
                uids.get_mut(g).unwrap().push(u64::from_le_bytes(b.try_into().unwrap()));
            }
        }

        for (t, &tid) in l.table_ids.iter().enumerate() {
            let base = acc_heights[t];
            let end = acc_heights.iter().copied().filter(|h| *h > base).min().unwrap_or((num_muls * num_rows) as u64);
            let height = end - base;
            if height == 0 {
                continue;
            }
            considered += 1;
            considered_ids.push(tid);
            let t_fit = std::time::Instant::now();

            // A table's entries occupy accumulator offsets [base, base+height), each an
            // (accumulator column, air row) pair.
            let mut rows_of_table: Vec<(usize, u64, u64)> = Vec::new(); // (air_row, row_in_table, group)
            let mut widths: Vec<usize> = Vec::new();

            // A table starts at column `base >> shift` and its groups, in ascending group order,
            // occupy consecutive columns. Derived, not searched (search is ambiguous across columns).
            let shift = num_rows.trailing_zeros() as u64;
            let c0 = base >> shift;
            let c_end = (base + height - 1) >> shift;
            let mut mine_by_group: Vec<(u64, Vec<usize>)> = Vec::new();
            for (g, gi) in groups.iter() {
                match gi.1 {
                    Some(_) => {
                        let rows_here: Vec<usize> = uids[g]
                            .iter()
                            .enumerate()
                            .filter_map(|(r, u)| if *u == tid { Some(r) } else { None })
                            .collect();
                        if !rows_here.is_empty() {
                            mine_by_group.push((*g, rows_here));
                        }
                    }
                    None => mine_by_group.push((*g, Vec::new())), // claimed below only if its column is ours
                }
            }
            mine_by_group.sort_by_key(|(g, _)| *g);

            // What the PIL says each of this table's groups holds.
            let hint_names = proves_side_tuple_names(pctx, setup, airgroup_id, air_id, tid);

            // Walk the table's columns and the groups together.
            let mut col = c0;
            let mut ok = true;
            for (g, rows_here) in mine_by_group.iter_mut() {
                if col > c_end {
                    break;
                }
                let has_uid = groups[g].1.is_some();
                debug_assert!(!has_uid || !rows_here.is_empty());
                if !has_uid {
                    // Only a table that already owns a UID-bearing group may absorb UID-less ones,
                    // or a range table would swallow one and override its working decoder.
                    if widths.is_empty() {
                        continue;
                    }

                    // A UID-less group's column is ours only if it lies inside the table's span.
                    let off0 = col * num_rows as u64;
                    if off0 < base || off0 >= end {
                        continue;
                    }
                    *rows_here = (0..num_rows).collect();
                }
                // A UID-less group of identical rows is filler, not entries. The column counter must
                // still advance past it, or every later group is off by one.
                if !has_uid && !rows_here.is_empty() {
                    let cset: Vec<usize> = groups[g].0.iter().map(|(_, c)| *c).collect();
                    let mut first: Option<Vec<u64>> = None;
                    let mut constant = true;
                    for probe in [0usize, 1, num_rows / 2, num_rows - 1] {
                        if probe >= complete_rows {
                            break;
                        }
                        let buf = &const_buf[probe * row_bytes..(probe + 1) * row_bytes];
                        let t: Vec<u64> = cset
                            .iter()
                            .map(|c| u64::from_le_bytes(buf[c * 8..c * 8 + 8].try_into().unwrap()))
                            .collect();
                        match &first {
                            None => first = Some(t),
                            Some(f0) => {
                                if *f0 != t {
                                    constant = false;
                                    break;
                                }
                            }
                        }
                    }
                    if constant {
                        col += 1;
                        continue;
                    }
                }
                for a in rows_here.iter() {
                    let off = col * num_rows as u64 + *a as u64;
                    if off < base || off >= end {
                        ok = false;
                        break;
                    }
                    rows_of_table.push((*a, off - base, *g));
                }
                // Width is the last column index + 1, not the surviving count (see
                // `proves_side_tuple_names`); holes are resolved from the hint below.
                widths.push(groups[g].0.last().map(|(j, _)| *j as usize + 1).unwrap_or(0));
                col += 1;
                if !ok {
                    break;
                }
            }

            if !ok || rows_of_table.is_empty() || widths.is_empty() || rows_of_table.len() as u64 > height {
                tracing::trace!(
                    "virtual table {tid}: SKIPPED before fitting (ok={ok} rows={} height={height} groups={})",
                    rows_of_table.len(),
                    widths.len()
                );
                continue;
            }

            // The lookup states how many elements it sends; that is the key width.
            let Some(width) = arity.get(&tid).copied() else {
                tracing::debug!("virtual table {tid}: no lookup found, not fitted");
                continue;
            };
            if width == 0 || width > 8 {
                continue;
            }
            // Bind each group to the proves-side hint whose `name_exprs` matches every column the
            // group kept, at the same position. Derived, never assumed from ordering.
            let mut tuple_cols: HashMap<u64, Vec<usize>> = HashMap::new();
            for (g, rows_here) in mine_by_group.iter() {
                if rows_here.is_empty() {
                    continue;
                }
                let matches: Vec<&Vec<String>> = hint_names
                    .iter()
                    .filter(|names| {
                        groups[g].0.iter().all(|(j, idx)| {
                            names.get(*j as usize).map(|n| n.as_str()) == Some(pol_map[*idx].name.as_str())
                        })
                    })
                    .collect();
                // Exactly one hint must match; otherwise leave the table to the std rather than guess.
                let [only] = matches[..] else {
                    tracing::trace!(
                        "virtual table {tid}: group {g} matches {} proves-side hints, not one -- \
                         not fitted",
                        matches.len()
                    );
                    ok = false;
                    break;
                };
                // Every element's real column, aliases included.
                let cols: Option<Vec<usize>> = only.iter().map(|n| name_to_idx.get(n.as_str()).copied()).collect();
                let Some(cols) = cols else {
                    tracing::debug!("virtual table {tid}: group {g} names a column the setup does not hold");
                    ok = false;
                    break;
                };
                tuple_cols.insert(*g, cols);
            }
            if !ok {
                continue;
            }

            // A padded group may carry more elements than the lookup sends (the key is the leading
            // `width`), but never fewer.
            if let Some(short) = tuple_cols.values().map(|c| c.len()).filter(|n| *n < width).min() {
                tracing::debug!(
                    "virtual table {tid}: lookup sends {width} elements but a group carries only \
                     {short} -- not fitted"
                );
                continue;
            }

            // Tuples for this table, in accumulator order.
            rows_of_table.sort_unstable_by_key(|(_, r, _)| *r);
            let mut samples: Vec<(Vec<u64>, u64)> = vec![(Vec::new(), 0); rows_of_table.len()];
            for (i, &(air_row, row_in_table, g)) in rows_of_table.iter().enumerate() {
                if air_row >= complete_rows {
                    ok = false;
                    break;
                }
                let buf = &const_buf[air_row * row_bytes..(air_row + 1) * row_bytes];
                let tuple: Vec<u64> = tuple_cols[&g][..width]
                    .iter()
                    .map(|c| u64::from_le_bytes(buf[c * 8..c * 8 + 8].try_into().unwrap()))
                    .collect();
                samples[i] = (tuple, row_in_table);
            }
            if !ok {
                tracing::debug!("virtual table {tid}: proves-side tuple unavailable, not fitted");
                continue;
            }

            if let Some((nkey, kv, slots)) = fit_exact_map(&samples, width) {
                // Collisions are genuine duplicate tuples; merging them is sound.
                exact_bytes += kv.len() as u64 * 8;
                tracing::debug!(
                    "virtual table {tid}: exact map over {} entries, {nkey} key columns, {slots} slots \
                     ({} MB, {} ms)",
                    samples.len(),
                    kv.len() * 8 / 1_000_000,
                    t_fit.elapsed().as_millis()
                );
                out.push(VtFittedMap { table_id: tid, map: (nkey, kv, slots) });
            } else {
                let ranges: Vec<u64> =
                    (0..width).map(|j| samples.iter().map(|(t, _)| t[j]).max().unwrap_or(0)).collect();
                tracing::debug!(
                    "virtual table {tid}: NO FIT ({} entries, {width} columns, {} ms) -- no exact map \
                     fits; column maxima {ranges:?}",
                    samples.len(),
                    t_fit.elapsed().as_millis()
                );
            }
        }
    }
    // Unfitted: range tables or std-owned; the caller tells them apart.
    let fitted_ids: std::collections::HashSet<u64> = out.iter().map(|m| m.table_id).collect();
    let unclaimed_ids: Vec<u64> = considered_ids.into_iter().filter(|tid| !fitted_ids.contains(tid)).collect();
    Ok((out, VtFitSummary { considered, exact_bytes, elapsed_ms: __t0.elapsed().as_millis(), unclaimed_ids }))
}

pub fn collect_virtual_table_layouts<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
) -> ProofmanResult<Vec<VtLayout>> {
    let global_hint = get_hint_ids_by_name(sctx.get_global_bin(), "virtual_table_data_global");
    if global_hint.is_empty() {
        return Ok(Vec::new());
    }
    let airgroup_ids = get_global_hint_field_constant_a_as::<usize, F>(sctx, global_hint[0], "airgroup_ids")?;
    let air_ids = get_global_hint_field_constant_a_as::<usize, F>(sctx, global_hint[0], "air_ids")?;

    let mut out = Vec::with_capacity(airgroup_ids.len());
    for i in 0..airgroup_ids.len() {
        let (airgroup_id, air_id) = (airgroup_ids[i], air_ids[i]);
        let setup = sctx.get_setup(airgroup_id, air_id)?;
        let hint_id = get_hint_ids_by_name(setup.p_setup.p_expressions_bin, "virtual_table_data")[0] as usize;
        let o = HintFieldOptions::default();
        let table_ids = get_hint_field_constant_a_as::<usize, F>(
            pctx,
            setup,
            airgroup_id,
            air_id,
            hint_id,
            "table_ids",
            o.clone(),
        )?;
        let acc_heights = get_hint_field_constant_a_as::<u64, F>(
            pctx,
            setup,
            airgroup_id,
            air_id,
            hint_id,
            "acc_heights",
            o.clone(),
        )?;
        let num_muls =
            get_hint_field_constant_as::<usize, F>(pctx, setup, airgroup_id, air_id, hint_id, "num_muls", o)?;
        out.push(VtLayout {
            airgroup_id: airgroup_id as u64,
            air_id: air_id as u64,
            num_rows: pctx.global_info.airs[airgroup_id][air_id].num_rows as u64,
            num_cols: num_muls as u64,
            table_ids: table_ids.iter().map(|id| *id as u64).collect(),
            acc_bases: acc_heights,
        });
    }
    Ok(out)
}

impl<F: PrimeField64> StdVirtualTable<F> {
    pub fn new(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>, shared_tables: bool) -> ProofmanResult<Arc<Self>> {
        // Get relevant data from the global hint
        let virtual_table_global_hint = get_hint_ids_by_name(sctx.get_global_bin(), "virtual_table_data_global");
        if virtual_table_global_hint.is_empty() {
            return Ok(Arc::new(Self {
                _phantom: std::marker::PhantomData,
                global_id_by_uid: HashMap::new(),
                indices_by_global_id: Vec::new(),
                virtual_table_airs: None,
            }));
        }

        let airgroup_ids =
            get_global_hint_field_constant_a_as::<usize, F>(sctx, virtual_table_global_hint[0], "airgroup_ids")?;
        let air_ids = get_global_hint_field_constant_a_as::<usize, F>(sctx, virtual_table_global_hint[0], "air_ids")?;

        let num_virtual_tables = airgroup_ids.len();
        let mut virtual_tables = Vec::with_capacity(num_virtual_tables);
        let mut global_id_by_uid = HashMap::new();
        let mut indices_by_global_id = Vec::new();
        let mut current_global_id = 0;
        for i in 0..num_virtual_tables {
            let airgroup_id = airgroup_ids[i];
            let air_id = air_ids[i];

            // Get the Virtual Table structure
            let setup = sctx.get_setup(airgroup_id, air_id)?;
            let hint_id = get_hint_ids_by_name(setup.p_setup.p_expressions_bin, "virtual_table_data")[0] as usize;

            let hint_opt = HintFieldOptions::default();
            let table_ids = get_hint_field_constant_a_as::<usize, F>(
                pctx,
                setup,
                airgroup_id,
                air_id,
                hint_id,
                "table_ids",
                hint_opt.clone(),
            )?;
            let acc_heights = get_hint_field_constant_a_as::<u64, F>(
                pctx,
                setup,
                airgroup_id,
                air_id,
                hint_id,
                "acc_heights",
                hint_opt.clone(),
            )?;
            let num_muls = get_hint_field_constant_as::<usize, F>(
                pctx,
                setup,
                airgroup_id,
                air_id,
                hint_id,
                "num_muls",
                hint_opt.clone(),
            )?;

            // Map each table_id to an ordered set of indexes
            let num_table_ids = table_ids.len();
            let mut idxs = vec![(0, 0); num_table_ids];
            for j in 0..num_table_ids {
                idxs[j] = (table_ids[j], acc_heights[j]);

                // Update global ID mapping: global_idx -> (air_idx, uid, uid_idx)
                // global_id_map.insert(current_global_id, (i, table_ids[j], j));
                global_id_by_uid.insert(table_ids[j], current_global_id);
                indices_by_global_id.push((i, j));
                current_global_id += 1;
            }

            let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;
            let multiplicities: Vec<AtomicU64> =
                (0..(num_muls as usize * num_rows)).into_par_iter().map(|_| AtomicU64::new(0)).collect();

            let buffer = vec![F::ZERO; num_muls as usize * num_rows];
            // The reclaim slot returns this same allocation every iteration, so one pin covers every H2D.
            let trace_buffer_pinned = if pctx.gpu { register_host_buffer(&buffer) } else { None };
            let trace_buffer = Arc::new(Mutex::new(Some(buffer)));

            let virtual_table_air = VirtualTableAir::<F> {
                airgroup_id,
                air_id,
                shift: num_rows.trailing_zeros() as u64,
                mask: (num_rows - 1) as u64,
                num_rows,
                num_cols: num_muls as usize,
                table_ids: idxs,
                prover_owned: std::sync::OnceLock::new(),
                multiplicities,
                table_instance_id: AtomicU64::new(0),
                calculated: AtomicBool::new(false),
                shared_tables,
                trace_buffer,
                trace_buffer_pinned,
            };
            virtual_tables.push(Arc::new(virtual_table_air));
        }

        Ok(Arc::new(Self {
            _phantom: std::marker::PhantomData,
            global_id_by_uid,
            indices_by_global_id,
            virtual_table_airs: Some(virtual_tables),
        }))
    }

    pub fn get_global_id(&self, id: usize) -> ProofmanResult<usize> {
        self.global_id_by_uid
            .get(&id)
            .copied()
            .ok_or_else(|| ProofmanError::StdError(format!("ID {id} not found in the global ID map")))
    }

    pub fn inc_virtual_row(&self, global_id: usize, row: u64, multiplicity: u64) {
        let (air_idx, uid_idx) = self.indices_by_global_id[global_id];
        self.virtual_table_airs.as_ref().unwrap()[air_idx].inc_virtual_row(uid_idx, row, multiplicity);
    }

    /// Generic over the caller's row/multiplicity width so the slices cross as-is: the
    /// widening happens per element inside the iterator, never through an intermediate Vec.
    pub fn inc_virtual_rows<R: RCMultiplicity, M: RCMultiplicity>(
        &self,
        global_id: usize,
        rows: &[R],
        multiplicities: &[M],
    ) {
        debug_assert!(!rows.is_empty() && rows.len() == multiplicities.len());
        let pairs = rows.iter().copied().zip(multiplicities.iter().copied()).map(|(r, m)| (r.to_u64(), m.to_u64()));
        self.inc_virtual_pairs(global_id, pairs);
    }

    pub fn inc_virtual_rows_same_mul<R: RCMultiplicity>(&self, global_id: usize, rows: &[R], multiplicity: u64) {
        let pairs = rows.iter().copied().map(move |r| (r.to_u64(), multiplicity));
        self.inc_virtual_pairs(global_id, pairs);
    }

    pub fn inc_virtual_rows_ranged<M: RCMultiplicity>(
        &self,
        global_id: usize,
        start: Option<u64>,
        multiplicities: &[M],
    ) {
        let start = start.unwrap_or(0);
        // Compute (row, multiplicity) pairs on the fly — no Vec allocation.
        let pairs = multiplicities.iter().copied().enumerate().map(move |(i, m)| (start + i as u64, m.to_u64()));
        self.inc_virtual_pairs(global_id, pairs);
    }

    /// Increment multiplicities directly from an iterator of (row, multiplicity) pairs.
    /// Lets callers avoid materializing an intermediate Vec<u64> of rows.
    pub fn inc_virtual_pairs(&self, global_id: usize, pairs: impl Iterator<Item = (u64, u64)>) {
        let (air_idx, uid_idx) = self.indices_by_global_id[global_id];
        self.virtual_table_airs.as_ref().unwrap()[air_idx].inc_virtual_pairs(uid_idx, pairs);
    }
}

#[cfg(test)]
impl<F: PrimeField64> StdVirtualTable<F> {
    /// Single-AIR virtual table holding `table_ids` (table_id, acc_height) entries.
    /// Global ids are handed out in the order given, matching `new`.
    pub(crate) fn for_test(num_rows: usize, num_cols: usize, table_ids: Vec<(usize, u64)>) -> Arc<Self> {
        let global_id_by_uid = table_ids.iter().enumerate().map(|(j, &(uid, _))| (uid, j)).collect();
        let indices_by_global_id = (0..table_ids.len()).map(|j| (0, j)).collect();
        let air = VirtualTableAir::<F> {
            airgroup_id: 0,
            air_id: 0,
            shift: num_rows.trailing_zeros() as u64,
            mask: (num_rows - 1) as u64,
            num_rows,
            num_cols,
            prover_owned: std::sync::OnceLock::new(),
            table_ids,
            multiplicities: (0..num_cols * num_rows).map(|_| AtomicU64::new(0)).collect(),
            table_instance_id: AtomicU64::new(0),
            calculated: AtomicBool::new(false),
            shared_tables: false,
            trace_buffer: Arc::new(Mutex::new(Some(vec![F::ZERO; num_cols * num_rows]))),
            trace_buffer_pinned: None,
        };
        Arc::new(Self {
            _phantom: std::marker::PhantomData,
            global_id_by_uid,
            indices_by_global_id,
            virtual_table_airs: Some(vec![Arc::new(air)]),
        })
    }

    pub(crate) fn snapshot(&self) -> Vec<u64> {
        self.virtual_table_airs.as_ref().unwrap()[0].multiplicities.iter().map(|m| m.load(Ordering::Relaxed)).collect()
    }
}

impl<F: PrimeField64 + Send + Sync + 'static> WitnessComponent<F> for StdVirtualTable<F> {
    fn pre_calculate_witness(
        &self,
        _stage: u32,
        _pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        Ok(())
    }
}

impl<F: PrimeField64> VirtualTableAir<F> {
    pub fn get_id(&self, id: usize) -> ProofmanResult<usize> {
        if let Some(pos) = self.table_ids.iter().position(|&(table_id, _)| table_id == id) {
            Ok(pos)
        } else {
            Err(ProofmanError::StdError("ID not found in the virtual table".to_string()))
        }
    }

    /// Whether the prover counts this table itself, resolved once on first use.
    fn is_prover_owned(&self, id: usize) -> bool {
        self.prover_owned.get_or_init(|| {
            let owned = prover_owned_tables();
            let flags: Vec<bool> = self.table_ids.iter().map(|(t, _)| owned.contains(&(*t as u64))).collect();
            let n = flags.iter().filter(|o| **o).count();
            if n > 0 {
                tracing::info!("VirtualTable air {}: {n}/{} tables counted by the prover", self.air_id, flags.len());
            }
            flags
        })[id]
    }

    /// Core update function: Updates multiplicities for row/multiplicity pairs
    fn update(&self, id: usize, iter: impl Iterator<Item = (u64, u64)>) {
        if self.is_prover_owned(id) || self.calculated.load(Ordering::Relaxed) {
            return;
        }
        let table_offset = self.table_ids[id].1;

        for (row, multiplicity) in iter {
            if multiplicity == 0 {
                continue;
            }

            // Get the offset
            let offset = table_offset + row;

            // Map it to the appropriate multiplicity
            let sub_table_idx = offset >> self.shift;

            // Get the row index
            let row_idx = offset & self.mask;

            // Update the multiplicity (col-major flat layout)
            self.multiplicities[sub_table_idx as usize * self.num_rows + row_idx as usize]
                .fetch_add(multiplicity, Ordering::Relaxed);
        }
    }

    pub fn inc_virtual_row(&self, id: usize, row: u64, multiplicity: u64) {
        self.update(id, std::iter::once((row, multiplicity)));
    }

    /// Increment multiplicities directly from an iterator of (row, multiplicity) pairs.
    pub fn inc_virtual_pairs(&self, id: usize, pairs: impl Iterator<Item = (u64, u64)>) {
        self.update(id, pairs);
    }
}

impl<F: PrimeField64 + Send + Sync + 'static> WitnessComponent<F> for VirtualTableAir<F> {
    fn execute(
        &self,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _global_ids: &RwLock<Vec<usize>>,
    ) -> ProofmanResult<()> {
        let (instance_found, mut table_instance_id) = pctx.dctx_find_process_table(self.airgroup_id, self.air_id)?;

        if !instance_found {
            if !self.shared_tables {
                table_instance_id = pctx.add_table_all(self.airgroup_id, self.air_id)?;
            } else {
                table_instance_id = pctx.add_table(self.airgroup_id, self.air_id)?;
            }
        }

        self.calculated.store(false, Ordering::Relaxed);
        self.multiplicities.par_iter().for_each(|v| {
            v.store(0, Ordering::Relaxed);
        });
        // The prover-side accumulators are reset in proofman.rs: `execute` is fork-exposed
        // (--asm spawns the microservices here) and an FFI call there kills the process silently.

        self.table_instance_id.store(table_instance_id as u64, Ordering::SeqCst);
        Ok(())
    }

    fn pre_calculate_witness(
        &self,
        _stage: u32,
        _pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        Ok(())
    }

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        _instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        if stage == 1 {
            let table_instance_id = self.table_instance_id.load(Ordering::Relaxed) as usize;

            let instance_id = pctx.dctx_get_table_instance_idx(table_instance_id)?;

            if !_instance_ids.contains(&instance_id) {
                return Ok(());
            }

            self.calculated.store(true, Ordering::Relaxed);

            // Device-owned air: the counts are already on the GPU, so skip the host path and the
            // emptiness check (which would see zeros and drop the instance).
            let device_owned = pctx.device_owned_table_airs.read().unwrap().contains(&self.air_id);

            // Before `distribute_multiplicities`, so the MPI path sees a complete accumulator.
            if let Some(counts) = pctx.prover_counts.read().unwrap().get(&self.air_id).filter(|_| !device_owned) {
                for (slot, add) in self.multiplicities.iter().zip(counts.iter()) {
                    if *add != 0 {
                        slot.fetch_add(*add, Ordering::Relaxed);
                    }
                }
            }

            // An assigned table is computed only on its single owner node; its
            // multiplicities are produced there, so there is no cross-rank reduction.
            let assigned = pctx.dctx_is_assigned_table(instance_id)?;

            if self.shared_tables && !assigned && !device_owned {
                let owner_idx = pctx.dctx_get_process_owner_instance(instance_id)?;
                pctx.mpi_ctx.distribute_multiplicities(&self.multiplicities, self.num_cols, self.num_rows, owner_idx);
            }

            if (!self.shared_tables && !assigned) || pctx.dctx_is_my_process_instance(instance_id)? {
                let buffer_size = self.num_cols * self.num_rows;
                // The slot is pre-populated by `new` and refilled by the reclaim hook
                // on every prior iteration's clear_traces / Drop. If it's empty here,
                // the reclaim path is broken.
                let mut buffer = self
                    .trace_buffer
                    .lock()
                    .unwrap()
                    .take()
                    .expect("VirtualTableAir trace_buffer must be populated by reclaim before calculate_witness");
                debug_assert_eq!(buffer.len(), buffer_size);
                let any_nonzero = std::sync::atomic::AtomicBool::new(device_owned);
                let num_rows = self.num_rows;
                if !device_owned {
                    buffer.par_chunks_mut(self.num_cols).enumerate().for_each(|(row, chunk)| {
                        for (col, slot) in chunk.iter_mut().enumerate() {
                            let v = self.multiplicities[col * num_rows + row].load(Ordering::Relaxed);
                            if v != 0 {
                                any_nonzero.store(true, Ordering::Relaxed);
                            }
                            *slot = F::from_u64(v);
                        }
                    });
                }
                if !any_nonzero.load(Ordering::Relaxed) {
                    tracing::info!(
                        "Skipping uninitialized virtual table (airgroup_id: {}, air_id: {})",
                        self.airgroup_id,
                        self.air_id
                    );
                    pctx.dctx_skip_process_instance(instance_id);
                    *self.trace_buffer.lock().unwrap() = Some(buffer);
                    return Ok(());
                }
                let setup = sctx.get_setup(self.airgroup_id, self.air_id)?;
                let n_cols = setup.stark_info.map_sections_n["cm1"] as usize;
                let air_instance = AirInstance::new(
                    TraceInfo::new(self.airgroup_id, self.air_id, n_cols, self.num_rows, buffer, false, false)
                        .with_reclaim_slot(self.trace_buffer.clone()),
                );
                pctx.add_air_instance(air_instance, instance_id);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proofman_fields::Goldilocks as F;

    const ROWS: usize = 64;
    const COLS: usize = 4;

    /// Two tables at different accumulated heights, so the offset is exercised too.
    fn table() -> Arc<StdVirtualTable<F>> {
        StdVirtualTable::<F>::for_test(ROWS, COLS, vec![(7, 0), (9, 100)])
    }

    fn assert_wrote_something(snapshot: &[u64]) {
        assert!(snapshot.iter().any(|&m| m != 0), "wrote nothing — the comparison would be vacuous");
    }

    #[test]
    fn inc_virtual_rows_matches_repeated_inc_virtual_row() {
        for global_id in 0..2 {
            let rows: Vec<u32> = vec![0, 1, 5, 63, 64, 99, 5];
            let muls: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7];

            let reference = table();
            for (&r, &m) in rows.iter().zip(muls.iter()) {
                reference.inc_virtual_row(global_id, r as u64, m as u64);
            }

            let batched = table();
            batched.inc_virtual_rows(global_id, &rows, &muls);

            let expected = reference.snapshot();
            assert_wrote_something(&expected);
            assert_eq!(expected, batched.snapshot(), "inc_virtual_rows diverged for global_id {global_id}");
        }
    }

    #[test]
    fn inc_virtual_rows_same_mul_matches_repeated_inc_virtual_row() {
        for global_id in 0..2 {
            let rows: Vec<u32> = vec![0, 1, 5, 63, 99, 5];
            let mul = 11u64;

            let reference = table();
            for &r in rows.iter() {
                reference.inc_virtual_row(global_id, r as u64, mul);
            }

            let batched = table();
            batched.inc_virtual_rows_same_mul(global_id, &rows, mul);

            let expected = reference.snapshot();
            assert_wrote_something(&expected);
            assert_eq!(expected, batched.snapshot(), "inc_virtual_rows_same_mul diverged for global_id {global_id}");
        }
    }

    #[test]
    fn inc_virtual_rows_ranged_matches_repeated_inc_virtual_row() {
        for global_id in 0..2 {
            let start = 5u64;
            let muls: Vec<u32> = (0..20u32).map(|i| i % 4).collect();

            let reference = table();
            for (i, &m) in muls.iter().enumerate() {
                reference.inc_virtual_row(global_id, start + i as u64, m as u64);
            }

            let batched = table();
            batched.inc_virtual_rows_ranged(global_id, Some(start), &muls);

            let expected = reference.snapshot();
            assert_wrote_something(&expected);
            assert_eq!(expected, batched.snapshot(), "inc_virtual_rows_ranged diverged for global_id {global_id}");
        }
    }

    /// `start: None` must mean row 0.
    #[test]
    fn inc_virtual_rows_ranged_defaults_start_to_zero() {
        let muls: Vec<u32> = (1..9u32).collect();

        let explicit = table();
        explicit.inc_virtual_rows_ranged(0, Some(0), &muls);

        let defaulted = table();
        defaulted.inc_virtual_rows_ranged(0, None, &muls);

        let expected = explicit.snapshot();
        assert_wrote_something(&expected);
        assert_eq!(expected, defaulted.snapshot());
    }

    /// A/B harness modelling Mem's 2^22-entry u32 histogram. Measured on this machine:
    /// 19.5 ms/call when the multiplicities were widened into a Vec<u64> first, 12.5 ms
    /// passing the u32 slice straight through.
    /// `cargo test -p pil2-std-lib --lib --release bench_ranged -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_ranged() {
        use std::time::Instant;

        const ROWS: usize = 4096;
        const COLS: usize = 1024; // ROWS * COLS = 2^22
        const REPS: usize = 20;

        // A realistic histogram: mostly small counts, ~30% of buckets untouched.
        let muls: Vec<u32> = (0..ROWS * COLS).map(|i| ((i * 2654435761) % 10) as u32).collect();
        let t = StdVirtualTable::<F>::for_test(ROWS, COLS, vec![(0, 0)]);

        // Warm the pages so the first rep doesn't dominate.
        t.inc_virtual_rows_ranged(0, None, &muls);

        let start = Instant::now();
        for _ in 0..REPS {
            t.inc_virtual_rows_ranged(0, None, &muls);
        }
        let elapsed = start.elapsed();
        println!("BENCH inc_virtual_rows_ranged: {:?} per call ({REPS} reps)", elapsed / REPS as u32);
    }

    /// The row/multiplicity widths must not change the result.
    #[test]
    fn row_and_multiplicity_widths_are_irrelevant() {
        let rows64: Vec<u64> = vec![0, 1, 5, 63, 99];
        let muls64: Vec<u64> = vec![1, 2, 3, 4, 5];

        let wide = table();
        wide.inc_virtual_rows(0, &rows64, &muls64);
        let expected = wide.snapshot();
        assert_wrote_something(&expected);

        let rows16: Vec<u16> = rows64.iter().map(|&r| r as u16).collect();
        let muls16: Vec<u16> = muls64.iter().map(|&m| m as u16).collect();
        let narrow = table();
        narrow.inc_virtual_rows(0, &rows16, &muls16);
        assert_eq!(expected, narrow.snapshot(), "u16 rows/multiplicities diverged");

        let rows_sz: Vec<usize> = rows64.iter().map(|&r| r as usize).collect();
        let muls32: Vec<u32> = muls64.iter().map(|&m| m as u32).collect();
        let mixed = table();
        mixed.inc_virtual_rows(0, &rows_sz, &muls32);
        assert_eq!(expected, mixed.snapshot(), "mixed usize/u32 widths diverged");
    }

    // -----------------------------------------------------------------------------------------
    // Row-map fitting on small hand-built (tuple, row) samples; no proving key needed.

    /// Probe a packed `fit_exact_map` result as the GPU decoder does.
    fn probe_packed_map(kv: &[u64], nkey: usize, slots: u64, tuple: &[u64]) -> Option<u64> {
        assert_eq!(kv[0], MUL_MAP_PACKED, "test setup: expected the packed layout");
        let head = 1 + nkey;
        let shifts: Vec<u32> = (0..nkey).map(|c| (kv[1 + c] >> 32) as u32).collect();
        let k = (0..nkey).fold(0u64, |acc, c| acc | (tuple[c] << shifts[c]));
        let slots = slots as usize;
        let mut i = (hash_key(&[k]) as usize) & (slots - 1);
        for _ in 0..slots.min(MUL_MAP_MAX_PROBE) {
            let slot = head + i * 2;
            if kv[slot] == u64::MAX {
                return None;
            }
            if kv[slot] == k {
                return Some(kv[slot + 1]);
            }
            i = (i + 1) & (slots - 1);
        }
        None
    }

    /// Five small, distinct tuples: every one of them must probe back to its own row.
    #[test]
    fn fit_exact_map_resolves_every_sample() {
        let samples: Vec<(Vec<u64>, u64)> =
            vec![(vec![1, 2], 10), (vec![3, 4], 20), (vec![5, 6], 30), (vec![7, 8], 40), (vec![9, 10], 50)];
        let (nkey, kv, slots) = fit_exact_map(&samples, 2).expect("a small exact table must fit");
        for (t, r) in samples.iter() {
            assert_eq!(probe_packed_map(&kv, nkey, slots, t), Some(*r));
        }
    }

    /// A tuple not in the table must probe as "not found", never alias onto another row.
    #[test]
    fn fit_exact_map_out_of_range_key_is_not_found() {
        let samples: Vec<(Vec<u64>, u64)> =
            vec![(vec![1, 2], 10), (vec![3, 4], 20), (vec![5, 6], 30), (vec![7, 8], 40), (vec![9, 10], 50)];
        let (nkey, kv, slots) = fit_exact_map(&samples, 2).expect("a small exact table must fit");
        assert_eq!(probe_packed_map(&kv, nkey, slots, &[100, 200]), None);
    }

    /// Duplicate tuples: all credit lands on the first row written.
    #[test]
    fn fit_exact_map_duplicate_tuple_credited_to_one_row() {
        let samples: Vec<(Vec<u64>, u64)> = vec![(vec![1, 2], 10), (vec![1, 2], 99), (vec![3, 4], 20)];
        let (nkey, kv, slots) = fit_exact_map(&samples, 2).expect("must still fit with a duplicate");
        assert_eq!(probe_packed_map(&kv, nkey, slots, &[1, 2]), Some(10), "credit goes to the first-written row");
    }
}
