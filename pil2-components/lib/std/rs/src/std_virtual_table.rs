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
    SetupCtx, TraceInfo,
};
use proofman_hints::{get_hint_ids_by_name, HintFieldOptions};

use crate::{get_global_hint_field_constant_a_as, get_hint_field_constant_a_as, get_hint_field_constant_as, RCMultiplicity};

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
    // Parallel to `table_ids`: the prover counts this table itself, so increments are dropped here
    // and the counts arrive through `ProofCtx::prover_counts`.
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

/// Parse every virtual-table air's geometry. Returned rather than registered -- see
/// `ProofCtx::prover_owned_tables`.
// ---------------------------------------------------------------------------------------------
// Fitting a table's row map from its own fixed columns.
//
// A virtual table row holds a tuple (its COL_* columns) and the id of the table it belongs to (its
// UID_* column). A lookup supplies the tuple; the prover has to reach the row. When the table's
// layout is a packing, that row is affine in the tuple, and this recovers the map from the table
// itself -- no table ids, no conventions, nothing written down. A fit is accepted only after it
// reproduces EVERY entry of the table, so a table whose layout is a filter simply fails and is left
// to whoever counted it before.
//
// The arithmetic is Goldilocks, not integers, because that is exactly what the scatter evaluates.
const GL_P: u128 = 0xFFFF_FFFF_0000_0001;

#[inline]
fn gl_red(a: u128) -> u64 {
    (a % GL_P) as u64
}
#[inline]
fn gl_add(a: u64, b: u64) -> u64 {
    gl_red(a as u128 + b as u128)
}
#[inline]
fn gl_sub(a: u64, b: u64) -> u64 {
    gl_red(GL_P + a as u128 - b as u128)
}
#[inline]
fn gl_mul(a: u64, b: u64) -> u64 {
    gl_red(a as u128 * b as u128)
}
fn gl_pow(mut b: u64, mut e: u64) -> u64 {
    let mut r = 1u64;
    while e > 0 {
        if e & 1 == 1 {
            r = gl_mul(r, b);
        }
        b = gl_mul(b, b);
        e >>= 1;
    }
    r
}
#[inline]
fn gl_inv(a: u64) -> u64 {
    gl_pow(a, (GL_P - 2) as u64)
}

/// Solve `target = sum(c_j * tuple_j) + k` from `samples`, then REQUIRE it on all of them.
/// Returns (coefficients, constant). None when no affine map reproduces the table.
fn fit_affine(samples: &[(Vec<u64>, u64)], width: usize) -> Option<(Vec<u64>, u64)> {
    let unknowns = width + 1;
    if samples.len() < unknowns {
        return None;
    }
    // Take a spread of rows rather than the first few: the head of a table is often all zeros,
    // which is rank-deficient and would fit anything.
    let take = (unknowns * 8).min(samples.len());
    let step = samples.len() / take;
    let mut m: Vec<Vec<u64>> = Vec::with_capacity(take);
    for i in 0..take {
        let (t, y) = &samples[i * step];
        let mut row: Vec<u64> = t.iter().map(|v| gl_red(*v as u128)).collect();
        row.push(1);
        row.push(*y);
        m.push(row);
    }
    // Gauss-Jordan over the field.
    let mut piv: Vec<usize> = Vec::new();
    let mut r = 0usize;
    for c in 0..unknowns {
        let Some(p) = (r..m.len()).find(|&rr| m[rr][c] != 0) else { continue };
        m.swap(r, p);
        let inv = gl_inv(m[r][c]);
        for v in m[r].iter_mut() {
            *v = gl_mul(*v, inv);
        }
        for rr in 0..m.len() {
            if rr != r && m[rr][c] != 0 {
                let f = m[rr][c];
                for cc in 0..=unknowns {
                    let s = gl_mul(f, m[r][cc]);
                    m[rr][cc] = gl_sub(m[rr][cc], s);
                }
            }
        }
        piv.push(c);
        r += 1;
        if r == m.len() {
            break;
        }
    }
    let mut sol = vec![0u64; unknowns];
    for (i, c) in piv.iter().enumerate() {
        sol[*c] = m[i][unknowns];
    }
    // Exhaustive check. A rank-deficient sample can yield a solution that suits the sample and
    // nothing else, so this is the whole guarantee -- never skip it.
    for (t, y) in samples.iter() {
        let mut acc = sol[width];
        for (j, v) in t.iter().enumerate() {
            acc = gl_add(acc, gl_mul(sol[j], gl_red(*v as u128)));
        }
        if acc != *y {
            return None;
        }
    }
    Some((sol[..width].to_vec(), sol[width]))
}

/// Tables the prover counts itself.
///
/// TEMPORARY. It exists only for the window where the prover has claimed a table but zisk still
/// calls `inc_virtual_row` for it; once a wave's call sites are deleted there is nothing to
/// suppress and this goes with them. Global rather than plumbed through `ProofCtx` because the
/// claim is global, and read lazily because the set is filled by the host binary AFTER the virtual
/// table airs are built -- reading it at construction (as this used to) yielded all-false and
/// silently double-counted.
static PROVER_OWNED_TABLES: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();

pub fn set_prover_owned_tables(tables: Vec<u64>) {
    let _ = PROVER_OWNED_TABLES.set(tables);
}

fn prover_owned_tables() -> &'static [u64] {
    PROVER_OWNED_TABLES.get().map(|v| v.as_slice()).unwrap_or(&[])
}

/// One fitted row map: (table_id, coefficients, constant).
pub struct VtFittedMap {
    pub table_id: u64,
    pub coef: Vec<u64>,
    pub konst: u64,
    /// When the row is not affine in the tuple, the coefficients above yield a KEY instead and this
    /// turns it into a row: `row = index[key - key_min]`, with `u32::MAX` for a key not in the table.
    pub index: Option<(u64, Vec<u32>)>,
    /// A change of base, when the table's layout is one: cheaper than any index.
    pub remap: Option<(u32, u32, u32, Vec<u32>)>,
}

/// Widest base a digit remap may use; mirrors MUL_MAX_DIGIT_BASE on the C side.
const VT_MAX_DIGIT_BASE: u64 = 64;

/// Is the row a change of base of the key, digit by digit?
///
/// `row = sum_i map[digit_i(key, base_in)] * base_out^i`. Keccak's chi table is exactly this: the
/// lookup commits `rc·28^5 + sum (tA + 8·tB)·28^x` while the row is `rc·16^5 + sum (tA + 4·tB)·16^x`
/// -- the same digits in a different base, through a fixed per-digit map. Recovering that costs a
/// few bytes where a direct index costs 137 MB, and replaces a dependent load with arithmetic.
///
/// Searched, not written down: the bases and the digit map all come from the table's own entries,
/// and the fit is accepted only if it reproduces every one of them.
fn fit_digit_remap(samples: &[(Vec<u64>, u64)], col: usize) -> Option<(u32, u32, u32, Vec<u32>)> {
    let max_key = samples.iter().map(|(t, _)| t[col]).max()?;
    let max_row = samples.iter().map(|(_, r)| *r).max()?;

    let digits = |mut v: u64, b: u64, n: u32| -> Option<Vec<u64>> {
        let mut d = Vec::with_capacity(n as usize);
        for _ in 0..n {
            d.push(v % b);
            v /= b;
        }
        if v == 0 {
            Some(d)
        } else {
            None
        }
    };
    let pow = |b: u64, e: u32| -> Option<u64> { b.checked_pow(e) };

    for nd in 1..=12u32 {
        for bi in 2..=VT_MAX_DIGIT_BASE {
            // base_in must be exactly wide enough for the keys, or the digits are not the real ones
            match (pow(bi, nd), pow(bi, nd - 1)) {
                (Some(hi), Some(lo)) if hi > max_key && (nd == 1 || lo <= max_key) => {}
                _ => continue,
            }
            for bo in 2..=VT_MAX_DIGIT_BASE {
                match (pow(bo, nd), pow(bo, nd - 1)) {
                    (Some(hi), Some(lo)) if hi > max_row && (nd == 1 || lo <= max_row) => {}
                    _ => continue,
                }
                let mut map = vec![u32::MAX; bi as usize];
                let mut ok = true;
                // cheap rejection first, then the exhaustive check that actually licenses the fit
                for (t, r) in samples.iter().take(64).chain(samples.iter()) {
                    let (Some(kd), Some(rd)) = (digits(t[col], bi, nd), digits(*r, bo, nd)) else {
                        ok = false;
                        break;
                    };
                    for (x, y) in kd.iter().zip(rd.iter()) {
                        let slot = &mut map[*x as usize];
                        if *slot == u32::MAX {
                            *slot = *y as u32;
                        } else if *slot != *y as u32 {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        break;
                    }
                }
                if ok {
                    return Some((bi as u32, bo as u32, nd, map));
                }
            }
        }
    }
    None
}

/// Largest key range worth a direct index: beyond this the array costs more than the table.
const VT_INDEX_MAX_RANGE: u64 = 1 << 26;

/// How empty a key index may be before the key is judged not to be the table's own.
const VT_INDEX_MAX_SPARSITY: u64 = 32;

/// When the row is not affine in the tuple, look for a linear key that merely SEPARATES the rows,
/// and invert it with a direct array. Tries single columns first (the cheapest key, and often a
/// natural one), then a mixed radix over every column.
///
/// Injectivity is checked on the real entries, not argued: a key that collides on any two rows is
/// rejected, so a table with duplicate tuples -- where tuple -> row is not even a function -- simply
/// gets no index and stays with whoever counted it.
fn fit_key_index(samples: &[(Vec<u64>, u64)], width: usize, tid: u64) -> Option<VtFittedMap> {
    let build = |coef: &[u64]| -> Option<(u64, u64, Vec<u64>)> {
        let mut keys = Vec::with_capacity(samples.len());
        for (t, _) in samples.iter() {
            let mut k: u128 = 0;
            for (j, v) in t.iter().enumerate() {
                k += coef[j] as u128 * *v as u128;
                if k > u64::MAX as u128 {
                    return None;
                }
            }
            keys.push(k as u64);
        }
        let (lo, hi) = (*keys.iter().min()?, *keys.iter().max()?);
        let range = hi.checked_sub(lo)?.checked_add(1)?;
        if range > VT_INDEX_MAX_RANGE {
            return None;
        }
        // A key whose index is almost all holes is not this table's natural key: the packing we
        // guessed separates the rows by accident, and a lookup's tuple will not reproduce it.
        // Table 331 (124 rows over a 2.9M key range) is exactly that, and it decoded out of range
        // on every lookup. Requiring the index to be mostly occupied rejects such keys up front
        // instead of at proving time.
        if range > samples.len() as u64 * VT_INDEX_MAX_SPARSITY {
            return None;
        }
        let mut seen = vec![false; range as usize];
        for k in keys.iter() {
            let off = (*k - lo) as usize;
            if seen[off] {
                return None; // not injective
            }
            seen[off] = true;
        }
        Some((lo, range, keys))
    };

    // Shortest PREFIX of the columns first. A lookup supplies its tuple in column order and may
    // supply fewer elements than the table has columns, so a key over columns [0, L) is usable by
    // any lookup with at least L elements -- while a key needing a late column may be unusable.
    // SINGLE column only, deliberately.
    //
    // Injectivity over the table is necessary but NOT sufficient: the key also has to be the one
    // lookups address the table by, and the fixed columns alone cannot tell us that. A multi-column
    // mixed radix is a guess about column order and radices; table 127 accepted such a key, passed
    // injectivity AND density (2.1x), decoded entirely in range -- and placed every count on the
    // wrong row, with a total that still matched. Only a position-sensitive check caught it.
    //
    // A single column that separates every row is not a guess: there is nothing to get wrong about
    // its order or radix. Widening this needs evidence of the lookup's key, not of the table's.
    let mut best: Option<(Vec<u64>, u64, u64, Vec<u64>)> = None;
    for len in 1..=1.min(width) {
        let mut coef = vec![0u64; width];
        let mut radix: u128 = 1;
        let mut ok = true;
        for (j, c) in coef.iter_mut().enumerate().take(len) {
            let max = samples.iter().map(|(t, _)| t[j]).max().unwrap_or(0) as u128;
            if radix > u64::MAX as u128 {
                ok = false;
                break;
            }
            *c = radix as u64;
            radix *= max + 1;
        }
        if !ok || radix > u64::MAX as u128 {
            continue;
        }
        if let Some((lo, range, keys)) = build(&coef) {
            best = Some((coef, lo, range, keys));
            break;
        }
    }

    let (coef, key_min, range, keys) = best?;
    let mut index = vec![u32::MAX; range as usize];
    for ((_, row), k) in samples.iter().zip(keys.iter()) {
        index[(*k - key_min) as usize] = *row as u32;
    }
    tracing::info!(
        "virtual table {tid}: no affine row map, but a linear key separates its {} entries -- \
         indexing {} keys ({} MB)",
        samples.len(),
        range,
        range * 4 / 1_000_000
    );
    Some(VtFittedMap { table_id: tid, coef, konst: 0, index: Some((key_min, index)), remap: None })
}

/// Recover a row map for every virtual table whose layout is a packing.
///
/// Scope, deliberately: a table whose entries live in ONE `COL_*` group that covers its whole
/// height and sits inside a single accumulator column. Tables split across groups of differing
/// widths have no single tuple shape to fit and are skipped, as are filtered layouts, whose fit
/// fails its exhaustive check. Skipping is always safe -- the std keeps counting whatever is not
/// claimed here.
pub fn fit_virtual_table_maps<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
) -> ProofmanResult<Vec<VtFittedMap>> {
    use std::io::{BufReader, Read};

    let global_hint = get_hint_ids_by_name(sctx.get_global_bin(), "virtual_table_data_global");
    if global_hint.is_empty() {
        return Ok(Vec::new());
    }
    let airgroup_ids = get_global_hint_field_constant_a_as::<usize, F>(sctx, global_hint[0], "airgroup_ids")?;
    let air_ids = get_global_hint_field_constant_a_as::<usize, F>(sctx, global_hint[0], "air_ids")?;

    let mut out = Vec::new();
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

        let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;
        let n_const = setup.stark_info.n_constants as usize;
        let Some(pol_map) = setup.stark_info.const_pols_map.as_ref() else { continue };

        // COL_<g>_<base>_<k> and UID_<g>: the group is the tuple, the uid says which table a row
        // belongs to. Parsed, not assumed -- a new group shape needs no change here.
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
        if groups.is_empty() {
            continue;
        }

        // The .const is raw row-major u64 with an n_constants stride. Read the UID columns first;
        // the tuples are only needed for the tables that survive the shape checks.
        let path = setup.const_pols_path.replace(".const_gpu", ".const");
        let Ok(file) = std::fs::File::open(&path) else {
            tracing::debug!("virtual table fit: cannot open {path}; skipping");
            continue;
        };
        let uid_idx: Vec<(u64, usize)> = groups.iter().filter_map(|(g, v)| v.1.map(|u| (*g, u))).collect();
        let mut uids: std::collections::BTreeMap<u64, Vec<u64>> =
            uid_idx.iter().map(|(g, _)| (*g, Vec::with_capacity(num_rows))).collect();
        {
            let mut rd = BufReader::with_capacity(1 << 22, &file);
            let mut row = vec![0u8; n_const * 8];
            for _ in 0..num_rows {
                if rd.read_exact(&mut row).is_err() {
                    break;
                }
                for (g, c) in uid_idx.iter() {
                    let b = &row[c * 8..c * 8 + 8];
                    uids.get_mut(g).unwrap().push(u64::from_le_bytes(b.try_into().unwrap()));
                }
            }
        }

        for (t, tid) in table_ids.iter().enumerate() {
            let tid = *tid as u64;
            let base = acc_heights[t];
            let end = acc_heights.iter().copied().filter(|h| *h > base).min().unwrap_or((num_muls * num_rows) as u64);
            let height = end - base;
            if height == 0 {
                continue;
            }

            // A table's entries occupy the accumulator offsets [base, base+height). Each offset is
            // an (accumulator column, air row) pair, and the COL_* group holding a row is the one
            // whose UID names this table there. Bind each group to the column whose air rows it
            // covers -- derived, so no convention about group numbering is assumed.
            let mut rows_of_table: Vec<(usize, u64, u64)> = Vec::new(); // (air_row, row_in_table, group)
            let mut widths: Vec<usize> = Vec::new();
            let mut ok = true;
            for (g, uid_col) in uids.iter() {
                let mine: Vec<usize> =
                    uid_col.iter().enumerate().filter_map(|(r, u)| if *u == tid { Some(r) } else { None }).collect();
                if mine.is_empty() {
                    continue;
                }
                // Which accumulator column places exactly these air rows inside the table's span?
                let mut col = None;
                for c in 0..num_muls as u64 {
                    if mine.iter().all(|a| {
                        let off = c * num_rows as u64 + *a as u64;
                        off >= base && off < end
                    }) {
                        col = Some(c);
                        break;
                    }
                }
                let Some(c) = col else {
                    ok = false;
                    break;
                };
                for a in mine.iter() {
                    let off = c * num_rows as u64 + *a as u64;
                    rows_of_table.push((*a, off - base, *g));
                }
                widths.push(groups[g].0.len());
            }
            if !ok || rows_of_table.len() as u64 != height || widths.is_empty() {
                tracing::info!(
                    "virtual table {tid}: SKIPPED before fitting (groups_ok={ok} rows={} height={height} groups={})",
                    rows_of_table.len(),
                    widths.len()
                );
                continue;
            }
            // The lookup has one tuple shape, so the key may only use columns every group has.
            let width = *widths.iter().min().unwrap();
            if width == 0 || width > 8 {
                continue;
            }

            // Tuples for this table, in accumulator order.
            rows_of_table.sort_unstable_by_key(|(_, r, _)| *r);
            let mut samples: Vec<(Vec<u64>, u64)> = Vec::with_capacity(height as usize);
            {
                use std::io::{Seek, SeekFrom};
                let mut f2 = std::fs::File::open(&path)?;
                let mut buf = vec![0u8; n_const * 8];
                for (air_row, row_in_table, g) in rows_of_table.iter() {
                    f2.seek(SeekFrom::Start(*air_row as u64 * n_const as u64 * 8))?;
                    if f2.read_exact(&mut buf).is_err() {
                        ok = false;
                        break;
                    }
                    let cols: Vec<usize> = groups[g].0.iter().map(|(_, c)| *c).take(width).collect();
                    let tuple: Vec<u64> =
                        cols.iter().map(|c| u64::from_le_bytes(buf[c * 8..c * 8 + 8].try_into().unwrap())).collect();
                    samples.push((tuple, *row_in_table));
                }
            }
            if !ok || samples.len() as u64 != height {
                continue;
            }

            // Shortest prefix that verifies, not the group's full width.
            //
            // A COL_* group is shared by several tables and padded to the widest of them, so the
            // trailing columns of a narrower table hold another table's data. Fitting over the
            // group width therefore fits over junk -- that is why 133 (a 6-element tuple in a
            // 7-wide group) came back "duplicate tuples". Any fit is still verified against every
            // entry, so a shorter prefix is accepted only when it reproduces the whole table.
            let mut fitted = None;
            for w in 1..=width {
                let narrowed: Vec<(Vec<u64>, u64)> = samples.iter().map(|(t, r)| (t[..w].to_vec(), *r)).collect();
                if let Some((coef, konst)) = fit_affine(&narrowed, w) {
                    let mut full = vec![0u64; width];
                    full[..w].copy_from_slice(&coef);
                    fitted = Some((full, konst, w));
                    break;
                }
            }
            if let Some((coef, konst, w)) = fitted {
                tracing::info!(
                    "virtual table {tid}: row map fitted from its fixed columns ({height} entries, \
                     {w} of {width} columns)"
                );
                out.push(VtFittedMap { table_id: tid, coef, konst, index: None, remap: None });
            } else if let Some((bi, bo, nd, map)) = fit_digit_remap(&samples, 0) {
                tracing::info!(
                    "virtual table {tid}: row is a change of base of its key ({bi} -> {bo}, {nd} digits) \
                     -- {height} entries, no index needed"
                );
                let mut coef = vec![0u64; width];
                coef[0] = 1; // the key is the first tuple element; the remap does the rest
                out.push(VtFittedMap { table_id: tid, coef, konst: 0, index: None, remap: Some((bi, bo, nd, map)) });
            } else if let Some(m) = fit_key_index(&samples, width, tid) {
                out.push(m);
            } else {
                tracing::info!(
                    "virtual table {tid}: NO FIT ({height} entries, {width} columns) -- \
                     affine/remap/index all declined"
                );
            }
        }
    }
    Ok(out)
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
            // Filled by the host binary before the witness library is registered.

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

            // Before `distribute_multiplicities`, so the MPI path sees a complete accumulator.
            if let Some(counts) = pctx.prover_counts.read().unwrap().get(&self.air_id) {
                for (slot, add) in self.multiplicities.iter().zip(counts.iter()) {
                    if *add != 0 {
                        slot.fetch_add(*add, Ordering::Relaxed);
                    }
                }
            }

            {
                // TEMP: per-table totals for the migration diff.
                let total = self.multiplicities.len() as u64;
                let mut ends: Vec<u64> = self.table_ids.iter().map(|(_, b)| *b).chain(std::iter::once(total)).collect();
                ends.sort_unstable();
                for (k, (id, base)) in self.table_ids.iter().enumerate() {
                    let end = *ends.iter().find(|e| **e > *base).unwrap_or(&total);
                    let mut sum = 0u64;
                    let mut chk = 0u64; // position-sensitive: same total on different rows differs
                    for i in *base..end {
                        let v = self.multiplicities[i as usize].load(Ordering::Relaxed);
                        sum = sum.wrapping_add(v);
                        chk = chk.wrapping_add(v.wrapping_mul(i - *base + 1));
                    }
                    tracing::info!(
                        "VTVOL table={} owned={} increments={} chk={}",
                        id,
                        self.is_prover_owned(k),
                        sum,
                        chk
                    );
                }
            }

            // An assigned table is computed only on its single owner node; its
            // multiplicities are produced there, so there is no cross-rank reduction.
            let assigned = pctx.dctx_is_assigned_table(instance_id)?;

            if self.shared_tables && !assigned {
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
                let any_nonzero = std::sync::atomic::AtomicBool::new(false);
                let num_rows = self.num_rows;
                buffer.par_chunks_mut(self.num_cols).enumerate().for_each(|(row, chunk)| {
                    for (col, slot) in chunk.iter_mut().enumerate() {
                        let v = self.multiplicities[col * num_rows + row].load(Ordering::Relaxed);
                        if v != 0 {
                            any_nonzero.store(true, Ordering::Relaxed);
                        }
                        *slot = F::from_u64(v);
                    }
                });
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
}
