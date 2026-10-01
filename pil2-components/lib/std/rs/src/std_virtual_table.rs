use std::{
    collections::{HashMap, HashSet},
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
use proofman_starks_lib_c::{mul_eval_proves_hint_c, mul_proves_hints_c};

use crate::{
    get_global_hint_field_constant_a_as, get_global_hint_field_constant_as, get_hint_field_constant_a_as,
    get_hint_field_constant_as, RCMultiplicity, PIOP_TYPE_PROVES,
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
    // Where the host publishes which tables it counts: the ProofCtx is shared with a dlopen'ed witness
    // library, a static is not. None in unit tests.
    pctx: Option<Arc<ProofCtx<F>>>,
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
    /// Why each of `unclaimed_ids` did not fit.
    pub unclaimed_why: HashMap<u64, String>,
    /// Of `unclaimed_ids`, the tables no lookup names by their own id (several tables sharing one
    /// opid): the prover matches lookups to tables by opid, so the std counts these.
    pub unaddressed_ids: Vec<u64>,
}

/// One fitted row map: an exact-match map `(key columns, table, slots)` for `table_id`. Slots is
/// explicit: the packed layout has a header, so it cannot be derived from the table length.
pub struct VtFittedMap {
    pub table_id: u64,
    pub map: (usize, Vec<u64>, u64),
}

/// Header word 0 of an exact map ("PACKKEY2"). Must match MUL_MAP_MAGIC on the C side.
const MUL_MAP_MAGIC: u64 = 0x5041_434B_4B45_5932;

/// Words before the per-column descriptors: magic, shape, slot count. Must match MUL_MAP_HEAD.
const MUL_MAP_HEAD: usize = 3;

/// Key words per slot, at most. Must match MUL_MAP_MAX_WORDS.
const MUL_MAP_MAX_WORDS: usize = 8;

/// Must match `MUL_MAP_MAX_PROBE` in `multiplicity_decoders.hpp`. The self-checks use the same bound,
/// or a longer chain would verify here but miss silently at runtime.
const MUL_MAP_MAX_PROBE: usize = 4096;

// Fitting a table's row map (tuple -> row) from its own fixed columns (COL_* tuple, UID_* table id).
// A fit is accepted only if it reproduces EVERY entry; otherwise the table must be std-owned.
//
// Map layout, in u64 words (read by mulMapPack / mulMapFind on the C side):
//   [MAGIC][nKey | nWords<<16 | slotWords<<24 | rowWord<<32 | rowShift<<40 | rowInline<<48][slots]
//   per key column: [min][word<<40 | shift<<32 | width]
//   slots * slotWords: the packed key (nWords words), the row inline above the key bits of rowWord
//   or in a word of its own. A slot is empty when its word 0 is u64::MAX, which word 0's clear top
//   bit keeps any key from being.

/// Where each key column and the row sit in a slot.
struct MapShape {
    nkey: usize,
    mins: Vec<u64>,
    place: Vec<(u32, u32, u32)>, // (word, shift, width) per column
    nwords: usize,
    slot_words: usize,
    row_word: usize,
    row_shift: u32,
    row_inline: bool,
}

impl MapShape {
    /// Each column stores `value - min` in the bits its range needs, widest first into the first
    /// word with room; the row shares the key word with the most room when it fits.
    fn new(keys: &[u64], rows: &[u64], width: usize, nkey: usize) -> Option<Self> {
        let (mins, maxs, max_row) = (0..rows.len())
            .into_par_iter()
            .fold(
                || (vec![u64::MAX; nkey], vec![0u64; nkey], 0u64),
                |(mut mn, mut mx, mr), i| {
                    let t = &keys[i * width..i * width + nkey];
                    for c in 0..nkey {
                        mn[c] = mn[c].min(t[c]);
                        mx[c] = mx[c].max(t[c]);
                    }
                    (mn, mx, mr.max(rows[i]))
                },
            )
            .reduce(
                || (vec![u64::MAX; nkey], vec![0u64; nkey], 0u64),
                |(a0, a1, a2), (b0, b1, b2)| {
                    let mn = a0.iter().zip(&b0).map(|(x, y)| *x.min(y)).collect();
                    let mx = a1.iter().zip(&b1).map(|(x, y)| *x.max(y)).collect();
                    (mn, mx, a2.max(b2))
                },
            );
        let bits = |x: u64| 64 - x.leading_zeros();
        let cap = |w: usize| if w == 0 { 63 } else { 64 };
        let widths: Vec<u32> = (0..nkey).map(|c| bits(maxs[c] - mins[c])).collect();
        let mut order: Vec<usize> = (0..nkey).collect();
        order.sort_by_key(|&c| std::cmp::Reverse(widths[c]));
        let mut used: Vec<u32> = vec![0];
        let mut place = vec![(0u32, 0u32, 0u32); nkey];
        for c in order {
            let w = widths[c];
            if w == 0 {
                continue; // a single value: checked, nothing stored
            }
            let word = match (0..used.len()).find(|&i| used[i] + w <= cap(i)) {
                Some(i) => i,
                None => {
                    used.push(0);
                    used.len() - 1
                }
            };
            place[c] = (word as u32, used[word], w);
            used[word] += w;
        }
        let nwords = used.len();
        if nwords > MUL_MAP_MAX_WORDS {
            return None;
        }
        let row_bits = bits(max_row).max(1);
        let spare = (0..nwords).filter(|&i| used[i] + row_bits <= cap(i)).max_by_key(|&i| cap(i) - used[i]);
        let (row_word, row_shift, row_inline, slot_words) = match spare {
            Some(i) => (i, used[i], true, nwords),
            None => (nwords, 0, false, nwords + 1),
        };
        Some(Self { nkey, mins, place, nwords, slot_words, row_word, row_shift, row_inline })
    }

    /// A table tuple's packed key (its values lie inside their columns' ranges by construction).
    fn pack_into(&self, t: &[u64], kw: &mut [u64]) {
        for (c, &(word, shift, width)) in self.place.iter().enumerate() {
            if width != 0 {
                kw[word as usize] |= (t[c] - self.mins[c]) << shift;
            }
        }
    }

    fn header(&self, slots: usize) -> Vec<u64> {
        let mut h = Vec::with_capacity(MUL_MAP_HEAD + 2 * self.nkey);
        h.push(MUL_MAP_MAGIC);
        h.push(
            self.nkey as u64
                | (self.nwords as u64) << 16
                | (self.slot_words as u64) << 24
                | (self.row_word as u64) << 32
                | (self.row_shift as u64) << 40
                | (self.row_inline as u64) << 48,
        );
        h.push(slots as u64);
        for c in 0..self.nkey {
            let (word, shift, width) = self.place[c];
            h.push(self.mins[c]);
            h.push((word as u64) << 40 | (shift as u64) << 32 | width as u64);
        }
        h
    }
}

/// Pack a lookup tuple as mulMapPack does. None: some element lies outside its column's range, so
/// the tuple is not in the table.
fn map_pack(kv: &[u64], tuple: &[u64]) -> Option<[u64; MUL_MAP_MAX_WORDS]> {
    let nkey = (kv[1] & 0xFFFF) as usize;
    let mut kw = [0u64; MUL_MAP_MAX_WORDS];
    for (c, &v) in tuple.iter().enumerate().take(nkey) {
        let (min, meta) = (kv[MUL_MAP_HEAD + 2 * c], kv[MUL_MAP_HEAD + 2 * c + 1]);
        let (width, shift, word) = ((meta & 0xFF) as u32, ((meta >> 32) & 0xFF) as u32, ((meta >> 40) & 0xFF) as usize);
        let d = v.checked_sub(min)?;
        if width < 64 && (d >> width) != 0 {
            return None;
        }
        kw[word] |= d << shift;
    }
    Some(kw)
}

fn map_start(h: u64, slots: usize) -> usize {
    ((h as u128 * slots as u128) >> 64) as usize
}

/// Look a packed key up as mulMapFind does, within MUL_MAP_MAX_PROBE probes.
fn map_find(kv: &[u64], kw: &[u64]) -> Option<u64> {
    let shape = kv[1];
    let nkey = (shape & 0xFFFF) as usize;
    let nwords = ((shape >> 16) & 0xFF) as usize;
    let slot_words = ((shape >> 24) & 0xFF) as usize;
    let row_word = ((shape >> 32) & 0xFF) as usize;
    let row_shift = ((shape >> 40) & 0xFF) as u32;
    let inline = (shape >> 48) & 1 == 1;
    let slots = kv[2] as usize;
    let table = &kv[MUL_MAP_HEAD + 2 * nkey..];
    let key_mask = if inline { (1u64 << row_shift) - 1 } else { u64::MAX };
    let mut i = map_start(hash_key(&kw[..nwords]), slots);
    for _ in 0..slots.min(MUL_MAP_MAX_PROBE) {
        let s = &table[i * slot_words..(i + 1) * slot_words];
        if s[0] == u64::MAX {
            return None;
        }
        let hit = (0..nwords).all(|w| (if w == row_word { s[w] & key_mask } else { s[w] }) == kw[w]);
        if hit {
            return Some(if inline { s[row_word] >> row_shift } else { s[nwords] });
        }
        i += 1;
        if i == slots {
            i = 0;
        }
    }
    None
}

/// Exact-match map over `rows.len()` tuples of `width` elements, flat in `keys`. Keys are the lookup's
/// whole tuple (a prefix that separates the table's rows need not be how lookups address it).
///
/// Duplicate tuples are allowed: the lookup constrains only the SUM of their multiplicities, so all
/// counts go to the first. The map is sized by distinct tuples, at most 80% full. Built in parallel,
/// so which slot an entry takes varies between runs; what it resolves to does not.
fn fit_exact_map(keys: &[u64], rows: &[u64], width: usize) -> Option<(usize, Vec<u64>, u64)> {
    let n = rows.len();
    let tuple = |i: usize| &keys[i * width..(i + 1) * width];
    // Constant columns stay in the key: they pack into zero bits, and their value is still checked.
    let nkey = width;

    let Some(shape) = MapShape::new(keys, rows, width, nkey) else {
        tracing::debug!("exact map: {nkey} key columns need more than {MUL_MAP_MAX_WORDS} words");
        return None;
    };
    let nw = shape.nwords;
    let mut packed = vec![0u64; n * nw];
    packed.par_chunks_mut(nw).enumerate().for_each(|(i, kw)| shape.pack_into(&tuple(i)[..nkey], kw));
    let key_of = |i: usize| &packed[i * nw..(i + 1) * nw];

    // First occurrence of each tuple: packing is injective over the table's own values.
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.par_sort_unstable_by(|&a, &b| key_of(a as usize).cmp(key_of(b as usize)).then(a.cmp(&b)));
    let uniq: Vec<u32> = order
        .iter()
        .enumerate()
        .filter(|&(k, &i)| k == 0 || key_of(order[k - 1] as usize) != key_of(i as usize))
        .map(|(_, &i)| i)
        .collect();
    drop(order);

    let sw = shape.slot_words;
    let word_of = |kw: &[u64], w: usize, r: u64| {
        if shape.row_inline && w == shape.row_word {
            kw[w] | r << shape.row_shift
        } else {
            kw[w]
        }
    };
    // A chain past MUL_MAP_MAX_PROBE would miss at runtime; a roomier map shortens it.
    for (num, den) in [(5usize, 4usize), (2, 1), (4, 1)] {
        let slots = (uniq.len() * num).div_ceil(den).max(1);
        let mut kv = shape.header(slots);
        let head = kv.len();
        kv.resize(head + slots * sw, u64::MAX);
        {
            // SAFETY: AtomicU64 has u64's layout, and `kv` is borrowed only through `table` here.
            let table: &[AtomicU64] = unsafe { &*(&mut kv[head..] as *mut [u64] as *const [AtomicU64]) };
            // Slots > entries, so every probe ends. Claiming word 0 owns the slot.
            uniq.par_iter().for_each(|&i| {
                let (kw, r) = (key_of(i as usize), rows[i as usize]);
                let mut s = map_start(hash_key(kw), slots);
                while table[s * sw]
                    .compare_exchange(u64::MAX, word_of(kw, 0, r), Ordering::Relaxed, Ordering::Relaxed)
                    .is_err()
                {
                    s += 1;
                    if s == slots {
                        s = 0;
                    }
                }
                for w in 1..nw {
                    table[s * sw + w].store(word_of(kw, w, r), Ordering::Relaxed);
                }
                if !shape.row_inline {
                    table[s * sw + nw].store(r, Ordering::Relaxed);
                }
            });
        }
        // Probe every tuple back with the decoder's algorithm (duplicated on the C side): a mismatch
        // there is a wrong count, here it fails loudly.
        let bad = uniq.par_iter().find_any(|&&i| {
            let i = i as usize;
            map_pack(&kv, &tuple(i)[..nkey]).and_then(|kw| map_find(&kv, &kw)) != Some(rows[i])
        });
        match bad {
            None => return Some((nkey, kv, slots as u64)),
            Some(&i) => tracing::debug!(
                "exact map at {slots} slots does not probe back key {:?} (row {}) -- retrying roomier",
                &tuple(i as usize)[..nkey],
                rows[i as usize]
            ),
        }
    }
    None
}

/// Must match mulMapHash on the C side.
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
/// the table's (padded) COL_* groups. `None` when two lookups disagree: a map keyed on either width
/// would miscount the other.
fn lookup_arity<F: PrimeField64>(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>) -> HashMap<u64, Option<usize>> {
    let mut arity: HashMap<u64, Option<usize>> = HashMap::new();
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
                    let e = arity.entry(id).or_insert(Some(len as usize));
                    if *e != Some(len as usize) {
                        *e = None;
                    }
                }
            }
        }
    }
    arity
}

/// Recover an exact row map for every virtual table from its air's proves-side lookups.
///
/// Tables that do not fit are skipped; the caller requires them to be std-owned. `range_ids`
/// (`collect_prover_owned_ranges`) are decoded by bias instead, so they are not fitted.
pub fn fit_virtual_table_maps<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    sctx: &SetupCtx<F>,
    layouts: &[VtLayout],
    range_ids: &HashSet<u64>,
) -> ProofmanResult<(Vec<VtFittedMap>, VtFitSummary)> {
    let arity = lookup_arity(pctx, sctx);
    let t0 = std::time::Instant::now();
    // Airs, and the tables inside each, fit independently.
    let per_air: Vec<Vec<(u64, Result<VtFittedMap, String>)>> =
        layouts.par_iter().map(|l| fit_air_maps(sctx, l, &arity, range_ids)).collect::<ProofmanResult<_>>()?;
    let mut out = Vec::new();
    // Unfitted: range tables or std-owned; the caller tells them apart.
    let (mut unclaimed_ids, mut unclaimed_why) = (Vec::new(), HashMap::new());
    for (tid, m) in per_air.into_iter().flatten() {
        match m {
            Ok(m) => out.push(m),
            Err(why) => {
                tracing::trace!("virtual table {tid}: not fitted -- {why}");
                unclaimed_ids.push(tid);
                unclaimed_why.insert(tid, why);
            }
        }
    }
    // A table id another air fitted is claimed.
    let fitted_ids: std::collections::HashSet<u64> = out.iter().map(|m| m.table_id).collect();
    unclaimed_ids.retain(|t| !fitted_ids.contains(t));
    let unaddressed_ids = unclaimed_ids.iter().copied().filter(|t| !arity.contains_key(t)).collect();
    let exact_bytes = out.iter().map(|m| m.map.1.len() as u64 * 8).sum();
    let considered = out.len() + unclaimed_ids.len();
    Ok((
        out,
        VtFitSummary {
            considered,
            exact_bytes,
            elapsed_ms: t0.elapsed().as_millis(),
            unclaimed_ids,
            unclaimed_why,
            unaddressed_ids,
        },
    ))
}

/// Tables a global sum (`gsum_debug_data_global`) looks up, as an assumes or a free sum (which assumes
/// when its multiplicity is negative). No per-air plan sees those lookups, so the prover cannot count
/// such a table.
pub fn global_sum_assumed_tables<F: PrimeField64>(sctx: &SetupCtx<F>) -> ProofmanResult<Vec<u64>> {
    let hints = get_hint_ids_by_name(sctx.get_global_bin(), "gsum_debug_data_global");
    let Some(&header) = hints.first() else { return Ok(Vec::new()) };
    let n = get_global_hint_field_constant_as::<usize, F>(sctx, header, "num_global_hints")?;
    let mut out = Vec::new();
    for &h in hints.iter().skip(1).take(n) {
        if get_global_hint_field_constant_as::<u64, F>(sctx, h, "type_piop")? != PIOP_TYPE_PROVES {
            out.extend(get_global_hint_field_constant_a_as::<u64, F>(sctx, h, "opids")?);
        }
    }
    Ok(out)
}

/// The air's `.const`, row-major.
fn read_const_pols(path: &str) -> std::io::Result<Vec<u64>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let words = f.metadata()?.len() as usize / 8;
    let mut v = vec![0u64; words];
    // SAFETY: the byte view covers exactly `v`, and every bit pattern is a u64.
    f.read_exact(unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, words * 8) })?;
    Ok(v)
}

/// Every table of one virtual-table air: `(table id, map)`, or why it does not fit. `skip`: range
/// tables, decoded by bias instead.
///
/// Accumulator offset `c * num_rows + r` is row `r` of the proves-side lookup crediting multiplicity
/// column `c`; a table owns the offsets `[base, end)` that carry its bus id. Tuples are evaluated from
/// the lookup's expressions, not read by `COL_*` name: `simplify_virtual_fixed` may have turned a
/// column into a constant, a row-index line or a rotated alias.
fn fit_air_maps<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    l: &VtLayout,
    arity: &HashMap<u64, Option<usize>>,
    skip: &HashSet<u64>,
) -> ProofmanResult<Vec<(u64, Result<VtFittedMap, String>)>> {
    let setup = sctx.get_setup(l.airgroup_id as usize, l.air_id as usize)?;
    let p_setup = &setup.p_setup; // Sync, unlike the raw pointer it converts to
    let num_rows = l.num_rows;
    let total = l.num_cols * num_rows;

    // (table, base, end, key width) for the tables the prover may own; the rest say why not.
    let mut out = Vec::new();
    let mut todo = Vec::new();
    for (&tid, &base) in l.table_ids.iter().zip(&l.acc_bases) {
        let end = l.acc_bases.iter().copied().filter(|h| *h > base).min().unwrap_or(total);
        if end == base {
            continue;
        }
        let width = match arity.get(&tid) {
            _ if skip.contains(&tid) => Err("a range table".to_string()),
            Some(Some(0)) => Err("its lookup sends no elements".into()),
            Some(Some(w)) => Ok(*w),
            Some(None) => Err("its lookups send tuples of different widths".into()),
            None => Err("no lookup into it was found".into()),
        };
        match width {
            Ok(w) => todo.push((tid, base, end, w)),
            Err(why) => out.push((tid, Err(why))),
        }
    }
    let fail_all = |mut out: Vec<_>, why: String| {
        out.extend(todo.iter().map(|t| (t.0, Err(why.clone()))));
        Ok(out)
    };
    if todo.is_empty() {
        return Ok(out);
    }

    let mut by_col: HashMap<u64, (usize, usize)> = HashMap::new();
    for (k, (col, len)) in mul_proves_hints_c(p_setup.into()).into_iter().enumerate() {
        if let Some(c) = col {
            if by_col.insert(c as u64, (k, len)).is_some() {
                return fail_all(out, format!("two proves-side lookups credit multiplicity column {c}"));
            }
        }
    }
    let path = setup.const_pols_path.replace(".const_gpu", ".const");
    let const_pols = match read_const_pols(&path) {
        Ok(v) => v,
        Err(e) => return fail_all(out, format!("cannot read {path} ({e})")),
    };
    let fit_table = |&(tid, base, end, width): &(u64, u64, u64, usize)| -> Result<VtFittedMap, String> {
        let t_fit = std::time::Instant::now();
        let height = (end - base) as usize;
        let (mut keys, mut bus) = (vec![0u64; height * width], vec![0u64; height]);
        // Disjoint pieces, none crossing a column, evaluated in parallel straight into place.
        const CHUNK: u64 = 1 << 14;
        let mut pieces = Vec::new();
        let (mut k_rest, mut b_rest) = (&mut keys[..], &mut bus[..]);
        let mut o = base;
        while o < end {
            let o1 = (o + CHUNK).min(end).min((o / num_rows + 1) * num_rows);
            let n = (o1 - o) as usize;
            let (k, kr) = std::mem::take(&mut k_rest).split_at_mut(n * width);
            let (b, br) = std::mem::take(&mut b_rest).split_at_mut(n);
            pieces.push((o, k, b));
            (k_rest, b_rest, o) = (kr, br, o1);
        }
        pieces.into_par_iter().try_for_each(|(o, k, b)| {
            let c = o / num_rows;
            let &(hint, len) = by_col.get(&c).ok_or_else(|| format!("no proves-side lookup credits column {c}"))?;
            if len < width {
                return Err(format!("its lookup sends {width} elements but column {c} carries {len}"));
            }
            if !mul_eval_proves_hint_c(p_setup.into(), hint, &const_pols, (o % num_rows) as usize, b, k, width) {
                return Err(format!("the lookup crediting column {c} does not evaluate from fixed columns"));
            }
            Ok(())
        })?;

        // Its rows carry its first row's bus id. Tables pack densely: only the air's last one may
        // end in padding, which carries another.
        let table_bus = bus[0];
        let n_mine = bus.iter().position(|b| *b != table_bus).unwrap_or(height);
        if (end < total && n_mine != height) || bus[n_mine..].contains(&table_bus) {
            return Err(format!("its rows with bus id {table_bus} are not one run from its base"));
        }
        keys.truncate(n_mine * width);
        let rows: Vec<u64> = (0..n_mine as u64).collect();
        let (nkey, kv, slots) = fit_exact_map(&keys, &rows, width).ok_or_else(|| {
            let maxima: Vec<u64> =
                (0..width).map(|j| keys.iter().skip(j).step_by(width).copied().max().unwrap_or(0)).collect();
            format!("no exact map fits its {} entries of {width} columns (column maxima {maxima:?})", rows.len())
        })?;
        tracing::debug!(
            "virtual table {tid}: exact map over {} entries, {nkey} key columns, {slots} slots ({} MB, {} ms)",
            rows.len(),
            kv.len() * 8 / 1_000_000,
            t_fit.elapsed().as_millis()
        );
        Ok(VtFittedMap { table_id: tid, map: (nkey, kv, slots) })
    };
    out.par_extend(todo.par_iter().map(|t| (t.0, fit_table(t))));
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
    pub fn new(pctx: &Arc<ProofCtx<F>>, sctx: &SetupCtx<F>, shared_tables: bool) -> ProofmanResult<Arc<Self>> {
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
                pctx: Some(pctx.clone()),
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
            pctx: None,
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

    /// Whether the prover counts this table itself, resolved once on first use: the host fills
    /// `prover_owned_tables` after these airs are built, so reading it at construction would
    /// double-count.
    fn is_prover_owned(&self, id: usize) -> bool {
        self.prover_owned.get_or_init(|| {
            let owned = self.pctx.as_ref().map(|p| p.prover_owned_tables.read().unwrap().clone()).unwrap_or_default();
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
            let key = (self.airgroup_id, self.air_id);
            let device_owned = pctx.device_owned_table_airs.read().unwrap().contains(&key);

            // Before `distribute_multiplicities`, so the MPI path sees a complete accumulator.
            if let Some(counts) = pctx.prover_counts.read().unwrap().get(&key).filter(|_| !device_owned) {
                self.multiplicities.par_iter().zip(counts.par_iter()).for_each(|(slot, add)| {
                    if *add != 0 {
                        slot.fetch_add(*add, Ordering::Relaxed);
                    }
                });
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

    /// Probe a `fit_exact_map` result as the GPU decoder does.
    fn probe(kv: &[u64], tuple: &[u64]) -> Option<u64> {
        map_pack(kv, tuple).and_then(|kw| map_find(kv, &kw))
    }

    fn fit(samples: &[(Vec<u64>, u64)], width: usize) -> Option<(usize, Vec<u64>, u64)> {
        let keys: Vec<u64> = samples.iter().flat_map(|(t, _)| t[..width].iter().copied()).collect();
        let rows: Vec<u64> = samples.iter().map(|(_, r)| *r).collect();
        fit_exact_map(&keys, &rows, width)
    }

    /// Five small, distinct tuples: every one of them must probe back to its own row.
    #[test]
    fn fit_exact_map_resolves_every_sample() {
        let samples: Vec<(Vec<u64>, u64)> =
            vec![(vec![1, 2], 10), (vec![3, 4], 20), (vec![5, 6], 30), (vec![7, 8], 40), (vec![9, 10], 50)];
        let (_, kv, _) = fit(&samples, 2).expect("a small exact table must fit");
        for (t, r) in samples.iter() {
            assert_eq!(probe(&kv, t), Some(*r));
        }
    }

    /// A tuple not in the table must probe as "not found", never alias onto another row.
    #[test]
    fn fit_exact_map_out_of_range_key_is_not_found() {
        let samples: Vec<(Vec<u64>, u64)> =
            vec![(vec![1, 2], 10), (vec![3, 4], 20), (vec![5, 6], 30), (vec![7, 8], 40), (vec![9, 10], 50)];
        let (_, kv, _) = fit(&samples, 2).expect("a small exact table must fit");
        assert_eq!(probe(&kv, &[100, 200]), None);
        // Inside each column's range but not a table tuple.
        assert_eq!(probe(&kv, &[1, 4]), None);
    }

    /// A constant trailing column is still part of the key: a lookup that differs only there is not in
    /// the table.
    #[test]
    fn fit_exact_map_checks_constant_trailing_columns() {
        let samples: Vec<(Vec<u64>, u64)> = (0..8u64).map(|i| (vec![i, 7], i)).collect();
        let (nkey, kv, _) = fit(&samples, 2).expect("must fit");
        assert_eq!(nkey, 2);
        assert_eq!(probe(&kv, &[3, 7]), Some(3));
        assert_eq!(probe(&kv, &[3, 8]), None, "same prefix, other constant");
    }

    /// Duplicate tuples: all credit lands on the first row written.
    #[test]
    fn fit_exact_map_duplicate_tuple_credited_to_one_row() {
        let samples: Vec<(Vec<u64>, u64)> = vec![(vec![1, 2], 10), (vec![1, 2], 99), (vec![3, 4], 20)];
        let (_, kv, slots) = fit(&samples, 2).expect("must still fit with a duplicate");
        assert_eq!(probe(&kv, &[1, 2]), Some(10), "credit goes to the first-written row");
        assert_eq!(slots, 3, "sized by distinct tuples at 80%");
    }

    /// More than eight columns, some a full field element wide: several key words, row in its own.
    #[test]
    fn fit_exact_map_many_wide_columns() {
        let p = 0xFFFF_FFFF_0000_0001u64;
        let samples: Vec<(Vec<u64>, u64)> = (0..500u64)
            .map(|i| {
                let t: Vec<u64> =
                    (0..11u64).map(|c| if c % 3 == 0 { (i * 0x9E37_79B9 + c) % p } else { i % (c + 2) }).collect();
                (t, i * 7)
            })
            .collect();
        let (nkey, kv, _) = fit(&samples, 11).expect("eleven columns must fit");
        assert_eq!(nkey, 11);
        assert!(((kv[1] >> 16) & 0xFF) > 1, "full-width columns need several key words");
        for (t, r) in samples.iter() {
            assert_eq!(probe(&kv, t), Some(*r));
        }
        let mut miss = samples[3].0.clone();
        miss[1] = (miss[1] + 1) % 3;
        assert_eq!(probe(&kv, &miss), None);
    }

    /// Column minima are subtracted: large but narrow columns still pack with the row in one word.
    #[test]
    fn fit_exact_map_offsets_by_min() {
        let samples: Vec<(Vec<u64>, u64)> = (0..256u64).map(|i| (vec![1_000_000 + i, 5 + i % 4], i)).collect();
        let (_, kv, _) = fit(&samples, 2).expect("must fit");
        assert_eq!((kv[1] >> 16) & 0xFF, 1, "one key word");
        assert_eq!((kv[1] >> 24) & 0xFF, 1, "row inline: one word per slot");
        for (t, r) in samples.iter() {
            assert_eq!(probe(&kv, t), Some(*r));
        }
        assert_eq!(probe(&kv, &[999_999, 5]), None, "below the column minimum");
    }
}
