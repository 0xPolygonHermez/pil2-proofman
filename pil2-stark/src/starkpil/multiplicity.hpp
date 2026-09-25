#ifndef MULTIPLICITY_HPP
#define MULTIPLICITY_HPP

#include <cstdint>
#include <vector>
#include <map>
#include <string>
#include <mutex>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include "zklog.hpp"
#include "exit_process.hpp"
#include "multiplicity_decoders.hpp"

// Virtual-table geometry, registered per run from StdVirtualTable::new. Integers only: the host
// accumulator pointer is passed per call, never retained here.
struct MulVtLayout {
    uint64_t airId = 0;
    uint64_t nCounters = 0;                  // numRows * num_muls
    std::map<uint64_t, uint64_t> accBase;    // table_id -> acc_height

    // Height = gap to the next table's offset; the last runs to the end.
    uint64_t tableHeight(uint64_t table_id) const {
        std::vector<uint64_t> offs;
        offs.reserve(accBase.size());
        for (const auto& kv : accBase) offs.push_back(kv.second);
        std::sort(offs.begin(), offs.end());
        const uint64_t base = accBase.at(table_id);
        auto it = std::upper_bound(offs.begin(), offs.end(), base);
        return (it == offs.end() ? nCounters : *it) - base;
    }
};

inline std::vector<MulVtLayout>& mulVtLayouts() {
    static std::vector<MulVtLayout> v;
    return v;
}

inline void mul_register_vt(uint64_t airgroupId, uint64_t airId,
                            uint64_t numRows, uint64_t numCols,
                            const uint64_t* tableIds, const uint64_t* accBases, uint64_t nTables) {
    MulVtLayout L;
    L.airId = airId;
    L.nCounters = numRows * numCols;
    for (uint64_t k = 0; k < nTables; ++k) L.accBase[tableIds[k]] = accBases[k];
    mulVtLayouts().push_back(L);
    zklog.trace("Virtual table air " + std::to_string(airgroupId) + "/" + std::to_string(airId) + ": "
               + std::to_string(numRows) + " rows x " + std::to_string(numCols) + " cols = "
               + std::to_string(L.nCounters) + " counters ("
               + std::to_string(L.nCounters * sizeof(uint64_t) / 1000000) + " MB), "
               + std::to_string(L.accBase.size()) + " tables");
}


// Range tables the prover owns, as (table_id, bias). Requests and layouts arrive in either order,
// so decoders are materialised lazily. Shared by both backends.
struct MulOwnedReq { uint64_t table_id; int64_t bias; };

inline std::vector<MulOwnedReq>& mulOwnedReqs() {
    static std::vector<MulOwnedReq> v;
    return v;
}

// A row map fitted and verified from a table's COL_* fixed columns. Attached to the decoder when
// it materialises, so registration order does not matter.
struct MulFittedMap { uint8_t nCoef; uint64_t coef[MUL_MAX_TUPLE]; uint64_t konst; };

inline std::map<uint64_t, MulFittedMap>& mulFittedMaps() {
    static std::map<uint64_t, MulFittedMap> m;
    return m;
}

// A table's row-map shape (fitted / exact-map / digit-rule / indexed-base) is meant to be
// singular: mulResolveRow picks among the last three by a fixed priority with no diagnostic, so a
// second, different shape registered for the same table would silently have one win and the other
// dropped with nothing to say why. Tracked separately from mulDecoders() because registration
// order is not fixed -- the decoder for this table may not exist yet when the conflict happens.
inline std::map<uint64_t, std::string>& mulShapeClaims() { static std::map<uint64_t, std::string> m; return m; }
inline std::mutex& mulShapeClaimsMutex() { static std::mutex m; return m; }

inline void mulClaimShape(uint64_t tableId, const char* shape) {
    std::lock_guard<std::mutex> lk(mulShapeClaimsMutex());
    auto& claimed = mulShapeClaims();
    auto it = claimed.find(tableId);
    if (it != claimed.end() && it->second != shape) {
        zklog.error("multiplicity: table " + std::to_string(tableId) + " already claimed as a \""
                    + it->second + "\" row map, now also registered as \"" + shape
                    + "\" -- a table can only have one shape");
        exitProcess();
    }
    claimed[tableId] = shape;
}

inline void mul_register_table_decode_impl(uint64_t tableId, const uint64_t* coef, uint64_t nCoef,
                                           uint64_t konst) {
    mulClaimShape(tableId, "fitted");
    if (nCoef == 0 || nCoef > MUL_MAX_TUPLE) {
        zklog.error("multiplicity: table " + std::to_string(tableId) + " fitted with "
                    + std::to_string(nCoef) + " coefficients; the cap is "
                    + std::to_string(MUL_MAX_TUPLE));
        exitProcess();
    }
    MulFittedMap f{};
    f.nCoef = (uint8_t)nCoef;
    f.konst = konst;
    for (uint64_t i = 0; i < nCoef; ++i) f.coef[i] = coef[i];
    mulFittedMaps()[tableId] = f;
    // A fitted map is itself a claim on the table: without this the table would have a row map and
    // no decoder, so nobody would count it. Bias is zero -- the fit's constant already places the
    // row.
    if (std::find_if(mulOwnedReqs().begin(), mulOwnedReqs().end(),
                     [&](const MulOwnedReq& o){ return o.table_id == tableId; }) == mulOwnedReqs().end())
        mulOwnedReqs().push_back({tableId, 0});
    // A decoder may already exist for this table (registration order is not fixed).
    for (auto& d : mulDecoders())
        if (d.table_id == tableId) { d.nCoef = f.nCoef; d.konst = f.konst;
                                     for (uint64_t i = 0; i < nCoef; ++i) d.coef[i] = coef[i]; }
}

// Exact-match maps, by table id.
inline std::map<uint64_t, std::vector<uint64_t>>& mulTableMaps() {
    static std::map<uint64_t, std::vector<uint64_t>> m;
    return m;
}

inline std::map<uint64_t, uint32_t>& mulTableMapKeys() { static std::map<uint64_t, uint32_t> m; return m; }

// Slots per map, stored: it cannot be derived from the buffer length once a layout has a header.
inline std::map<uint64_t, uint64_t>& mulTableMapSlots() { static std::map<uint64_t, uint64_t> m; return m; }

inline void mul_register_table_map_impl(uint64_t tableId, const uint64_t* kv, uint64_t n,
                                        uint64_t slots, uint64_t nKey) {
    mulClaimShape(tableId, "map");
    // The probe masks with `slots - 1`, so slots must be a power of two.
    if (slots == 0 || (slots & (slots - 1)) != 0) {
        zklog.error("multiplicity: table " + std::to_string(tableId) + " map has " + std::to_string(slots)
                    + " slots, which is not a power of two -- the probe mask would be meaningless");
        exitProcess();
    }
    // Length is given, never re-derived: a packed map is `1 + nKey + slots*2` words.
    auto& v = mulTableMaps()[tableId];
    v.assign(kv, kv + n);
    mulTableMapKeys()[tableId] = (uint32_t)nKey;
    mulTableMapSlots()[tableId] = slots;
    // A map is a claim on the table, exactly as a fitted row map is: without this the table gets a
    // map and no decoder, so nobody counts it.
    if (std::find_if(mulOwnedReqs().begin(), mulOwnedReqs().end(),
                     [&](const MulOwnedReq& o){ return o.table_id == tableId; }) == mulOwnedReqs().end())
        mulOwnedReqs().push_back({tableId, 0});
    for (auto& d : mulDecoders())
        if (d.table_id == tableId) { d.mapSlots = slots; d.mapKV = v.data(); d.nKey = (uint32_t)nKey; }
    zklog.trace("Multiplicity map: table " + std::to_string(tableId) + " " + std::to_string(slots)
               + " slots x " + std::to_string(nKey) + " key cols"
               + (v.empty() || v[0] != MUL_MAP_PACKED ? "" : ", packed")
               + " (" + std::to_string(v.size() * 8 / 1000000) + " MB)");
}

// Digit-recoding tables, by table id.
inline std::map<uint64_t, std::vector<uint64_t>>& mulTableDigits() {
    static std::map<uint64_t, std::vector<uint64_t>> m;
    return m;
}
struct MulDigitShape { uint32_t cols; uint32_t col[MUL_MAX_TUPLE]; };
inline std::map<uint64_t, MulDigitShape>& mulTableDigitShape() {
    static std::map<uint64_t, MulDigitShape> m;
    return m;
}

inline void mul_register_table_digits_impl(uint64_t tableId, const uint64_t* tab, uint64_t n,
                                           const uint32_t* cols, uint64_t nCols) {
    mulClaimShape(tableId, "digits");
    auto& v = mulTableDigits()[tableId];
    v.assign(tab, tab + n);
    MulDigitShape sh{};
    sh.cols = (uint32_t)nCols;
    if (nCols > MUL_MAX_TUPLE) {
        zklog.error("multiplicity: table " + std::to_string(tableId) + " separable over "
                    + std::to_string(nCols) + " columns, more than the tuple holds");
        exitProcess();
    }
    for (uint64_t i = 0; i < nCols; ++i) sh.col[i] = cols[i];
    mulTableDigitShape()[tableId] = sh;
    // A digit rule is a claim on the table, exactly as a fitted row map or an exact map is.
    if (std::find_if(mulOwnedReqs().begin(), mulOwnedReqs().end(),
                     [&](const MulOwnedReq& o){ return o.table_id == tableId; }) == mulOwnedReqs().end())
        mulOwnedReqs().push_back({tableId, 0});
    for (auto& d : mulDecoders())
        if (d.table_id == tableId) {
            d.digitCols = sh.cols;
            for (uint32_t i = 0; i < sh.cols; ++i) d.digitCol[i] = sh.col[i];
            d.digitTab = v.data();
        }
    zklog.trace("Multiplicity separable: table " + std::to_string(tableId) + " over "
               + std::to_string(nCols) + " columns (" + std::to_string(n * 8)
               + " bytes) -- no map needed");
}

// Indexed-base row maps, by table id: a per-block base plus uniform strides (table 125's shape).
inline std::map<uint64_t, std::vector<uint64_t>>& mulTableIndexedBase() {
    static std::map<uint64_t, std::vector<uint64_t>> m;
    return m;
}
struct MulIndexedBaseShape {
    uint32_t nSel = 0;
    uint32_t selCol[MUL_MAX_TUPLE]   = {0};
    uint32_t selShift[MUL_MAX_TUPLE] = {0};
    uint64_t selMask[MUL_MAX_TUPLE]  = {0};
    uint32_t nStride = 0;
    uint32_t strideCol[MUL_MAX_TUPLE] = {0};
    uint64_t strideVal[MUL_MAX_TUPLE] = {0};
    // Lower bound on the tuple width the rule reads (max selCol/strideCol + 1). The registration
    // call carries no explicit tuple width, so this is a derived floor -- exact for the plan
    // builder, which evaluates the hint's own `fEx->values.size()` instead and never reads this;
    // it exists so the interpreter fallback (hints.cu, expressions_gpu.cu), which has no access to
    // the hint at that point, knows how many tuple columns to evaluate before calling mulResolveRow.
    uint32_t nKey = 0;
};
inline std::map<uint64_t, MulIndexedBaseShape>& mulTableIndexedBaseShape() {
    static std::map<uint64_t, MulIndexedBaseShape> m;
    return m;
}

inline void mul_register_table_indexed_base_impl(uint64_t tableId, const uint64_t* sel, uint64_t nSel,
                                                  const uint64_t* base, uint64_t nBase,
                                                  const uint64_t* stride, uint64_t nStride) {
    mulClaimShape(tableId, "indexed-base");
    if (nSel == 0 || nSel > MUL_MAX_TUPLE || nStride > MUL_MAX_TUPLE) {
        zklog.error("multiplicity: table " + std::to_string(tableId) + " indexed-base rule has "
                    + std::to_string(nSel) + " selector fields and " + std::to_string(nStride)
                    + " strides; the cap is " + std::to_string(MUL_MAX_TUPLE));
        exitProcess();
    }
    // The selector packs its fields most-significant-first, so the space it addresses is the
    // product of (mask+1) over every field -- `base` must cover exactly that, or an in-range
    // selector would read past the table (or a live entry would never be reached).
    uint64_t want = 1;
    for (uint64_t i = 0; i < nSel; ++i) want *= sel[3 * i + 2] + 1;
    if (want != nBase) {
        zklog.error("multiplicity: table " + std::to_string(tableId) + " indexed-base has "
                    + std::to_string(nBase) + " base entries, selector space is "
                    + std::to_string(want));
        exitProcess();
    }
    auto& v = mulTableIndexedBase()[tableId];
    v.assign(base, base + nBase);
    MulIndexedBaseShape sh{};
    sh.nSel = (uint32_t)nSel;
    for (uint64_t i = 0; i < nSel; ++i) {
        sh.selCol[i]   = (uint32_t)sel[3 * i];
        sh.selShift[i] = (uint32_t)sel[3 * i + 1];
        sh.selMask[i]  = sel[3 * i + 2];
    }
    sh.nStride = (uint32_t)nStride;
    for (uint64_t i = 0; i < nStride; ++i) {
        sh.strideCol[i] = (uint32_t)stride[2 * i];
        sh.strideVal[i] = stride[2 * i + 1];
    }
    for (uint64_t i = 0; i < nSel; ++i) sh.nKey = std::max(sh.nKey, sh.selCol[i] + 1);
    for (uint64_t i = 0; i < nStride; ++i) sh.nKey = std::max(sh.nKey, sh.strideCol[i] + 1);
    mulTableIndexedBaseShape()[tableId] = sh;
    // An indexed-base rule is a claim on the table, exactly as a fitted row map, an exact map, or a
    // digit rule is: without this the table gets a row map and no decoder, so nobody counts it.
    if (std::find_if(mulOwnedReqs().begin(), mulOwnedReqs().end(),
                     [&](const MulOwnedReq& o){ return o.table_id == tableId; }) == mulOwnedReqs().end())
        mulOwnedReqs().push_back({tableId, 0});
    // A decoder may already exist for this table (registration order is not fixed).
    for (auto& d : mulDecoders())
        if (d.table_id == tableId) {
            d.nSel = sh.nSel;
            for (uint32_t i = 0; i < sh.nSel; ++i) {
                d.selCol[i] = sh.selCol[i]; d.selShift[i] = sh.selShift[i]; d.selMask[i] = sh.selMask[i];
            }
            d.nStride = sh.nStride;
            for (uint32_t i = 0; i < sh.nStride; ++i) {
                d.strideCol[i] = sh.strideCol[i]; d.strideVal[i] = sh.strideVal[i];
            }
            d.baseTab = v.data();
            d.nKey = sh.nKey;
        }
    // Per-table, like the map and digit registrations above -- trace, not info.
    zklog.trace("Multiplicity indexed-base: table " + std::to_string(tableId) + " "
               + std::to_string(nSel) + " selector fields, " + std::to_string(nBase) + " bases ("
               + std::to_string(nBase * 8 / 1000000) + " MB), " + std::to_string(nStride) + " strides");
}

inline void mul_materialize_decoders() {
    for (const auto& L : mulVtLayouts()) {
        for (const auto& o : mulOwnedReqs()) {
            if (!L.accBase.count(o.table_id) || mulDecoderFor(o.table_id) != nullptr) continue;
            MulDecoder d{};
            d.table_id = (uint32_t)o.table_id;
            d.acc_base = L.accBase.at(o.table_id);
            d.n_rows   = L.tableHeight(o.table_id);
            d.bias     = o.bias;
            auto dg = mulTableDigits().find(o.table_id);
            if (dg != mulTableDigits().end() && !dg->second.empty()) {
                const MulDigitShape& sh = mulTableDigitShape()[o.table_id];
                d.digitCols = sh.cols;
                for (uint32_t i = 0; i < sh.cols; ++i) d.digitCol[i] = sh.col[i];
                d.digitTab = dg->second.data();
            }
            auto ib = mulTableIndexedBase().find(o.table_id);
            if (ib != mulTableIndexedBase().end() && !ib->second.empty()) {
                const MulIndexedBaseShape& sh = mulTableIndexedBaseShape()[o.table_id];
                d.nSel = sh.nSel;
                for (uint32_t i = 0; i < sh.nSel; ++i) {
                    d.selCol[i] = sh.selCol[i]; d.selShift[i] = sh.selShift[i]; d.selMask[i] = sh.selMask[i];
                }
                d.nStride = sh.nStride;
                for (uint32_t i = 0; i < sh.nStride; ++i) {
                    d.strideCol[i] = sh.strideCol[i]; d.strideVal[i] = sh.strideVal[i];
                }
                d.baseTab = ib->second.data();
                d.nKey = sh.nKey;
            }
            auto mp = mulTableMaps().find(o.table_id);
            if (mp != mulTableMaps().end()) {
                d.nKey = mulTableMapKeys()[o.table_id];
                d.mapSlots = mulTableMapSlots()[o.table_id];
                d.mapKV = mp->second.data();
            }
            auto fit = mulFittedMaps().find(o.table_id);
            if (fit != mulFittedMaps().end()) {
                d.nCoef = fit->second.nCoef;
                d.konst = fit->second.konst;
                for (uint8_t c = 0; c < d.nCoef; ++c) d.coef[c] = fit->second.coef[c];
            }
            if (d.mapSlots != 0 && (d.mapSlots & (d.mapSlots - 1)) != 0) {
                zklog.error("multiplicity: table " + std::to_string(o.table_id) + " map has "
                            + std::to_string(d.mapSlots) + " slots, which is not a power of two");
                exitProcess();
            }
            mulDecoders().push_back(d);
            zklog.trace("Multiplicity decoder: table " + std::to_string(o.table_id)
                       + " acc_base=" + std::to_string(d.acc_base)
                       + " n_rows=" + std::to_string(d.n_rows)
                       + " bias=" + std::to_string(d.bias) + " air=" + std::to_string(L.airId));
        }
    }
}

// Every virtual range-check table, from the std's `range_def` hints: row = `value + bias`.
inline void mul_register_range_tables_impl(const uint64_t* tableIds, const int64_t* biases, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) {
        auto it = std::find_if(mulOwnedReqs().begin(), mulOwnedReqs().end(),
                               [&](const MulOwnedReq& o){ return o.table_id == tableIds[i]; });
        if (it == mulOwnedReqs().end()) {
            mulOwnedReqs().push_back({tableIds[i], biases[i]});
        } else if (it->bias != biases[i]) {
            // Two AIRs disagreeing about a range's minimum is a setup bug.
            zklog.error("multiplicity: table " + std::to_string(tableIds[i])
                        + " registered with two different biases");
            exitProcess();
        }
    }
    mul_materialize_decoders();
    zklog.trace("Multiplicity: " + std::to_string(n) + " range tables requested, "
               + std::to_string(mulDecoders().size()) + " decoders registered");
}

// Tables whose multiplicities the prover owns, so Std stops counting them.
inline uint64_t mul_migrated_tables_impl(uint64_t* out, uint64_t cap) {
    uint64_t n = 0;
    for (const auto& d : mulDecoders()) { if (n >= cap) break; out[n++] = d.table_id; }
    return n;
}

// Whether the prover counts anything in this air; if not, the host builds no accumulator for it.
inline bool mul_air_has_owned_tables(uint64_t airId) {
    for (const auto& l : mulVtLayouts()) {
        if (l.airId != airId) continue;
        for (const auto& kv : l.accBase) if (mulDecoderFor(kv.first) != nullptr) return true;
    }
    return false;
}

// One aggregate line of what is still Rust-owned across all virtual-table airs, with the height a
// decoder would have to cover. Shared by both backends' `mul_alloc`.
inline void mul_log_coverage() {
    if (mulVtLayouts().empty()) return;
    size_t have = 0, total = 0;
    std::string todo;
    for (const auto& L : mulVtLayouts()) {
        for (const auto& kv : L.accBase) {
            ++total;
            if (mulDecoderFor(kv.first) != nullptr) ++have;
            else todo += " " + std::to_string(kv.first) + "(h=" + std::to_string(L.tableHeight(kv.first)) + ")";
        }
    }
    zklog.info("Multiplicity coverage: " + std::to_string(have) + "/" + std::to_string(total)
               + " tables prover-owned across " + std::to_string(mulVtLayouts().size())
               + " airs; remaining:" + (todo.empty() ? " none" : todo));
}

// Commits completed this proof. The fold runs inside CALCULATING_TABLES, before the commit
// workers are joined, so it cannot assume every instance has scattered.
inline std::atomic<uint64_t>& mulCommits() { static std::atomic<uint64_t> n{0}; return n; }

inline void mul_note_commit() { mulCommits().fetch_add(1, std::memory_order_release); }

// Block until `expected` instances have committed. Bounded, so a wrong expectation fails loudly.

inline bool mul_await_commits(uint64_t expected) {
    using namespace std::chrono;
    const auto deadline = steady_clock::now() + seconds(120);
    while (mulCommits().load(std::memory_order_acquire) < expected) {
        if (steady_clock::now() > deadline) {
            zklog.error("multiplicity: only " + std::to_string(mulCommits().load())
                        + " of " + std::to_string(expected)
                        + " instances committed before the table fold -- counts would be short");
            return false;
        }
        std::this_thread::sleep_for(milliseconds(1));
    }
    return true;
}

inline void mul_reset_commits() { mulCommits().store(0, std::memory_order_release); }

#endif
