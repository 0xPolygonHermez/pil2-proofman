#ifndef MULTIPLICITY_HPP
#define MULTIPLICITY_HPP

#include <cstdint>
#include <vector>
#include <map>
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
    zklog.info("Virtual table air " + std::to_string(airgroupId) + "/" + std::to_string(airId) + ": "
               + std::to_string(numRows) + " rows x " + std::to_string(numCols) + " cols = "
               + std::to_string(L.nCounters) + " counters ("
               + std::to_string(L.nCounters * sizeof(uint64_t) / 1000000) + " MB), "
               + std::to_string(L.accBase.size()) + " tables");
}


// Range tables the prover owns, as (table_id, bias). Requests and layouts arrive in either order,
// so decoders are materialised lazily from whatever is known. Shared by both backends' C entry
// points, which differ only in what they do with the resulting decoders.
struct MulOwnedReq { uint64_t table_id; int64_t bias; };

inline std::vector<MulOwnedReq>& mulOwnedReqs() {
    static std::vector<MulOwnedReq> v;
    return v;
}

// A row map fitted from a table's own COL_* fixed columns and verified there, handed down by the
// side that can read them. Recorded per table id and attached to the decoder when it materialises,
// so registration order does not matter.
struct MulFittedMap { uint8_t nCoef; uint64_t coef[MUL_MAX_TUPLE]; uint64_t konst; };

inline std::map<uint64_t, MulFittedMap>& mulFittedMaps() {
    static std::map<uint64_t, MulFittedMap> m;
    return m;
}

inline void mul_register_table_decode_impl(uint64_t tableId, const uint64_t* coef, uint64_t nCoef,
                                           uint64_t konst) {
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

// Key->row indexes, owned here so they outlive every plan. Keyed by table id; the host copy is what
// the CPU scatter reads and what each device copy is made from.
struct MulTableIndex { uint64_t keyMin; std::vector<uint32_t> rows; };

inline std::map<uint64_t, MulTableIndex>& mulTableIndexes() {
    static std::map<uint64_t, MulTableIndex> m;
    return m;
}

inline void mul_register_table_index_impl(uint64_t tableId, uint64_t keyMin, const uint32_t* rows,
                                          uint64_t len) {
    MulTableIndex ix;
    ix.keyMin = keyMin;
    ix.rows.assign(rows, rows + len);
    mulTableIndexes()[tableId] = std::move(ix);
    const MulTableIndex& stored = mulTableIndexes()[tableId];
    for (auto& d : mulDecoders())
        if (d.table_id == tableId) {
            d.keyMin = stored.keyMin;
            d.indexLen = stored.rows.size();
            d.index = stored.rows.data();
        }
    zklog.info("Multiplicity index: table " + std::to_string(tableId) + " key range "
               + std::to_string(len) + " (" + std::to_string(len * sizeof(uint32_t) / 1000000)
               + " MB), keyMin=" + std::to_string(keyMin));
}

// Digit remaps, by table id. Tiny, so carried by value into every job.
struct MulRemap { uint32_t baseIn, baseOut, nDigits; uint32_t map[MUL_MAX_DIGIT_BASE]; };

inline std::map<uint64_t, MulRemap>& mulRemaps() { static std::map<uint64_t, MulRemap> m; return m; }

inline void mul_register_table_remap_impl(uint64_t tableId, uint64_t baseIn, uint64_t baseOut,
                                          uint64_t nDigits, const uint32_t* map, uint64_t mapLen) {
    if (baseIn == 0 || baseIn > MUL_MAX_DIGIT_BASE || mapLen > MUL_MAX_DIGIT_BASE) {
        zklog.error("multiplicity: table " + std::to_string(tableId) + " remap base "
                    + std::to_string(baseIn) + " exceeds the cap");
        exitProcess();
    }
    MulRemap r{};
    r.baseIn = (uint32_t)baseIn; r.baseOut = (uint32_t)baseOut; r.nDigits = (uint32_t)nDigits;
    for (uint32_t i = 0; i < MUL_MAX_DIGIT_BASE; ++i) r.map[i] = MUL_INDEX_NONE;
    for (uint64_t i = 0; i < mapLen; ++i) r.map[i] = map[i];
    mulRemaps()[tableId] = r;
    for (auto& d : mulDecoders())
        if (d.table_id == tableId) {
            d.baseIn = r.baseIn; d.baseOut = r.baseOut; d.nDigits = r.nDigits;
            for (uint32_t i = 0; i < MUL_MAX_DIGIT_BASE; ++i) d.digitMap[i] = r.map[i];
        }
    zklog.info("Multiplicity remap: table " + std::to_string(tableId) + " base "
               + std::to_string(baseIn) + " -> " + std::to_string(baseOut) + ", "
               + std::to_string(nDigits) + " digits (no index needed)");
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
            auto rm = mulRemaps().find(o.table_id);
            if (rm != mulRemaps().end()) {
                d.baseIn = rm->second.baseIn; d.baseOut = rm->second.baseOut;
                d.nDigits = rm->second.nDigits;
                for (uint32_t i = 0; i < MUL_MAX_DIGIT_BASE; ++i) d.digitMap[i] = rm->second.map[i];
            }
            auto ix = mulTableIndexes().find(o.table_id);
            if (ix != mulTableIndexes().end()) {
                d.keyMin = ix->second.keyMin;
                d.indexLen = ix->second.rows.size();
                d.index = ix->second.rows.data();
            }
            auto fit = mulFittedMaps().find(o.table_id);
            if (fit != mulFittedMaps().end()) {
                d.nCoef = fit->second.nCoef;
                d.konst = fit->second.konst;
                for (uint8_t c = 0; c < d.nCoef; ++c) d.coef[c] = fit->second.coef[c];
            }
            if (d.acc_base + d.n_rows > L.nCounters) {
                zklog.error("multiplicity decoder " + std::to_string(o.table_id)
                            + " span exceeds its virtual table");
                exitProcess();
            }
            mulDecoders().push_back(d);
            zklog.info("Multiplicity decoder: table " + std::to_string(o.table_id)
                       + " acc_base=" + std::to_string(d.acc_base)
                       + " n_rows=" + std::to_string(d.n_rows)
                       + " bias=" + std::to_string(d.bias) + " air=" + std::to_string(L.airId));
        }
    }
}

// Every virtual range-check table, resolved by the std from its `range_def` hints: the row a lookup
// addresses is `value + bias`. Nothing here is per-table, so a new range needs no code change.
inline void mul_register_range_tables_impl(const uint64_t* tableIds, const int64_t* biases, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) {
        auto it = std::find_if(mulOwnedReqs().begin(), mulOwnedReqs().end(),
                               [&](const MulOwnedReq& o){ return o.table_id == tableIds[i]; });
        if (it == mulOwnedReqs().end()) {
            mulOwnedReqs().push_back({tableIds[i], biases[i]});
        } else if (it->bias != biases[i]) {
            // Two AIRs disagreeing about a range's minimum is a setup bug; taking either would
            // silently corrupt the table.
            zklog.error("multiplicity: table " + std::to_string(tableIds[i])
                        + " registered with two different biases");
            exitProcess();
        }
    }
    mul_materialize_decoders();
    zklog.info("Multiplicity: " + std::to_string(n) + " range tables requested, "
               + std::to_string(mulDecoders().size()) + " decoders registered");
}

// Tables whose multiplicities the prover owns, so Std stops counting them.
inline uint64_t mul_migrated_tables_impl(uint64_t* out, uint64_t cap) {
    uint64_t n = 0;
    for (const auto& d : mulDecoders()) { if (n >= cap) break; out[n++] = d.table_id; }
    return n;
}

// What is still Rust-owned, with the height a decoder would have to cover.
inline void mul_log_coverage() {
    for (const auto& L : mulVtLayouts()) {
        size_t have = 0;
        std::string todo;
        for (const auto& kv : L.accBase) {
            if (mulDecoderFor(kv.first) != nullptr) ++have;
            else todo += " " + std::to_string(kv.first) + "(h=" + std::to_string(L.tableHeight(kv.first)) + ")";
        }
        zklog.info("Multiplicity coverage air " + std::to_string(L.airId) + ": " + std::to_string(have)
                   + "/" + std::to_string(L.accBase.size()) + " tables prover-owned; remaining:" + todo);
    }
}

// Commits completed this proof. The fold cannot assume every instance has scattered: proofman
// queues instances for commit during CALCULATING_WITNESS but joins the commit workers only after
// CALCULATING_TABLES, and the fold runs inside CALCULATING_TABLES.
inline std::atomic<uint64_t>& mulCommits() { static std::atomic<uint64_t> n{0}; return n; }

inline void mul_note_commit() { mulCommits().fetch_add(1, std::memory_order_release); }

// Block until `expected` instances have committed. Bounded: a count that never arrives is a bug in
// the expectation, and hanging the prover would hide it.
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
