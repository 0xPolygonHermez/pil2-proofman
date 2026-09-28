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

inline const MulVtLayout* mulLayoutFor(uint64_t airId) {
    for (const auto& l : mulVtLayouts()) if (l.airId == airId) return &l;
    return nullptr;
}

// True iff a registered decoder targets a table in this layout.
inline bool mulLayoutHostsMigrated(const MulVtLayout& L) {
    for (const auto& d : mulDecoders())
        if (L.accBase.count(d.table_id)) return true;
    return false;
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

// Every decode rule is also a claim on the table; without one nobody counts it.
inline void mulClaimOwned(uint64_t tableId) {
    if (std::find_if(mulOwnedReqs().begin(), mulOwnedReqs().end(),
                     [&](const MulOwnedReq& o){ return o.table_id == tableId; }) == mulOwnedReqs().end())
        mulOwnedReqs().push_back({tableId, 0});
}

// Exact-match map of one table. Slots are stored: they cannot be derived from the buffer length
// once a layout has a header.
struct MulTableMap {
    std::vector<uint64_t> kv;
    uint32_t nKey = 0;
    uint64_t slots = 0;
};

inline std::map<uint64_t, MulTableMap>& mulTableMaps() {
    static std::map<uint64_t, MulTableMap> m;
    return m;
}

inline void mul_register_table_map_impl(uint64_t tableId, const uint64_t* kv, uint64_t n,
                                        uint64_t slots, uint64_t nKey) {
    // The probe masks with `slots - 1`, so slots must be a power of two.
    if (slots == 0 || (slots & (slots - 1)) != 0) {
        zklog.error("multiplicity: table " + std::to_string(tableId) + " map has " + std::to_string(slots)
                    + " slots, which is not a power of two -- the probe mask would be meaningless");
        exitProcess();
    }
    // Length is given, never re-derived: a packed map is `1 + nKey + slots*2` words.
    MulTableMap& m = mulTableMaps()[tableId];
    m.kv.assign(kv, kv + n);
    m.nKey = (uint32_t)nKey;
    m.slots = slots;
    const std::vector<uint64_t>& v = m.kv;
    mulClaimOwned(tableId);
    for (auto& d : mulDecoders())
        if (d.table_id == tableId) { d.mapSlots = slots; d.mapKV = v.data(); d.nKey = (uint32_t)nKey; }
    zklog.trace("Multiplicity map: table " + std::to_string(tableId) + " " + std::to_string(slots)
               + " slots x " + std::to_string(nKey) + " key cols"
               + (v.empty() || v[0] != MUL_MAP_PACKED ? "" : ", packed")
               + " (" + std::to_string(v.size() * 8 / 1000000) + " MB)");
}

inline void mul_materialize_decoders() {
    for (const auto& L : mulVtLayouts()) {
        for (const auto& o : mulOwnedReqs()) {
            if (!L.accBase.count(o.table_id) || mulDecoderFor(o.table_id) != nullptr) continue;
            MulDecoder d{};
            d.table_id = (uint32_t)o.table_id;
            d.hostAirId = L.airId;
            d.acc_base = L.accBase.at(o.table_id);
            d.n_rows   = L.tableHeight(o.table_id);
            d.bias     = o.bias;
            auto mp = mulTableMaps().find(o.table_id);
            if (mp != mulTableMaps().end()) {
                d.nKey = mp->second.nKey;
                d.mapSlots = mp->second.slots;
                d.mapKV = mp->second.kv.data();
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
    const MulVtLayout* L = mulLayoutFor(airId);
    return L != nullptr && mulLayoutHostsMigrated(*L);
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

// Commits completed this proof. mul_sync_commits runs inside CALCULATING_TABLES, before the commit
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
