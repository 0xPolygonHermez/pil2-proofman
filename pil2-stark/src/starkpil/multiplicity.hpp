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

inline void mul_materialize_decoders() {
    for (const auto& L : mulVtLayouts()) {
        for (const auto& o : mulOwnedReqs()) {
            if (!L.accBase.count(o.table_id) || mulDecoderFor(o.table_id) != nullptr) continue;
            MulDecoder d{};
            d.table_id = (uint32_t)o.table_id;
            d.acc_base = L.accBase.at(o.table_id);
            d.n_rows   = L.tableHeight(o.table_id);
            d.bias     = o.bias;
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
