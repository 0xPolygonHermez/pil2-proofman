// Registry of airs whose stage-1 witness a caller-supplied kernel writes
// straight into the commit slot, instead of the prover uploading a host trace.
// Backend-agnostic (compiles into the CPU library, which never registers any).
//
// Keyed by the owning prover's device buffers, so one prover's registration never replaces or
// clears another's; every access holds one mutex and lookups return a copy.

#ifndef GPU_WITNESS_HPP
#define GPU_WITNESS_HPP

#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

/// What a kernel writes into the commit slot. Must match the air's packing:
/// the transform that runs afterwards decodes it unchecked.
enum GpuWitnessLayout : int {
    GPU_WITNESS_PACKED_CM1 = 0,
    GPU_WITNESS_PLAIN_CM1 = 1,
};

/// A registered kernel. Enqueue-only against caller-owned memory: no
/// allocation, no sync, no new stream, exactly one launch on `stream` (keeps it
/// legal inside the commit path's CUDA graph capture).
/// `device_id` -1 leaves the current device alone. Returns 0 on success; a
/// negative value aborts the commit.
typedef int (*GpuWitnessFillFn)(const void *d_ops, uint64_t num_ops, uint64_t *d_dst,
                                int device_id, void *stream);

struct GpuWitnessAirReg {
    uint64_t airgroupId;
    uint64_t airId;
    /// Size of one staged operation, so an instance's op count becomes a byte count.
    uint64_t bytesPerOp;
    int emits;
    GpuWitnessFillFn fill;
};

inline std::map<const void *, std::vector<GpuWitnessAirReg>> &gpuWitnessRegs() {
    static std::map<const void *, std::vector<GpuWitnessAirReg>> regs;
    return regs;
}

inline std::mutex &gpuWitnessMutex() {
    static std::mutex m;
    return m;
}

/// Register (or replace) an air's kernel for `owner`; a later registration for the same air wins.
inline void gpu_witness_register_impl(const void *owner, uint64_t airgroupId, uint64_t airId,
                                      uint64_t bytesPerOp, int emits, GpuWitnessFillFn fill) {
    std::lock_guard<std::mutex> lk(gpuWitnessMutex());
    auto &regs = gpuWitnessRegs()[owner];
    for (auto &r : regs) {
        if (r.airgroupId == airgroupId && r.airId == airId) {
            r.bytesPerOp = bytesPerOp;
            r.emits = emits;
            r.fill = fill;
            return;
        }
    }
    regs.push_back(GpuWitnessAirReg{airgroupId, airId, bytesPerOp, emits, fill});
}

/// Drop `owner`'s registrations: at its start (a freed owner's address can be reused) and teardown.
inline void gpu_witness_clear_impl(const void *owner) {
    std::lock_guard<std::mutex> lk(gpuWitnessMutex());
    gpuWitnessRegs().erase(owner);
}

/// Copies `owner`'s kernel for this air into `out`; false when the host still fills it.
inline bool gpu_witness_for(const void *owner, uint64_t airgroupId, uint64_t airId, GpuWitnessAirReg *out) {
    std::lock_guard<std::mutex> lk(gpuWitnessMutex());
    auto it = gpuWitnessRegs().find(owner);
    if (it == gpuWitnessRegs().end()) return false;
    for (const auto &r : it->second) {
        if (r.airgroupId == airgroupId && r.airId == airId) {
            *out = r;
            return true;
        }
    }
    return false;
}

inline uint64_t gpu_witness_count_impl(const void *owner) {
    std::lock_guard<std::mutex> lk(gpuWitnessMutex());
    auto it = gpuWitnessRegs().find(owner);
    return it == gpuWitnessRegs().end() ? 0 : (uint64_t)it->second.size();
}

#endif  // GPU_WITNESS_HPP
