#ifndef PILFFLONK_HINTS_GPU_HPP
#define PILFFLONK_HINTS_GPU_HPP

// The columns of the stages on a key on the GPU (pilfflonk/docs/performance.md#gpu) that are not the
// witness's: the std's prover hints of a stage s >= 2 and the im pols of every stage, computed on the
// device where the stage's commit reads them (InstanceGpu::commitStage), by the kernels of
// pilfflonk_hints.cu, the PLONK GPU prover's running product (gpu_plonk_prefix_scan_multiply) and the
// interpreter on the device (GpuAirKey::expressions). Only a library built with the GPU
// (__USE_CUDA__) has it, in pilfflonk_hints_gpu.cpp.
//
// Every column is the CPU's bit for bit (Instance::computeHintColumns and computeImPols): the same
// field elements from the same operands, in the CPU's order (AirKey::stdHints, imPolOrder), with
// the CPU's checks and errors (zeroDenominatorError, the interpreter's).

#include <cstdint>
#include <vector>

#include "pilfflonk_expressions.hpp"
#include "pilfflonk_key_gpu.hpp"

namespace PilFflonk {

// What computeStageColumns keeps of stage `stage` in a proof's arena, from ArenaLayout::hints on:
// byte offsets from there, each a multiple of 256.
struct StageScratch {
    // The fixed columns the stage's hints and im pols read (constPolsMap indices, in increasing
    // order), on H: column fixed[k] at fixedValues + k·N elements, which computeStageColumns
    // computes from their coefficients on the device: they are not kept there ("recomputed, not
    // kept", pilfflonk/docs/performance.md#rules-of-the-device-path).
    std::vector<uint64_t> fixed;
    uint64_t fixedValues = 0;
    // Those of the hints, if the stage has any (stage 1 has none: 0 bytes from `denominator` on).
    uint64_t denominator = 0; // N elements: a hint's denominator, when it is an expression
    uint64_t work = 0;        // the scans' work (pilfflonk_gpu_prefix_scan_work_elements)
    uint64_t zeroRow = 0;     // a 64-bit row: the first where a hint's denominator is 0
    uint64_t bytes = 0;
};

// That of stage `stage` (1 … nStages) of `air`.
StageScratch stageScratch(const AirKey &air, uint64_t stage);

// The most bytes of the scratch of a stage of `air`: ArenaLayout's.
uint64_t stageScratchBytes(const AirKey &air);

// Instance::computeHintColumns and computeImPols of stage `stage` (1 … nStages) on the device, for
// the instance that holds `air`'s arena (GpuKey::Lease) with the stages before committed (and, of
// stage 1, the witness columns in the arena, as InstanceGpu leaves them), and `scalars` the scalars
// the bytecode reads (its columns are not read): the fixed columns the stage reads on H
// (StageScratch::fixed), then each hint of the stage in the order of AirKey::stdHints,
//   - its numerator and denominator, each a column at its offset, a number, or an expression the
//     interpreter evaluates on H (the numerator into the hint's column, the denominator into the
//     scratch),
//   - the quotient into its column (pilfflonk_gpu_hint_quotient), whose first row with a zero
//     denominator, if any, throws zeroDenominatorError before the next hint, as the CPU,
//   - and its running product (gprod_col) or sum (gsum_col);
// then the im pols, in the order of imPolOrder, with the interpreter. Every column of the stage goes
// to its place in the arena (ArenaLayout::evaluations), and none to the host. Logs the CPU's timers,
// PILFFLONK_HINT_COLUMNS_<stage> and PILFFLONK_IM_POLS_<stage>, and returns once the device is done.
// Throws what the CPU's would, with its messages.
void computeStageColumns(const GpuAirKey &air, uint64_t stage, const ProverValues &scalars);

// The N values on H of the column of stage `stage` (1 … nStages) at stagePos, where the arena has it
// (a witness column once the instance is made, the others once computeStageColumns has run), into
// `out` (host memory), for the holder of the arena before Q's phase reuses their memory.
void stageColumnToHost(const GpuAirKey &air, uint64_t stage, uint64_t stagePos, FrElement *out);

} // namespace PilFflonk

#endif
