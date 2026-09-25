#ifndef MULTIPLICITY_JOB_HPP
#define MULTIPLICITY_JOB_HPP

#include <cstdint>
#include "multiplicity_linear.hpp"
#include "multiplicity_decoders.hpp"

// One range-check lookup, resolved. Backend-neutral: the GPU kernel and the CPU scatter evaluate
// the SAME jobs. Addresses are resolved at registration against flat column-major buffers
// (goldilocks_trace_layout.cuh). Row-varying sources index by `elemBase + row`, uniform ones
// (publics, value pools) by `elemBase`, known only once the proof starts.
enum MulSrc : uint32_t {
    MUL_SRC_CONST = 0, MUL_SRC_TRACE = 1, MUL_SRC_AUX = 2,
    MUL_SRC_PUBLIC = 3, MUL_SRC_AIRVALUE = 4, MUL_SRC_PROOFVALUE = 5, MUL_SRC_AIRGROUPVALUE = 6,
    // A custom commit's fixed section. Row-addressed like the trace, in its own buffer.
    MUL_SRC_CUSTOM = 7,
    MUL_SRC_N = 8
};
#define MUL_SRC_IS_UNIFORM(s) ((s) >= MUL_SRC_PUBLIC)

// 32 bytes: every thread reads its job's terms, so this struct's size is L1 traffic.
// The column is kept apart from the section offset because the two backends store a section
// DIFFERENTLY: the GPU is flat column-major (`col * nRows + row`, goldilocks_trace_layout.cuh) and
// the CPU is row-major (`row * nCols + col`, expressions_pack.hpp). Baking either into one offset
// silently reads the wrong cells on the other backend.
struct MulTermDev {
    uint64_t sectionOffset;  // start of the section inside its buffer; also the index of a uniform
    uint64_t coef;           // canonical field element; 1 in the overwhelming majority
    int32_t  rowStride;      // nextStridesExps[rowOffsetIndex], for `'`-shifted references
    uint32_t src;            // MulSrc
    uint32_t col;            // column within the section
    uint32_t nCols;          // columns in the section, for the row-major backend
};
static_assert(sizeof(MulTermDev) == 32, "MulTermDev size drifted -- update the L1-traffic comment above");

// `sum(t) + konst`, or that multiplied by `sum(t2) + konst2` when hasProduct -- the shape a
// selector built from two flags compiles to.
struct MulFormDev {
    uint64_t   konst;
    uint32_t   n;
    uint32_t   hasProduct;
    uint64_t   konst2;
    uint32_t   n2;
    uint32_t   pad;
    MulTermDev t[MUL_MAX_TERMS];
    MulTermDev t2[MUL_MAX_TERMS];
};


// A sum of products; `p[0]` alone is the plain linear / single-product case.
struct MulPolyDev {
    uint32_t   nProd;
    uint32_t   pad;
    MulFormDev p[MUL_MAX_PRODUCTS];
};

struct MulJobDev {
    // Hot fields first: every thread reads them for every job on every row, so keep them in one or
    // two cache lines.
    uint64_t   rows;       // 1 for a degree-0 term, else the air's row count
    uint64_t   accBase;    // first counter of this table inside the air's accumulator
    uint64_t   nTableRows;
    uint64_t   biasFE;     // -min, in the field
    uint64_t   mapSlots;
    const uint64_t* mapKV;
    // Digit recoding of the first tuple column; see MulDecoder. Needs one column, not the tuple.
    const uint64_t* digitTab;
    uint32_t   digitCols;
    // Indexed-base rule (table 125's shape). `hasIndexedBase` is a plain bool, copied by value at
    // plan-build time, so it can never dangle. `dec` starts null and is filled in only later, at
    // device-plan-build time (mulPlanDevice), with a stable device-resident pointer -- it must NOT
    // be cached from mulDecoders() at plan-build time, because that vector keeps growing via
    // push_back from other airs' registration calls and reallocates without warning; the CPU
    // scatter instead resolves its own copy via mulDecoderFor(tableId) once registration is known
    // to be complete (at proving time), not through this field.
    uint32_t   hasIndexedBase;
    const MulDecoder* dec;
    uint64_t   hostAirId;  // the air whose virtual table holds this lookup's counters
    uint32_t   nKey;
    uint32_t   tableId;
    uint32_t   selConstOne;
    uint32_t   hasBus;
    // Compiled programs, for the fields no closed form can express. A non-zero length means the
    // program is authoritative and the MulPolyDev beside it is unused; the offsets index the air's
    // single instruction buffer. See multiplicity_program.hpp.
    uint32_t   valProgOff, valProgLen;
    uint32_t   selProgOff, selProgLen;
    uint32_t   busProgOff, busProgLen;
    uint32_t   keyProgOff[MUL_MAX_TUPLE], keyProgLen[MUL_MAX_TUPLE];

    // Cold: read only once a job survives the checks above.
    MulPolyDev value;      // the range-checked expression
    MulPolyDev sel;        // the multiplicity to add; skipped when selConstOne
    MulPolyDev bus;        // the bus id, when the opid is chosen per row
    // An exact map looks the lookup's own tuple up verbatim, so the job carries a form per key
    // column rather than the single folded value the affine path uses.
    MulPolyDev key[MUL_MAX_TUPLE];
};

// `bases` is indexed by MulSrc; a null entry is fine as long as no term names it.

// Rows one block covers. Here, not with the kernel, because the plan sizes its grid from it.
#define MUL_SCATTER_BLOCK 256

#endif
