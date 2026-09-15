#ifndef MULTIPLICITY_JOB_HPP
#define MULTIPLICITY_JOB_HPP

#include <cstdint>
#include "multiplicity_linear.hpp"

// One range-check lookup, resolved. Backend-neutral on purpose: the GPU kernel and the CPU scatter
// evaluate the SAME jobs, so a lookup cannot be counted differently depending on where it runs.
// Addresses are resolved on the host at registration -- the prover's buffers are flat column-major
// (goldilocks_trace_layout.cuh), so a column reference is one element offset plus the row.
//
// Row-varying sources index by `elemBase + row`; the uniform ones (publics and the three value
// pools) index by `elemBase` alone, but are only known once the proof starts.
enum MulSrc : uint32_t {
    MUL_SRC_CONST = 0, MUL_SRC_TRACE = 1, MUL_SRC_AUX = 2,
    MUL_SRC_PUBLIC = 3, MUL_SRC_AIRVALUE = 4, MUL_SRC_PROOFVALUE = 5, MUL_SRC_AIRGROUPVALUE = 6,
    // A custom commit's fixed section. Row-addressed like the trace, in its own buffer.
    MUL_SRC_CUSTOM = 7,
    MUL_SRC_N = 8
};
#define MUL_SRC_IS_UNIFORM(s) ((s) >= MUL_SRC_PUBLIC)

// 24 bytes, not 32: every thread reads its job's terms, so this struct's size is L1 traffic.
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


struct MulJobDev {
    MulFormDev value;      // the range-checked expression
    MulFormDev sel;        // the multiplicity to add; skipped when selConstOne
    MulFormDev bus;        // the bus id, when the opid is chosen per row
    uint64_t   accBase;    // first counter of this table inside the air's accumulator
    uint64_t   nTableRows;
    uint64_t   biasFE;     // -min, in the field
    uint64_t   rows;       // 1 for a degree-0 term, else the air's row count
    uint64_t   hostAirId;  // the air whose virtual table holds this lookup's counters
    uint32_t   tableId;
    uint32_t   selConstOne;
    uint32_t   hasBus;
    uint32_t   pad;
};

// `bases` is indexed by MulSrc; a null entry is fine as long as no term names it.

// Rows one block covers. Lives here rather than with the kernel because the plan sizes its grid
// from it.
#define MUL_SCATTER_BLOCK 256

#endif
