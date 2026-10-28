// The prover mode of the interpreter on the device (pilfflonk_expressions_kernels.hpp), in the GPU
// library only: the bytecode's ops over BN128 scalars in sppark's Montgomery arithmetic
// (BN128GPUScalarField), which keeps every result fully reduced, as ffiasm does, so that the same
// field operation gives the same bytes on either; and the zerofiers of a part of the coset.
#include "pilfflonk_cuda.cuh"
#include "pilfflonk_expressions_kernels.hpp"

namespace {

using PilFflonk::ExpressionLaunch;

static_assert(sizeof(Element) == PilFflonk::OPERAND_BYTES, "an operand is one element");

// The zerofiers' blocks are of THREADS threads (pilfflonk_cuda.cuh); the code's, of EXPRESSION_ROWS.

__device__ __forceinline__ Element source(const ExpressionLaunch &p, uint32_t type, uint32_t arg1, uint32_t arg2,
                                          uint64_t i, const Element *tmp) {
    if (type == p.tmpType) {
        return tmp[arg1 * blockDim.x + threadIdx.x];
    }
    return *static_cast<const Element *>(PilFflonk::operandAddress(p.tables, type, arg1, arg2, i));
}

// Each block evaluates the code on EXPRESSION_ROWS points at a time, a point per thread, each
// thread with its temporaries apart: tmp[t·blockDim + thread]. An op is computed into a register
// before its temporary is written, as on the CPU, since it may be one of its sources. Its args are
// the same for every thread, and come through the read-only cache, once per warp. The threads past
// the domain's end, where it has fewer points than a block, compute on points the masks keep in it,
// and write nothing.
__global__ void calculateExpression(const ExpressionLaunch p) {
    extern __shared__ __align__(16) uint8_t shared[];
    Element *tmp = p.temporaries != nullptr
                       ? static_cast<Element *>(p.temporaries) + uint64_t(blockIdx.x) * p.nTemp * blockDim.x
                       : reinterpret_cast<Element *>(shared);
    const uint4 *code = reinterpret_cast<const uint4 *>(p.args);
    for (uint64_t row = uint64_t(blockIdx.x) * blockDim.x; row < p.size; row += stride()) {
        const uint64_t i = row + threadIdx.x;
        Element value;
        for (uint32_t k = 0; k < p.nOps; ++k) {
            // opType dest aType aArg1, then aArg2 bType bArg1 bArg2.
            const uint4 head = __ldg(code + 2 * k), tail = __ldg(code + 2 * k + 1);
            const Element a = source(p, head.z, head.w, tail.x, i, tmp);
            const Element b = source(p, tail.y, tail.z, tail.w, i, tmp);
            switch (head.x) {
            case PilFflonk::EXPRESSION_ADD:
                value = Fr::add(a, b);
                break;
            case PilFflonk::EXPRESSION_SUB:
                value = Fr::sub(a, b);
                break;
            case PilFflonk::EXPRESSION_MUL:
                value = Fr::mul(a, b);
                break;
            default: // EXPRESSION_SUB_SWAP: the reader refuses any other
                value = Fr::sub(b, a);
                break;
            }
            tmp[head.y * blockDim.x + threadIdx.x] = value;
        }
        if (i < p.size) {
            static_cast<Element *>(p.dest)[i * p.stride] = value;
        }
    }
}

__device__ __forceinline__ Element pointOfPart(const Element &shift, const Element *bases, const Element *powers,
                                               uint64_t i) {
    return Fr::mul(shift, Fr::mul(bases[i >> 8], powers[i & 255]));
}

// Every thread of a block runs every round of the loop, past the end too, on point 0: sppark's
// reciprocal pairs the lanes of a warp.
__global__ void oneRowZerofier(Element *out, uint64_t size, const Element shift, const Element *bases,
                               const Element *powers, const Element root, const Element *zh, uint64_t zhMask) {
    for (uint64_t first = uint64_t(blockIdx.x) * blockDim.x; first < size; first += stride()) {
        const uint64_t i = first + threadIdx.x;
        const bool inside = i < size;
        const uint64_t point = inside ? i : 0;
        const Element inverse = Fr::reciprocal(Fr::sub(pointOfPart(shift, bases, powers, point), root));
        if (inside) {
            out[i] = Fr::mul(zh[i & zhMask], inverse);
        }
    }
}

__global__ void frameZerofier(Element *out, uint64_t size, const Element shift, const Element *bases,
                              const Element *powers, const Element *roots, uint64_t nRoots) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size; i += stride()) {
        const Element x = pointOfPart(shift, bases, powers, i);
        Element product = Fr::one();
        for (uint64_t j = 0; j < nRoots; ++j) {
            product = Fr::mul(product, Fr::sub(x, roots[j]));
        }
        out[i] = product;
    }
}

} // namespace

extern "C" void pilfflonk_gpu_calculate_expression(const ExpressionLaunch *launch) {
    if (launch->size == 0) {
        return;
    }
    const size_t shared =
        launch->temporaries != nullptr ? 0 : size_t(launch->nTemp) * PilFflonk::EXPRESSION_ROWS * sizeof(Element);
    calculateExpression<<<launch->blocks, PilFflonk::EXPRESSION_ROWS, shared>>>(*launch);
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void pilfflonk_gpu_one_row_zerofier(void *out, uint64_t size, const void *shift, const void *bases,
                                               const void *powers, const void *root, const void *zh,
                                               uint64_t zhMask) {
    if (size == 0) {
        return;
    }
    oneRowZerofier<<<blocksFor(size), THREADS>>>(
        static_cast<Element *>(out), size, *static_cast<const Element *>(shift), static_cast<const Element *>(bases),
        static_cast<const Element *>(powers), *static_cast<const Element *>(root), static_cast<const Element *>(zh),
        zhMask);
    CHECKCUDAERR(cudaGetLastError());
}

extern "C" void pilfflonk_gpu_frame_zerofier(void *out, uint64_t size, const void *shift, const void *bases,
                                             const void *powers, const void *roots, uint64_t nRoots) {
    if (size == 0) {
        return;
    }
    frameZerofier<<<blocksFor(size), THREADS>>>(
        static_cast<Element *>(out), size, *static_cast<const Element *>(shift), static_cast<const Element *>(bases),
        static_cast<const Element *>(powers), static_cast<const Element *>(roots), nRoots);
    CHECKCUDAERR(cudaGetLastError());
}
