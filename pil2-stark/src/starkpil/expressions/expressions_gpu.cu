#include "expressions_gpu.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "goldilocks_tooling.cuh"
#include "goldilocks_cubic_extension.cuh"
#include "expressions_codegen.cuh"

extern __shared__ Goldilocks::Element scratchpad[];

ExpressionsGPU::ExpressionsGPU(SetupCtx &setupCtx, uint32_t nRowsPack, uint32_t nBlocks) : ExpressionsCtx(setupCtx), nRowsPack(nRowsPack), nBlocks(nBlocks)
{
    
    uint32_t ns = 1 + setupCtx.starkInfo.nStages + 1;
    uint32_t nCustoms = setupCtx.starkInfo.customCommits.size();
    uint32_t nOpenings = setupCtx.starkInfo.openingPoints.size();
    uint32_t nStages_ = setupCtx.starkInfo.nStages;
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

    bufferCommitSize = 1 + nStages_ + 3 + nCustoms;

    h_deviceArgs.N = N;
    h_deviceArgs.NExtended = NExtended;
    h_deviceArgs.nBlocks = nBlocks;
    h_deviceArgs.nStages = nStages_;
    h_deviceArgs.nCustomCommits = nCustoms;
    h_deviceArgs.bufferCommitSize = bufferCommitSize;
    
    h_deviceArgs.zi_offset = setupCtx.starkInfo.mapOffsets[std::make_pair("zi", true)];

    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.mapOffsets, ns * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.mapOffsetsExtended, ns * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.mapOffsetsCustomFixed, nCustoms * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.mapOffsetsCustomFixedExtended, nCustoms * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.nextStrides, nOpenings * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.nextStridesExtended, nOpenings * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.mapSectionsN, ns * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.mapSectionsNCustomFixed, nCustoms * sizeof(uint64_t)));

    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.mapOffsets, mapOffsets, ns * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.mapOffsetsExtended, mapOffsetsExtended, ns * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.mapOffsetsCustomFixed, mapOffsetsCustomFixed, nCustoms * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.mapOffsetsCustomFixedExtended, mapOffsetsCustomFixedExtended, nCustoms * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.nextStrides, nextStrides, nOpenings * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.nextStridesExtended, nextStridesExtended, nOpenings * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.mapSectionsN, mapSectionsN, ns * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.mapSectionsNCustomFixed, mapSectionsNCustomFixed, nCustoms * sizeof(uint64_t), cudaMemcpyHostToDevice));


    ParserArgs parserArgs = setupCtx.expressionsBin.expressionsBinArgsExpressions;
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.numbers, parserArgs.nNumbers * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.numbers, (Goldilocks::Element *)parserArgs.numbers, parserArgs.nNumbers * sizeof(Goldilocks::Element),cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.ops, setupCtx.expressionsBin.nOpsTotal * sizeof(uint8_t)));   
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.args, setupCtx.expressionsBin.nArgsTotal * sizeof(uint16_t))); 
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.ops, parserArgs.ops, setupCtx.expressionsBin.nOpsTotal * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.args, parserArgs.args, setupCtx.expressionsBin.nArgsTotal * sizeof(uint16_t), cudaMemcpyHostToDevice));

    ParserArgs parserArgsConstraints = setupCtx.expressionsBin.expressionsBinArgsConstraints;
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.numbersConstraints, parserArgsConstraints.nNumbers * sizeof(Goldilocks::Element)));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.numbersConstraints, (Goldilocks::Element *)parserArgsConstraints.numbers, parserArgsConstraints.nNumbers * sizeof(Goldilocks::Element),cudaMemcpyHostToDevice));

    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.opsConstraints, setupCtx.expressionsBin.nOpsDebug * sizeof(uint8_t)));   
    CHECKCUDAERR(cudaMalloc(&h_deviceArgs.argsConstraints, setupCtx.expressionsBin.nArgsDebug * sizeof(uint16_t))); 
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.opsConstraints, parserArgsConstraints.ops, setupCtx.expressionsBin.nOpsDebug * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(h_deviceArgs.argsConstraints, parserArgsConstraints.args, setupCtx.expressionsBin.nArgsDebug * sizeof(uint16_t), cudaMemcpyHostToDevice));


    CHECKCUDAERR(cudaMalloc(&d_deviceArgs, sizeof(DeviceArguments)));
    CHECKCUDAERR(cudaMemcpy(d_deviceArgs, &h_deviceArgs, sizeof(DeviceArguments), cudaMemcpyHostToDevice));

    ExpsKernel ek = expsOpenForAir(setupCtx);
    expsLib = ek.lib;
    expsFn = (void *)ek.fn;
    expsMinScratch = ek.minScratch;
};

ExpressionsGPU::~ExpressionsGPU()
{
    CHECKCUDAERR(cudaFree(h_deviceArgs.mapOffsets));
    CHECKCUDAERR(cudaFree(h_deviceArgs.mapOffsetsExtended));
    CHECKCUDAERR(cudaFree(h_deviceArgs.nextStrides));
    CHECKCUDAERR(cudaFree(h_deviceArgs.nextStridesExtended));
    CHECKCUDAERR(cudaFree(h_deviceArgs.mapOffsetsCustomFixed));
    CHECKCUDAERR(cudaFree(h_deviceArgs.mapOffsetsCustomFixedExtended));
    CHECKCUDAERR(cudaFree(h_deviceArgs.mapSectionsN));
    CHECKCUDAERR(cudaFree(h_deviceArgs.mapSectionsNCustomFixed));
    CHECKCUDAERR(cudaFree(h_deviceArgs.numbers));
    CHECKCUDAERR(cudaFree(h_deviceArgs.ops));
    CHECKCUDAERR(cudaFree(h_deviceArgs.args));
    CHECKCUDAERR(cudaFree(h_deviceArgs.numbersConstraints));
    CHECKCUDAERR(cudaFree(h_deviceArgs.opsConstraints));
    CHECKCUDAERR(cudaFree(h_deviceArgs.argsConstraints));

    CHECKCUDAERR(cudaFree(d_deviceArgs));

    expsClose(expsLib);
}

// Stage one launch's params/args into pinned slot `countId` and enqueue the H2D
// copies. Slot strides are in BYTES: pinned_exps_* are Element* only by type, so
// Element* arithmetic would stride 8x too far.
static void stageExpsSlot(Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args,
                          uint64_t countId, const DestParamsGPU *h_dest_params, const ExpsArguments &h_expsArgs,
                          DestParamsGPU *d_destParams, ExpsArguments *d_expsArgs, cudaStream_t stream)
{
    if (countId >= PINNED_EXPS_SLOTS) {
        zklog.error("ExpressionsGPU: expression launch count " + std::to_string(countId) +
                    " exceeds pinned slot capacity " + std::to_string(PINNED_EXPS_SLOTS));
        exitProcess();
    }
    // Each slot spans 2 DestParamsGPU (stride 2*sizeof) and d_destParams is sized for 2;
    // more params would overrun both the pinned slot and the device buffer.
    if (h_expsArgs.dest_nParams > 2) {
        zklog.error("ExpressionsGPU: dest_nParams " + std::to_string(h_expsArgs.dest_nParams) +
                    " exceeds slot capacity 2");
        exitProcess();
    }
    uint8_t *paramsSlot = (uint8_t *)pinned_exps_params + countId * 2 * sizeof(DestParamsGPU);
    memcpy(paramsSlot, h_dest_params, h_expsArgs.dest_nParams * sizeof(DestParamsGPU));
    CHECKCUDAERR(cudaMemcpyAsync(d_destParams, paramsSlot, h_expsArgs.dest_nParams * sizeof(DestParamsGPU), cudaMemcpyHostToDevice, stream));

    uint8_t *argsSlot = (uint8_t *)pinned_exps_args + countId * sizeof(ExpsArguments);
    memcpy(argsSlot, &h_expsArgs, sizeof(ExpsArguments));
    CHECKCUDAERR(cudaMemcpyAsync(d_expsArgs, argsSlot, sizeof(ExpsArguments), cudaMemcpyHostToDevice, stream));
}

void ExpressionsGPU::calculateExpressions_gpu(StepsParams *d_params, Dest dest, uint64_t domainSize, bool domainExtended, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU &timer, cudaStream_t stream, bool constraints)
{
    ExpsArguments h_expsArgs;

    uint32_t nrowsPack = std::min(static_cast<uint32_t>(nRowsPack), static_cast<uint32_t>(domainSize));
    h_expsArgs.nRowsPack = nrowsPack;
    
    h_expsArgs.mapOffsetsExps = domainExtended ? h_deviceArgs.mapOffsetsExtended : h_deviceArgs.mapOffsets;            
    h_expsArgs.mapOffsetsCustomExps = domainExtended ? h_deviceArgs.mapOffsetsCustomFixedExtended : h_deviceArgs.mapOffsetsCustomFixed;
    h_expsArgs.nextStridesExps = domainExtended ? h_deviceArgs.nextStridesExtended : h_deviceArgs.nextStrides;

    h_expsArgs.k_min = domainExtended
                             ? uint64_t((minRowExtended + h_expsArgs.nRowsPack - 1) / h_expsArgs.nRowsPack) * h_expsArgs.nRowsPack
                             : uint64_t((minRow + h_expsArgs.nRowsPack - 1) / h_expsArgs.nRowsPack) * h_expsArgs.nRowsPack;
    h_expsArgs.k_max = domainExtended
                             ? uint64_t(maxRowExtended / h_expsArgs.nRowsPack) * h_expsArgs.nRowsPack
                             : uint64_t(maxRow / h_expsArgs.nRowsPack) * h_expsArgs.nRowsPack;

    h_expsArgs.maxTemp1Size = 0;
    h_expsArgs.maxTemp3Size = 0;

    h_expsArgs.offsetTmp1 = setupCtx.starkInfo.mapOffsets[std::make_pair("tmp1", false)];
    h_expsArgs.offsetTmp3 = setupCtx.starkInfo.mapOffsets[std::make_pair("tmp3", false)];
    h_expsArgs.offsetDestVals = setupCtx.starkInfo.mapOffsets[std::make_pair("destVals", false)];

    for (uint64_t k = 0; k < dest.params.size(); ++k)
    {
        ParserParams &parserParams = constraints 
            ? setupCtx.expressionsBin.constraintsInfoDebug[dest.params[k].expId]
            : setupCtx.expressionsBin.expressionsInfo[dest.params[k].expId];
        if (parserParams.nTemp1*h_expsArgs.nRowsPack > h_expsArgs.maxTemp1Size) {
            h_expsArgs.maxTemp1Size = parserParams.nTemp1*h_expsArgs.nRowsPack;
        }
        if (parserParams.nTemp3*h_expsArgs.nRowsPack*FIELD_EXTENSION > h_expsArgs.maxTemp3Size) {
            h_expsArgs.maxTemp3Size = parserParams.nTemp3*h_expsArgs.nRowsPack*FIELD_EXTENSION;
        }
    }

    h_expsArgs.domainSize = domainSize;
    h_expsArgs.domainExtended = domainExtended;

    h_expsArgs.dest_gpu = dest.dest_gpu;
    h_expsArgs.dest_domainSize = dest.domainSize;
    h_expsArgs.dest_stageCols = dest.stageCols;
    h_expsArgs.dest_stagePos = dest.stagePos;
    h_expsArgs.dest_dim = dest.dim;
    h_expsArgs.dest_expr = dest.expr;
    h_expsArgs.dest_nParams = dest.params.size();

    assert(dest.params.size() == 1 || dest.params.size() == 2);

    DestParamsGPU* h_dest_params = new DestParamsGPU[h_expsArgs.dest_nParams];
    for (uint64_t j = 0; j < h_expsArgs.dest_nParams; ++j){

        ParserParams &parserParams = constraints 
            ? setupCtx.expressionsBin.constraintsInfoDebug[dest.params[j].expId]
            : setupCtx.expressionsBin.expressionsInfo[dest.params[j].expId];
        h_dest_params[j].dim = dest.params[j].dim;
        h_dest_params[j].stage = dest.params[j].stage;
        h_dest_params[j].stagePos = dest.params[j].stagePos;
        h_dest_params[j].polsMapId = dest.params[j].polsMapId;
        h_dest_params[j].rowOffsetIndex = dest.params[j].rowOffsetIndex;
        h_dest_params[j].inverse = dest.params[j].inverse;
        h_dest_params[j].op = dest.params[j].op;
        h_dest_params[j].value = dest.params[j].value;
        h_dest_params[j].nOps = parserParams.nOps;
        h_dest_params[j].opsOffset = parserParams.opsOffset;
        h_dest_params[j].nArgs = parserParams.nArgs;
        h_dest_params[j].argsOffset =parserParams.argsOffset;
    }

    stageExpsSlot(pinned_exps_params, pinned_exps_args, countId, h_dest_params, h_expsArgs, d_destParams, d_expsArgs, stream);
    delete[] h_dest_params;

    uint32_t nblocks_ = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(nBlocks),(domainSize + nrowsPack - 1) / nrowsPack));
    uint32_t nthreads_ = nblocks_ == 1 ? domainSize : nrowsPack;
    dim3 nBlocks_ =  nblocks_;
    dim3 nThreads_ = nthreads_;

    assert(bufferCommitSize  + 9  < 32);
    size_t ptrMem = 32 * sizeof(Goldilocks::Element);
    size_t tmpMem = (h_expsArgs.maxTemp1Size + h_expsArgs.maxTemp3Size) * sizeof(Goldilocks::Element);
    bool useTmpInShared = tmpMem <= 40960 && tmpMem > 0;
    size_t sharedMem = useTmpInShared ? (ptrMem + tmpMem) : ptrMem;

    TimerStartCategoryGPU(timer, EXPRESSIONS);
    computeExpressions_<<<nBlocks_, nThreads_, sharedMem, stream>>>(d_params, d_deviceArgs, d_expsArgs, d_destParams, constraints);
    TimerStopCategoryGPU(timer, EXPRESSIONS);
}

void ExpressionsGPU::calculateExpressionsQ_gpu(StepsParams *d_params, Dest dest, uint64_t domainSize, bool domainExtended, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *pinned_exps_params, Goldilocks::Element *pinned_exps_args, uint64_t& countId, TimerGPU &timer, cudaStream_t stream)
{
    ExpsArguments h_expsArgs;

    uint32_t nrowsPack = std::min(static_cast<uint32_t>(nRowsPack), static_cast<uint32_t>(domainSize));
    h_expsArgs.nRowsPack = nrowsPack;
    
    h_expsArgs.mapOffsetsExps = domainExtended ? h_deviceArgs.mapOffsetsExtended : h_deviceArgs.mapOffsets;            
    h_expsArgs.mapOffsetsCustomExps = domainExtended ? h_deviceArgs.mapOffsetsCustomFixedExtended : h_deviceArgs.mapOffsetsCustomFixed;
    h_expsArgs.nextStridesExps = domainExtended ? h_deviceArgs.nextStridesExtended : h_deviceArgs.nextStrides;

    h_expsArgs.k_min = domainExtended
                             ? uint64_t((minRowExtended + h_expsArgs.nRowsPack - 1) / h_expsArgs.nRowsPack) * h_expsArgs.nRowsPack
                             : uint64_t((minRow + h_expsArgs.nRowsPack - 1) / h_expsArgs.nRowsPack) * h_expsArgs.nRowsPack;
    h_expsArgs.k_max = domainExtended
                             ? uint64_t(maxRowExtended / h_expsArgs.nRowsPack) * h_expsArgs.nRowsPack
                             : uint64_t(maxRow / h_expsArgs.nRowsPack) * h_expsArgs.nRowsPack;

    h_expsArgs.maxTemp1Size = 0;
    h_expsArgs.maxTemp3Size = 0;

    h_expsArgs.offsetTmp1 = setupCtx.starkInfo.mapOffsets[std::make_pair("tmp1", false)];
    h_expsArgs.offsetTmp3 = setupCtx.starkInfo.mapOffsets[std::make_pair("tmp3", false)];
    h_expsArgs.offsetDestVals = setupCtx.starkInfo.mapOffsets[std::make_pair("destVals", false)];

    for (uint64_t k = 0; k < dest.params.size(); ++k)
    {
        ParserParams &parserParams = setupCtx.expressionsBin.expressionsInfo[dest.params[k].expId];
        if (parserParams.nTemp1*h_expsArgs.nRowsPack > h_expsArgs.maxTemp1Size) {
            h_expsArgs.maxTemp1Size = parserParams.nTemp1*h_expsArgs.nRowsPack;
        }
        if (parserParams.nTemp3*h_expsArgs.nRowsPack*FIELD_EXTENSION > h_expsArgs.maxTemp3Size) {
            h_expsArgs.maxTemp3Size = parserParams.nTemp3*h_expsArgs.nRowsPack*FIELD_EXTENSION;
        }
    }

    h_expsArgs.domainSize = domainSize;
    h_expsArgs.domainExtended = domainExtended;

    h_expsArgs.dest_gpu = dest.dest_gpu;
    h_expsArgs.dest_domainSize = dest.domainSize;
    h_expsArgs.dest_stageCols = dest.stageCols;
    h_expsArgs.dest_stagePos = dest.stagePos;
    h_expsArgs.dest_dim = dest.dim;
    h_expsArgs.dest_expr = dest.expr;
    h_expsArgs.dest_nParams = dest.params.size();

    // The pinned slot and d_destParams hold at most 2 entries.
    assert(dest.params.size() == 1 || dest.params.size() == 2);

    DestParamsGPU* h_dest_params = new DestParamsGPU[h_expsArgs.dest_nParams];
    for (uint64_t j = 0; j < h_expsArgs.dest_nParams; ++j){

        ParserParams &parserParams = setupCtx.expressionsBin.expressionsInfo[dest.params[j].expId];
        h_dest_params[j].dim = dest.params[j].dim;
        h_dest_params[j].stage = dest.params[j].stage;
        h_dest_params[j].stagePos = dest.params[j].stagePos;
        h_dest_params[j].polsMapId = dest.params[j].polsMapId;
        h_dest_params[j].rowOffsetIndex = dest.params[j].rowOffsetIndex;
        h_dest_params[j].inverse = dest.params[j].inverse;
        h_dest_params[j].op = dest.params[j].op;
        h_dest_params[j].value = dest.params[j].value;
        h_dest_params[j].nOps = parserParams.nOps;
        h_dest_params[j].opsOffset = parserParams.opsOffset;
        h_dest_params[j].nArgs = parserParams.nArgs;
        h_dest_params[j].argsOffset =parserParams.argsOffset;
    }

    stageExpsSlot(pinned_exps_params, pinned_exps_args, countId, h_dest_params, h_expsArgs, d_destParams, d_expsArgs, stream);
    delete[] h_dest_params;

    uint32_t nblocks_ = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(nBlocks),(domainSize + nrowsPack - 1) / nrowsPack));
    uint32_t nthreads_ = nblocks_ == 1 ? domainSize : nrowsPack;
    dim3 nBlocks_ =  nblocks_;
    dim3 nThreads_ = nthreads_;
    
    assert(bufferCommitSize  + 9  < 32);
    // Include temp buffers in dynamic shared memory if they fit in 40KB budget
    size_t ptrMem = 32 * sizeof(Goldilocks::Element);
    size_t tmpMem = (h_expsArgs.maxTemp1Size + h_expsArgs.maxTemp3Size) * sizeof(Goldilocks::Element);
    bool useTmpInShared = tmpMem <= 40960 && tmpMem > 0;
    size_t sharedMem = useTmpInShared ? (ptrMem + tmpMem) : ptrMem;

    TimerStartCategoryGPU(timer, EXPRESSIONS);
    // If this AIR has a generated Q kernel launch it instead of the bytecode interpreter.
    bool computed = false;
    if (dest.dest_gpu != nullptr && expsFn != nullptr) {
        computed = tryLaunchExps(setupCtx, (ExpsKernelFn)expsFn, expsMinScratch, d_params, (gl64_t*)dest.dest_gpu, stream);
    }
    if (!computed) {
        computeExpression_<<<nBlocks_, nThreads_, sharedMem, stream>>>(d_params, d_deviceArgs, d_expsArgs, d_destParams);
    }
    CHECKCUDAERR(cudaGetLastError());
    TimerStopCategoryGPU(timer, EXPRESSIONS);
}

__device__ __forceinline__ void load__(
    const DeviceArguments* __restrict__ dArgs,
    const ExpsArguments* __restrict__ dExpsArgs,
    const StepsParams* __restrict__ dParams,
    Goldilocks::Element** __restrict__ exprParams,
    const uint16_t type,
    const uint16_t argIdx,
    const uint16_t argOffset,
    const uint64_t row,
    const uint64_t dim,
    const bool isCyclic,
    gl64_t*& out0,
    gl64_t*& out1,
    gl64_t*& out2

) {
    const uint32_t r = row + threadIdx.x;
    const uint64_t base = dArgs->bufferCommitSize;
    const uint64_t domainSize = dExpsArgs->domainSize;

    // Fast-path: temporary/intermediate buffers
    if (type == base || type == base + 1) {
        //return &exprParams[type][argIdx * blockDim.x];
        if(dim == 1 ){
            out0 = (gl64_t*)&exprParams[type][argIdx * blockDim.x + threadIdx.x];
            out1 = nullptr;
            out2 = nullptr;
            return;
        } else {
            out0 =  (gl64_t*)&exprParams[type][argIdx * blockDim.x + threadIdx.x];
            out1 =  (gl64_t*)&exprParams[type][argIdx * blockDim.x + threadIdx.x + blockDim.x];
            out2 =  (gl64_t*)&exprParams[type][argIdx * blockDim.x + threadIdx.x + 2*blockDim.x];
            return;
        }
    }

    // Fast-path: constants
    if (type >= base + 2) {
        if(dim == 1 ){
            out0 = (gl64_t*)&exprParams[type][argIdx];
            out1 = nullptr;
            out2 = nullptr;
            return;
        } else {
            out0 = (gl64_t*)&exprParams[type][argIdx];
            out1 = (gl64_t*)&exprParams[type][argIdx + 1];
            out2 = (gl64_t*)&exprParams[type][argIdx + 2];
            return;
        }
    }

    const int64_t stride = dExpsArgs->nextStridesExps[argOffset];
    const uint64_t logicalRow = isCyclic ? (r + stride) % domainSize : (r + stride);

    // Use pack256 fast path when non-cyclic, stride==0, blockDim==TILE_HEIGHT, and TILE_WIDTH==4
    const bool usePack256 = !isCyclic && stride == 0 && blockDim.x == TILE_HEIGHT && TILE_WIDTH == 4;
    const uint64_t chunkBase = row; // row is always blockDim.x-aligned

    // ConstPols
    if (type == 0) {
        const Goldilocks::Element* basePtr = dExpsArgs->domainExtended
            ? dParams->pConstPolsExtendedTreeAddress
            : dParams->pConstPolsAddress;

        const uint64_t nCols0 = dArgs->mapSectionsN[0];
        // Const sections are stored fixedLayout() (ColMajorTiled) -- match the const-tree build's layout.
        const Layout lytC = fixedLayout();
        const uint64_t pos = usePack256
            ? getBufferOffset_pack256(chunkBase, argIdx, domainSize, nCols0, lytC)
            : getBufferOffset(logicalRow, argIdx, domainSize, nCols0, lytC);
        out0 = (gl64_t*)&basePtr[pos];
        out1 = nullptr;
        out2 = nullptr;
        return;
    }

    // Trace and aux_trace (committed pols). A committed section's storage layout is resolveLayout(nBits,
    // sectionNCols) keyed on the AIR's small-domain nBits = log2(N) -- identical to what the commit/LDE
    // and Merkle used, so reads agree with writes. ColMajor (flat) puts a column's 3 extension
    // components domainSize apart; ColMajorTiled addresses each component via getBufferOffset.
    if (type >= 1 && type <= 3) {
        const uint64_t offset = dExpsArgs->mapOffsetsExps[type];
        const uint64_t nCols = dArgs->mapSectionsN[type];
        const uint64_t nBits = 63 - __clzll(dArgs->N);
        const Layout lyt = resolveLayout(nBits, nCols);

        if (type == 1 && !dExpsArgs->domainExtended) {
            const uint64_t pos = usePack256
                ? getBufferOffset_pack256(chunkBase, argIdx, domainSize, nCols, lyt)
                : getBufferOffset(logicalRow, argIdx, domainSize, nCols, lyt);
            out0 = (gl64_t*)&dParams->trace[pos];
            out1 = nullptr;
            out2 = nullptr;
            return;
        } else if (dim == 3 && lyt == Layout::ColMajor) {
            // Flat: the 3 extension components of column argIdx are domainSize apart.
            const uint64_t pos0 = usePack256
                ? getBufferOffset_pack256(chunkBase, argIdx, domainSize, nCols, lyt)
                : getBufferOffset(logicalRow, argIdx, domainSize, nCols, lyt);
            out0 = (gl64_t*)&dParams->aux_trace[offset + pos0];
            out1 = (gl64_t*)&dParams->aux_trace[offset + pos0 + domainSize];
            out2 = (gl64_t*)&dParams->aux_trace[offset + pos0 + 2 * domainSize];
            return;
        } else if (dim == 1) {
            const uint64_t pos0 = usePack256
                ? getBufferOffset_pack256(chunkBase, argIdx, domainSize, nCols, lyt)
                : getBufferOffset(logicalRow, argIdx, domainSize, nCols, lyt);
            out0 = (gl64_t*)&dParams->aux_trace[offset + pos0];
            out1 = nullptr;
            out2 = nullptr;
            return;
        } else {
            // dim==3 general (incl. ColMajorTiled): each component addressed independently.
            const uint64_t pos0 = usePack256
                    ? getBufferOffset_pack256(chunkBase, argIdx, domainSize, nCols, lyt)
                    : getBufferOffset(logicalRow, argIdx, domainSize, nCols, lyt);
            out0 = (gl64_t*)&dParams->aux_trace[offset + pos0];
            const uint64_t pos1 = usePack256
                    ? getBufferOffset_pack256(chunkBase, argIdx+1, domainSize, nCols, lyt)
                    : getBufferOffset(logicalRow, argIdx+1, domainSize, nCols, lyt);
            out1 = (gl64_t*)&dParams->aux_trace[offset + pos1];
            const uint64_t pos2 = usePack256
                    ? getBufferOffset_pack256(chunkBase, argIdx+2, domainSize, nCols, lyt)
                    : getBufferOffset(logicalRow, argIdx+2, domainSize, nCols, lyt);
            out2 = (gl64_t*)&dParams->aux_trace[offset + pos2];
            return;
        }
    }

    // Special case: zi
    if (type == 4) {
        //return &dParams->aux_trace[dArgs->zi_offset + (argIdx - 1) * domainSize + row];
        out0 = (gl64_t*)&dParams->aux_trace[dArgs->zi_offset + (argIdx - 1) * domainSize + row + threadIdx.x];
        out1 = nullptr;
        out2 = nullptr;
        return;
    }
    // Custom commits -- fixed/preprocessed section, stored fixedLayout() (ColMajorTiled).
    const uint64_t idx = type - (dArgs->nStages + 4);
    const uint64_t offset = dExpsArgs->mapOffsetsCustomExps[idx];
    const uint64_t nCols = dArgs->mapSectionsNCustomFixed[idx];
    const uint64_t pos = getBufferOffset(logicalRow, argIdx, domainSize, nCols, fixedLayout());

    out0 = (gl64_t*)&dParams->pCustomCommitsFixed[offset + pos];
    out1 = nullptr;
    out2 = nullptr;
    return;
}

__device__ __noinline__ void storePolynomial__(ExpsArguments *d_expsArgs, Goldilocks::Element *destVals, uint64_t row)
{
    // Writing into a committed section -> must match that section's storage layout (same resolveLayout
    // the reads use), so a tiled cm section round-trips. dest_domainSize is the section's row count.
    const Layout lyt = resolveLayout(63 - __clzll(d_expsArgs->dest_domainSize), d_expsArgs->dest_stageCols);
    #pragma unroll
    for (uint32_t i = 0; i < d_expsArgs->dest_dim; i++) {
        if (!d_expsArgs->dest_expr) {
            uint64_t col = d_expsArgs->dest_stagePos + i;
            uint64_t nRows = d_expsArgs->dest_domainSize;
            uint64_t nCols = d_expsArgs->dest_stageCols;
            uint64_t idx = getBufferOffset(row + threadIdx.x, col, nRows, nCols, lyt);
            d_expsArgs->dest_gpu[idx] = destVals[i * blockDim.x + threadIdx.x];
        } else {
            d_expsArgs->dest_gpu[(row + threadIdx.x) * d_expsArgs->dest_dim + i] = destVals[i * blockDim.x + threadIdx.x];
        }
    }
}

__device__ __noinline__ void multiplyPolynomials__(ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, DeviceArguments *d_deviceArgs, gl64_t *destVals, uint64_t row)
{
    if (d_expsArgs->dest_dim == 1)
    {
        gl64_gpu::op_gpu(2, &destVals[0], &destVals[0], false, &destVals[FIELD_EXTENSION * blockDim.x], false);
    }
    else
    {
        if (d_destParams[0].dim == FIELD_EXTENSION && d_destParams[1].dim == FIELD_EXTENSION)
        {
            Goldilocks3GPU::mul_gpu_no_const(&destVals[0], &destVals[0], &destVals[FIELD_EXTENSION * blockDim.x]);
        }
        else if (d_destParams[0].dim == FIELD_EXTENSION && d_destParams[1].dim == 1)
        {
            Goldilocks3GPU::mul_31_gpu_no_const(&destVals[0], &destVals[0], &destVals[FIELD_EXTENSION * blockDim.x]);
        }
        else
        {
            Goldilocks3GPU::mul_31_gpu_no_const(&destVals[FIELD_EXTENSION * blockDim.x], &destVals[FIELD_EXTENSION * blockDim.x], &destVals[0]);
            destVals[threadIdx.x] = destVals[FIELD_EXTENSION * blockDim.x + threadIdx.x];
            destVals[blockDim.x + threadIdx.x] = destVals[(FIELD_EXTENSION + 1) * blockDim.x + threadIdx.x];
            destVals[2 * blockDim.x + threadIdx.x] = destVals[(FIELD_EXTENSION + 2) * blockDim.x + threadIdx.x];
        }
    }
    storePolynomial__(d_expsArgs, (Goldilocks::Element *)destVals, row);
}

__device__ __noinline__ void getInversePolinomial__(gl64_t *polynomial, uint64_t dim)
{
    int idx = threadIdx.x;
    if (dim == 1)
    {
        polynomial[idx] = polynomial[idx].reciprocal();
    }
    else if (dim == FIELD_EXTENSION)
    {
        Goldilocks3GPU::Element aux;
        aux[0] = polynomial[idx];
        aux[1] = polynomial[blockDim.x + idx];
        aux[2] = polynomial[2 * blockDim.x + idx];
        Goldilocks3GPU::inv(aux, aux);
        polynomial[idx] = aux[0];
        polynomial[blockDim.x + idx] = aux[1];
        polynomial[2 * blockDim.x + idx] = aux[2];
    }
}

__device__ __noinline__ bool caseNoOperations__(StepsParams *d_params, DeviceArguments *d_deviceArgs, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, Goldilocks::Element *destVals, uint32_t k, uint64_t row)
{

    uint32_t r = row + threadIdx.x;

    if (d_destParams[k].op == opType::cm || d_destParams[k].op == opType::const_)
    { // roger: assumeixes k==0 en aqeusta part?
        uint64_t openingPointIndex = d_destParams[k].rowOffsetIndex;
        uint64_t stagePos = d_destParams[k].stagePos;
        int64_t o = d_expsArgs->nextStridesExps[openingPointIndex];
        uint64_t l = (r + o) % d_expsArgs->domainSize;
        uint64_t nCols = d_deviceArgs->mapSectionsN[0];
        Goldilocks::Element *slot = &destVals[k * FIELD_EXTENSION * blockDim.x];
        if (d_destParams[k].op == opType::const_)
        {
            // Const stored fixedLayout() (ColMajorTiled) -- match the const-tree build.
            uint64_t pos = getBufferOffset(l, stagePos, d_expsArgs->domainSize, nCols, fixedLayout());
            slot[threadIdx.x] = d_params->pConstPolsAddress[pos];
        }
        else
        {
            // Committed section read: match its storage layout (resolveLayout keyed on the AIR's
            // small-domain nBits = log2(N) and the section's column count).
            uint64_t offset = d_expsArgs->mapOffsetsExps[d_destParams[k].stage];
            uint64_t nCols = d_deviceArgs->mapSectionsN[d_destParams[k].stage];
            Layout lyt = resolveLayout(63 - __clzll(d_deviceArgs->N), nCols);
            if (d_destParams[k].stage == 1)
            {
                uint64_t pos = getBufferOffset(l, stagePos, d_expsArgs->domainSize, nCols, lyt);
                slot[threadIdx.x] = d_params->trace[pos];
            }
            else
            {
                for (uint64_t d = 0; d < d_destParams[k].dim; ++d)
                {
                    uint64_t pos = getBufferOffset(l, stagePos + d, d_expsArgs->domainSize, nCols, lyt);
                    slot[threadIdx.x + d * blockDim.x] = d_params->aux_trace[offset + pos];
                }
            }
        }

        if (d_destParams[k].inverse)
        {
            getInversePolinomial__((gl64_t*) &destVals[k * FIELD_EXTENSION * blockDim.x], d_destParams[k].dim);
        }
        return true;
    }
    else if (d_destParams[k].op == opType::number)
    {
        destVals[k * FIELD_EXTENSION * blockDim.x + threadIdx.x].fe = d_destParams[k].value;
        return true;
    }
    else if (d_destParams[k].op == opType::airvalue)
    {
        if(d_destParams[k].dim == 1) {
            destVals[k * FIELD_EXTENSION * blockDim.x + threadIdx.x] = d_params->airValues[d_destParams[k].polsMapId];
        } else {
            destVals[k * FIELD_EXTENSION * blockDim.x + threadIdx.x] = d_params->airValues[d_destParams[k].polsMapId];
            destVals[k * FIELD_EXTENSION * blockDim.x + threadIdx.x + blockDim.x] = d_params->airValues[d_destParams[k].polsMapId + 1];
            destVals[k * FIELD_EXTENSION * blockDim.x + threadIdx.x + 2 * blockDim.x] = d_params->airValues[d_destParams[k].polsMapId + 2];
        }
        return true;
    }
    return false;
}

__device__ __forceinline__ void op_gpu_p2(uint64_t op, gl64_t *C, const gl64_t *a, const gl64_t *b)
{
    const gl64_t A = *a;
    const gl64_t B = *b;

    switch(op)
    {
        case 0: C[threadIdx.x] = A + B; return;
        case 1: C[threadIdx.x] = A - B; return;
        case 2: C[threadIdx.x] = A * B; return;
        case 3: C[threadIdx.x] = B - A; return;
    }
}

__device__ __forceinline__
void op_31_gpu_p2(
    uint64_t op,
    gl64_t * C,
    const gl64_t * a0,
    const gl64_t * a1,
    const gl64_t * a2,
    const gl64_t * b)
{
    // -----------------------------
    // LOAD ONCE (critical improvement)
    // -----------------------------
    const gl64_t A0 = *a0;
    const gl64_t A1 = *a1;
    const gl64_t A2 = *a2;
    const gl64_t B  = *b;

    const int lane = threadIdx.x;
    const int stride = blockDim.x;

    switch (op)
    {
    case 0:
        C[lane] = A0 + B;
        C[stride + lane] = A1;
        C[2 * stride + lane] = A2;
        return;

    case 1:
        C[lane] = A0 - B;
        C[stride + lane] = A1;
        C[2 * stride + lane] = A2;
        return;
    case 2:
    {
        // compute once per thread
        const gl64_t t0 = A0 * B;
        const gl64_t t1 = A1 * B;
        const gl64_t t2 = A2 * B;

        C[lane] = t0;
        C[stride + lane] = t1;
        C[2 * stride + lane] = t2;
        return;
    }
    case 3:
        C[lane] = B - A0;
        C[stride + lane] = -A1;
        C[2 * stride + lane] = -A2;
        return;
    }
}

__device__ __forceinline__
void op_33_gpu_p2(
    uint64_t op,
    gl64_t * C,
    const gl64_t * a0,
    const gl64_t * a1,
    const gl64_t * a2,
    const gl64_t * b0,
    const gl64_t * b1,
    const gl64_t * b2)
{
    // ----------------------------
    // LOAD ONCE (register reuse)
    // ----------------------------
    const gl64_t A0 = *a0;
    const gl64_t A1 = *a1;
    const gl64_t A2 = *a2;

    const gl64_t B0 = *b0;
    const gl64_t B1 = *b1;
    const gl64_t B2 = *b2;

    switch (op)
    {
    case 0:
        C[threadIdx.x] = A0 + B0;
        C[blockDim.x + threadIdx.x] = A1 + B1;
        C[2 * blockDim.x + threadIdx.x] = A2 + B2;
        return;

    case 1:
        C[threadIdx.x] = A0 - B0;
        C[blockDim.x + threadIdx.x] = A1 - B1;
        C[2 * blockDim.x + threadIdx.x] = A2 - B2;
        return;

    case 2:
    {
        const gl64_t A01 = A0 + A1;
        const gl64_t A02 = A0 + A2;
        const gl64_t A12 = A1 + A2;

        const gl64_t B01 = B0 + B1;
        const gl64_t B02 = B0 + B2;
        const gl64_t B12 = B1 + B2;

        const gl64_t D = A0 * B0;
        const gl64_t E = A1 * B1;
        const gl64_t F = A2 * B2;

        const gl64_t G = D - E;

        const gl64_t R0 = A12 * B12;
        const gl64_t R1 = A01 * B01;
        const gl64_t R2 = A02 * B02;

        C[threadIdx.x] = (R0 + G) - F;
        C[blockDim.x + threadIdx.x] = ((R1 + R0) - E - E) - D;
        C[2 * blockDim.x + threadIdx.x] = R2 - G;
        return;
    }

    case 3:
        C[threadIdx.x] = B0 - A0;
        C[blockDim.x + threadIdx.x] = B1 - A1;
        C[2 * blockDim.x + threadIdx.x] = B2 - A2;
        return;
    }
}

__global__  void computeExpressions_(StepsParams *d_params, DeviceArguments *d_deviceArgs, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams, bool constraints)
{

    int chunk_idx = blockIdx.x;
    uint64_t nchunks = d_expsArgs->domainSize / blockDim.x;

    uint32_t bufferCommitsSize = d_deviceArgs->bufferCommitSize;
    Goldilocks::Element **expressions_params = (Goldilocks::Element **)scratchpad;

    // Use temp buffers in dynamic shared memory if launch allocated space for them
    Goldilocks::Element *smem_after_ptrs_s = scratchpad + 32;
    uint64_t tmpTotal_s = d_expsArgs->maxTemp1Size + d_expsArgs->maxTemp3Size;
    bool useTmpSmem_s = tmpTotal_s > 0 && tmpTotal_s <= 5120;

    if (threadIdx.x == 0)
    {
        if (useTmpSmem_s) {
            expressions_params[bufferCommitsSize + 0] = smem_after_ptrs_s;
            expressions_params[bufferCommitsSize + 1] = smem_after_ptrs_s + d_expsArgs->maxTemp1Size;
        } else {
            expressions_params[bufferCommitsSize + 0] = (&d_params->aux_trace[d_expsArgs->offsetTmp1 + blockIdx.x * d_expsArgs->maxTemp1Size]);
            expressions_params[bufferCommitsSize + 1] = (&d_params->aux_trace[d_expsArgs->offsetTmp3 + blockIdx.x * d_expsArgs->maxTemp3Size]);
        }
        expressions_params[bufferCommitsSize + 2] = d_params->publicInputs;
        expressions_params[bufferCommitsSize + 3] = constraints ? d_deviceArgs->numbersConstraints : d_deviceArgs->numbers;
        expressions_params[bufferCommitsSize + 4] = d_params->airValues;
        expressions_params[bufferCommitsSize + 5] = d_params->proofValues;
        expressions_params[bufferCommitsSize + 6] = d_params->airgroupValues;
        expressions_params[bufferCommitsSize + 7] = d_params->challenges;
        expressions_params[bufferCommitsSize + 8] = d_params->evals;
    }
    __syncthreads();
    Goldilocks::Element *destVals = &(d_params->aux_trace[d_expsArgs->offsetDestVals + blockIdx.x * d_expsArgs->dest_nParams * blockDim.x * FIELD_EXTENSION]); 

    while (chunk_idx < nchunks)
    {
        uint64_t i = chunk_idx * blockDim.x;
        bool isCyclic = i < d_expsArgs->k_min || i >= d_expsArgs->k_max;
#pragma unroll 1
        for (uint64_t k = 0; k < d_expsArgs->dest_nParams; ++k)
        {
            if(caseNoOperations__(d_params, d_deviceArgs, d_expsArgs, d_destParams, destVals, k, i)){
                continue;
            }
            uint8_t *ops = constraints ? &d_deviceArgs->opsConstraints[d_destParams[k].opsOffset] : &d_deviceArgs->ops[d_destParams[k].opsOffset];
            uint16_t *args = constraints ? &d_deviceArgs->argsConstraints[d_destParams[k].argsOffset] : &d_deviceArgs->args[d_destParams[k].argsOffset];
            gl64_t *a0, *a1, *a2, *b0, *b1, *b2;

            uint64_t i_args = 0;
            uint64_t nOps = d_destParams[k].nOps;
            for (uint64_t kk = 0; kk < nOps; ++kk)

            {

                switch (ops[kk])
                {
                case 0:
                {
                    // OPERATION WITH DEST: dim1 - SRC0: dim1 - SRC1: dim1
                    load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 1, isCyclic, a0, a1, a2);
                    load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, isCyclic, b0, b1, b2);
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize][args[i_args + 1] * blockDim.x]);
                    op_gpu_p2(args[i_args], res, a0, b0);
                    i_args += 8;
                    break;
                }
                case 1:
                {
                    // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim1
                    load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3, isCyclic, a0, a1, a2);
                    load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, isCyclic, b0, b1, b2);
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize + 1][args[i_args + 1] * blockDim.x]);
                    op_31_gpu_p2(args[i_args], res, a0, a1, a2, b0);
                    i_args += 8;
                    break;
                }
                case 2:
                {
                    // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim3
                    load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3, isCyclic, a0, a1, a2);
                    load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 3, isCyclic, b0, b1, b2);
                    gl64_t *res = (gl64_t*) (kk == nOps - 1 ? &destVals[k * FIELD_EXTENSION * blockDim.x] : &expressions_params[bufferCommitsSize + 1][args[i_args + 1] * blockDim.x]);
                    op_33_gpu_p2(args[i_args], res, a0, a1, a2, b0, b1, b2);
                    i_args += 8;
                    break;
                }
                default:
                {
                    printf(" Wrong operation! %d \n", ops[kk]);
                }
                }
            }
            if (i_args !=  d_destParams[k].nArgs){
                printf(" %lu consumed args - %lu expected args \n", i_args, d_destParams[k].nArgs);
            }
            if (d_destParams[k].inverse)
            {
                getInversePolinomial__((gl64_t*) &destVals[k * FIELD_EXTENSION * blockDim.x], d_destParams[k].dim);
            }

        }

        if (d_expsArgs->dest_nParams == 2)
        {
            
            multiplyPolynomials__(d_expsArgs, d_destParams, d_deviceArgs, (gl64_t*) destVals, i);
        } else {
            storePolynomial__(d_expsArgs, destVals, i);
        }

        chunk_idx += gridDim.x;
    }

}


template<bool IsCyclic>
__device__ __forceinline__ void computeExpression_chunk_(
    StepsParams *d_params, DeviceArguments *d_deviceArgs, ExpsArguments *d_expsArgs,
    DestParamsGPU *d_destParams, Goldilocks::Element **expressions_params,
    uint32_t bufferCommitsSize, uint64_t i,
    const uint8_t * __restrict__ ops, const uint16_t * __restrict__ args)
{
    gl64_t *a0, *a1, *a2, *b0, *b1, *b2;
    gl64_t *res = nullptr;

    uint64_t i_args = 0;
    uint64_t nOps = d_destParams[0].nOps;
    for (uint64_t kk = 0; kk < nOps; ++kk)
    {
        switch (ops[kk])
        {
        case 0:
        {
            load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 1, IsCyclic, a0, a1, a2);
            load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, IsCyclic, b0, b1, b2);
            res = (gl64_t*)&expressions_params[bufferCommitsSize][args[i_args + 1] * blockDim.x];
            op_gpu_p2(args[i_args], res, a0, b0);
            i_args += 8;
            break;
        }
        case 1:
        {
            load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3, IsCyclic, a0, a1, a2);
            load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 1, IsCyclic, b0, b1, b2);
            res = (gl64_t*)&expressions_params[bufferCommitsSize + 1][args[i_args + 1] * blockDim.x];
            op_31_gpu_p2(args[i_args], res, a0, a1, a2, b0);
            i_args += 8;
            break;
        }
        case 2:
        {
            load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 2], args[i_args + 3], args[i_args + 4], i, 3, IsCyclic, a0, a1, a2);
            load__(d_deviceArgs, d_expsArgs, d_params, expressions_params, args[i_args + 5], args[i_args + 6], args[i_args + 7], i, 3, IsCyclic, b0, b1, b2);
            res = (gl64_t*)&expressions_params[bufferCommitsSize + 1][args[i_args + 1] * blockDim.x];
            op_33_gpu_p2(args[i_args], res, a0, a1, a2, b0, b1, b2);
            i_args += 8;
            break;
        }
        default:
        {
            printf(" Wrong operation! %d \n", ops[kk]);
        }
        }
    }
    if (i_args != d_destParams[0].nArgs){
        printf(" %lu consumed args - %lu expected args \n", i_args, d_destParams[0].nArgs);
    }

    if (res != nullptr) {
        storePolynomial__(d_expsArgs, (Goldilocks::Element *)res, i);
    }
}

__global__  void computeExpression_(StepsParams *d_params, DeviceArguments *d_deviceArgs, ExpsArguments *d_expsArgs, DestParamsGPU *d_destParams)
{

    int chunk_idx = blockIdx.x;
    uint64_t nchunks = d_expsArgs->domainSize / blockDim.x;

    uint32_t bufferCommitsSize = d_deviceArgs->bufferCommitSize;
    Goldilocks::Element **expressions_params = (Goldilocks::Element **)scratchpad;

    // Static shared memory for ops/args staging
    __shared__ uint8_t ops_staged[256];
    __shared__ uint16_t args_staged[2048];

    uint64_t nOps = d_destParams[0].nOps;
    uint64_t nArgs = d_destParams[0].nArgs;

    // Use temp buffers in dynamic shared memory if launch allocated space for them
    Goldilocks::Element *smem_after_ptrs = scratchpad + 32;
    uint64_t tmpTotal = d_expsArgs->maxTemp1Size + d_expsArgs->maxTemp3Size;
    bool useTmpSmem = tmpTotal > 0 && tmpTotal <= 5120;

    if (threadIdx.x == 0)
    {
        if (useTmpSmem) {
            expressions_params[bufferCommitsSize + 0] = smem_after_ptrs;
            expressions_params[bufferCommitsSize + 1] = smem_after_ptrs + d_expsArgs->maxTemp1Size;
        } else {
            expressions_params[bufferCommitsSize + 0] = (&d_params->aux_trace[d_expsArgs->offsetTmp1 + blockIdx.x * d_expsArgs->maxTemp1Size]);
            expressions_params[bufferCommitsSize + 1] = (&d_params->aux_trace[d_expsArgs->offsetTmp3 + blockIdx.x * d_expsArgs->maxTemp3Size]);
        }
        expressions_params[bufferCommitsSize + 2] = d_params->publicInputs;
        expressions_params[bufferCommitsSize + 3] = d_deviceArgs->numbers;
        expressions_params[bufferCommitsSize + 4] = d_params->airValues;
        expressions_params[bufferCommitsSize + 5] = d_params->proofValues;
        expressions_params[bufferCommitsSize + 6] = d_params->airgroupValues;
        expressions_params[bufferCommitsSize + 7] = d_params->challenges;
        expressions_params[bufferCommitsSize + 8] = d_params->evals;
    }
    // Stage ops and args cooperatively
    const uint8_t *g_ops = &d_deviceArgs->ops[d_destParams[0].opsOffset];
    const uint16_t *g_args = &d_deviceArgs->args[d_destParams[0].argsOffset];
    for (uint32_t t = threadIdx.x; t < nOps && t < 256; t += blockDim.x) ops_staged[t] = g_ops[t];
    for (uint32_t t = threadIdx.x; t < nArgs && t < 2048; t += blockDim.x) args_staged[t] = g_args[t];
    __syncthreads();

    const uint8_t *active_ops = (nOps <= 256) ? ops_staged : g_ops;
    const uint16_t *active_args = (nArgs <= 2048) ? args_staged : g_args;

    // k_min and k_max are multiples of nRowsPack (== blockDim.x when nblocks_ > 1).
    // Chunk-level dispatch avoids per-iteration branching for the ~99% non-cyclic interior.
    // For single-block launches (nchunks == 1), use the safe cyclic path unconditionally.
    uint64_t k_min_chunk = d_expsArgs->k_min / blockDim.x;
    uint64_t k_max_chunk = d_expsArgs->k_max / blockDim.x;

    while (chunk_idx < nchunks)
    {
        uint64_t i = chunk_idx * blockDim.x;
        if (nchunks == 1 || chunk_idx < k_min_chunk || chunk_idx >= k_max_chunk) {
            computeExpression_chunk_<true>(d_params, d_deviceArgs, d_expsArgs, d_destParams, expressions_params, bufferCommitsSize, i, active_ops, active_args);
        } else {
            computeExpression_chunk_<false>(d_params, d_deviceArgs, d_expsArgs, d_destParams, expressions_params, bufferCommitsSize, i, active_ops, active_args);
        }

        chunk_idx += gridDim.x;
    }

}