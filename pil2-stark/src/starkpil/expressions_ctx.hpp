#ifndef EXPRESSIONS_CTX_HPP
#define EXPRESSIONS_CTX_HPP
#include "expressions_bin.hpp"
#include "const_pols.hpp"
#include "stark_info.hpp"
#include "steps.hpp"
#include "setup_ctx.hpp"

struct Params {
    uint64_t expId;
    uint64_t dim;
    uint64_t stage;
    uint64_t stagePos;
    uint64_t polsMapId;
    uint64_t rowOffsetIndex;
    bool inverse = false;
    bool batch = true;
    opType op;
    uint64_t value;
    
    Params(uint64_t expId_, uint64_t dim_, bool inverse_ = false, bool batch_ = true) : expId(expId_), dim(dim_), inverse(inverse_), batch(batch_), op(opType::tmp) {
        op = opType::tmp;
    }

    Params(PolMap& polMap, uint64_t rowOffsetIndex_, bool inverse_ = false, bool committed = false): dim(polMap.dim), stage(polMap.stage), stagePos(polMap.stagePos), polsMapId(polMap.polsMapId), rowOffsetIndex(rowOffsetIndex_), inverse(inverse_) {
        op = committed ? opType::cm : opType::const_;
    }

    Params(uint64_t value_, bool inverse_ = false) : inverse(inverse_) {
        dim = 1;
        op = opType::number;
        value = value_;
    }

    Params(uint64_t stage_, uint64_t dim_, uint64_t id, opType op_, bool inverse_ = false) : dim(dim_), stage(stage_), polsMapId(id), inverse(inverse_), op(op_) {}
};

struct Dest {
    Goldilocks::Element *dest = nullptr;
    int64_t expId = -1;
    uint64_t offset = 0;
    uint64_t dim = 1;
    uint64_t domainSize;
    std::vector<Params> params;

    Dest(Goldilocks::Element *dest_, uint64_t domainSize_, uint64_t offset_ = 0, int64_t expId_ = -1) : dest(dest_), expId(expId_), offset(offset_), domainSize(domainSize_) {}

    void addParams(uint64_t expId, uint64_t dimExp, bool inverse_ = false, bool batch_ = true) {
        params.push_back(Params(expId, dimExp, inverse_, batch_));
        dim = std::max(dim, dimExp);
    }

    void addCmPol(PolMap& cmPol, uint64_t rowOffsetIndex, bool inverse_ = false) {
        params.push_back(Params(cmPol, rowOffsetIndex, inverse_, true));
        dim = std::max(dim, cmPol.dim);
    }

    void addAirValue(uint64_t stage, uint64_t id, bool inverse_ = false) {
        uint64_t airvalueDim = stage == 1 ? 1 : FIELD_EXTENSION;
        params.push_back(Params(stage, dim, id, opType::airvalue, inverse_));
        dim = std::max(dim, airvalueDim);
    }

    void addConstPol(PolMap& constPol, uint64_t rowOffsetIndex, bool inverse_ = false) {
        params.push_back(Params(constPol, rowOffsetIndex, inverse_, false));
        dim = std::max(dim, constPol.dim);
    }

    void addNumber(uint64_t value, bool inverse_ = false) {
        if(inverse_) value = Goldilocks::inv(Goldilocks::fromU64(value)).fe;
        params.push_back(Params(value, inverse_));
    }
};

class ExpressionsCtx {
public:

    SetupCtx &setupCtx;
    ProverHelpers &proverHelpers;

    Goldilocks::Element *xis;
    int64_t *nextStrides;
    int64_t *nextStridesExtended;
    uint64_t *mapOffsets;
    uint64_t *mapOffsetsExtended;
    uint64_t *mapSectionsN;
    uint64_t *mapOffsetsCustomFixed;
    uint64_t *mapOffsetsCustomFixedExtended;
    uint64_t *mapSectionsNCustomFixed;
    uint64_t nrowsPack_;
    uint64_t minRow;
    uint64_t maxRow;
    uint64_t minRowExtended;
    uint64_t maxRowExtended;
    uint64_t bufferCommitsSize;
    uint64_t nStages;
    uint64_t nPublics;
    uint64_t nChallenges;
    uint64_t nEvals;

    ExpressionsCtx(SetupCtx& _setupCtx, ProverHelpers& proverHelpers_) : setupCtx(_setupCtx), proverHelpers(proverHelpers_) {
        nextStrides = new int64_t[setupCtx.starkInfo.openingPoints.size()];
        nextStridesExtended = new int64_t[setupCtx.starkInfo.openingPoints.size()];
        mapOffsets = new uint64_t[1 + setupCtx.starkInfo.nStages + 1];
        mapOffsetsExtended = new uint64_t[1 + setupCtx.starkInfo.nStages + 1];
        mapOffsetsCustomFixed = new uint64_t[setupCtx.starkInfo.customCommits.size()];
        mapOffsetsCustomFixedExtended = new uint64_t[setupCtx.starkInfo.customCommits.size()];
        mapSectionsN = new uint64_t[1 + setupCtx.starkInfo.nStages + 1];
        mapSectionsNCustomFixed = new uint64_t[setupCtx.starkInfo.customCommits.size()];

        int64_t extend = (1 << (setupCtx.starkInfo.starkStruct.nBitsExt - setupCtx.starkInfo.starkStruct.nBits));

        uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
        uint64_t NExtended = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;

        minRow = 0;
        maxRow = N;
        minRowExtended = 0;
        maxRowExtended = NExtended;

        for(uint64_t i = 0; i < setupCtx.starkInfo.openingPoints.size(); ++i) {
            nextStrides[i] = setupCtx.starkInfo.verify ? 0 : setupCtx.starkInfo.openingPoints[i];
            nextStridesExtended[i] = setupCtx.starkInfo.verify ? 0 : setupCtx.starkInfo.openingPoints[i] * extend;
            if(setupCtx.starkInfo.openingPoints[i] < 0) {
                minRow = std::max(minRow, uint64_t(std::abs(nextStrides[i])));
                minRowExtended = std::max(minRowExtended, uint64_t(std::abs(nextStridesExtended[i])));
            } else {
                maxRow = std::min(maxRow, N - nextStrides[i]);
                maxRowExtended = std::min(maxRowExtended, NExtended - nextStridesExtended[i]);
            }
        }

        mapOffsets[0] = setupCtx.starkInfo.mapOffsets[std::make_pair("const", false)];
        mapOffsetsExtended[0] = setupCtx.starkInfo.mapOffsets[std::make_pair("const", true)];
        mapSectionsN[0] = setupCtx.starkInfo.mapSectionsN["const"];

        for(uint64_t i = 0; i < setupCtx.starkInfo.nStages + 1; ++i) {
            mapSectionsN[i + 1] = setupCtx.starkInfo.mapSectionsN["cm" + std::to_string(i + 1)];
            mapOffsets[i + 1] = setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + std::to_string(i + 1), false)];
            mapOffsetsExtended[i + 1] = setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + std::to_string(i + 1), true)];
        }

        for(uint64_t i = 0; i < setupCtx.starkInfo.customCommits.size(); ++i) {
            mapSectionsNCustomFixed[i] = setupCtx.starkInfo.mapSectionsN[setupCtx.starkInfo.customCommits[i].name + "0"];
            mapOffsetsCustomFixed[i] = setupCtx.starkInfo.mapOffsets[std::make_pair(setupCtx.starkInfo.customCommits[i].name + "0", false)];
            mapOffsetsCustomFixedExtended[i] = setupCtx.starkInfo.mapOffsets[std::make_pair(setupCtx.starkInfo.customCommits[i].name + "0", true)];
        }

        bufferCommitsSize = 1 + setupCtx.starkInfo.nStages + 3 + setupCtx.starkInfo.customCommits.size();
        nStages = setupCtx.starkInfo.nStages;
        nPublics = setupCtx.starkInfo.nPublics;
        nChallenges = setupCtx.starkInfo.challengesMap.size();
        nEvals = setupCtx.starkInfo.evMap.size();
    };

    virtual ~ExpressionsCtx() {
        delete[] nextStrides;
        delete[] nextStridesExtended;
        delete[] mapOffsets;
        delete[] mapOffsetsExtended;
        delete[] mapOffsetsCustomFixed;
        delete[] mapOffsetsCustomFixedExtended;
        delete[] mapSectionsN;
        delete[] mapSectionsNCustomFixed;
    };
    
    virtual void calculateExpressions(StepsParams& params, Dest &dest, uint64_t domainSize, bool domainExtended, bool compilationTime = false, bool verify_constraints = false) {};
 
    void calculateExpression(StepsParams& params, Goldilocks::Element* dest, uint64_t expressionId, bool inverse = false, bool compilation_time = false) {
        uint64_t domainSize;
        bool domainExtended;
        if (compilation_time) {
            domainSize = 1;
            domainExtended = false;
        } else if(expressionId == setupCtx.starkInfo.cExpId || expressionId == setupCtx.starkInfo.friExpId) {
            setupCtx.expressionsBin.expressionsInfo[expressionId].destDim = 3;
            domainSize = 1 << setupCtx.starkInfo.starkStruct.nBitsExt;
            domainExtended = true;
        } else {
            domainSize = 1 << setupCtx.starkInfo.starkStruct.nBits;
            domainExtended = false;
        }
        Dest destStruct(dest, domainSize, 0, expressionId);
        destStruct.addParams(expressionId, setupCtx.expressionsBin.expressionsInfo[expressionId].destDim, inverse);
        calculateExpressions(params, destStruct, domainSize, domainExtended, compilation_time);
    }

    void setXi(Goldilocks::Element *xi) {
        xis = xi;
    }
};

#endif