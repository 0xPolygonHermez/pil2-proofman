#ifndef SETUP_CTX_HPP
#define SETUP_CTX_HPP

#include "stark_info.hpp"
#include "const_pols.hpp"
#include "expressions_bin.hpp"

class ProverHelpers {
    public: 
    Goldilocks::Element *zi = nullptr;
    Goldilocks::Element *x = nullptr;
    Goldilocks::Element *x_n = nullptr; // Needed for PIL1 compatibility

    ProverHelpers() {}

    ProverHelpers(StarkInfo &starkInfo, bool pil1) {
        uint64_t nBits = starkInfo.starkStruct.nBits;
        uint64_t nBitsExt = starkInfo.starkStruct.nBitsExt;
        vector<Boundary> boundaries = starkInfo.boundaries;
        computeX(nBits, nBitsExt, pil1);
        computeZerofier(nBits, nBitsExt, boundaries);
    }

    void computeZerofier(uint64_t nBits, uint64_t nBitsExt, vector<Boundary> boundaries) {
        uint64_t N = 1 << nBits;
        uint64_t NExtended = 1 << nBitsExt;
        zi = new Goldilocks::Element[boundaries.size() * NExtended];

        for(uint64_t i = 0; i < boundaries.size(); ++i) {
            Boundary boundary = boundaries[i];
            if(boundary.name == "everyRow") {
                buildZHInv(nBits, nBitsExt);
            } else if(boundary.name == "firstRow") {
                buildOneRowZerofierInv(nBits, nBitsExt, i, 0);
            } else if(boundary.name == "lastRow") {
                buildOneRowZerofierInv(nBits, nBitsExt, i, N);
            } else if(boundary.name == "everyFrame") {
                buildFrameZerofierInv(nBits, nBitsExt, i, boundary.offsetMin, boundary.offsetMax);
            }
        }
    }

    void computeX(uint64_t nBits, uint64_t nBitsExt, bool pil1) {
        uint64_t NExtended = 1 << nBitsExt;
        uint64_t N = 1 << nBits;
        x = new Goldilocks::Element[NExtended];
        if(pil1) x_n = new Goldilocks::Element[N];
    #pragma omp parallel for
        for (uint64_t k = 0; k < NExtended; k+=4096) {
            if(pil1 && k < N) x_n[k] = Goldilocks::pow(Goldilocks::w(nBits), k);
            x[k] = Goldilocks::mul(Goldilocks::shift(), Goldilocks::pow(Goldilocks::w(nBitsExt), k));
            for(uint64_t j = k+1; j < std::min(k + 4096, NExtended); ++j) {
                if(pil1 && j < N) x_n[j] = x_n[j-1] * Goldilocks::w(nBits);
                x[j] = x[j-1] * Goldilocks::w(nBitsExt);
            }
        }
    }

    void buildZHInv(uint64_t nBits, uint64_t nBitsExt)
    {
        uint64_t NExtended = 1 << nBitsExt;
        uint64_t extendBits = nBitsExt - nBits;
        uint64_t extend = (1 << extendBits);
        
        Goldilocks::Element w = Goldilocks::one();
        Goldilocks::Element sn = Goldilocks::shift();
        for (uint64_t i = 0; i < nBits; i++) Goldilocks::square(sn, sn);

        for (uint64_t i=0; i<extend; i++) {
            Goldilocks::inv(zi[i], (sn * w) - Goldilocks::one());
            Goldilocks::mul(w, w, Goldilocks::w(extendBits));
        }

        #pragma omp parallel for
        for (uint64_t i=extend; i<NExtended; i++) {
            zi[i] = zi[i % extend];
        }
    };

    void buildOneRowZerofierInv(uint64_t nBits, uint64_t nBitsExt, uint64_t offset, uint64_t rowIndex)
    {
        uint64_t NExtended = 1 << nBitsExt;
        Goldilocks::Element root = Goldilocks::one();

        for(uint64_t i = 0; i < rowIndex; ++i) {
            root = root * Goldilocks::w(nBits);
        }

    #pragma omp parallel for
        for(uint64_t i = 0; i < NExtended; ++i) {
            Goldilocks::inv(zi[i + offset * NExtended], (x[i] - root) * zi[i]);
        }
    }

    void buildFrameZerofierInv(uint64_t nBits, uint64_t nBitsExt, uint64_t offset, uint64_t offsetMin, uint64_t offsetMax)
    {
        uint64_t NExtended = 1 << nBitsExt;
        uint64_t N = 1 << nBits;
        uint64_t nRoots = offsetMin + offsetMax;
        Goldilocks::Element roots[nRoots];

        for(uint64_t i = 0; i < offsetMin; ++i) {
            roots[i] = Goldilocks::one();
            for(uint64_t j = 0; j < i; ++j) {
                roots[i] = roots[i] * Goldilocks::w(nBits);
            }
        }

        for(uint64_t i = 0; i < offsetMax; ++i) {
            roots[i + offsetMin] = Goldilocks::one();
            for(uint64_t j = 0; j < (N - i - 1); ++j) {
                roots[i + offsetMin] = roots[i + offsetMin] * Goldilocks::w(nBits);
            }
        }

    #pragma omp parallel for
        for(uint64_t i = 0; i < NExtended; ++i) {
            zi[i + offset*NExtended] = Goldilocks::one();
            for(uint64_t j = 0; j < nRoots; ++j) {
                zi[i + offset*NExtended] = zi[i + offset*NExtended] * (x[i] - roots[j]);
            }
        }
    }

    ~ProverHelpers() {
        if(zi != nullptr) delete[] zi;
        if(x != nullptr) delete[] x;
        if(x_n != nullptr) delete[] x_n;
    };
};

class SetupCtx {
public:

    StarkInfo &starkInfo;
    ExpressionsBin &expressionsBin;
    
    SetupCtx(StarkInfo &_starkInfo, ExpressionsBin& _expressionsBin) : starkInfo(_starkInfo), expressionsBin(_expressionsBin)  {};
};

#endif