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
        computeX(nBits, nBitsExt, pil1);
        computeZerofier(nBits, nBitsExt);
    }

    // everyRow only: StarkInfo::load refuses any other boundary.
    void computeZerofier(uint64_t nBits, uint64_t nBitsExt) {
        zi = new Goldilocks::Element[1 << nBitsExt];
        buildZHInv(nBits, nBitsExt);
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