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
    bool zeroDenominator = false; // Zh(z) or some V_O(z) is zero: the verifier must reject

    ProverHelpers() {}

    ProverHelpers(StarkInfo &starkInfo, bool pil1) {
        uint64_t nBits = starkInfo.starkStruct.nBits;
        uint64_t nBitsExt = starkInfo.starkStruct.nBitsExt;
        computeX(nBits, nBitsExt, pil1);
        computeZerofier(starkInfo);
    }

    // zi[b] at z: Zh(z)^-1 for everyRow, Zh(z)/V_O(z) for a domain, V_O(z) for its complement.
    ProverHelpers(StarkInfo& starkInfo, Goldilocks::Element* z) { 
        uint64_t nBits = starkInfo.starkStruct.nBits;
        zi = new Goldilocks::Element[starkInfo.boundaries.size() * FIELD_EXTENSION];

        // zPow[i] = z^(2^i)
        std::vector<Goldilocks::Element> zPowBuf((nBits + 1) * FIELD_EXTENSION);
        Goldilocks3::Element *zPow = (Goldilocks3::Element *)zPowBuf.data();
        Goldilocks3::copy(zPow[0], *(Goldilocks3::Element *)z);
        for(uint64_t i = 1; i <= nBits; ++i) {
            Goldilocks3::square(zPow[i], zPow[i - 1]);
        }

        x_n = new Goldilocks::Element[FIELD_EXTENSION];
        x_n[0] = z[0];
        x_n[1] = z[1];
        x_n[2] = z[2];

        // Every V_O divides Zh, so Zh(z) != 0 rules out every zero denominator (inv throws on zero).
        Goldilocks3::Element zN;
        Goldilocks3::sub(zN, zPow[nBits], Goldilocks::one());
        if(Goldilocks::isZero(zN[0]) && Goldilocks::isZero(zN[1]) && Goldilocks::isZero(zN[2])) {
            zeroDenominator = true;
            std::memset(zi, 0, starkInfo.boundaries.size() * FIELD_EXTENSION * sizeof(Goldilocks::Element));
            return;
        }
        Goldilocks3::inv((Goldilocks3::Element *)&zi[0], &zN);

        for(uint64_t i = 1; i < starkInfo.boundaries.size(); ++i) {
            const Boundary &boundary = starkInfo.boundaries[i];
            const Goldilocks3::Element &zm = zPow[nBits - boundary.cycleBits];
            Goldilocks3::Element v = {Goldilocks::one(), Goldilocks::zero(), Goldilocks::zero()};
            for(uint64_t o : boundary.offsets) {
                Goldilocks3::Element f;
                Goldilocks3::sub(f, zm, Goldilocks::exp(Goldilocks::w(boundary.cycleBits), o));
                Goldilocks3::mul(v, v, f);
            }
            Goldilocks3::Element &dst = *(Goldilocks3::Element *)&zi[i * FIELD_EXTENSION];
            if(boundary.complement) {
                Goldilocks3::copy(dst, v);
            } else {
                Goldilocks3::inv(&dst, &v);
                Goldilocks3::mul(dst, dst, zN);
            }
        }
    };

    // One period of each boundary's zerofier, laid out by starkInfo.ziOffsets.
    void computeZerofier(StarkInfo &starkInfo) {
        uint64_t nBits = starkInfo.starkStruct.nBits;
        uint64_t nBitsExt = starkInfo.starkStruct.nBitsExt;
        zi = new Goldilocks::Element[starkInfo.ziSize];

        for(uint64_t i = 0; i < starkInfo.boundaries.size(); ++i) {
            const Boundary &boundary = starkInfo.boundaries[i];
            if(boundary.name == "everyRow") {
                buildZHInv(nBits, nBitsExt, &zi[starkInfo.ziOffsets[i]]);
            } else {
                buildDomainZerofier(nBits, nBitsExt, boundary, &zi[starkInfo.ziOffsets[i]]);
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

    // 1/Zh over its period 2^(nBitsExt - nBits).
    void buildZHInv(uint64_t nBits, uint64_t nBitsExt, Goldilocks::Element *dst)
    {
        uint64_t extendBits = nBitsExt - nBits;
        uint64_t extend = (1 << extendBits);
        
        Goldilocks::Element w = Goldilocks::one();
        Goldilocks::Element sn = Goldilocks::shift();
        for (uint64_t i = 0; i < nBits; i++) Goldilocks::square(sn, sn);

        for (uint64_t i=0; i<extend; i++) {
            Goldilocks::inv(dst[i], (sn * w) - Goldilocks::one());
            Goldilocks::mul(w, w, Goldilocks::w(extendBits));
        }
    };

    // Zh/V_O (or V_O for a complement) over its period P: x^m = shift^m * w(log P)^i, m = N/2^c.
    void buildDomainZerofier(uint64_t nBits, uint64_t nBitsExt, const Boundary &boundary, Goldilocks::Element *dst)
    {
        uint64_t pBits = periodBits(boundary, nBits, nBitsExt);
        uint64_t P = uint64_t(1) << pBits;
        uint64_t c = boundary.cycleBits;
        Goldilocks::Element shiftM = Goldilocks::shift();
        for (uint64_t i = 0; i < nBits - c; i++) Goldilocks::square(shiftM, shiftM);
        std::vector<Goldilocks::Element> roots;
        for (uint64_t o : boundary.offsets) roots.push_back(Goldilocks::exp(Goldilocks::w(c), o));

    #pragma omp parallel for
        for (uint64_t k = 0; k < P; k += 4096) {
            Goldilocks::Element xm = shiftM * Goldilocks::exp(Goldilocks::w(pBits), k);
            for (uint64_t i = k; i < std::min(k + 4096, P); ++i) {
                Goldilocks::Element v = Goldilocks::one();
                for (const Goldilocks::Element &r : roots) v = v * (xm - r);
                if (boundary.complement) {
                    dst[i] = v;
                } else {
                    Goldilocks::Element zh = xm;
                    for (uint64_t j = 0; j < c; j++) Goldilocks::square(zh, zh);
                    dst[i] = (zh - Goldilocks::one()) * Goldilocks::inv(v);
                }
                xm = xm * Goldilocks::w(pBits);
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