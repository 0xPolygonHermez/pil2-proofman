#ifndef CPOLYNOMIAL_C_HPP
#define CPOLYNOMIAL_C_HPP

#include "cpolynomial.hpp"

using namespace CPlusPlusLogging;

template<typename Engine>
CPolynomial<Engine>::CPolynomial(Engine &_E, int n) : E(_E), n(n) {
    this->polynomials = new Polynomial<Engine> *[n];
    for (int i = 0; i < n; i++) {
        this->polynomials[i] = NULL;
    }
}

template<typename Engine>
CPolynomial<Engine>::~CPolynomial() {
    delete[] this->polynomials;
}

template<typename Engine>
void CPolynomial<Engine>::addPolynomial(int position, Polynomial<Engine> *polynomial) {
    if (position > this->n - 1) {
        throw std::invalid_argument(
                "CPolynomial:addPolynomial, cannot add a polynomial to a position greater than n-1");
    }

    this->polynomials[position] = polynomial;
}

template<typename Engine>
u_int64_t CPolynomial<Engine>::getDegree() const {
    u_int64_t degree = 0;
    for (int i = 0; i < n; i++) {
        if (this->polynomials[i] != NULL) {
            degree = std::max(degree, this->polynomials[i]->getDegree() * n + i);
        }
    }
    return degree;
}

template<typename Engine>
Polynomial<Engine> *CPolynomial<Engine>::getPolynomial(FrElement *reservedBuffer) const {
    u_int64_t degrees[n];

    for (int i = 0; i < n; i++) {
        degrees[i] = polynomials[i] == NULL ? 0 : polynomials[i]->getDegree();
    }

    u_int64_t maxDegree = this->getDegree();
    u_int64_t lengthBuffer = std::pow(2, ((u_int64_t)log2(maxDegree - 1)) + 1);
    Polynomial<Engine> *polynomial = new Polynomial<Engine>(E, reservedBuffer, lengthBuffer);

    #pragma omp parallel for
    for (u_int64_t i = 0; i < maxDegree+1; i++) {
        for (int j = 0; j < n; j++) {
            if (polynomials[j] != NULL) {
                if (i <= degrees[j]) polynomial->coef[i * n + j] = polynomials[j]->coef[i];
            }
        }
    }

    polynomial->fixDegree();

    return polynomial;
}

template<typename Engine>
typename Engine::G1Point CPolynomial<Engine>::multiExponentiation(G1PointAffine *PTau) const {
//    LOG_TRACE("> Computing C2 multi exponentiation");
//    u_int64_t lengths[3] = {polynomials["Z"]->getDegree() + 1,
//                            polynomials["T1"]->getDegree() + 1,
//                            polynomials["T2"]->getDegree() + 1};
//    G1Point C2 = multiExponentiation(polynomials["C2"], 3, lengths);
}

template<typename Engine>
u_int64_t CPolynomial<Engine>::getCoefficients(FrElement *buffer) const {
    // The coefficients of each p_i to copy, deg p_i + 1, or none where no polynomial was added.
    std::vector<const FrElement *> coefs(n);
    std::vector<u_int64_t> lengths(n);
    for (int i = 0; i < n; i++) {
        coefs[i] = polynomials[i] == NULL ? NULL : polynomials[i]->coef;
        lengths[i] = polynomials[i] == NULL ? 0 : polynomials[i]->getDegree() + 1;
    }

    const u_int64_t nCoefs = this->getDegree() + 1;
    const FrElement zero = E.fr.zero();
    if (n == 0) {
        buffer[0] = zero;
        return nCoefs;
    }
    const u_int64_t k = n;

    // Row c of f, its coefficients c * n + i, is coefficient c of every p_i.
    #pragma omp parallel for
    for (u_int64_t c = 0; c < (nCoefs + k - 1) / k; c++) {
        const u_int64_t end = std::min(k, nCoefs - c * k);
        for (u_int64_t i = 0; i < end; i++) {
            buffer[c * k + i] = c < lengths[i] ? coefs[i][c] : zero;
        }
    }

    return nCoefs;
}

#endif
