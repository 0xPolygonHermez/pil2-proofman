#ifndef CPOLYNOMIAL_HPP
#define CPOLYNOMIAL_HPP

#include <algorithm>
#include <vector>

#include "polynomial.hpp"


template<typename Engine>
class CPolynomial {
    using FrElement = typename Engine::FrElement;
    using G1Point = typename Engine::G1Point;
    using G1PointAffine = typename Engine::G1PointAffine;

    Polynomial<Engine> **polynomials;
    Engine &E;

    int n;

public:
    CPolynomial(Engine &_E, int n);

    ~CPolynomial();

    void addPolynomial(int position, Polynomial<Engine> * polynomial);

    u_int64_t getDegree() const;

    Polynomial<Engine> * getPolynomial(FrElement *reservedBuffer) const;

    // Writes f(X) = sum_i p_i(X^n) * X^i, p_i the polynomial added at position i (zero if none was),
    // to buffer[0, getDegree() + 1): coefficient c * n + i of f is coefficient c of p_i up to its
    // degree, and zero above it. Each p_i's degree must be up to date. Returns getDegree() + 1, the
    // count of coefficients written, which is above deg f + 1 if the top ones are zero.
    // Unlike getPolynomial, it writes f's coefficients only, in one parallel pass: it neither clears
    // a power-of-two prefix of the buffer nor scans it for f's degree, it writes nothing beyond the
    // count, and it is defined at every degree.
    u_int64_t getCoefficients(FrElement *buffer) const;

    typename Engine::G1Point multiExponentiation(G1PointAffine *PTau) const;
};

#include "cpolynomial.c.hpp"

#endif
