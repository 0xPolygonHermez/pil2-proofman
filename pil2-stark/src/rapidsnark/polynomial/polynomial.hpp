#ifndef POLYNOMIAL_HPP
#define POLYNOMIAL_HPP

#include "assert.h"
#include <sstream>
#include <algorithm>
#include <vector>
#include <gmp.h>
#include "fft.hpp"


template<typename Engine>
class Polynomial {
    using FrElement = typename Engine::FrElement;
    using G1Point = typename Engine::G1Point;

    bool createBuffer;
    u_int64_t length;
    u_int64_t degree;

    Engine &E;

    void initialize(u_int64_t length, u_int64_t blindLength = 0, bool createBuffer = true);

    static Polynomial<Engine>* computeLagrangePolynomial(u_int64_t i, FrElement xArr[], FrElement yArr[], u_int32_t length);

    // The steps of divByMonicInPlace
    template<bool Write>
    void divByMonicRecurrence(u_int64_t lo, u_int64_t hi, u_int64_t m, const FrElement &beta, FrElement *q,
                              FrElement *above);

    void divByMonicSlots(FrElement *slots, u_int64_t from, u_int64_t m) const;
public:
    FrElement *coef;

    Polynomial(Engine &_E, u_int64_t length, u_int64_t blindLength = 0);

    Polynomial(Engine &_E, FrElement *reservedBuffer, u_int64_t length, u_int64_t blindLength = 0);

    // From coefficients
    static Polynomial<Engine>* fromPolynomial(Engine &_E, Polynomial<Engine> &polynomial, u_int64_t blindLength = 0);

    static Polynomial<Engine>* fromPolynomial(Engine &_E, Polynomial<Engine> &polynomial, FrElement *reservedBuffer, u_int64_t blindLength = 0);

    // Over the coefficients already in reservedBuffer[0, length), which it neither clears (as the
    // constructor on a reserved buffer does) nor copies, and does not free. The degree is fixed.
    static Polynomial<Engine>* fromReservedBuffer(Engine &_E, FrElement *reservedBuffer, u_int64_t length);

    // From evaluations
    static Polynomial<Engine>* fromEvaluations(Engine &_E, FFT<typename Engine::Fr> *fft, FrElement *evaluations, u_int64_t length, u_int64_t blindLength = 0);

    static Polynomial<Engine>* fromEvaluations(Engine &_E, FFT<typename Engine::Fr> *fft, FrElement *evaluations, FrElement *reservedBuffer, u_int64_t length, u_int64_t blindLength = 0);

    ~Polynomial();

    void fixDegree();

    void fixDegreeFrom(uint64_t initial);

    bool isEqual(const Polynomial<Engine> &other) const;

    void blindCoefficients(FrElement blindingFactors[], u_int32_t length);

    typename Engine::FrElement getCoef(u_int64_t index) const;

    void setCoef(u_int64_t index, FrElement value);

    u_int64_t getLength() const;

    u_int64_t getDegree() const;

    inline typename Engine::FrElement evaluate(FrElement point) const {
        FrElement result = E.fr.zero();

        for (u_int64_t i = degree + 1; i > 0; i--) {
            result = E.fr.add(coef[i - 1], E.fr.mul(result, point));
        }
        return result;
    }

    typename Engine::FrElement fastEvaluate(FrElement point) const;

    void add(Polynomial<Engine> &polynomial);

    void addBlinding(Polynomial<Engine> &polynomial, FrElement &blindingValue);

    void sub(Polynomial<Engine> &polynomial);

    void subBlinding(Polynomial<Engine> &polynomial, FrElement &blindingValue);

    void mulScalar(FrElement &value);

    void addScalar(FrElement &value);

    void subScalar(FrElement &value);

    // Multiply current polynomial by the polynomial (X - value)
    void byXSubValue(FrElement &value);

    void byXNSubValue(int n, FrElement &value);

    // Euclidean division, returns reminder polygon
    Polynomial<Engine>* divBy(Polynomial<Engine> &polynomial);

    void divByMonic(uint32_t m, FrElement beta);

    // Division by X^m - beta, 1 <= m <= d for the degree d, in place: the quotient q replaces the
    // coefficients, q_j at j <= d - m and zeros at d - m < j <= d, and the degree is fixed. Returns
    // whether the remainder, a_j + beta * q_j for j < m, is zero: whether the division is exact.
    // The degree must be up to date: coefficients above it are neither read nor written. Throws
    // std::runtime_error if m is 0 or above d.
    // Unlike divByMonic, which runs on m threads (one per residue of j mod m), moves to a buffer of
    // its own, does not compute the remainder and needs d >= 2m - 1, it runs on every thread, in
    // parallel over blocks of the quotient's coefficients, keeps its buffer, owned or reserved, and
    // computes the remainder. The quotient is the same, coefficient by coefficient.
    bool divByMonicInPlace(u_int64_t m, const FrElement &beta);

    // The coefficients of a block of divByMonicInPlace(m, beta), m >= 1: a whole number of rows of
    // m, about 2^12.
    static u_int64_t divByMonicInPlaceBlockLength(u_int64_t m);

    Polynomial<Engine>* divByVanishing(uint32_t m, FrElement beta);

    Polynomial<Engine>* divByVanishing(FrElement *reservedBuffer, uint64_t m, FrElement beta);

    void fastDivByVanishing(FrElement *reservedBuffer, uint32_t m, FrElement beta);

    void divZh(u_int64_t domainSize, int extension = 4);

    void divByZerofier(u_int64_t n, FrElement beta);

    void byX();

    static Polynomial<Engine>* lagrangePolynomialInterpolation(FrElement xArr[], FrElement yArr[], u_int32_t length);

    static Polynomial<Engine>* zerofierPolynomial(FrElement xArr[], u_int32_t length);

    void print();
};

#include "polynomial.c.hpp"

#endif
