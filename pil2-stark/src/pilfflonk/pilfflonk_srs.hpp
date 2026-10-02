#ifndef PILFFLONK_SRS_HPP
#define PILFFLONK_SRS_HPP

#include <cstdint>
#include <memory>
#include <string>

#include "alt_bn128.hpp"

namespace PilFflonk {

using FrElement = AltBn128::Engine::FrElement;
using G1Point = AltBn128::Engine::G1Point;
using G1PointAffine = AltBn128::Engine::G1PointAffine;
using G2PointAffine = AltBn128::Engine::G2PointAffine;

// A point as a snarkjs ptau, a zkey and pilfflonk.srs.bin store it: affine, x‖y, every Fq
// coordinate in ffiasm's Montgomery form (x·2^256 mod q), little-endian. A G2 coordinate is an Fq2
// element c0 + c1·u, stored c0‖c1. These are ffiasm's in-memory affine points, byte for byte.
constexpr uint64_t SRS_G1_BYTES = 64;
constexpr uint64_t SRS_G2_BYTES = 128;
static_assert(sizeof(G1PointAffine) == SRS_G1_BYTES, "ffiasm's affine G1 point must be the stored one");
static_assert(sizeof(G2PointAffine) == SRS_G2_BYTES, "ffiasm's affine G2 point must be the stored one");

// The most powers [τ^i]₁ an SRS holds: ffiasm's MSM counts its points in an unsigned int. The
// largest ptau there is, of power 28, has 2^29 - 1.
constexpr uint64_t MAX_SRS_G1 = 0xffffffffu;

// Section 1 of a snarkjs powers-of-tau file, and the sizes of the two sections pilfflonk reads.
// A file of power p made by snarkjs has 2^(p+1) - 1 powers [τ^i]₁ and 2^p powers [τ^i]₂; the
// reader relies only on the sizes of the sections, not on p.
struct PtauHeader {
    uint32_t power;
    uint32_t ceremonyPower;
    uint64_t nG1; // [τ^i]₁ in section 2
    uint64_t nG2; // [τ^i]₂ in section 3
};

// Reads and checks section 1 of the snarkjs ptau at `path` (4-byte n8q = 32, q in n8q bytes
// little-endian, which must be BN254's base field modulus, then 4-byte power and ceremonyPower)
// and the sizes of sections 2 and 3, which must hold whole points. Reads nothing else.
// Throws IoError if the file cannot be opened or read, FormatError if it is not such a file, and
// std::runtime_error where ffiasm has no assembly backend.
PtauHeader readPtauHeader(const std::string &path);

// What checkG2 finds wrong with a point of the G2 twist, if anything.
enum class G2Error {
    None,
    Infinity,   // (0, 0), ffiasm's affine point at infinity
    NotOnTwist, // a coordinate not below q, or off y^2 = x^3 + 3/(9+u)
    NotInG2,    // on the twist, but r times it is not the point at infinity
};

// Whether `p` (Montgomery form) is a point of G2 other than the point at infinity: its coordinates
// below q, on the twist, and in its r-torsion group G2. What the JS verifier requires of the vkey's
// X_2 (pilfflonk/js/src/elements.js, g2FromObject), and so of the SRS's [τ]₂.
G2Error checkG2(const G2PointAffine &p);

// The structured reference string of a proof (pilfflonk/docs/README.md#setup-pilfflonk): the
// powers [τ^i]₁ for i < nG1, and [1]₂ and [τ]₂, as a snarkjs ptau holds them. KZG commitments are
// MSMs over its G1 powers.
//
// Every point is checked when it is read: each coordinate below q, on its curve, [1]₁ and [1]₂
// equal to the generators, and [τ]₂ in the r-torsion group G2 of the twist, as the JS verifier
// requires of the vkey's X_2. That catches a corrupt or truncated file; it cannot tell whether the
// points are consistent powers of one τ, which takes pairings (snarkjs's `powersoftau verify`).
//
// pilfflonk.srs.bin, version 1 (pilfflonk/docs/formats.md#srs): a binfile container, as
// rapidsnark's BinFile and snarkjs read them (4-byte type, u32 version, u32 number of sections,
// then each section as u32 id, u64 size in bytes and its contents; integers little-endian), of
// type "pfsr" and with three sections, the ids of their counterparts in the ptau:
//
//   1  header, 88 bytes:
//        u32  n8q = 32
//        q    n8q bytes, little-endian: BN254's base field modulus
//        u32  n8r = 32
//        r    n8r bytes, little-endian: BN254's scalar field modulus
//        u64  nG1, the points in section 2: 1 <= nG1 <= MAX_SRS_G1
//        u64  nG2, the points in section 3: 2
//   2  [τ^i]₁ for i < nG1: nG1 · SRS_G1_BYTES bytes
//   3  [τ^i]₂ for i < nG2, that is [1]₂ and [τ]₂: nG2 · SRS_G2_BYTES bytes
//
// The points are those of sections 2 and 3 of the ptau, copied as they are (the format above).
//
// Immutable once built, so its const functions may run concurrently. A key on the GPU copies the
// powers [τ^i]₁ to the device (Gpu, pilfflonk_gpu.hpp), which commits there (GpuKey::commit).
class Srs {
public:
    // The powers [τ^i]₂ an SRS holds: [1]₂ and [τ]₂.
    static constexpr uint64_t N_G2 = 2;

    // The first nG1 powers [τ^i]₁, and [1]₂ and [τ]₂, of the snarkjs ptau at ptauPath (sections 2
    // and 3, pilfflonk/docs/README.md#setup-pilfflonk; section 12 is not needed). Reads only those
    // points, never the whole file. Throws std::invalid_argument unless 1 <= nG1 <= MAX_SRS_G1 and
    // the file holds nG1 powers [τ^i]₁ (checked before any point is read); IoError and FormatError
    // as readPtauHeader, and FormatError if a point read is not valid or section 3 has fewer than
    // N_G2 points. Throws std::runtime_error where ffiasm has no assembly backend.
    static Srs fromPtau(const std::string &ptauPath, uint64_t nG1);

    // Reads pilfflonk.srs.bin. Throws IoError if the file cannot be opened or read, FormatError if
    // it is not such a file (any size or header field that does not match, or a point that is not
    // valid), and std::runtime_error where ffiasm has no assembly backend.
    static Srs load(const std::string &path);

    // The nG1 of the pilfflonk.srs.bin at `path`, from its header, checked as load checks it: no point
    // is read. Throws as load does.
    static uint64_t powersIn(const std::string &path);

    // Writes pilfflonk.srs.bin to `path`, replacing any file there. It writes to `path` + ".tmp"
    // first and renames that into place, so `path` is never left half written. Throws IoError if
    // the file cannot be written.
    void save(const std::string &path) const;

    uint64_t nG1() const { return nPowers; }

    // [τ^i]₁, for i < nG1().
    const G1PointAffine &g1(uint64_t i) const { return g1Powers[i]; }

    // [τ^i]₂, for i < N_G2.
    const G2PointAffine &g2(uint64_t i) const { return g2Powers[i]; }

    // The KZG commitment [p(τ)]₁ = Σ_i coefs[i]·[τ^i]₁ of the polynomial with the nCoefs
    // coefficients in `coefs` (Montgomery form, increasing degree): ffiasm's MSM, whose scalars
    // must be canonical, converted from Montgomery right before it
    // (pilfflonk/docs/protocol.md#commitments). With nCoefs = 0 it is the point at infinity. Throws
    // std::invalid_argument if nCoefs > nG1().
    G1Point commit(const FrElement *coefs, uint64_t nCoefs) const;

private:
    explicit Srs(uint64_t nG1);

    // Throws FormatError, naming `source`, unless every point is valid.
    void checkPoints(const std::string &source) const;

    uint64_t nPowers;
    // Behind a pointer because ffiasm's multiMulByScalar takes its bases as non-const, although
    // it only reads them.
    std::unique_ptr<G1PointAffine[]> g1Powers;
    G2PointAffine g2Powers[N_G2];
};

} // namespace PilFflonk

#endif
