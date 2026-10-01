#include "pilfflonk_srs.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <system_error>
#include <vector>

#include "binfile_utils.hpp"
#include "binfile_writer.hpp"
#include "pilfflonk_error.hpp"
#include "pilfflonk_gpu.hpp"

namespace PilFflonk {

// The MSM reads each scalar's limbs as the bytes of a little-endian integer, and the files store
// ffiasm's limbs as they are in memory.
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "the SRS and its files assume a little-endian host");

namespace {

using Engine = AltBn128::Engine;
using BinFileUtils::BinFile;

// Bytes of an element of Fq or Fr.
constexpr uint32_t N8 = 32;
static_assert(N8 == RawFq::N64 * sizeof(uint64_t) && N8 == RawFr::N64 * sizeof(uint64_t),
              "both BN254 fields have 32-byte elements");

// A snarkjs ptau (powersoftau_new.js): the header, [τ^i]₁ and [τ^i]₂.
constexpr const char *PTAU_TYPE = "ptau";
constexpr uint32_t PTAU_VERSION = 1;
constexpr uint32_t PTAU_HEADER_SECTION = 1;
constexpr uint32_t PTAU_G1_SECTION = 2;
constexpr uint32_t PTAU_G2_SECTION = 3;
// n8q, q, power, ceremonyPower.
constexpr uint64_t PTAU_HEADER_BYTES = 4 + N8 + 4 + 4;

// pilfflonk.srs.bin: the format in pilfflonk_srs.hpp.
constexpr const char *SRS_TYPE = "pfsr";
constexpr uint32_t SRS_VERSION = 1;
constexpr uint32_t SRS_N_SECTIONS = 3;
constexpr uint32_t SRS_HEADER_SECTION = 1;
constexpr uint32_t SRS_G1_SECTION = 2;
constexpr uint32_t SRS_G2_SECTION = 3;
// n8q, q, n8r, r, nG1, nG2.
constexpr uint64_t SRS_HEADER_BYTES = 4 + N8 + 4 + N8 + 8 + 8;
// A binfile starts with its type, version and number of sections; each section with its id and size.
constexpr uint64_t BINFILE_HEADER_BYTES = 4 + 4 + 4;
constexpr uint64_t BINFILE_SECTION_HEADER_BYTES = 4 + 8;

std::invalid_argument invalid(const char *function, const std::string &message) {
    return std::invalid_argument(std::string("Srs::") + function + ": " + message);
}

// Runs a call on BinFile, which throws standard exceptions: std::system_error when the OS refuses
// (open, fstat, pread), and other types for what the file holds (the wrong type or version,
// sections beyond its end, a missing section). Rethrows them as IoError and FormatError, naming
// `path`. Keep `call` to BinFile itself, so that no other failure is taken for one of the file.
template <typename Call>
auto onBinFile(const std::string &path, Call call) -> decltype(call()) {
    try {
        return call();
    } catch (const std::bad_alloc &) {
        throw;
    } catch (const std::system_error &e) {
        throw IoError(path + ": " + e.what());
    } catch (const std::exception &e) {
        throw FormatError(path + ": " + e.what());
    }
}

// Opened in BinFile's direct-read mode, which reads a section's bytes only when asked: a ptau can
// hold 2^29 points, of which the SRS may need a handful. In that mode only readSectionTo reads.
std::unique_ptr<BinFile> openBinFile(const std::string &path, const char *type, uint32_t version) {
    return onBinFile(path, [&] { return std::make_unique<BinFile>(path, type, version, true); });
}

uint64_t sectionSize(BinFile &file, const std::string &path, uint32_t id, const char *contents) {
    if (!file.sectionExists(id)) {
        throw FormatError(path + ": no section " + std::to_string(id) + " (" + contents + ")");
    }
    return onBinFile(path, [&] { return file.getSectionSize(id); });
}

// The first `bytes` bytes of section `id`.
void readSection(BinFile &file, const std::string &path, uint32_t id, void *out, uint64_t bytes) {
    onBinFile(path, [&] { file.readSectionTo(out, id, 0, bytes); });
}

uint64_t readLittleEndian(const uint8_t *bytes, size_t n) {
    uint64_t value = 0;
    for (size_t i = n; i > 0; --i) {
        value = (value << 8) | bytes[i - 1];
    }
    return value;
}

// ffiasm's raw modulus (64-bit limbs, least significant first) as N8 little-endian bytes.
void modulusBytes(const uint64_t *modulus, uint8_t out[N8]) {
    for (size_t i = 0; i < N8; ++i) {
        out[i] = static_cast<uint8_t>(modulus[i / 8] >> (8 * (i % 8)));
    }
}

bool isModulus(const uint8_t bytes[N8], const uint64_t *modulus) {
    uint8_t expected[N8];
    modulusBytes(modulus, expected);
    return std::memcmp(bytes, expected, N8) == 0;
}

std::string pointsMessage(uint64_t bytes, uint64_t pointBytes, const char *group) {
    return std::to_string(bytes) + " bytes, not a whole number of " + std::to_string(pointBytes) + "-byte " + group +
           " points";
}

// Whether the limbs of `e` are below q: the Montgomery form of an element of Fq as ffiasm keeps
// it, and the only one its arithmetic is defined on.
bool isReduced(const Engine::F1Element &e) {
    for (int limb = RawFq::N64 - 1; limb >= 0; --limb) {
        if (e.v[limb] != Fq_rawq[limb]) {
            return e.v[limb] < Fq_rawq[limb];
        }
    }
    return false;
}

// Whether `p` is a point of G1: reduced coordinates with y^2 = x^3 + 3. BN254's G1 has cofactor 1.
// (0, 0), ffiasm's affine point at infinity, is not: no power of τ != 0 is that point.
bool isValidG1(const G1PointAffine &p) {
    if (!isReduced(p.x) || !isReduced(p.y)) {
        return false;
    }
    Engine &E = Engine::engine;
    Engine::F1Element y2, x3, rhs;
    E.f1.square(y2, p.y);
    E.f1.square(x3, p.x);
    E.f1.mul(x3, x3, p.x);
    E.f1.add(rhs, x3, E.g1.b());
    return E.f1.eq(y2, rhs);
}

// Whether `p` is on the twist y^2 = x^3 + 3/(9+u) with reduced coordinates. That is not enough to
// be a point of G2, the order-r subgroup of the twist (whose cofactor is not 1): inG2 checks that.
bool isValidG2(const G2PointAffine &p) {
    if (!isReduced(p.x.a) || !isReduced(p.x.b) || !isReduced(p.y.a) || !isReduced(p.y.b)) {
        return false;
    }
    Engine &E = Engine::engine;
    // F2Field takes its operands by non-const reference.
    Engine::F2Element x = p.x, y = p.y, y2, x2, x3, rhs;
    E.f2.square(y2, y);
    E.f2.square(x2, x);
    E.f2.mul(x3, x2, x);
    E.f2.add(rhs, x3, E.g2.b());
    return E.f2.eq(y2, rhs);
}

// Whether `p`, a point of the twist (isValidG2), is in G2: r·p is the point at infinity. The JS
// verifier refuses a vkey whose [τ]₂ is not (pilfflonk/js/src/elements.js, g2FromObject), so the
// setup must not write one. A point of the twist outside G2 is not a power of any τ: only a pairing
// check against [τ]₁ tells a wrong τ from the right one, but this is a malformed file.
bool inG2(const G2PointAffine &p) {
    Engine &E = Engine::engine;
    uint8_t r[N8];
    modulusBytes(Fr_rawq, r);
    // Curve::mulByScalar takes its base by non-const reference.
    G2PointAffine base = p;
    Engine::G2Point product;
    E.g2.mulByScalar(product, base, r, N8);
    return E.g2.isZero(product);
}

// The index of the first point that is not a point of G1, or n if every one is.
uint64_t firstInvalidG1(const G1PointAffine *points, uint64_t n) {
    uint64_t first = n;
#pragma omp parallel for reduction(min : first)
    for (uint64_t i = 0; i < n; ++i) {
        if (!isValidG1(points[i])) {
            first = std::min(first, i);
        }
    }
    return first;
}

// Section 1 of the ptau and the sizes of sections 2 and 3.
PtauHeader readHeaderOf(BinFile &file, const std::string &path) {
    const uint64_t headerBytes = sectionSize(file, path, PTAU_HEADER_SECTION, "the header");
    if (headerBytes < 4) {
        throw FormatError(path + ": the header (section 1) has " + std::to_string(headerBytes) + " bytes");
    }
    // n8q first, whatever the size, to tell a ptau of another curve from a broken one.
    uint8_t header[PTAU_HEADER_BYTES];
    readSection(file, path, PTAU_HEADER_SECTION, header, 4);
    const uint64_t n8q = readLittleEndian(header, 4);
    if (n8q != N8) {
        throw FormatError(path + ": n8q = " + std::to_string(n8q) + ", not the " + std::to_string(N8) +
                          " bytes of BN254's base field: not a BN254 ptau");
    }
    if (headerBytes != PTAU_HEADER_BYTES) {
        throw FormatError(path + ": the header (section 1) has " + std::to_string(headerBytes) + " bytes, not " +
                          std::to_string(PTAU_HEADER_BYTES));
    }
    readSection(file, path, PTAU_HEADER_SECTION, header, sizeof(header));
    if (!isModulus(header + 4, Fq_rawq)) {
        throw FormatError(path + ": q is not BN254's base field modulus: not a BN254 ptau");
    }

    PtauHeader result;
    result.power = static_cast<uint32_t>(readLittleEndian(header + 4 + N8, 4));
    result.ceremonyPower = static_cast<uint32_t>(readLittleEndian(header + 8 + N8, 4));

    const uint64_t g1Bytes = sectionSize(file, path, PTAU_G1_SECTION, "[τ^i]₁");
    if (g1Bytes % SRS_G1_BYTES != 0) {
        throw FormatError(path + ": section 2 ([τ^i]₁) has " + pointsMessage(g1Bytes, SRS_G1_BYTES, "G1"));
    }
    const uint64_t g2Bytes = sectionSize(file, path, PTAU_G2_SECTION, "[τ^i]₂");
    if (g2Bytes % SRS_G2_BYTES != 0) {
        throw FormatError(path + ": section 3 ([τ^i]₂) has " + pointsMessage(g2Bytes, SRS_G2_BYTES, "G2"));
    }
    result.nG1 = g1Bytes / SRS_G1_BYTES;
    result.nG2 = g2Bytes / SRS_G2_BYTES;
    return result;
}

} // namespace

PtauHeader readPtauHeader(const std::string &path) {
#ifndef __USE_ASSEMBLY__
    throw std::runtime_error("reading a ptau needs ffiasm's assembly backend, not built on this platform");
#else
    const std::unique_ptr<BinFile> file = openBinFile(path, PTAU_TYPE, PTAU_VERSION);
    return readHeaderOf(*file, path);
#endif
}

Srs::Srs(uint64_t nG1) : nPowers(nG1), g1Powers(new G1PointAffine[nG1]) {}

Srs Srs::fromPtau(const std::string &ptauPath, uint64_t nG1) {
    if (nG1 == 0) {
        throw invalid("fromPtau", "nG1 = 0: an SRS holds at least [1]₁");
    }
    if (nG1 > MAX_SRS_G1) {
        throw invalid("fromPtau", "nG1 = " + std::to_string(nG1) + " exceeds " + std::to_string(MAX_SRS_G1) +
                                      ", the most points ffiasm's MSM takes");
    }
#ifndef __USE_ASSEMBLY__
    throw std::runtime_error("the SRS needs ffiasm's assembly backend, not built on this platform");
#else
    const std::unique_ptr<BinFile> file = openBinFile(ptauPath, PTAU_TYPE, PTAU_VERSION);
    const PtauHeader header = readHeaderOf(*file, ptauPath);
    if (header.nG1 < nG1) {
        throw invalid("fromPtau", ptauPath + " holds " + std::to_string(header.nG1) + " powers [τ^i]₁, fewer than the " +
                                      std::to_string(nG1) + " requested");
    }
    if (header.nG2 < N_G2) {
        throw FormatError(ptauPath + ": section 3 holds " + std::to_string(header.nG2) +
                          " powers [τ^i]₂, fewer than [1]₂ and [τ]₂");
    }

    Srs srs(nG1);
    readSection(*file, ptauPath, PTAU_G1_SECTION, srs.g1Powers.get(), nG1 * SRS_G1_BYTES);
    readSection(*file, ptauPath, PTAU_G2_SECTION, srs.g2Powers, N_G2 * SRS_G2_BYTES);
    srs.checkPoints(ptauPath);
    return srs;
#endif
}

Srs Srs::load(const std::string &path) {
#ifndef __USE_ASSEMBLY__
    throw std::runtime_error("the SRS needs ffiasm's assembly backend, not built on this platform");
#else
    const std::unique_ptr<BinFile> file = openBinFile(path, SRS_TYPE, SRS_VERSION);

    const uint64_t headerBytes = sectionSize(*file, path, SRS_HEADER_SECTION, "the header");
    if (headerBytes != SRS_HEADER_BYTES) {
        throw FormatError(path + ": the header (section 1) has " + std::to_string(headerBytes) + " bytes, not " +
                          std::to_string(SRS_HEADER_BYTES));
    }
    uint8_t header[SRS_HEADER_BYTES];
    readSection(*file, path, SRS_HEADER_SECTION, header, sizeof(header));
    const uint8_t *field = header;
    if (readLittleEndian(field, 4) != N8 || !isModulus(field + 4, Fq_rawq)) {
        throw FormatError(path + ": n8q and q are not those of BN254's base field");
    }
    field += 4 + N8;
    if (readLittleEndian(field, 4) != N8 || !isModulus(field + 4, Fr_rawq)) {
        throw FormatError(path + ": n8r and r are not those of BN254's scalar field");
    }
    field += 4 + N8;
    const uint64_t nG1 = readLittleEndian(field, 8);
    const uint64_t nG2 = readLittleEndian(field + 8, 8);
    if (nG1 == 0 || nG1 > MAX_SRS_G1) {
        throw FormatError(path + ": nG1 = " + std::to_string(nG1) + ", not between 1 and " +
                          std::to_string(MAX_SRS_G1));
    }
    if (nG2 != N_G2) {
        throw FormatError(path + ": nG2 = " + std::to_string(nG2) + ", not " + std::to_string(N_G2));
    }
    const uint64_t g1Bytes = sectionSize(*file, path, SRS_G1_SECTION, "[τ^i]₁");
    if (g1Bytes != nG1 * SRS_G1_BYTES) {
        throw FormatError(path + ": section 2 ([τ^i]₁) has " + std::to_string(g1Bytes) + " bytes, not the " +
                          std::to_string(nG1 * SRS_G1_BYTES) + " of nG1 = " + std::to_string(nG1) + " points");
    }
    const uint64_t g2Bytes = sectionSize(*file, path, SRS_G2_SECTION, "[τ^i]₂");
    if (g2Bytes != N_G2 * SRS_G2_BYTES) {
        throw FormatError(path + ": section 3 ([τ^i]₂) has " + std::to_string(g2Bytes) + " bytes, not the " +
                          std::to_string(N_G2 * SRS_G2_BYTES) + " of [1]₂ and [τ]₂");
    }

    Srs srs(nG1);
    readSection(*file, path, SRS_G1_SECTION, srs.g1Powers.get(), g1Bytes);
    readSection(*file, path, SRS_G2_SECTION, srs.g2Powers, g2Bytes);
    srs.checkPoints(path);
    return srs;
#endif
}

void Srs::save(const std::string &path) const {
    const std::string temporary = path + ".tmp";
    // BinFileWriter never checks the std::ofstream it writes to, so it cannot fail. The file is
    // created here first, where a failure comes with an errno, and its size checked once written:
    // a write that failed midway leaves it short.
    const int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) {
        throw IoError(temporary + ": " + std::strerror(errno));
    }
    ::close(fd);

    uint8_t q[N8], r[N8];
    modulusBytes(Fq_rawq, q);
    modulusBytes(Fr_rawq, r);
    // BinFileWriter::write takes a non-const buffer.
    G2PointAffine g2[N_G2];
    std::copy(g2Powers, g2Powers + N_G2, g2);
    try {
        BinFileUtils::BinFileWriter writer(temporary, SRS_TYPE, SRS_VERSION, SRS_N_SECTIONS);
        writer.startWriteSection(SRS_HEADER_SECTION);
        writer.writeU32LE(N8);
        writer.write(q, N8);
        writer.writeU32LE(N8);
        writer.write(r, N8);
        writer.writeU64LE(nPowers);
        writer.writeU64LE(N_G2);
        writer.endWriteSection();
        writer.startWriteSection(SRS_G1_SECTION);
        writer.write(g1Powers.get(), nPowers * SRS_G1_BYTES);
        writer.endWriteSection();
        writer.startWriteSection(SRS_G2_SECTION);
        writer.write(g2, N_G2 * SRS_G2_BYTES);
        writer.endWriteSection();
        writer.close();
    } catch (...) {
        // Out of memory, say: the temporary file must not outlive the failure.
        std::remove(temporary.c_str());
        throw;
    }

    const uint64_t expected = BINFILE_HEADER_BYTES + SRS_N_SECTIONS * BINFILE_SECTION_HEADER_BYTES + SRS_HEADER_BYTES +
                              nPowers * SRS_G1_BYTES + N_G2 * SRS_G2_BYTES;
    struct stat written;
    if (::stat(temporary.c_str(), &written) != 0 || static_cast<uint64_t>(written.st_size) != expected) {
        std::remove(temporary.c_str());
        throw IoError(temporary + ": could not write the " + std::to_string(expected) + " bytes of the SRS");
    }
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        const int error = errno;
        std::remove(temporary.c_str());
        throw IoError(path + ": " + std::strerror(error));
    }
}

void Srs::checkPoints(const std::string &source) const {
    Engine &E = Engine::engine;
    const uint64_t invalidG1 = firstInvalidG1(g1Powers.get(), nPowers);
    if (invalidG1 < nPowers) {
        throw FormatError(source + ": [τ^" + std::to_string(invalidG1) +
                          "]₁ is not a point of G1 (a coordinate not below q, or off the curve)");
    }
    G1PointAffine one = g1Powers[0];
    if (!E.g1.eq(one, E.g1.oneAffine())) {
        throw FormatError(source + ": [1]₁ is not the generator (1, 2) of G1");
    }
    // [1]₂ here; [τ]₂ below, with checkG2.
    if (!isValidG2(g2Powers[0])) {
        throw FormatError(source + ": [τ^0]₂ is not a point of the G2 twist (a coordinate not below q, or off the curve)");
    }
    G2PointAffine oneG2 = g2Powers[0];
    if (!E.g2.eq(oneG2, E.g2.oneAffine())) {
        throw FormatError(source + ": [1]₂ is not the generator of G2");
    }
    // [1]₂, the generator, is in G2; [τ]₂ must be too, and not the point at infinity (which the
    // twist's equation already refuses: (0, 0) is not on it; said here by its name).
    for (uint64_t i = 1; i < N_G2; ++i) {
        switch (checkG2(g2Powers[i])) {
        case G2Error::None:
            break;
        case G2Error::Infinity:
            throw FormatError(source + ": [τ^" + std::to_string(i) + "]₂ is the point at infinity (τ = 0)");
        case G2Error::NotOnTwist:
            throw FormatError(source + ": [τ^" + std::to_string(i) +
                              "]₂ is not a point of the G2 twist (a coordinate not below q, or off the curve)");
        case G2Error::NotInG2:
            throw FormatError(source + ": [τ^" + std::to_string(i) +
                              "]₂ is a point of the G2 twist not in the r-torsion group (r times it is not the "
                              "point at infinity)");
        }
    }
}

G2Error checkG2(const G2PointAffine &p) {
    Engine &E = Engine::engine;
    // Curve::isZero takes its point by non-const reference.
    G2PointAffine point = p;
    if (E.g2.isZero(point)) {
        return G2Error::Infinity;
    }
    if (!isValidG2(p)) {
        return G2Error::NotOnTwist;
    }
    if (!inG2(p)) {
        return G2Error::NotInG2;
    }
    return G2Error::None;
}

G1Point Srs::commit(const FrElement *coefs, uint64_t nCoefs) const {
    if (nCoefs > nPowers) {
        throw invalid("commit", std::to_string(nCoefs) + " coefficients exceed the " + std::to_string(nPowers) +
                                    " powers [τ^i]₁ of the SRS");
    }
    if (coefs == nullptr && nCoefs != 0) {
        throw invalid("commit", "coefs is null");
    }
#ifdef __USE_CUDA__
    if (device != nullptr) {
        return device->msm(coefs, nCoefs);
    }
#endif
    Engine &E = Engine::engine;
    // ffiasm's MSM reads each scalar as a little-endian integer, so it must be the canonical value:
    // Montgomery limbs would commit to p·2^256 instead (spec §4.4, "Escalars").
    std::unique_ptr<FrElement[]> scalars(new FrElement[nCoefs]);
#pragma omp parallel for
    for (uint64_t i = 0; i < nCoefs; ++i) {
        E.fr.fromMontgomery(scalars[i], coefs[i]);
    }
    G1Point result;
    E.g1.multiMulByScalar(result, g1Powers.get(), reinterpret_cast<uint8_t *>(scalars.get()), sizeof(FrElement),
                          static_cast<unsigned int>(nCoefs));
    return result;
}

} // namespace PilFflonk
