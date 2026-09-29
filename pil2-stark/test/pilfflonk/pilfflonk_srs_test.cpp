// Tests for PilFflonk::Srs and the SRS entry points of the C API: the ptau reader must return the
// file's points and only read what it needs, pilfflonk.srs.bin must round-trip in the format
// pilfflonk_srs.hpp documents, and every malformed file must come back as an exception of the
// right type, or a status, not an abort.
#include "pilfflonk_test.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

#include "alt_bn128.hpp"
#include "pilfflonk_api.hpp"
#include "pilfflonk_error.hpp"
#include "pilfflonk_srs.hpp"
#include "pilfflonk_test_ptau.hpp"

namespace PilFflonkTest {

namespace {

using Engine = AltBn128::Engine;
using FrElement = Engine::FrElement;
using G1PointAffine = Engine::G1PointAffine;
using G2PointAffine = Engine::G2PointAffine;
using PilFflonk::FormatError;
using PilFflonk::IoError;
using PilFflonk::Srs;
using Bytes = std::vector<uint8_t>;
using Sections = std::vector<std::pair<uint32_t, Bytes>>;

Engine &E = Engine::engine;

// The generators of G1 and G2, written out independently of ffiasm.
G1PointAffine g1Generator() {
    G1PointAffine g;
    E.f1.fromString(g.x, "1");
    E.f1.fromString(g.y, "2");
    return g;
}

G2PointAffine g2Generator() {
    G2PointAffine g;
    E.f1.fromString(g.x.a, "10857046999023057135944570762232829481370756359578518086990519993285655852781");
    E.f1.fromString(g.x.b, "11559732032986387107991004021392285783925812861821192530917403151452391805634");
    E.f1.fromString(g.y.a, "8495653923123431417604973247489272438418190587263600148770280649306958101930");
    E.f1.fromString(g.y.b, "4082367875863433681332203403145435568316851327593401208105741076214120093531");
    return g;
}

// Bit for bit: the files store points as ffiasm keeps them.
template <typename Point>
bool identical(const Point &a, const Point &b) {
    return std::memcmp(&a, &b, sizeof(Point)) == 0;
}

FrElement power(const FrElement &base, uint64_t exponent) {
    FrElement result = E.fr.one();
    for (uint64_t i = 0; i < exponent; ++i) {
        E.fr.mul(result, result, base);
    }
    return result;
}

// k·G1 or k·G2 for the generator, affine.
G1PointAffine g1Times(const FrElement &k) {
    FrElement canonical;
    E.fr.fromMontgomery(canonical, k);
    Engine::G1Point p;
    E.g1.mulByScalar(p, E.g1.oneAffine(), reinterpret_cast<uint8_t *>(canonical.v), 32);
    G1PointAffine affine;
    E.g1.copy(affine, p);
    return affine;
}

G2PointAffine g2Times(const FrElement &k) {
    FrElement canonical;
    E.fr.fromMontgomery(canonical, k);
    Engine::G2Point p;
    E.g2.mulByScalar(p, E.g2.oneAffine(), reinterpret_cast<uint8_t *>(canonical.v), 32);
    G2PointAffine affine;
    E.g2.copy(affine, p);
    return affine;
}

Bytes readFile(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    assert(in);
    return Bytes(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void writeFile(const std::string &path, const Bytes &bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    assert(out);
}

// A copy of `from` at `to`, with `bytes` written over it at `offset`.
void patchedCopy(const std::string &from, const std::string &to, uint64_t offset, const Bytes &bytes) {
    Bytes contents = readFile(from);
    assert(offset + bytes.size() <= contents.size());
    std::copy(bytes.begin(), bytes.end(), contents.begin() + offset);
    writeFile(to, contents);
}

Bytes littleEndian(uint64_t value, size_t n) {
    Bytes bytes(n);
    for (size_t i = 0; i < n; ++i) {
        bytes[i] = static_cast<uint8_t>(value >> (8 * i));
    }
    return bytes;
}

uint64_t readLittleEndian(const Bytes &bytes, uint64_t offset, size_t n) {
    uint64_t value = 0;
    for (size_t i = n; i > 0; --i) {
        value = (value << 8) | bytes[offset + i - 1];
    }
    return value;
}

bool exists(const std::string &path) {
    return access(path.c_str(), F_OK) == 0;
}

template <typename Exception, typename Call>
void expectThrows(Call call, const char *message) {
    try {
        call();
    } catch (const Exception &e) {
        if (std::strstr(e.what(), message) == nullptr) {
            std::fprintf(stderr, "unexpected message: %s\n", e.what());
            assert(!"the exception does not say what was expected");
        }
        return;
    }
    assert(!"expected an exception");
}

// A test ptau of nG1 points [τ^i]₁ and two [τ^i]₂, as sections.
Sections ptauSections(uint64_t nG1, uint64_t nG2 = 2) {
    const PtauSections s = testPtauSections(nG1, nG2, testTau());
    return {{1, s.header}, {2, s.g1}, {3, s.g2}};
}

// The byte offsets of a ptau written with its sections in order 1, 2, 3 (a 44-byte header).
constexpr uint64_t PTAU_Q = 12 + 12 + 4;
constexpr uint64_t PTAU_G1 = 12 + 12 + 44 + 12;

// The byte offsets of pilfflonk.srs.bin, as Srs::save writes it (sections 1, 2, 3).
constexpr uint64_t SRS_N8Q = 12 + 12;
constexpr uint64_t SRS_Q = SRS_N8Q + 4;
constexpr uint64_t SRS_N8R = SRS_Q + 32;
constexpr uint64_t SRS_R = SRS_N8R + 4;
constexpr uint64_t SRS_NG1 = SRS_R + 32;
constexpr uint64_t SRS_NG2 = SRS_NG1 + 8;
constexpr uint64_t SRS_G1 = SRS_NG2 + 8 + 12;
uint64_t srsG2(uint64_t nG1) {
    return SRS_G1 + nG1 * 64 + 12;
}

void testTestPtauHeader() {
    TestDir dir;
    // The smallest power of a real ptau with that many powers: 2^(p+1) - 1 >= nG1, 2^p >= nG2.
    const std::vector<std::pair<std::pair<uint64_t, uint64_t>, uint32_t>> powers = {
        {{1, 1}, 0}, {{1, 2}, 1}, {{3, 2}, 1}, {{4, 2}, 2}, {{7, 2}, 2}, {{8, 2}, 3}, {{64, 2}, 6}, {{2, 5}, 3}};
    for (const auto &test : powers) {
        const PtauSections s = testPtauSections(test.first.first, test.first.second, testTau());
        assert(s.power == test.second);
    }

    const std::string path = dir.file("header.ptau");
    writeTestPtau(path, 5, 3);
    const PilFflonk::PtauHeader header = PilFflonk::readPtauHeader(path);
    assert(header.power == 2 && header.ceremonyPower == 2 && header.nG1 == 5 && header.nG2 == 3);

    // snarkjs's header: n8q, q, power, ceremonyPower.
    const Bytes file = readFile(path);
    assert(std::memcmp(file.data(), "ptau", 4) == 0);
    assert(readLittleEndian(file, 4, 4) == 1 && readLittleEndian(file, 8, 4) == 3);
    assert(readLittleEndian(file, 12, 4) == 1 && readLittleEndian(file, 16, 8) == 44);
    assert(readLittleEndian(file, 24, 4) == 32);
    assert(std::memcmp(file.data() + PTAU_Q, Bytes32(Q_HEX).bytes, 32) == 0);
    assert(readLittleEndian(file, PTAU_Q + 32, 4) == 2 && readLittleEndian(file, PTAU_Q + 36, 4) == 2);
}

void testFromPtauReadsThePowers() {
    TestDir dir;
    const std::string path = dir.file("powers.ptau");
    writeTestPtau(path, 64, 4);
    const FrElement tau = testTau();

    const Srs srs = Srs::fromPtau(path, 40);
    assert(srs.nG1() == 40);
    // [1]₁ and [1]₂ are the generators: the points are ffiasm's, and ffiasm's are snarkjs's.
    assert(identical(srs.g1(0), g1Generator()));
    assert(identical(srs.g2(0), g2Generator()));
    for (uint64_t i : {1, 2, 3, 20, 39}) {
        assert(identical(srs.g1(i), g1Times(power(tau, i))));
    }
    assert(identical(srs.g2(1), g2Times(tau)));

    // Exactly the file's first points.
    const Bytes file = readFile(path);
    for (uint64_t i = 0; i < 40; ++i) {
        assert(std::memcmp(&srs.g1(i), file.data() + PTAU_G1 + 64 * i, 64) == 0);
    }

    const Srs all = Srs::fromPtau(path, 64);
    assert(all.nG1() == 64 && identical(all.g1(63), g1Times(power(tau, 63))));
    const Srs one = Srs::fromPtau(path, 1);
    assert(one.nG1() == 1 && identical(one.g1(0), g1Generator()) && identical(one.g2(1), g2Times(tau)));

    expectThrows<std::invalid_argument>([&] { Srs::fromPtau(path, 65); },
                                        "holds 64 powers [τ^i]₁, fewer than the 65 requested");
    expectThrows<std::invalid_argument>([&] { Srs::fromPtau(path, 0); }, "nG1 = 0");
    expectThrows<std::invalid_argument>([&] { Srs::fromPtau(path, uint64_t(1) << 32); },
                                        "nG1 = 4294967296 exceeds 4294967295");
    // Checked before the file is opened.
    expectThrows<std::invalid_argument>([&] { Srs::fromPtau(dir.path() + "/missing.ptau", 0); }, "nG1 = 0");
}

// A ptau whose section 2 claims 2^28 points, as the largest real ones do, in a sparse file that
// only holds the first few: reading those must not touch the rest. Also, the sections are not in
// the usual order: readers must go by their ids.
void testFromPtauReadsOnlyWhatItNeeds() {
    TestDir dir;
    const std::string path = dir.file("sparse.ptau");
    const PtauSections s = testPtauSections(16, 2, testTau());
    writeBinFile(path, "ptau", 1, {{1, s.header}, {3, s.g2}, {2, s.g1}});

    const uint64_t nClaimed = uint64_t(1) << 28;
    const uint64_t g1SizeField = 12 + (12 + 44) + (12 + 256) + 4;
    Bytes file = readFile(path);
    assert(readLittleEndian(file, g1SizeField - 4, 4) == 2 && readLittleEndian(file, g1SizeField, 8) == 16 * 64);
    const Bytes size = littleEndian(nClaimed * 64, 8);
    std::copy(size.begin(), size.end(), file.begin() + g1SizeField);
    writeFile(path, file);
    assert(truncate(path.c_str(), static_cast<off_t>(g1SizeField + 8 + nClaimed * 64)) == 0);

    assert(PilFflonk::readPtauHeader(path).nG1 == nClaimed);
    const Srs srs = Srs::fromPtau(path, 16);
    assert(srs.nG1() == 16 && identical(srs.g1(15), g1Times(power(testTau(), 15))));
    assert(identical(srs.g2(1), g2Times(testTau())));
    // Beyond what was written, the file reads as zeros: (0, 0) is not a point of G1.
    expectThrows<FormatError>([&] { Srs::fromPtau(path, 17); }, "[τ^16]₁ is not a point of G1");
}

void testPtauErrors() {
    TestDir dir;
    const std::string good = dir.file("good.ptau");
    writeTestPtau(good, 8);
    const std::string bad = dir.file("bad.ptau");
    const auto fromPtau = [&] { Srs::fromPtau(bad, 8); };

    expectThrows<IoError>([&] { Srs::fromPtau(dir.path() + "/missing.ptau", 8); }, "missing.ptau: open");
    expectThrows<IoError>([&] { PilFflonk::readPtauHeader(dir.path() + "/missing.ptau"); }, "open");

    // The container.
    writeBinFile(bad, "zkey", 1, ptauSections(8));
    expectThrows<FormatError>(fromPtau, "Invalid file type");
    writeBinFile(bad, "ptau", 2, ptauSections(8));
    expectThrows<FormatError>(fromPtau, "Invalid version");
    writeFile(bad, Bytes{'p', 't', 'a', 'u', 1, 0});
    expectThrows<FormatError>(fromPtau, "Failed to read BinFile header");
    Bytes cut = readFile(good);
    cut.pop_back();
    writeFile(bad, cut);
    expectThrows<FormatError>(fromPtau, "Section data exceeds file size");

    // The header.
    Sections sections = ptauSections(8);
    sections.erase(sections.begin());
    writeBinFile(bad, "ptau", 1, sections);
    expectThrows<FormatError>(fromPtau, "no section 1 (the header)");
    sections = ptauSections(8);
    sections[0].second[0] = 48;
    writeBinFile(bad, "ptau", 1, sections);
    expectThrows<FormatError>(fromPtau, "n8q = 48, not the 32 bytes of BN254's base field: not a BN254 ptau");
    sections = ptauSections(8);
    sections[0].second.push_back(0);
    writeBinFile(bad, "ptau", 1, sections);
    expectThrows<FormatError>(fromPtau, "the header (section 1) has 45 bytes, not 44");
    sections[0].second = {32, 0, 0};
    writeBinFile(bad, "ptau", 1, sections);
    expectThrows<FormatError>(fromPtau, "the header (section 1) has 3 bytes");
    patchedCopy(good, bad, PTAU_Q, {0x48}); // q + 1
    expectThrows<FormatError>(fromPtau, "q is not BN254's base field modulus");
    const Bytes32 r(R_HEX);
    patchedCopy(good, bad, PTAU_Q, Bytes(r.bytes, r.bytes + 32));
    expectThrows<FormatError>(fromPtau, "q is not BN254's base field modulus");

    // The points' sections.
    sections = ptauSections(8);
    sections.erase(sections.begin() + 1);
    writeBinFile(bad, "ptau", 1, sections);
    expectThrows<FormatError>(fromPtau, "no section 2 ([τ^i]₁)");
    sections = ptauSections(8);
    sections.pop_back();
    writeBinFile(bad, "ptau", 1, sections);
    expectThrows<FormatError>(fromPtau, "no section 3 ([τ^i]₂)");
    sections = ptauSections(8);
    sections[1].second.pop_back();
    writeBinFile(bad, "ptau", 1, sections);
    expectThrows<FormatError>(fromPtau, "section 2 ([τ^i]₁) has 511 bytes, not a whole number of 64-byte G1 points");
    sections = ptauSections(8);
    sections[2].second.push_back(0);
    writeBinFile(bad, "ptau", 1, sections);
    expectThrows<FormatError>(fromPtau, "section 3 ([τ^i]₂) has 257 bytes, not a whole number of 128-byte G2");
    writeBinFile(bad, "ptau", 1, ptauSections(8, 1));
    expectThrows<FormatError>(fromPtau, "section 3 holds 1 powers [τ^i]₂, fewer than [1]₂ and [τ]₂");

    // The points.
    patchedCopy(good, bad, PTAU_G1 + 5 * 64 + 32, {static_cast<uint8_t>(readFile(good)[PTAU_G1 + 5 * 64 + 32] ^ 1)});
    expectThrows<FormatError>(fromPtau, "[τ^5]₁ is not a point of G1");
    // Beyond the points read, nothing is checked: they were never read.
    const Srs five = Srs::fromPtau(bad, 5);
    assert(five.nG1() == 5);
    const Bytes32 q(Q_HEX);
    patchedCopy(good, bad, PTAU_G1 + 3 * 64, Bytes(q.bytes, q.bytes + 32)); // x = q: not reduced
    expectThrows<FormatError>(fromPtau, "[τ^3]₁ is not a point of G1");
    sections = ptauSections(9);
    sections[1].second.erase(sections[1].second.begin(), sections[1].second.begin() + 64); // [τ]₁ first
    writeBinFile(bad, "ptau", 1, sections);
    expectThrows<FormatError>(fromPtau, "[1]₁ is not the generator (1, 2) of G1");
    sections = ptauSections(8);
    sections[2].second[128 + 64] ^= 1; // [τ]₂'s y
    writeBinFile(bad, "ptau", 1, sections);
    expectThrows<FormatError>(fromPtau, "[τ^1]₂ is not a point of the G2 twist");
    sections = ptauSections(8, 3);
    sections[2].second.erase(sections[2].second.begin(), sections[2].second.begin() + 128); // [τ]₂ first
    writeBinFile(bad, "ptau", 1, sections);
    expectThrows<FormatError>(fromPtau, "[1]₂ is not the generator of G2");
}

void testSrsFileRoundTrip() {
    TestDir dir;
    const std::string ptau = dir.file("roundtrip.ptau");
    writeTestPtau(ptau, 40);
    const std::string path = dir.file("pilfflonk.srs.bin");

    const Srs srs = Srs::fromPtau(ptau, 33);
    srs.save(path);
    assert(!exists(path + ".tmp"));
    const Srs loaded = Srs::load(path);
    assert(loaded.nG1() == 33);
    for (uint64_t i = 0; i < 33; ++i) {
        assert(identical(loaded.g1(i), srs.g1(i)));
    }
    assert(identical(loaded.g2(0), g2Generator()) && identical(loaded.g2(1), srs.g2(1)));
    assert(identical(loaded.g1(0), g1Generator()));

    // The format of pilfflonk_srs.hpp, byte for byte: the points are the ptau's, copied.
    const Bytes file = readFile(path);
    const Bytes ptauFile = readFile(ptau);
    assert(file.size() == 12 + (12 + 88) + (12 + 33 * 64) + (12 + 256));
    assert(std::memcmp(file.data(), "pfsr", 4) == 0);
    assert(readLittleEndian(file, 4, 4) == 1 && readLittleEndian(file, 8, 4) == 3);
    assert(readLittleEndian(file, 12, 4) == 1 && readLittleEndian(file, 16, 8) == 88);
    assert(readLittleEndian(file, SRS_N8Q, 4) == 32 && std::memcmp(file.data() + SRS_Q, Bytes32(Q_HEX).bytes, 32) == 0);
    assert(readLittleEndian(file, SRS_N8R, 4) == 32 && std::memcmp(file.data() + SRS_R, Bytes32(R_HEX).bytes, 32) == 0);
    assert(readLittleEndian(file, SRS_NG1, 8) == 33 && readLittleEndian(file, SRS_NG2, 8) == 2);
    assert(readLittleEndian(file, SRS_G1 - 12, 4) == 2 && readLittleEndian(file, SRS_G1 - 8, 8) == 33 * 64);
    assert(std::memcmp(file.data() + SRS_G1, ptauFile.data() + PTAU_G1, 33 * 64) == 0);
    const uint64_t g2 = srsG2(33);
    assert(readLittleEndian(file, g2 - 12, 4) == 3 && readLittleEndian(file, g2 - 8, 8) == 256);
    assert(std::memcmp(file.data() + g2, ptauFile.data() + PTAU_G1 + 40 * 64 + 12, 256) == 0);

    // Saving again replaces the file.
    Srs::fromPtau(ptau, 2).save(path);
    assert(Srs::load(path).nG1() == 2 && !exists(path + ".tmp"));
}

void testSrsFileErrors() {
    TestDir dir;
    const std::string ptau = dir.file("errors.ptau");
    writeTestPtau(ptau, 16);
    const std::string good = dir.file("good.srs.bin");
    const Srs srs = Srs::fromPtau(ptau, 8);
    srs.save(good);
    const Bytes goodFile = readFile(good);
    const std::string bad = dir.file("bad.srs.bin");
    const auto load = [&] { Srs::load(bad); };
    const auto expectPatched = [&](uint64_t offset, const Bytes &bytes, const char *message) {
        patchedCopy(good, bad, offset, bytes);
        expectThrows<FormatError>(load, message);
    };
    const auto flipped = [&](uint64_t offset) { return Bytes{static_cast<uint8_t>(goodFile[offset] ^ 1)}; };
    const auto slice = [&](uint64_t offset, uint64_t n) {
        return Bytes(goodFile.begin() + offset, goodFile.begin() + offset + n);
    };

    expectThrows<IoError>([&] { Srs::load(dir.path() + "/missing.srs.bin"); }, "missing.srs.bin: open");
    expectThrows<FormatError>([&] { Srs::load(ptau); }, "Invalid file type");
    expectPatched(4, {2}, "Invalid version");
    writeFile(bad, slice(0, goodFile.size() - 1));
    expectThrows<FormatError>(load, "Section data exceeds file size");

    // The header.
    expectPatched(SRS_N8Q, {31}, "n8q and q are not those of BN254's base field");
    expectPatched(SRS_Q + 31, {0x31}, "n8q and q are not those of BN254's base field");
    expectPatched(SRS_N8R, {48}, "n8r and r are not those of BN254's scalar field");
    expectPatched(SRS_R, {0x02}, "n8r and r are not those of BN254's scalar field");
    expectPatched(SRS_NG1, littleEndian(0, 8), "nG1 = 0, not between 1 and 4294967295");
    expectPatched(SRS_NG1, littleEndian(uint64_t(1) << 32, 8), "nG1 = 4294967296, not between 1 and 4294967295");
    expectPatched(SRS_NG1, littleEndian(9, 8), "section 2 ([τ^i]₁) has 512 bytes, not the 576 of nG1 = 9 points");
    expectPatched(SRS_NG1, littleEndian(7, 8), "section 2 ([τ^i]₁) has 512 bytes, not the 448 of nG1 = 7 points");
    expectPatched(SRS_NG2, littleEndian(3, 8), "nG2 = 3, not 2");

    // The sections.
    const Bytes header = slice(SRS_N8Q, 88);
    const Bytes g1 = slice(SRS_G1, 8 * 64);
    const Bytes g2 = slice(srsG2(8), 256);
    writeBinFile(bad, "pfsr", 1, {{1, Bytes(header.begin(), header.end() - 1)}, {2, g1}, {3, g2}});
    expectThrows<FormatError>(load, "the header (section 1) has 87 bytes, not 88");
    writeBinFile(bad, "pfsr", 1, {{1, header}, {2, g1}});
    expectThrows<FormatError>(load, "no section 3 ([τ^i]₂)");
    writeBinFile(bad, "pfsr", 1, {{2, g1}, {3, g2}});
    expectThrows<FormatError>(load, "no section 1 (the header)");
    writeBinFile(bad, "pfsr", 1, {{1, header}, {3, g2}});
    expectThrows<FormatError>(load, "no section 2 ([τ^i]₁)");
    writeBinFile(bad, "pfsr", 1, {{1, header}, {2, g1}, {3, Bytes(g2.begin(), g2.begin() + 128)}});
    expectThrows<FormatError>(load, "section 3 ([τ^i]₂) has 128 bytes, not the 256 of [1]₂ and [τ]₂");
    writeBinFile(bad, "pfsr", 1, {{3, g2}, {1, header}, {2, g1}}); // any order
    assert(Srs::load(bad).nG1() == 8);

    // The points.
    expectPatched(SRS_G1 + 5 * 64 + 32, flipped(SRS_G1 + 5 * 64 + 32), "[τ^5]₁ is not a point of G1");
    const Bytes32 q(Q_HEX);
    expectPatched(SRS_G1 + 7 * 64 + 32, Bytes(q.bytes, q.bytes + 32), "[τ^7]₁ is not a point of G1");
    expectPatched(SRS_G1, slice(SRS_G1 + 64, 64), "[1]₁ is not the generator (1, 2) of G1");
    expectPatched(srsG2(8) + 128 + 100, flipped(srsG2(8) + 128 + 100), "[τ^1]₂ is not a point of the G2 twist");
    expectPatched(srsG2(8), slice(srsG2(8) + 128, 128), "[1]₂ is not the generator of G2");

    // Writing.
    const std::string nowhere = dir.path() + "/no/such/dir/pilfflonk.srs.bin";
    expectThrows<IoError>([&] { srs.save(nowhere); }, "No such file or directory");
    assert(!exists(nowhere) && !exists(nowhere + ".tmp"));
    // A directory in the way: the temporary file is written, the rename fails, and it is removed.
    const std::string directory = dir.path() + "/a-directory";
    assert(mkdir(directory.c_str(), 0700) == 0);
    expectThrows<IoError>([&] { srs.save(directory); }, "a-directory: Is a directory");
    assert(!exists(directory + ".tmp"));
    assert(rmdir(directory.c_str()) == 0);
}

bool lastErrorMentions(const char *text) {
    return std::strstr(pilfflonk_last_error(), text) != nullptr;
}

void expectStatus(int status, int expected, const char *text) {
    if (status != expected || pilfflonk_last_status() != expected || !lastErrorMentions(text)) {
        std::fprintf(stderr, "status %d (last %d), expected %d: %s\n", status, pilfflonk_last_status(), expected,
                     pilfflonk_last_error());
        assert(!"unexpected status");
    }
}

void expectOk(int status) {
    assert(status == PILFFLONK_OK);
    assert(pilfflonk_last_status() == PILFFLONK_OK && pilfflonk_last_error()[0] == '\0');
}

void expectLoadFails(const char *path, int expected, const char *text) {
    assert(pilfflonk_srs_load(path) == nullptr);
    expectStatus(pilfflonk_last_status(), expected, text);
    assert(lastErrorMentions("pilfflonk_srs_load"));
}

void testApi() {
    TestDir dir;
    const std::string ptau = dir.file("api.ptau");
    writeTestPtau(ptau, 64);
    const std::string srs = dir.file("api.srs.bin");
    const std::string zkey = dir.file("api.zkey");
    writeBinFile(zkey, "zkey", 1, ptauSections(8));
    const std::string missing = dir.path() + "/missing";
    const std::string nowhere = dir.path() + "/no/such/dir/srs.bin";

    expectStatus(pilfflonk_srs_from_ptau(nullptr, 8, srs.c_str()), PILFFLONK_ERR_INVALID_ARGUMENT,
                 "pilfflonk_srs_from_ptau: ptau_path is NULL");
    expectStatus(pilfflonk_srs_from_ptau(ptau.c_str(), 8, nullptr), PILFFLONK_ERR_INVALID_ARGUMENT,
                 "srs_path is NULL");
    expectStatus(pilfflonk_srs_from_ptau(ptau.c_str(), 0, srs.c_str()), PILFFLONK_ERR_INVALID_ARGUMENT, "nG1 = 0");
    expectStatus(pilfflonk_srs_from_ptau(ptau.c_str(), 65, srs.c_str()), PILFFLONK_ERR_INVALID_ARGUMENT,
                 "fewer than the 65 requested");
    expectStatus(pilfflonk_srs_from_ptau(missing.c_str(), 8, srs.c_str()), PILFFLONK_ERR_IO, "missing: open");
    expectStatus(pilfflonk_srs_from_ptau(zkey.c_str(), 8, srs.c_str()), PILFFLONK_ERR_FORMAT, "Invalid file type");
    expectStatus(pilfflonk_srs_from_ptau(ptau.c_str(), 8, nowhere.c_str()), PILFFLONK_ERR_IO,
                 "No such file or directory");
    assert(!exists(srs));
    expectOk(pilfflonk_srs_from_ptau(ptau.c_str(), 64, srs.c_str()));

    expectLoadFails(nullptr, PILFFLONK_ERR_INVALID_ARGUMENT, "srs_path is NULL");
    expectLoadFails(missing.c_str(), PILFFLONK_ERR_IO, "missing: open");
    expectLoadFails(ptau.c_str(), PILFFLONK_ERR_FORMAT, "Invalid file type");
    void *handle = pilfflonk_srs_load(srs.c_str());
    assert(handle != nullptr);
    expectOk(pilfflonk_last_status());
    const Srs &loaded = *static_cast<const Srs *>(handle);
    assert(loaded.nG1() == 64 && identical(loaded.g1(0), g1Generator()));
    assert(identical(loaded.g1(63), g1Times(power(testTau(), 63))) && identical(loaded.g2(1), g2Times(testTau())));

    // Freeing clears the last error, like every other call.
    expectLoadFails(missing.c_str(), PILFFLONK_ERR_IO, "open");
    pilfflonk_srs_free(handle);
    expectOk(pilfflonk_last_status());
    pilfflonk_srs_free(nullptr);
}

} // namespace

void runSrsTests() {
    testTestPtauHeader();
    testFromPtauReadsThePowers();
    testFromPtauReadsOnlyWhatItNeeds();
    testPtauErrors();
    testSrsFileRoundTrip();
    testSrsFileErrors();
    testApi();
}

} // namespace PilFflonkTest
