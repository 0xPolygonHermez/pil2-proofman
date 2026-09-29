#include "pilfflonk_test.hpp"
#include "pilfflonk_test_ptau.hpp"

#include <limits.h>
#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>

#include "binfile_writer.hpp"

namespace PilFflonkTest {

namespace {

using Engine = AltBn128::Engine;
using FrElement = Engine::FrElement;

Engine &E = Engine::engine;

// The canonical little-endian bytes of `e`, as ffiasm's scalar multiplication reads a scalar.
void canonicalBytes(const FrElement &e, uint8_t out[32]) {
    FrElement canonical;
    E.fr.fromMontgomery(canonical, e);
    std::memcpy(out, canonical.v, 32);
}

void appendU32(std::vector<uint8_t> &bytes, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}

template <typename Point>
void appendPoints(std::vector<uint8_t> &bytes, const std::vector<Point> &points) {
    const uint8_t *begin = reinterpret_cast<const uint8_t *>(points.data());
    bytes.insert(bytes.end(), begin, begin + points.size() * sizeof(Point));
}

} // namespace

TestDir::TestDir() {
    char exe[PATH_MAX];
    const ssize_t length = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    assert(length > 0);
    exe[length] = '\0';
    std::string pattern(exe);
    pattern = pattern.substr(0, pattern.rfind('/') + 1) + "pilfflonk_test.XXXXXX";
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    assert(mkdtemp(buffer.data()) != nullptr);
    dir = buffer.data();
}

TestDir::~TestDir() {
    for (const std::string &name : files) {
        std::remove(name.c_str());
    }
    rmdir(dir.c_str());
}

std::string TestDir::file(const std::string &name) {
    const std::string path = dir + "/" + name;
    // SRS writers go through path + ".tmp": gone if they succeed, not always if they fail.
    files.push_back(path);
    files.push_back(path + ".tmp");
    return path;
}

FrElement testTau() {
    FrElement tau;
    E.fr.fromString(tau, "1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f809", 16);
    return tau;
}

PtauSections testPtauSections(uint64_t nG1, uint64_t nG2, const FrElement &tau) {
    PtauSections sections;
    sections.power = 0;
    while ((uint64_t(2) << sections.power) - 1 < nG1 || (uint64_t(1) << sections.power) < nG2) {
        ++sections.power;
    }

    appendU32(sections.header, 32);
    const Bytes32 q(Q_HEX);
    sections.header.insert(sections.header.end(), q.bytes, q.bytes + 32);
    appendU32(sections.header, sections.power);
    appendU32(sections.header, sections.power);

    std::vector<FrElement> powers(std::max(nG1, nG2));
    if (!powers.empty()) {
        powers[0] = E.fr.one();
    }
    for (size_t i = 1; i < powers.size(); ++i) {
        E.fr.mul(powers[i], powers[i - 1], tau);
    }

    std::vector<Engine::G1PointAffine> g1(nG1);
#pragma omp parallel for
    for (uint64_t i = 0; i < nG1; ++i) {
        uint8_t scalar[32];
        canonicalBytes(powers[i], scalar);
        Engine::G1Point p;
        E.g1.mulByScalar(p, E.g1.oneAffine(), scalar, sizeof(scalar));
        E.g1.copy(g1[i], p);
    }
    std::vector<Engine::G2PointAffine> g2(nG2);
#pragma omp parallel for
    for (uint64_t i = 0; i < nG2; ++i) {
        uint8_t scalar[32];
        canonicalBytes(powers[i], scalar);
        Engine::G2Point p;
        E.g2.mulByScalar(p, E.g2.oneAffine(), scalar, sizeof(scalar));
        E.g2.copy(g2[i], p);
    }
    appendPoints(sections.g1, g1);
    appendPoints(sections.g2, g2);
    return sections;
}

void writeBinFile(const std::string &path, const std::string &type, uint32_t version,
                  const std::vector<std::pair<uint32_t, std::vector<uint8_t>>> &sections) {
    BinFileUtils::BinFileWriter writer(path, type, version, sections.size());
    for (const auto &section : sections) {
        // write() takes a non-const buffer.
        std::vector<uint8_t> bytes = section.second;
        writer.startWriteSection(section.first);
        writer.write(bytes.data(), bytes.size());
        writer.endWriteSection();
    }
    writer.close();
}

void writeTestPtau(const std::string &path, uint64_t nG1, uint64_t nG2) {
    const PtauSections sections = testPtauSections(nG1, nG2, testTau());
    writeBinFile(path, "ptau", 1, {{1, sections.header}, {2, sections.g1}, {3, sections.g2}});
}

} // namespace PilFflonkTest
