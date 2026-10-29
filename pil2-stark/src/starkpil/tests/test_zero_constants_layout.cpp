// A zero-fixed air's proof and buffer layouts drop exactly the const tree's sections.
#include <gtest/gtest.h>
#include <cmath>
#include "stark_info.hpp"
#include "merkleTreeGL.hpp"
#include "goldilocks_cubic_extension.hpp"

namespace {

const uint64_t NBITS = 6, NBITS_EXT = 7, ARITY = 4, LLV = 2, NQUERIES = 8, NCONST = 5;

json starkInfoJson(uint64_t nConstants) {
    json j;
    j["starkStruct"] = {{"nBits", NBITS}, {"nBitsExt", NBITS_EXT}, {"verificationHashType", "GL"},
                        {"merkleTreeArity", ARITY}, {"transcriptArity", 4}, {"merkleTreeCustom", true},
                        {"lastLevelVerification", LLV}, {"hashCommits", true}, {"numQueries", NQUERIES},
                        {"logDomainSizes", {NBITS_EXT, 4}}};
    j["nPublics"] = 0;
    j["nConstants"] = nConstants;
    j["nStages"] = 2;
    j["qDeg"] = 1;
    j["qDim"] = 3;
    j["nConstraints"] = 1;
    j["deepExpId"] = 0;
    j["cExpId"] = 0;
    j["customCommits"] = json::array();
    j["openingPoints"] = {0, 1};
    j["boundaries"] = {{{"name", "everyRow"}}};
    j["evMap"] = {{{"type", "cm"}, {"id", 0}, {"prime", 0}, {"openingPos", 0}},
                  {{"type", "cm"}, {"id", 0}, {"prime", 1}, {"openingPos", 1}}};
    j["mapSectionsN"] = {{"const", nConstants}, {"cm1", 7}, {"cm2", 6}, {"cm3", 3}};
    j["constPolsMap"] = json::array();
    for (uint64_t i = 0; i < nConstants; i++) {
        j["constPolsMap"].push_back({{"stage", 0}, {"name", "C"}, {"dim", 1}, {"stageId", i}, {"polsMapId", i}});
        j["evMap"].push_back({{"type", "const"}, {"id", i}, {"prime", 0}, {"openingPos", 0}});
    }
    return j;
}

StarkInfo load(uint64_t nConstants, bool gpu) {
    StarkInfo si;
    si.gpu = gpu;
    si.load(starkInfoJson(nConstants));
    return si;
}

uint64_t lastLevelWords() { return (uint64_t)std::pow(ARITY, LLV) * HASH_SIZE; }

TEST(ZeroConstantsLayout, ProofSizeDropsConstValsSiblingsAndLastLevel) {
    StarkInfo with = load(NCONST, false), without = load(0, false);
    EXPECT_TRUE(with.hasConstants());
    EXPECT_FALSE(without.hasConstants());
    EXPECT_EQ(with.numInitialTrees(), without.numInitialTrees() + 1);
    uint64_t nLevels = merkleProofLevels(NBITS_EXT, ARITY, LLV, false);
    uint64_t constEvalWords = NCONST * FIELD_EXTENSION;  // the fixture's const evMap entries
    EXPECT_EQ(with.proofSize - without.proofSize,
              NQUERIES * (NCONST + nLevels * (ARITY - 1) * HASH_SIZE) + lastLevelWords() + constEvalWords);
}

TEST(ZeroConstantsLayout, PinnedProofSizeDropsTheConstLastLevel) {
    StarkInfo with = load(NCONST, false), without = load(0, false);
    // The query block is scratch with one slot per tree, const included; only the last level is a section.
    EXPECT_EQ(with.getPinnedProofSize() - without.getPinnedProofSize(), lastLevelWords() + NCONST * FIELD_EXTENSION);
}

TEST(ZeroConstantsLayout, GpuMapReservesNoConstRegion) {
    StarkInfo with = load(NCONST, true), without = load(0, true);
    auto at = [](StarkInfo &si, const char *name, bool ext) { return si.mapOffsets.at(std::make_pair(std::string(name), ext)); };
    EXPECT_EQ(at(without, "const", true), 0u);
    EXPECT_EQ(at(without, "const", false), 0u);
    EXPECT_FALSE(without.calculateFixedExtended);
    EXPECT_TRUE(with.calculateFixedExtended);
    uint64_t NExt = 1 << NBITS_EXT, N = 1 << NBITS;
    uint64_t constTree = (NExt * NCONST + with.getNumNodesMTCommit(NExt) + 31) & ~uint64_t(31);
    EXPECT_EQ(at(with, "const", false), constTree);
    EXPECT_EQ(at(with, "custom_fixed", false) - at(without, "custom_fixed", false), constTree + N * NCONST);
}

TEST(ZeroConstantsLayout, LoadRefusesAConstEvalWithoutConstants) {
    json j = starkInfoJson(0);
    j["evMap"].push_back({{"type", "const"}, {"id", 0}, {"prime", 0}, {"openingPos", 0}});
    StarkInfo si;
    EXPECT_EXIT(si.load(j), ::testing::ExitedWithCode(255), "");
}

TEST(ZeroConstantsLayout, LoadRefusesAConstPolsMapWithoutConstants) {
    json j = starkInfoJson(0);
    j["constPolsMap"].push_back({{"stage", 0}, {"name", "C"}, {"dim", 1}, {"stageId", 0}, {"polsMapId", 0}});
    StarkInfo si;
    EXPECT_EXIT(si.load(j), ::testing::ExitedWithCode(255), "");
}

}  // namespace
