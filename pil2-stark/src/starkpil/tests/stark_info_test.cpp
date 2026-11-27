// The starkinfo loader refuses a key of another keyVersion, or one with a boundary other than everyRow.
#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>

#include "logger.hpp"
#include "stark_info.hpp"

namespace
{

void expectRejected(const std::string &json, const std::string &needle = "regenerate the setup")
{
    std::string path = testing::TempDir() + "old.starkinfo.json";
    std::ofstream(path) << json;
    // the logger writes to stdout once enabled; the death test reads stderr
    auto load = [&] {
        CPlusPlusLogging::Logger::getInstance()->enableConsoleLogging();
        dup2(2, 1);
        StarkInfo(path, false, false, true);
    };
    EXPECT_DEATH(load(), needle);
    std::remove(path.c_str());
}

TEST(StarkInfoKeyVersion, RejectsOldKey)
{
    expectRejected(R"({"nPublics": 0})");
    expectRejected(R"({"keyVersion": 1})");
}

TEST(StarkInfoBoundaries, RejectsAnyButEveryRow)
{
    const std::string head =
        R"({"keyVersion": 2, "nPublics": 0, "nConstants": 0, "nStages": 2, "qDeg": 1, "qDim": 3, "nConstraints": 1,
            "cExpId": 0, "starkStruct": {"nBits": 3, "nBitsExt": 4, "verificationHashType": "GL", "merkleTreeArity": 4,
            "transcriptArity": 4, "merkleTreeCustom": true, "lastLevelVerification": 0, "numQueries": 1,
            "logDomainSizes": [4]}, "boundaries": )";
    expectRejected(head + R"([{"name": "firstRow"}]})", "everyRow boundary only");
    expectRejected(head + R"([{"name": "everyRow"}, {"name": "lastRow"}]})", "everyRow boundary only");
    expectRejected(head + R"(["everyRow"]})", "everyRow boundary only");
}

} // namespace
