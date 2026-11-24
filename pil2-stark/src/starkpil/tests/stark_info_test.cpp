// The starkinfo loader refuses a key of another keyVersion.
#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>

#include "logger.hpp"
#include "stark_info.hpp"

namespace
{

void expectRejected(const std::string &json)
{
    std::string path = testing::TempDir() + "old.starkinfo.json";
    std::ofstream(path) << json;
    // the logger writes to stdout once enabled; the death test reads stderr
    auto load = [&] {
        CPlusPlusLogging::Logger::getInstance()->enableConsoleLogging();
        dup2(2, 1);
        StarkInfo(path, false, false, true);
    };
    EXPECT_DEATH(load(), "regenerate the setup");
    std::remove(path.c_str());
}

TEST(StarkInfoKeyVersion, RejectsOldKey)
{
    expectRejected(R"({"nPublics": 0})");
    expectRejected(R"({"keyVersion": 1})");
}

} // namespace
