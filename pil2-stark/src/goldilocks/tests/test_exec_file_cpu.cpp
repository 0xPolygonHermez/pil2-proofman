// The exec file's header: which format versions this build refuses, and how it says so.

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "../../starkpil/recursion_trace/exec_layout.hpp"

// A BN128 exec, which plonk2pil writes for the pilfflonk wrap, is never read at this build's
// offsets, and the error that refuses it says what it is: regenerating the key would not help.
TEST(ExecFile, ABn128ExecIsRefusedByName)
{
    // magic|3, nAdds, mapRows, mapCols, the coefficient width and the r1cs's wire count a version 2
    // header lacks, then a 1 x 1 map and an empty band section.
    const std::vector<uint64_t> exec{exec_layout::EXEC_MAGIC | exec_layout::EXEC_FORMAT_VERSION_WIDE, 0, 1, 1, 4,
                                     8, 7, 2, 0, 0};
    const exec_layout::Header h = exec_layout::header(exec.data(), exec.size());
    ASSERT_TRUE(h.magic);
    ASSERT_FALSE(h.versionOk);
    ASSERT_FALSE(h.valid);
    ASSERT_EQ(h.version, exec_layout::EXEC_FORMAT_VERSION_WIDE);

    const std::string bn128 = exec_layout::refused_version(h.version);
    ASSERT_NE(bn128.find("format version 3, the BN128 exec plonk2pil writes for the pilfflonk wrap"), std::string::npos)
        << bn128;
    ASSERT_EQ(bn128.find("regenerate"), std::string::npos) << bn128;

    const std::string newer = exec_layout::refused_version(exec_layout::EXEC_FORMAT_VERSION_WIDE + 1);
    ASSERT_NE(newer.find("format version 4, but this build reads version 2; regenerate"), std::string::npos)
        << newer;
}
