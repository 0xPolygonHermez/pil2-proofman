// The readers moved from pil-fflonk, against the vendored artifacts.
//
// Both files are byte-identical to the originals -- the shims (zklog, zkassert,
// exit_process, utils) exist so they compile here unedited, which keeps them
// diffable against the source they came from. These tests check the move
// preserved behaviour rather than merely compiling.
#include <gtest/gtest.h>

#include <string>

#include "fflonk_info.hpp"
#include "zkey_pilfflonk.hpp"

namespace {

const std::string FIXTURES = "../tests/fixtures/reference/";

} // namespace

TEST(MOVED_READERS, readsTheFflonkInfo) {
    FflonkInfo::FflonkInfo info(AltBn128::Engine::engine, FIXTURES + "pilfflonk.fflonkinfo.json");

    // The reference AIR: 9 constants, 3 publics, quotient degree 3.
    EXPECT_EQ(9u, info.nConstants);
    EXPECT_EQ(3u, info.nPublics);
    EXPECT_EQ(3u, info.qDeg);

    // BN254 is a prime field, so the quotient has dimension one -- unlike a
    // Goldilocks AIR, where it is three.
    EXPECT_EQ(1u, info.qDim);

    // maxPolsOpenings is read, not derived: it sizes the quotient's domain.
    EXPECT_EQ(3u, info.maxPolsOpenings);

    // The committed widths the trace and later stages have.
    EXPECT_EQ(15u, info.nCm1);
    EXPECT_EQ(38u, info.evMap.size());
}

TEST(MOVED_READERS, readsThePilFflonkZkey) {
    auto fd = BinFileUtils::openExisting(FIXTURES + "pilfflonk.zkey", "zkey", 1);
    auto zkey = PilFflonkZkey::loadPilFflonkZkey(fd.get());

    ASSERT_NE(nullptr, zkey);
    EXPECT_EQ(8u, zkey->power);
    EXPECT_EQ(12u, zkey->powerW);
    EXPECT_EQ(3u, zkey->nPublics);

    // Nine combined polynomials, f0..f8.
    EXPECT_EQ(9u, zkey->f.size());

    // f0 packs six constants at one opening point; f8 is the quotient alone.
    EXPECT_EQ(6u, zkey->f[0]->nPols);
    EXPECT_EQ(1u, zkey->f[0]->nOpeningPoints);
    EXPECT_EQ(1u, zkey->f[8]->nPols);
    EXPECT_EQ(std::string("Q"), zkey->f[8]->pols[0]);

    delete zkey;
}

// The two agree about the AIR they describe -- they are written by different
// parts of the setup, so this is a cross-check rather than a tautology.
TEST(MOVED_READERS, theKeyAndTheInfoAgree) {
    FflonkInfo::FflonkInfo info(AltBn128::Engine::engine, FIXTURES + "pilfflonk.fflonkinfo.json");
    auto fd = BinFileUtils::openExisting(FIXTURES + "pilfflonk.zkey", "zkey", 1);
    auto zkey = PilFflonkZkey::loadPilFflonkZkey(fd.get());

    EXPECT_EQ(info.nPublics, zkey->nPublics);
    EXPECT_EQ(info.nCm1, zkey->polsNamesStage[1]->size());

    delete zkey;
}
