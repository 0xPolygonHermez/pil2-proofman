// The moved prover, end to end, against the vendored artifacts.
//
// pilfflonk_prover.cpp is byte-identical to pil-fflonk's; this runs it here to
// show the move preserved behaviour, not merely that it compiles. It reads the
// same proving key, AIR description and committed trace the reference did, and
// must produce a structurally complete proof.
//
// The proof is blinded, so it differs run to run and cannot be compared to a
// stored one -- what is checked is its shape, and that every claimed opening
// and commitment the key calls for is present. The fixed-blinding capture in
// ../../tests/fixtures/pilfflonk.deterministic.proof.json is what the exact
// per-stage comparisons use.
#include <gtest/gtest.h>

#include <string>

#include "pilfflonk_prover.hpp"

namespace {

const std::string FIXTURES = "../tests/fixtures/reference/";

} // namespace

TEST(MOVED_PROVER, producesAStructurallyCompleteProof) {
    PilFflonk::PilFflonkProver prover(AltBn128::Engine::engine, FIXTURES + "pilfflonk.zkey",
                                      FIXTURES + "pilfflonk.fflonkinfo.json");

    auto [proof, publics] = prover.prove(FIXTURES + "pilfflonk.commit");

    EXPECT_EQ("pilfflonk", proof["protocol"]);
    EXPECT_EQ("bn128", proof["curve"]);

    // Nine combined polynomials plus W and W'.
    ASSERT_TRUE(proof.contains("polynomials"));
    EXPECT_EQ(11u, proof["polynomials"].size());
    for (const auto &name : {"f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "W", "Wp"}) {
        ASSERT_TRUE(proof["polynomials"].contains(name)) << "missing commitment " << name;
        EXPECT_EQ(3u, proof["polynomials"][name].size()) << name << " is not [x, y, z]";
        EXPECT_EQ("1", proof["polynomials"][name][2]) << name << " is not affine";
    }

    // 42 opening slots, less the quotient the verifier reconstructs, plus the
    // two inverse hints.
    ASSERT_TRUE(proof.contains("evaluations"));
    EXPECT_EQ(43u, proof["evaluations"].size());
    EXPECT_FALSE(proof["evaluations"].contains("Q")) << "the quotient's evaluation must not be sent";
    EXPECT_TRUE(proof["evaluations"].contains("inv"));
    EXPECT_TRUE(proof["evaluations"].contains("invZh"));

    EXPECT_EQ(3u, publics.size());
}

// Blinding is drawn afresh each run, so two proofs of the same statement differ
// -- which is what makes them zero-knowledge, and why the exact per-stage
// checks use the fixed-blinding capture instead.
TEST(MOVED_PROVER, blindingMakesEachProofDifferent) {
    PilFflonk::PilFflonkProver prover(AltBn128::Engine::engine, FIXTURES + "pilfflonk.zkey",
                                      FIXTURES + "pilfflonk.fflonkinfo.json");

    auto [first, _] = prover.prove(FIXTURES + "pilfflonk.commit");
    auto [second, __] = prover.prove(FIXTURES + "pilfflonk.commit");

    EXPECT_NE(first["polynomials"]["f8"], second["polynomials"]["f8"]);
    EXPECT_NE(first["polynomials"]["W"], second["polynomials"]["W"]);
}
