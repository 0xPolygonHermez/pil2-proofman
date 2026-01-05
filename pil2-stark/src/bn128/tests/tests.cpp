#include <gtest/gtest.h>
#include <vector>
#include "../src/ffiasm/fr.hpp"
#include "../src/poseidon/poseidon_bn128.hpp"



TEST(BN128_POSEIDON_TEST, poseidon_seq_widths_sanity) {
	PoseidonBN128 p;
    RawFrP field;

    vector<RawFrP::Element> state4(4);
    for (size_t i = 0; i < 4; i++)
    {
      field.fromUI(state4[i], (unsigned long int)(i + 1));
    }
    p.hash(state4);        
    ASSERT_EQ(field.toString(state4[0]), "13396720590959022673407666684009157274418812886936013644289745590247574720927");
    ASSERT_EQ(field.toString(state4[1]), "645294696644312786871998535449330391979706471414642698167450873247550170759");
    ASSERT_EQ(field.toString(state4[2]), "11049874281846710177095486239918952109260667041819690150235487093940630535956");
    ASSERT_EQ(field.toString(state4[3]), "4267180453229154970052132411041502189070482586987859275426916420442974788984");
	
}

