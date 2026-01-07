#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include "../src/ffiasm/fr.hpp"
#include "../src/poseidon/poseidon_bn128.hpp"
#include "../src/poseidon2/poseidon2_bn128.hpp"


TEST(BN128_POSEIDON_TEST, hash) {
  PoseidonBN128 p;
  RawFrP field;

  vector<RawFrP::Element> state4(4);
  for (size_t i = 0; i < 4; i++)
  {
    field.fromUI(state4[i], (unsigned long int)(i));
  }
  p.hash(state4);        
  ASSERT_EQ(field.toString(state4[0],16), "e7732d89e6939c0ff03d5e58dab6302f3230e269dc5b968f725df34ab36d732");
  ASSERT_EQ(field.toString(state4[1],16), "7b0b86b41ec7fdfe6c17ee6ccdddce4e47e748e493e542f9a435b0dde022a0d");
  ASSERT_EQ(field.toString(state4[2],16), "4362e50fcc8be421898d47ace20eab18b0a6efab0e12ade49f2df609fec4209");
  ASSERT_EQ(field.toString(state4[3],16), "1a779bd9781d3a8354eae5ed74e7fa44fa0e458e45a1407524bddf3b9f2bf2d7");
}

TEST(BN128_POSEIDON2_TEST, hash) {
  Poseidon2BN128 p;
  RawFrP field;
  size_t t = 2;
  vector<RawFrP::Element> state(t);
  for (size_t i = 0; i < t; i++)
  {
    field.fromUI(state[i], (unsigned long int)(i));
  }
  p.hash(state);        
  ASSERT_EQ(field.toString(state[0],16), 
  "1d01e56f49579cec72319e145f06f6177f6c5253206e78c2689781452a31878b");
  ASSERT_EQ(field.toString(state[1],16), 
  "d189ec589c41b8cffa88cfc523618a055abe8192c70f75aa72fc514560f6c61");
  
  state.resize(16);
  for (size_t i = 0; i < 16; i++)
  {
    field.fromUI(state[i], (unsigned long int)(i));
  }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), 
  "fc2e6b758f493969e1d860f9a44ee3bdffdf796f382aa4ffb16fa4e9bcc333f");
  ASSERT_EQ(field.toString(state[15],16), 
  "e2ceb1f8fde5f80be1f41bd239fabdc2f6133a6a98920a55c42891c3a925152"); 
}

#if 0
// This test is just to generate the constants for Poseidon2BN128 in montgomery form
TEST(CONVERTER, poseidon_seq_widths_sanity) {
	  Poseidon2BN128 p;
    RawFrP field;

    //takes a number in exadecimal
    RawFrP::Element number;
    int nstrings = 1000;
    std::string strs[nstrings];
    int k=0;
    strs[k++]="1d066a255517b7fd8bddd3a93f7804ef7f8fcde48bb4c37a59a09a1a97052816";
    strs[k++]="29daefb55f6f2dc6ac3f089cebcc6120b7c6fef31367b68eb7238547d32c1610";
    strs[k++]="1f2cb1624a78ee001ecbd88ad959d7012572d76f08ec5c4f9e8b7ad7b0b4e1d1";
    strs[k++]="0aad2e79f15735f2bd77c0ed3d14aa27b11f092a53bbc6e1db0672ded84f31e5";

    std::cout << "k: " << k << std::endl;
    assert(nstrings>=k);
    nstrings = k; 
    std::cout <<"{";
    for(int i = 0; i < nstrings; i++)
     {
        field.fromString(number, strs[i], 16);
        //prints the four components in hexadecimal (16 hex digits each limb)
        if(i != 0)
            std::cout << " ";
        std::cout << std::hex << std::uppercase
          << "{0x" << std::setw(16) << std::setfill('0') << (uint64_t)number.v[0]
          << ", 0x" << std::setw(16) << std::setfill('0') << (uint64_t)number.v[1]
          << ", 0x" << std::setw(16) << std::setfill('0') << (uint64_t)number.v[2]
          << ", 0x" << std::setw(16) << std::setfill('0') << (uint64_t)number.v[3]
          << "}" << std::dec;
        if(i != nstrings -1)
            std::cout << ","<<std::endl;
     }  
    std::cout << "}" << std::endl;
}
#endif