#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include "../src/ffiasm/fr.hpp"
#include "../src/ffiasm/alt_bn128.hpp"
#include "../src/ffiasm/multiexp.hpp"
#include "../src/ffiasm/fft.hpp"
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
  
  t=16;
  state.resize(t);
  for (size_t i = 0; i < t; i++)
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

TEST(BN128_MULTIEXP_TEST, multiexp_4_operands) {
  typedef AltBn128::Engine Engine;
  Engine engine;
  RawFrP field;
  
  uint64_t n = 4;
  uint64_t scalarSize = 32;
  
  // Create bases using doubling approach: base[i] = 2^i * G
  std::vector<Engine::G1::PointAffine> bases(n);
  Engine::G1::Point tempPoint;
  engine.g1.copy(tempPoint, engine.g1.oneAffine());  
  for (uint64_t i = 0; i < n; i++) {
    engine.g1.copy(bases[i], tempPoint);
    engine.g1.dbl(tempPoint, tempPoint);  // Double for next iteration
  }
  
  // Create scalars from strings: 253-bit large numbers (diverse)
  std::string scalarStrs[4] = {
    "5708990770823839524233143877797980545530985996",
    "8563486156235759286349715816696970818296478975",
    "9234567890123456789012345678901234567890123456",
    "10876543210987654321098765432109876543210987654"
  };
  std::vector<uint8_t> scalars(n * scalarSize, 0);
  
  for (uint64_t i = 0; i < n; i++) {
    Engine::Fr::Element scalarElem;
    engine.fr.fromString(scalarElem, scalarStrs[i], 10);
    // Convert to big-endian bytes, then reverse to get little-endian
    std::vector<uint8_t> beBytes(scalarSize, 0);
    engine.fr.toRprBE(scalarElem, beBytes.data(), scalarSize);
    // Reverse to little-endian (mulByScalar expects LE)
    for (uint64_t j = 0; j < scalarSize; j++) {
      scalars[i * scalarSize + j] = beBytes[scalarSize - 1 - j];
    }
  }
  
  // Perform multiexp
  ParallelMultiexp<Engine::G1> pme(engine.g1);
  Engine::G1::Point result;
  pme.multiexp(result, bases.data(), scalars.data(), scalarSize, n);
  
  // Verify multiexp result by manual point accumulation loop in reverse order
  Engine::G1::Point manualSum;
  engine.g1.copy(manualSum, engine.g1.zero());
  
  for (int i = n-1; i >= 0; i--) {
    Engine::G1::Point term;
    engine.g1.mulByScalar(term, bases[i], &scalars[i * scalarSize], scalarSize);
    engine.g1.add(manualSum, manualSum, term);
  }
  
  ASSERT_TRUE(engine.g1.eq(result, manualSum)) << "Multiexp result does not match manual point accumulation";
  
  // Verify multiexp by combining scalars into a single scalar
  // combinedScalar = s0 + s1*2 + s2*4 + s3*8
  // Parse scalars directly from strings
  Engine::Fr::Element rawScalars[4];
  for (uint64_t i = 0; i < n; i++) {
    engine.fr.fromString(rawScalars[i], scalarStrs[i], 10);
  }
  
  Engine::Fr::Element combinedScalar;
  engine.fr.fromUI(combinedScalar, 0);
  
  for (uint64_t i = 0; i < n; i++) {
    Engine::Fr::Element powerOfTwo;
    engine.fr.fromUI(powerOfTwo, 1ULL << i);
    Engine::Fr::Element term;
    engine.fr.mul(term, rawScalars[i], powerOfTwo);
    engine.fr.add(combinedScalar, combinedScalar, term);
  }
  
  // Convert combined scalar to little-endian bytes (mulByScalar expects LE)
  std::vector<uint8_t> combinedScalarBE(scalarSize, 0);
  engine.fr.toRprBE(combinedScalar, combinedScalarBE.data(), scalarSize);
  std::vector<uint8_t> combinedScalarLE(scalarSize, 0);
  for (uint64_t j = 0; j < scalarSize; j++) {
    combinedScalarLE[j] = combinedScalarBE[scalarSize - 1 - j];
  }
  
  // Compute expected result: (s0 + s1*2 + s2*4 + s3*8) * G
  Engine::G1::Point expected;
  engine.g1.mulByScalar(expected, engine.g1.oneAffine(), combinedScalarLE.data(), scalarSize);
  
  // Verify: multiexp(bases=[G,2G,4G,8G], scalars=[s0,s1,s2,s3]) == (s0 + s1*2 + s2*4 + s3*8) * G
  ASSERT_TRUE(engine.g1.eq(result, expected)) << "Combined scalar verification failed: multiexp result does not match (s0 + s1*2 + s2*4 + s3*8)*G";
}

// =====================
// FFT Tests
// =====================

TEST(BN128_FFT_TEST, fft_then_ifft_roundtrip) {
  // Test: fft followed by ifft should recover the original data
  RawFrP field;
  const uint64_t n = 16;
  
  FFT<RawFrP> fft(n);
  
  std::vector<RawFrP::Element> data(n);
  std::vector<RawFrP::Element> original(n);
  for (uint64_t i = 0; i < n; i++) {
    field.fromUI(data[i], i);
    field.copy(original[i], data[i]);
  }
  
  // Apply fft then ifft
  fft.fft(data.data(), n);
  fft.ifft(data.data(), n);
  
  // Verify result matches original
  for (uint64_t i = 0; i < n; i++) {
    ASSERT_TRUE(field.eq(data[i], original[i])) 
      << "Mismatch at index " << i 
      << ": expected " << field.toString(original[i], 10)
      << ", got " << field.toString(data[i], 10);
  }
}

TEST(BN128_FFT_TEST, ifft_then_fft_roundtrip) {
  // Test: ifft followed by fft should recover the original data
  RawFrP field;
  const uint64_t n = 16;
  
  FFT<RawFrP> fft(n);
  
  std::vector<RawFrP::Element> data(n);
  std::vector<RawFrP::Element> original(n);
  for (uint64_t i = 0; i < n; i++) {
    field.fromUI(data[i], i);
    field.copy(original[i], data[i]);
  }
  
  // Apply ifft then fft
  fft.ifft(data.data(), n);
  fft.fft(data.data(), n);
  
  // Verify result matches original
  for (uint64_t i = 0; i < n; i++) {
    ASSERT_TRUE(field.eq(data[i], original[i])) 
      << "Mismatch at index " << i 
      << ": expected " << field.toString(original[i], 10)
      << ", got " << field.toString(data[i], 10);
  }
}

TEST(BN128_FFT_TEST, fft_linearity) {
  // Test: fft(a + b) == fft(a) + fft(b)  (FFT is a linear operation)
  RawFrP field;
  const uint64_t n = 16;
  
  FFT<RawFrP> fft(n);
  
  // Create two input vectors
  std::vector<RawFrP::Element> a(n);
  std::vector<RawFrP::Element> b(n);
  std::vector<RawFrP::Element> a_plus_b(n);
  
  for (uint64_t i = 0; i < n; i++) {
    field.fromUI(a[i], i + 1);           // a = [1, 2, 3, ..., 16]
    field.fromUI(b[i], (i * 7) % 13);    // b = some different pattern
    field.add(a_plus_b[i], a[i], b[i]);  // a_plus_b = a + b
  }
  
  // Compute fft(a), fft(b), fft(a+b)
  std::vector<RawFrP::Element> fft_a(a);
  std::vector<RawFrP::Element> fft_b(b);
  std::vector<RawFrP::Element> fft_a_plus_b(a_plus_b);
  
  fft.fft(fft_a.data(), n);
  fft.fft(fft_b.data(), n);
  fft.fft(fft_a_plus_b.data(), n);
  
  // Verify: fft(a+b) == fft(a) + fft(b)
  for (uint64_t i = 0; i < n; i++) {
    RawFrP::Element expected_sum;
    field.add(expected_sum, fft_a[i], fft_b[i]);
    
    ASSERT_TRUE(field.eq(fft_a_plus_b[i], expected_sum)) 
      << "Linearity failed at index " << i 
      << ": fft(a+b) = " << field.toString(fft_a_plus_b[i], 10)
      << ", fft(a) + fft(b) = " << field.toString(expected_sum, 10);
  }
}

