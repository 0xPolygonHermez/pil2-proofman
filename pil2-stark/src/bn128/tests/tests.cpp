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

// ==============================================================================
// Poseidon2 tests for all supported t values (t=2,3,4,8,12,16)
// Input: state[i] = i for i in 0..t-1
// ==============================================================================

TEST(BN128_POSEIDON2_TEST, hash_t2) {
  Poseidon2BN128 p;
  RawFrP field;
  const size_t t = 2;
  vector<RawFrP::Element> state(t);
  for (size_t i = 0; i < t; i++) {
    field.fromUI(state[i], (unsigned long int)(i));
  }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "1d01e56f49579cec72319e145f06f6177f6c5253206e78c2689781452a31878b");
  ASSERT_EQ(field.toString(state[1],16), "d189ec589c41b8cffa88cfc523618a055abe8192c70f75aa72fc514560f6c61");
}

TEST(BN128_POSEIDON2_TEST, hash_t3) {
  Poseidon2BN128 p;
  RawFrP field;
  const size_t t = 3;
  vector<RawFrP::Element> state(t);
  for (size_t i = 0; i < t; i++) {
    field.fromUI(state[i], (unsigned long int)(i));
  }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "bb61d24daca55eebcb1929a82650f328134334da98ea4f847f760054f4a3033");
  ASSERT_EQ(field.toString(state[1],16), "303b6f7c86d043bfcbcc80214f26a30277a15d3f74ca654992defe7ff8d03570");
  ASSERT_EQ(field.toString(state[2],16), "1ed25194542b12eef8617361c3ba7c52e660b145994427cc86296242cf766ec8");
}

TEST(BN128_POSEIDON2_TEST, hash_t4) {
  Poseidon2BN128 p;
  RawFrP field;
  const size_t t = 4;
  vector<RawFrP::Element> state(t);
  for (size_t i = 0; i < t; i++) {
    field.fromUI(state[i], (unsigned long int)(i));
  }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "1bd538c2ee014ed5141b29e9ae240bf8db3fe5b9a38629a9647cf8d76c01737");
  ASSERT_EQ(field.toString(state[1],16), "239b62e7db98aa3a2a8f6a0d2fa1709e7a35959aa6c7034814d9daa90cbac662");
  ASSERT_EQ(field.toString(state[2],16), "4cbb44c61d928ed06808456bf758cbf0c18d1e15a7b6dbc8245fa7515d5e3cb");
  ASSERT_EQ(field.toString(state[3],16), "2e11c5cff2a22c64d01304b778d78f6998eff1ab73163a35603f54794c30847a");
}

TEST(BN128_POSEIDON2_TEST, hash_t8) {
  Poseidon2BN128 p;
  RawFrP field;
  const size_t t = 8;
  vector<RawFrP::Element> state(t);
  for (size_t i = 0; i < t; i++) {
    field.fromUI(state[i], (unsigned long int)(i));
  }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "1d1a50bcde871247856df135d56a4ca61af575f1140ed9b1503c77528cf345df");
  ASSERT_EQ(field.toString(state[1],16), "2d3943cf476ed49fd8a636660d8a76c83b55f07d06bc082005ad7eb1a21791c5");
  ASSERT_EQ(field.toString(state[2],16), "2fcda2dd846fadfde8104b1d05175dcf3cf8bd698ed8ea3ad2fbcf9c06e00310");
  ASSERT_EQ(field.toString(state[3],16), "28811ac7e0829171f9d3d81f1c0ff8f34b360d407a16b331a1cb6b5d992de094");
  ASSERT_EQ(field.toString(state[4],16), "2c07c1817cfccb67c1297935514885c07abad5a0e15477f6c076c0b0fb1ad6f3");
  ASSERT_EQ(field.toString(state[5],16), "1b6114397199bc44e37437dd3ba1754dff007d3315bfcdcdc14ec27d02452f52");
  ASSERT_EQ(field.toString(state[6],16), "1431250baf36fb61a07618caee4dd2f500da339a05c553e8f529a3349e617aa2");
  ASSERT_EQ(field.toString(state[7],16), "b19bfa00c8f1d505074130e7f8b49a8624b1905e280ceca5ba11099b081b265");
}

TEST(BN128_POSEIDON2_TEST, hash_t12) {
  Poseidon2BN128 p;
  RawFrP field;
  const size_t t = 12;
  vector<RawFrP::Element> state(t);
  for (size_t i = 0; i < t; i++) {
    field.fromUI(state[i], (unsigned long int)(i));
  }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "3014e0ec17029f7e4f5cfe8c7c54fc3df6a5f7539f6aa304b2f3c747a9105618");
  ASSERT_EQ(field.toString(state[1],16), "2f90753e7aaf46c158cd12346da7dd37c3136353ec51525cabbaaf2b2350f9b2");
  ASSERT_EQ(field.toString(state[2],16), "2e28bdc8b2c68b09da0cb653ee7e54eca909cf2ae010784554aa3e165b1a105f");
  ASSERT_EQ(field.toString(state[3],16), "1d6a97ef87dbd3476a848af45beebe6b5d79cb047b37212e3e5839f1e80b397a");
  ASSERT_EQ(field.toString(state[4],16), "24e23df24b19b75f44218a08d107709d35561bc1b982cfc317d54568cd496519");
  ASSERT_EQ(field.toString(state[5],16), "185a08e623b85e797844191a1f184f7b8fc486253919eb20f1186a8331757018");
  ASSERT_EQ(field.toString(state[6],16), "69ed78df853a105c8949dae5b4e81cbe370e8f6e25735a688aa8ff3df9659eb");
  ASSERT_EQ(field.toString(state[7],16), "284395d79b64123211a4a59b81a90f9cfa8d8314dccde4cef22ec1e31431efd3");
  ASSERT_EQ(field.toString(state[8],16), "f24be5a8c95e3504ead0da9e792b77d7056f94461d69b04b33ea5d239f8e444");
  ASSERT_EQ(field.toString(state[9],16), "22469ccfef0ce5a237518c38dec31fc2804e633b3b365c23a9f703ca31ef393");
  ASSERT_EQ(field.toString(state[10],16), "1fcdcee218d5a0101bd233d572f184964854d445ca08d2bd6df6ceba5651e322");
  ASSERT_EQ(field.toString(state[11],16), "905469a776b7d5a3f18841edb90fa0d8c6de479c2789c042dafefb367ad1a2b");
}

TEST(BN128_POSEIDON2_TEST, hash_t16) {
  Poseidon2BN128 p;
  RawFrP field;
  const size_t t = 16;
  vector<RawFrP::Element> state(t);
  for (size_t i = 0; i < t; i++) {
    field.fromUI(state[i], (unsigned long int)(i));
  }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "fc2e6b758f493969e1d860f9a44ee3bdffdf796f382aa4ffb16fa4e9bcc333f");
  ASSERT_EQ(field.toString(state[15],16), "e2ceb1f8fde5f80be1f41bd239fabdc2f6133a6a98920a55c42891c3a925152");
  // Note: state[1]-state[14] can be added if needed for more thorough validation
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

