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


// ==============================================================================
// Poseidon tests for all supported t values (t=2..17)
// ==============================================================================

TEST(BN128_POSEIDON_TEST, hash_t2) {
  PoseidonBN128 p;
  RawFr field;
  const size_t t = 2;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) { field.fromUI(state[i], (unsigned long int)(i)); }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "29176100eaa962bdc1fe6c654d6a3c130e96a4d1168b33848b897dc502820133");
  ASSERT_EQ(field.toString(state[1],16), "112a4f9241e384b0ede4655e6d2bbf7ebd9595775de9e7536df87cd487852fc4");
}

TEST(BN128_POSEIDON_TEST, hash_t3) {
  PoseidonBN128 p;
  RawFr field;
  const size_t t = 3;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) { field.fromUI(state[i], (unsigned long int)(i)); }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "115cc0f5e7d690413df64c6b9662e9cf2a3617f2743245519e19607a4417189a");
  ASSERT_EQ(field.toString(state[2],16), "e7ae82e40091e63cbd4f16a6d16310b3729d4b6e138fcf54110e2867045a30c");
}

TEST(BN128_POSEIDON_TEST, hash_t4) {
  PoseidonBN128 p;
  RawFr field;
  const size_t t = 4;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) { field.fromUI(state[i], (unsigned long int)(i)); }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "e7732d89e6939c0ff03d5e58dab6302f3230e269dc5b968f725df34ab36d732");
  ASSERT_EQ(field.toString(state[3],16), "1a779bd9781d3a8354eae5ed74e7fa44fa0e458e45a1407524bddf3b9f2bf2d7");
}

TEST(BN128_POSEIDON_TEST, hash_t5) {
  PoseidonBN128 p;
  RawFr field;
  const size_t t = 5;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) { field.fromUI(state[i], (unsigned long int)(i)); }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "299c867db6c1fdd79dcefa40e4510b9837e60ebb1ce0663dbaa525df65250465");
  ASSERT_EQ(field.toString(state[4],16), "7748bc6877c9b82c8b98666ee9d0626ec7f5be4205f79ee8528ef1c4a376fc7");
}

TEST(BN128_POSEIDON_TEST, hash_t6) {
  PoseidonBN128 p;
  RawFr field;
  const size_t t = 6;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) { field.fromUI(state[i], (unsigned long int)(i)); }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "dab9449e4a1398a15224c0b15a49d598b2174d305a316c918125f8feeb123c0");
  ASSERT_EQ(field.toString(state[5],16), "208adf8d7f4ac061f00db710aef42f3b2f13176de26674b0a5f4436b883db6bc");
}

TEST(BN128_POSEIDON_TEST, hash_t7) {
  PoseidonBN128 p;
  RawFr field;
  const size_t t = 7;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) { field.fromUI(state[i], (unsigned long int)(i)); }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "2d1a03850084442813c8ebf094dea47538490a68b05f2239134a4cca2f6302e1");
  ASSERT_EQ(field.toString(state[6],16), "2ac1d41181b675cbbfe7801457f882bfcd0d9994a37a6a105452b48a71f3c810");
}

TEST(BN128_POSEIDON_TEST, hash_t8) {
  PoseidonBN128 p;
  RawFr field;
  const size_t t = 8;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) { field.fromUI(state[i], (unsigned long int)(i)); }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "1c2f3482dbb140c4ebb9ada49abdbc374a9a85fcfc6533ec2e9df45b4921c318");
  ASSERT_EQ(field.toString(state[7],16), "73534f0cedf2b30a870814eee062903ce751e545270c3cbfc5e4732c450ba9c");
}

TEST(BN128_POSEIDON_TEST, hash_t9) {
  PoseidonBN128 p;
  RawFr field;
  const size_t t = 9;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) { field.fromUI(state[i], (unsigned long int)(i)); }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "2921ab9bd0140cbc98e40395c0fefb40337a4d54fbbecd9a4d43b3d8d0c4d8d1");
  ASSERT_EQ(field.toString(state[8],16), "2c8e23a3569963447e55619f1d1462f63ea2e40d3d405c18bbf394f13c253749");
}

TEST(BN128_POSEIDON_TEST, hash_t10) {
  PoseidonBN128 p;
  RawFr field;
  const size_t t = 10;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) { field.fromUI(state[i], (unsigned long int)(i)); }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "1e0b893aa2ad802275e749d260330b7675b22bb3aaa4461d204af32e60cd9078");
  ASSERT_EQ(field.toString(state[9],16), "315afa225921ebb807ba0f33feef2bb5b74c51b740b58faa205dc127e8aa7ac");
}

TEST(BN128_POSEIDON_TEST, hash_t11) {
  PoseidonBN128 p;
  RawFr field;
  const size_t t = 11;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) { field.fromUI(state[i], (unsigned long int)(i)); }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "816126a09c29ecfcc0628461dacfb9459816fc60d6738b78db9ad07206fdc21");
  ASSERT_EQ(field.toString(state[10],16), "10f779eb86c66f6e316473976ca0b6b81e8c0c2cadf917ce84bf9cce1b72c45e");
}

TEST(BN128_POSEIDON_TEST, hash_t12) {
  PoseidonBN128 p;
  RawFr field;
  const size_t t = 12;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) { field.fromUI(state[i], (unsigned long int)(i)); }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "7e5b070aa2dba008f30a6b785b6c5ae2429e211f71cacdbdae0e07fc05b47a8");
  ASSERT_EQ(field.toString(state[11],16), "1941a33364c6d1904c0e540b5170c73567d31cb038d5d6b83cd769412139321a");
}

TEST(BN128_POSEIDON_TEST, hash_t13) {
  PoseidonBN128 p;
  RawFr field;
  const size_t t = 13;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) { field.fromUI(state[i], (unsigned long int)(i)); }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "58814945232937db248a01e7cc55b3d681cc08702c8168494e856c1ef7693b5");
  ASSERT_EQ(field.toString(state[12],16), "1a6df4eadbafbed2a14f78606ca1326f4bef58a348cffc2a0e8c050dab9cff94");
}

TEST(BN128_POSEIDON_TEST, hash_t14) {
  PoseidonBN128 p;
  RawFr field;
  const size_t t = 14;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) { field.fromUI(state[i], (unsigned long int)(i)); }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "f918939632fadca6456a2fe6e65a124828d4c3920d379cc744e90a666887806");
  ASSERT_EQ(field.toString(state[13],16), "5a2ad96bd0cec0ed170ae830c1800d3e83a72d3fb84673213aab431fc578cb7");
}

TEST(BN128_POSEIDON_TEST, hash_t15) {
  PoseidonBN128 p;
  RawFr field;
  const size_t t = 15;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) { field.fromUI(state[i], (unsigned long int)(i)); }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "1278779aaafc5ca58bf573151005830cdb4683fb26591c85a7464d4f0e527776");
  ASSERT_EQ(field.toString(state[14],16), "2c24786e78a255df1c1f11c09c5bea75c4ac1f96ad7978e6867f033363ed6bda");
}

TEST(BN128_POSEIDON_TEST, hash_t16) {
  PoseidonBN128 p;
  RawFr field;
  const size_t t = 16;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) { field.fromUI(state[i], (unsigned long int)(i)); }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "94ae33b67a845998abb55e917642d4022d078d96f7c36ea11da4273ecf20f50");
  ASSERT_EQ(field.toString(state[15],16), "254e179b1f643318769c2480e0bdbc9f8e0aaeda3bb50be1284c184c0ce9d2a4");
}

TEST(BN128_POSEIDON_TEST, hash_t17) {
  PoseidonBN128 p;
  RawFr field;
  const size_t t = 17;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) { field.fromUI(state[i], (unsigned long int)(i)); }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "16159a551cbb66108281a48099fff949ae08afd7f1f2ec06de2ffb96b919b765");
  ASSERT_EQ(field.toString(state[16],16), "ffa1bd9b53dbedee9ab5742283c8968d0435c3b3a566fcb66ca61ce04a5b5bf");
}

// ==============================================================================
// Poseidon2 tests for all supported t values (t=2,3,4,8,12,16)
// ==============================================================================

TEST(BN128_POSEIDON2_TEST, hash_t2) {
  Poseidon2BN128 p;
  RawFr field;
  const size_t t = 2;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) {
    field.fromUI(state[i], (unsigned long int)(i));
  }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "1d01e56f49579cec72319e145f06f6177f6c5253206e78c2689781452a31878b");
  ASSERT_EQ(field.toString(state[1],16), "d189ec589c41b8cffa88cfc523618a055abe8192c70f75aa72fc514560f6c61");
}

TEST(BN128_POSEIDON2_TEST, hash_t3) {
  Poseidon2BN128 p;
  RawFr field;
  const size_t t = 3;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) {
    field.fromUI(state[i], (unsigned long int)(i));
  }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "bb61d24daca55eebcb1929a82650f328134334da98ea4f847f760054f4a3033");
  ASSERT_EQ(field.toString(state[2],16), "1ed25194542b12eef8617361c3ba7c52e660b145994427cc86296242cf766ec8");
}

TEST(BN128_POSEIDON2_TEST, hash_t4) {
  Poseidon2BN128 p;
  RawFr field;
  const size_t t = 4;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) {
    field.fromUI(state[i], (unsigned long int)(i));
  }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "1bd538c2ee014ed5141b29e9ae240bf8db3fe5b9a38629a9647cf8d76c01737");
  ASSERT_EQ(field.toString(state[3],16), "2e11c5cff2a22c64d01304b778d78f6998eff1ab73163a35603f54794c30847a");
}

TEST(BN128_POSEIDON2_TEST, hash_t8) {
  Poseidon2BN128 p;
  RawFr field;
  const size_t t = 8;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) {
    field.fromUI(state[i], (unsigned long int)(i));
  }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "1d1a50bcde871247856df135d56a4ca61af575f1140ed9b1503c77528cf345df");
  ASSERT_EQ(field.toString(state[7],16), "b19bfa00c8f1d505074130e7f8b49a8624b1905e280ceca5ba11099b081b265");
}

TEST(BN128_POSEIDON2_TEST, hash_t12) {
  Poseidon2BN128 p;
  RawFr field;
  const size_t t = 12;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) {
    field.fromUI(state[i], (unsigned long int)(i));
  }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "3014e0ec17029f7e4f5cfe8c7c54fc3df6a5f7539f6aa304b2f3c747a9105618");
  ASSERT_EQ(field.toString(state[11],16), "905469a776b7d5a3f18841edb90fa0d8c6de479c2789c042dafefb367ad1a2b");
}

TEST(BN128_POSEIDON2_TEST, hash_t16) {
  Poseidon2BN128 p;
  RawFr field;
  const size_t t = 16;
  vector<RawFr::Element> state(t);
  for (size_t i = 0; i < t; i++) {
    field.fromUI(state[i], (unsigned long int)(i));
  }
  p.hash(state);
  ASSERT_EQ(field.toString(state[0],16), "fc2e6b758f493969e1d860f9a44ee3bdffdf796f382aa4ffb16fa4e9bcc333f");
  ASSERT_EQ(field.toString(state[15],16), "e2ceb1f8fde5f80be1f41bd239fabdc2f6133a6a98920a55c42891c3a925152");
}

#if 0
// This test is just to generate the constants for Poseidon2BN128 in montgomery form
TEST(CONVERTER, poseidon_seq_widths_sanity) {
	  Poseidon2BN128 p;
    RawFr field;

    //takes a number in exadecimal
    RawFr::Element number;
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

// ==============================================================================
// Multiexp Tests
// ==============================================================================

TEST(BN128_MULTIEXP_TEST, multiexp_4_operands) {
  typedef AltBn128::Engine Engine;
  Engine engine;
  RawFr field;
  
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

// ==============================================================================
// FFT Tests
// ============================================================================== 

TEST(BN128_FFT_TEST, fft_then_ifft_roundtrip) {
  // Test: fft followed by ifft should recover the original data
  RawFr field;
  const uint64_t n = 16;
  
  FFT<RawFr> fft(n);
  
  std::vector<RawFr::Element> data(n);
  std::vector<RawFr::Element> original(n);
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
  RawFr field;
  const uint64_t n = 16;
  
  FFT<RawFr> fft(n);
  
  std::vector<RawFr::Element> data(n);
  std::vector<RawFr::Element> original(n);
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
  RawFr field;
  const uint64_t n = 16;
  
  FFT<RawFr> fft(n);
  
  // Create two input vectors
  std::vector<RawFr::Element> a(n);
  std::vector<RawFr::Element> b(n);
  std::vector<RawFr::Element> a_plus_b(n);
  
  for (uint64_t i = 0; i < n; i++) {
    field.fromUI(a[i], i + 1);           // a = [1, 2, 3, ..., 16]
    field.fromUI(b[i], (i * 7) % 13);    // b = some different pattern
    field.add(a_plus_b[i], a[i], b[i]);  // a_plus_b = a + b
  }
  
  // Compute fft(a), fft(b), fft(a+b)
  std::vector<RawFr::Element> fft_a(a);
  std::vector<RawFr::Element> fft_b(b);
  std::vector<RawFr::Element> fft_a_plus_b(a_plus_b);
  
  fft.fft(fft_a.data(), n);
  fft.fft(fft_b.data(), n);
  fft.fft(fft_a_plus_b.data(), n);
  
  // Verify: fft(a+b) == fft(a) + fft(b)
  for (uint64_t i = 0; i < n; i++) {
    RawFr::Element expected_sum;
    field.add(expected_sum, fft_a[i], fft_b[i]);
    
    ASSERT_TRUE(field.eq(fft_a_plus_b[i], expected_sum)) 
      << "Linearity failed at index " << i 
      << ": fft(a+b) = " << field.toString(fft_a_plus_b[i], 10)
      << ", fft(a) + fft(b) = " << field.toString(expected_sum, 10);
  }
}

// ==============================================================================
// Poseidon linearHash test
// ==============================================================================

TEST(BN128_POSEIDON_TEST, linearHash_100_elements) {
  PoseidonBN128 p;
  RawFr field;
  
  // Create 100 Goldilocks elements [0, 1, 2, ..., 99]
  Goldilocks::Element input[100];
  for (size_t i = 0; i < 100; i++) {
    input[i] = Goldilocks::fromU64(i);
  }
  
  RawFr::Element output;
  p.linearHash(&output, input, 100, 17, false);
  
  ASSERT_EQ(field.toString(output, 16), "f51f3d0104201ef2bf7424be924330f937e4504111c6b26f7c195afbcb9d6cd");
}

TEST(BN128_POSEIDON_TEST, linearHash_trace_4rows_100cols) {
  PoseidonBN128 p;
  RawFr field;
  
  const size_t rows = 4;
  const size_t cols = 100;
  
  // Create trace: 4 rows × 100 cols Goldilocks elements
  // Row i contains values [i*cols, i*cols+1, ..., i*cols+cols-1]
  std::vector<Goldilocks::Element> trace(rows * cols);
  for (size_t i = 0; i < rows * cols; i++) {
    trace[i] = Goldilocks::fromU64(i);
  }
  
  std::vector<RawFr::Element> output(rows);
  p.linearHash(output.data(), trace.data(), rows, cols, 17, false);
  
  // Verify all 4 rows
  ASSERT_EQ(field.toString(output[0], 16), "f51f3d0104201ef2bf7424be924330f937e4504111c6b26f7c195afbcb9d6cd");
  ASSERT_EQ(field.toString(output[1], 16), "12b276d381cf64df6b1732ec6e91ae30f1f8c60cc08959aa9f81ae3b00cae371");
  ASSERT_EQ(field.toString(output[2], 16), "160dcdd70c78a86409e231df7f4add4e32c9a92fe74ac92dff516d3d8fa728");
  ASSERT_EQ(field.toString(output[3], 16), "2393e4f4bbaadaee5acaf60590547ef52c7dfd553f856283726497304ad47b5b");
}

// ==============================================================================
// Poseidon merkletree test
// ==============================================================================

TEST(BN128_POSEIDON_TEST, merkletree_8rows_100cols) {
  PoseidonBN128 p;
  RawFr field;
  
  const size_t rows = 8;
  const size_t cols = 100;
  const size_t arity = 16;
  
  std::vector<Goldilocks::Element> trace(rows * cols);
  for (size_t i = 0; i < rows * cols; i++) {
    trace[i] = Goldilocks::fromU64(i);
  }
  
  const size_t numNodes = 17; 
  std::vector<RawFr::Element> tree(numNodes);
  
  p.merkletree(tree.data(), trace.data(), rows, cols, arity, false);
  
  ASSERT_EQ(field.toString(tree[0], 16), "f51f3d0104201ef2bf7424be924330f937e4504111c6b26f7c195afbcb9d6cd");
  ASSERT_EQ(field.toString(tree[1], 16), "12b276d381cf64df6b1732ec6e91ae30f1f8c60cc08959aa9f81ae3b00cae371");
  ASSERT_EQ(field.toString(tree[2], 16), "160dcdd70c78a86409e231df7f4add4e32c9a92fe74ac92dff516d3d8fa728");
  ASSERT_EQ(field.toString(tree[3], 16), "2393e4f4bbaadaee5acaf60590547ef52c7dfd553f856283726497304ad47b5b");
  ASSERT_EQ(field.toString(tree[4], 16), "69539124331a9758e99c6f6d9f4f86088a4e48aa51d643a87c85f2536820c91");
  ASSERT_EQ(field.toString(tree[5], 16), "3e5fef46738269d441b69cb79ed05afbc4cb319e81cb8500da2be30800fd478");
  ASSERT_EQ(field.toString(tree[6], 16), "26a631438c157a61dbfcf090f4da49ef1d7e05b3a0f1ace53df8aa5966334987");
  ASSERT_EQ(field.toString(tree[7], 16), "17bcde7b057013734be16c97b6397967fb42c57f10f9a85dbe92436bf1446f60");
  
  // Verify root (last node)
  ASSERT_EQ(field.toString(tree[numNodes - 1], 16), "b6d02c9e5ea48185580c371837a1ecc09d7cf40774e1ac6f8491819deb3824d");
}
// ==============================================================================
// Poseidon BN128 grinding test
// ==============================================================================

TEST(BN128_POSEIDON_TEST, grinding_cpu) {
  PoseidonBN128 p;
  RawFr field;
  
  constexpr uint8_t n_bits = 8;
  
  // Create input state with 3 elements
  vector<RawFr::Element> state(3);
  field.fromUI(state[0], 0x1234567890abcdefULL);
  field.fromUI(state[1], 0xfedcba0987654321ULL);
  field.fromUI(state[2], 0x0123456789abcdefULL);
  
  uint64_t nonce = UINT64_MAX;
  
  // Call CPU grinding function
  p.grinding(nonce, state, n_bits);
  
  // Verify we found a valid nonce
  ASSERT_NE(nonce, UINT64_MAX);
  
  // Verify the expected nonce value
  ASSERT_EQ(nonce, 1530ULL);
  
  // Verify the hash at nonce satisfies the grinding requirement
  uint64_t level = (1ULL << (64 - n_bits));
  
  // Compute the hash with the found nonce to verify
  vector<RawFr::Element> verifyState(state.size() + 2);
  verifyState[0] = field.zero();
  std::memcpy(&verifyState[1], &state[0], state.size() * sizeof(RawFr::Element));
  
  // Append nonce
  RawFr::Element tmp = field.zero();
  tmp.v[0] = nonce;
  field.toMontgomery(tmp, tmp);
  verifyState[state.size() + 1] = tmp;
  
  // Compute hash
  p.hash(verifyState);
  
  // Convert from Montgomery and check
  RawFr::Element res;
  field.fromMontgomery(res, verifyState[0]);
  
  // Check that res.v[0] < level
  ASSERT_LT(res.v[0], level);
}