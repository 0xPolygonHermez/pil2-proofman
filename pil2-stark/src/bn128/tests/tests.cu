
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "bn128.cuh"
#include "poseidon2_bn128.cuh"

__global__ void kernel_add_one_one(int* ok);

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
__global__ void kernel_add_one_one(int* ok)
{
    BN128GPUScalarField::Element a;
    BN128GPUScalarField::Element b;
    BN128GPUScalarField::Element r;

    a.v[0] = 1;
    b.v[0] = 1;
    for(int i = 1; i < 8; ++i) {
        a.v[i] = 0;
        b.v[i] = 0;
    }    
    a.v.to(); // Convert to montgomery form
    b.v.to(); // Convert to montgomery form

    BN128GPUScalarField::add(r, a, b);

    r.v.from(); // Convert back from montgomery form
    
    bool same = true;    
    for(int i = 0; i < 8; ++i) {
        uint32_t expected = (i == 0) ? 2 : 0;
        if(r.v[i] != expected) {
            same = false;
        }
    }
    *ok = same;
}
#endif

TEST(BN128, add)
{
    int *d_ok = nullptr;
    int h_ok = 0;
    cudaMalloc(&d_ok, sizeof(int));
    cudaMemset(d_ok, 0, sizeof(int));

    kernel_add_one_one<<<1,1>>>(d_ok);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_ok, d_ok, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_ok);

    EXPECT_EQ(h_ok, 1);
}

/*TEST(BN128_POSEIDON2_TEST, hash) {
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
}*/

TEST(BN128_POSEIDON2_TEST, hash_gpu) {
  Poseidon2BN128GPU p;
  BN128GPUScalarField field;
  /*size_t t = 2;
  vector<BN128GPUScalarField::Element> state(t);
  for (size_t i = 0; i < t; i++)
  {
    field.fromUI(state[i], (unsigned long int)(i));
  }
  p.hash(state);        
  ASSERT_EQ(field.toString(state[0],16), 
  "1d01e56f49579cec72319e145f06f6177f6c5253206e78c2689781452a31878b");
  ASSERT_EQ(field.toString(state[1],16), 
  "d189ec589c41b8cffa88cfc523618a055abe8192c70f75aa72fc514560f6c61");*/
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
