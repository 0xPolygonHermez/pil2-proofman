#include "test_helpers.hpp"

#define FFT_SIZE (1 << 4)
#define NUM_REPS 5
#define BLOWUP_FACTOR 1
#define NUM_COLUMNS 8
#define NPHASES 4
#define NBLOCKS 1

TEST(GOLDILOCKS_TEST, ntt)
{
    Goldilocks::Element *a = (Goldilocks::Element *)malloc(FFT_SIZE * sizeof(Goldilocks::Element));
    Goldilocks::Element *initial = (Goldilocks::Element *)malloc(FFT_SIZE * sizeof(Goldilocks::Element));
    NTT_Goldilocks gntt(FFT_SIZE);

    a[0] = Goldilocks::one();
    a[1] = Goldilocks::one();
    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        a[i] = a[i - 1] + a[i - 2];
    }

    std::memcpy(initial, a, FFT_SIZE * sizeof(Goldilocks::Element));

    for (int i = 0; i < NUM_REPS; i++)
    {
        gntt.NTT(a, a, FFT_SIZE);
        gntt.INTT(a, a, FFT_SIZE);
    }

    for (int i = 0; i < FFT_SIZE; i++)
    {
        ASSERT_EQ(Goldilocks::toU64(a[i]), Goldilocks::toU64(initial[i]));
    }
    free(a);
    free(initial);
}
TEST(GOLDILOCKS_TEST, ntt_block)
{
    Goldilocks::Element *a = (Goldilocks::Element *)malloc(FFT_SIZE * NUM_COLUMNS * sizeof(Goldilocks::Element));
    Goldilocks::Element *initial = (Goldilocks::Element *)malloc(FFT_SIZE * NUM_COLUMNS * sizeof(Goldilocks::Element));
    NTT_Goldilocks gntt(FFT_SIZE);

    for (uint i = 0; i < 2; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            Goldilocks::add(a[i * NUM_COLUMNS + j], Goldilocks::one(), Goldilocks::fromU64(j));
        }
    }

    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            a[i * NUM_COLUMNS + j] = a[NUM_COLUMNS * (i - 1) + j] + a[NUM_COLUMNS * (i - 2) + j];
        }
    }

    std::memcpy(initial, a, FFT_SIZE * NUM_COLUMNS * sizeof(Goldilocks::Element));

    // Option 1: dst is a NULL pointer
    for (int i = 0; i < NUM_REPS; i++)
    {
        gntt.NTT(NULL, a, FFT_SIZE, NUM_COLUMNS);
        gntt.INTT(NULL, a, FFT_SIZE, NUM_COLUMNS);
    }

    for (int i = 0; i < FFT_SIZE * NUM_COLUMNS; i++)
    {
        ASSERT_EQ(Goldilocks::toU64(a[i]), Goldilocks::toU64(initial[i]));
    }

    // Option 2: dst = src
    for (int i = 0; i < NUM_REPS; i++)
    {
        gntt.NTT(a, a, FFT_SIZE, NUM_COLUMNS);
        gntt.INTT(a, a, FFT_SIZE, NUM_COLUMNS);
    }

    for (int i = 0; i < FFT_SIZE * NUM_COLUMNS; i++)
    {
        ASSERT_EQ(Goldilocks::toU64(a[i]), Goldilocks::toU64(initial[i]));
    }

    // Option 3: dst != src
    Goldilocks::Element *dst = (Goldilocks::Element *)malloc(FFT_SIZE * NUM_COLUMNS * sizeof(Goldilocks::Element));
    for (int i = 0; i < NUM_REPS; i++)
    {
        gntt.NTT(dst, a, FFT_SIZE, NUM_COLUMNS);
        for (uint64_t k = 0; k < FFT_SIZE * NUM_COLUMNS; ++k)
            a[k] = Goldilocks::zero();
        gntt.INTT(a, dst, FFT_SIZE, NUM_COLUMNS);
    }

    for (int i = 0; i < FFT_SIZE * NUM_COLUMNS; i++)
    {
        ASSERT_EQ(Goldilocks::toU64(a[i]), Goldilocks::toU64(initial[i]));
    }

    // Option 4: different configurations of phases and blocks
    for (int i = 0; i < NUM_REPS; i++)
    {
        gntt.NTT(a, a, FFT_SIZE, NUM_COLUMNS, NULL, 3, 5);
        gntt.INTT(a, a, FFT_SIZE, NUM_COLUMNS, NULL, 4, 3);
    }

    for (int i = 0; i < FFT_SIZE * NUM_COLUMNS; i++)
    {
        ASSERT_EQ(Goldilocks::toU64(a[i]), Goldilocks::toU64(initial[i]));
    }
    
    // Option 5: same than 4 but with different src and dst buffers
    for (int i = 0; i < NUM_REPS; i++)
    {
        gntt.NTT(dst, a, FFT_SIZE, NUM_COLUMNS, NULL, 3, 5);
        gntt.INTT(a, dst, FFT_SIZE, NUM_COLUMNS, NULL, 4, 3);
    }

    for (int i = 0; i < FFT_SIZE * NUM_COLUMNS; i++)
    {
        ASSERT_EQ(Goldilocks::toU64(a[i]), Goldilocks::toU64(initial[i]));
    }

    // Option 6: out of range parameters
    for (int i = 0; i < NUM_REPS; i++)
    {
        gntt.NTT(a, a, FFT_SIZE, NUM_COLUMNS, NULL, 3, 3000);
        gntt.INTT(a, a, FFT_SIZE, NUM_COLUMNS, NULL, 4, -1);
    }

    for (int i = 0; i < FFT_SIZE * NUM_COLUMNS; i++)
    {
        ASSERT_EQ(Goldilocks::toU64(a[i]), Goldilocks::toU64(initial[i]));
    }
    free(a);
    free(initial);

    // Edge case:Try to call ntt with FFT_SIZE = 1 ncols=3
    uint64_t fft_size = 1;
    uint64_t ncols = 3;
    Goldilocks::Element a1[3] = {{1}, {2}, {3}};
    Goldilocks::Element b1[3];

    gntt.NTT(b1, a1, fft_size, ncols);
    ASSERT_EQ(Goldilocks::toU64(b1[0]), 1);
    ASSERT_EQ(Goldilocks::toU64(b1[1]), 2);
    ASSERT_EQ(Goldilocks::toU64(b1[2]), 3);

    gntt.INTT(a1, b1, fft_size, ncols);

    ASSERT_EQ(Goldilocks::toU64(a1[0]), 1);
    ASSERT_EQ(Goldilocks::toU64(a1[1]), 2);
    ASSERT_EQ(Goldilocks::toU64(a1[2]), 3);

    // Edge case:Try to call ntt with FFT_SIZE = 2 ncols=3
    fft_size = 2;
    ncols = 3;
    Goldilocks::Element a2[6] = {{1}, {2}, {3}, {4}, {5}, {6}};
    Goldilocks::Element b2[6];

    gntt.NTT(b2, a2, fft_size, ncols);
    gntt.INTT(a2, b2, fft_size, ncols);

    ASSERT_EQ(Goldilocks::toU64(a2[0]), 1);
    ASSERT_EQ(Goldilocks::toU64(a2[1]), 2);
    ASSERT_EQ(Goldilocks::toU64(a2[2]), 3);
    ASSERT_EQ(Goldilocks::toU64(a2[3]), 4);
    ASSERT_EQ(Goldilocks::toU64(a2[4]), 5);
    ASSERT_EQ(Goldilocks::toU64(a2[5]), 6);

    // Edge case: It does not crash with size==0 or ncols==0
    fft_size = 0;
    ncols = 3;
    gntt.NTT(b2, a2, fft_size, ncols);
    gntt.INTT(a2, b2, fft_size, ncols);
    fft_size = 1;
    ncols = 0;
    gntt.NTT(b2, a2, fft_size, ncols);
    gntt.INTT(a2, b2, fft_size, ncols);
}
TEST(GOLDILOCKS_TEST, LDE)
{
    Goldilocks::Element *a = (Goldilocks::Element *)malloc((FFT_SIZE << BLOWUP_FACTOR) * sizeof(Goldilocks::Element));
    NTT_Goldilocks gntt(FFT_SIZE);
    NTT_Goldilocks gntt_extension((FFT_SIZE << BLOWUP_FACTOR));

    Goldilocks::Element *zeros_array = (Goldilocks::Element *)malloc(((FFT_SIZE << BLOWUP_FACTOR) - FFT_SIZE) * sizeof(Goldilocks::Element));
#pragma omp parallel for
    for (uint i = 0; i < ((FFT_SIZE << BLOWUP_FACTOR) - FFT_SIZE); i++)
    {
        zeros_array[i] = Goldilocks::zero();
    }

    a[0] = Goldilocks::one();
    a[1] = Goldilocks::one();
    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        a[i] = a[i - 1] + a[i - 2];
    }

    Goldilocks::Element shift = Goldilocks::fromU64(49); // TODO: ask for this number, where to put it how to calculate it
    gntt.INTT(a, a, FFT_SIZE);

    // TODO: This can be pre-generated
    Goldilocks::Element *r = (Goldilocks::Element *)malloc(FFT_SIZE * sizeof(Goldilocks::Element));
    r[0] = Goldilocks::one();
    for (int i = 1; i < FFT_SIZE; i++)
    {
        r[i] = r[i - 1] * shift;
    }

#pragma omp parallel for
    for (int i = 0; i < FFT_SIZE; i++)
    {
        a[i] = a[i] * r[i];
    }

    std::memcpy(&a[FFT_SIZE], zeros_array, ((FFT_SIZE << BLOWUP_FACTOR) - FFT_SIZE) * sizeof(Goldilocks::Element));

    gntt_extension.NTT(a, a, (FFT_SIZE << BLOWUP_FACTOR));

    /*for (int k = 0; k < 32; ++k)
    {
        std::cout << std::showbase << std::hex << std::uppercase << Goldilocks::toU64(a[k]) << std::endl;
    }*/
    ASSERT_EQ(Goldilocks::toU64(a[0]), 0XCBA857825D02DA98);
    ASSERT_EQ(Goldilocks::toU64(a[1]), 0X46B25F2EB8DC45C6);
    ASSERT_EQ(Goldilocks::toU64(a[2]), 0X53CD52572B82CE93);
    ASSERT_EQ(Goldilocks::toU64(a[3]), 0X6A1C4033524890BC);
    ASSERT_EQ(Goldilocks::toU64(a[4]), 0XA9103D6B086AC1F6);
    ASSERT_EQ(Goldilocks::toU64(a[5]), 0XF9EDB8DE1C59C93D);
    ASSERT_EQ(Goldilocks::toU64(a[6]), 0XDAF72007263AED14);
    ASSERT_EQ(Goldilocks::toU64(a[7]), 0X4761FD742111A2C6);
    ASSERT_EQ(Goldilocks::toU64(a[8]), 0X91998C571BDAFBFE);
    ASSERT_EQ(Goldilocks::toU64(a[9]), 0X89B28028BF5894EC);
    ASSERT_EQ(Goldilocks::toU64(a[10]), 0XDD2FD6CB9F5A0A28);
    ASSERT_EQ(Goldilocks::toU64(a[11]), 0X43C4A931E1A7D68B);
    ASSERT_EQ(Goldilocks::toU64(a[12]), 0X88EB7870B0E49F21);
    ASSERT_EQ(Goldilocks::toU64(a[13]), 0X99A28535EABA76E9);
    ASSERT_EQ(Goldilocks::toU64(a[14]), 0XC05CC85A86046420);
    ASSERT_EQ(Goldilocks::toU64(a[15]), 0XE1DED0726EC6AB22);
    ASSERT_EQ(Goldilocks::toU64(a[16]), 0XFF4F0AFB9C48AA53);
    ASSERT_EQ(Goldilocks::toU64(a[17]), 0X2B3524757554A236);
    ASSERT_EQ(Goldilocks::toU64(a[18]), 0XB867D06B39F63E5B);
    ASSERT_EQ(Goldilocks::toU64(a[19]), 0X9D65B701D0DC0203);
    ASSERT_EQ(Goldilocks::toU64(a[20]), 0XDB653DED8EB0E8B1);
    ASSERT_EQ(Goldilocks::toU64(a[21]), 0X6431B1E66D89DEB8);
    ASSERT_EQ(Goldilocks::toU64(a[22]), 0XF1CB543225A25142);
    ASSERT_EQ(Goldilocks::toU64(a[23]), 0X199DD3926164C43A);
    ASSERT_EQ(Goldilocks::toU64(a[24]), 0XA7B8E1EFC3CFBBF5);
    ASSERT_EQ(Goldilocks::toU64(a[25]), 0X186D4972B303DB54);
    ASSERT_EQ(Goldilocks::toU64(a[26]), 0X249276F9AF9641DF);
    ASSERT_EQ(Goldilocks::toU64(a[27]), 0X2B1235BB52390A00);
    ASSERT_EQ(Goldilocks::toU64(a[28]), 0XEE3147DB1601B67B);
    ASSERT_EQ(Goldilocks::toU64(a[29]), 0XB8B579BA5E655721);
    ASSERT_EQ(Goldilocks::toU64(a[30]), 0X650D467042BCD196);
    ASSERT_EQ(Goldilocks::toU64(a[31]), 0X8249D169442CB677);

    free(a);
    free(zeros_array);
    free(r);
}
TEST(GOLDILOCKS_TEST, LDE_block)
{
    Goldilocks::Element *a = (Goldilocks::Element *)malloc((FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
    NTT_Goldilocks gntt(FFT_SIZE);
    NTT_Goldilocks gntt_extension((FFT_SIZE << BLOWUP_FACTOR));

    for (uint i = 0; i < 2; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            Goldilocks::add(a[i * NUM_COLUMNS + j], Goldilocks::one(), Goldilocks::fromU64(j));
        }
    }

    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            a[i * NUM_COLUMNS + j] = a[NUM_COLUMNS * (i - 1) + j] + a[NUM_COLUMNS * (i - 2) + j];
        }
    }

    gntt.INTT(a, a, FFT_SIZE, NUM_COLUMNS, NULL, NPHASES);

    // TODO: This can be pre-generated
    Goldilocks::Element *r = (Goldilocks::Element *)malloc(FFT_SIZE * sizeof(Goldilocks::Element));
    r[0] = Goldilocks::one();
    for (int i = 1; i < FFT_SIZE; i++)
    {
        r[i] = r[i - 1] * Goldilocks::shift();
    }

#pragma omp parallel for
    for (uint64_t i = 0; i < FFT_SIZE; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            a[i * NUM_COLUMNS + j] = a[NUM_COLUMNS * i + j] * r[i];
        }
    }
#pragma omp parallel for schedule(static)
    for (uint i = FFT_SIZE * NUM_COLUMNS; i < (FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS; i++)
    {
        a[i] = Goldilocks::zero();
    }

    gntt_extension.NTT(a, a, (FFT_SIZE << BLOWUP_FACTOR), NUM_COLUMNS, NULL, NUM_PHASES);
    /*for (int k = 0; k < 32; ++k)
    {
        std::cout << std::showbase << std::hex << std::uppercase << Goldilocks::toU64(a[k * NUM_COLUMNS]) << std::endl;
    }*/
    ASSERT_EQ(Goldilocks::toU64(a[0 * NUM_COLUMNS]), 0X3E7CA26D67147C31);
    ASSERT_EQ(Goldilocks::toU64(a[1 * NUM_COLUMNS]), 0X1310720153E0ABE4);
    ASSERT_EQ(Goldilocks::toU64(a[2 * NUM_COLUMNS]), 0X20446D2EA50E8F96);
    ASSERT_EQ(Goldilocks::toU64(a[3 * NUM_COLUMNS]), 0XEAB91008C3444102);
    ASSERT_EQ(Goldilocks::toU64(a[4 * NUM_COLUMNS]), 0X68523AC1294A2);
    ASSERT_EQ(Goldilocks::toU64(a[5 * NUM_COLUMNS]), 0X8A0BB8A3EBA8260A);
    ASSERT_EQ(Goldilocks::toU64(a[6 * NUM_COLUMNS]), 0X515CEC478A438B2);
    ASSERT_EQ(Goldilocks::toU64(a[7 * NUM_COLUMNS]), 0XA087431602851263);
    ASSERT_EQ(Goldilocks::toU64(a[8 * NUM_COLUMNS]), 0XF09629139EA12C82);
    ASSERT_EQ(Goldilocks::toU64(a[9 * NUM_COLUMNS]), 0X175DC5A131392734);
    ASSERT_EQ(Goldilocks::toU64(a[10 * NUM_COLUMNS]), 0X72991CA43B50D824);
    ASSERT_EQ(Goldilocks::toU64(a[11 * NUM_COLUMNS]), 0XDE85A385ABE2A817);
    ASSERT_EQ(Goldilocks::toU64(a[12 * NUM_COLUMNS]), 0X281F1BF7178650C);
    ASSERT_EQ(Goldilocks::toU64(a[13 * NUM_COLUMNS]), 0XA0C663876DFF41A7);
    ASSERT_EQ(Goldilocks::toU64(a[14 * NUM_COLUMNS]), 0XD49C07EA43D3806C);
    ASSERT_EQ(Goldilocks::toU64(a[15 * NUM_COLUMNS]), 0XBCEB714F2E6B299A);
    ASSERT_EQ(Goldilocks::toU64(a[16 * NUM_COLUMNS]), 0XC46EE848F93207D8);
    ASSERT_EQ(Goldilocks::toU64(a[17 * NUM_COLUMNS]), 0XF70EC69883DEE2A);
    ASSERT_EQ(Goldilocks::toU64(a[18 * NUM_COLUMNS]), 0XEE28CDAF6C30F9D9);
    ASSERT_EQ(Goldilocks::toU64(a[19 * NUM_COLUMNS]), 0X6356B93C02C259B3);
    ASSERT_EQ(Goldilocks::toU64(a[20 * NUM_COLUMNS]), 0XD19A89639BC31A16);
    ASSERT_EQ(Goldilocks::toU64(a[21 * NUM_COLUMNS]), 0XB097AE217FC93344);
    ASSERT_EQ(Goldilocks::toU64(a[22 * NUM_COLUMNS]), 0X29BB681AF743F8F6);
    ASSERT_EQ(Goldilocks::toU64(a[23 * NUM_COLUMNS]), 0X8E874011A158B00B);
    ASSERT_EQ(Goldilocks::toU64(a[24 * NUM_COLUMNS]), 0XC95F0B718235B6D7);
    ASSERT_EQ(Goldilocks::toU64(a[25 * NUM_COLUMNS]), 0XFE51B4A575AFECA0);
    ASSERT_EQ(Goldilocks::toU64(a[26 * NUM_COLUMNS]), 0XC68CF305A6F17F4F);
    ASSERT_EQ(Goldilocks::toU64(a[27 * NUM_COLUMNS]), 0XC7912AE75E2DD36D);
    ASSERT_EQ(Goldilocks::toU64(a[28 * NUM_COLUMNS]), 0X6EFC40795CF38959);
    ASSERT_EQ(Goldilocks::toU64(a[29 * NUM_COLUMNS]), 0X6BD4745D238824D9);
    ASSERT_EQ(Goldilocks::toU64(a[30 * NUM_COLUMNS]), 0XB4FF76AAC16372AA);
    ASSERT_EQ(Goldilocks::toU64(a[31 * NUM_COLUMNS]), 0XA0705C72DD9F9A2F);

    free(a);
    free(r);
}
TEST(GOLDILOCKS_TEST, extendePol)
{

    Goldilocks::Element *a = (Goldilocks::Element *)malloc((FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));
    Goldilocks::Element *b = (Goldilocks::Element *)malloc((FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(Goldilocks::Element));

    NTT_Goldilocks ntt(FFT_SIZE);

    for (uint i = 0; i < 2; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            Goldilocks::add(a[i * NUM_COLUMNS + j], Goldilocks::one(), Goldilocks::fromU64(j));
        }
    }

    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            a[i * NUM_COLUMNS + j] = a[NUM_COLUMNS * (i - 1) + j] + a[NUM_COLUMNS * (i - 2) + j];
        }
    }

    ntt.LDE(a, a, FFT_SIZE << BLOWUP_FACTOR, FFT_SIZE, NUM_COLUMNS, b);

    ASSERT_EQ(Goldilocks::toU64(a[0 * NUM_COLUMNS]), 0X3E7CA26D67147C31);
    ASSERT_EQ(Goldilocks::toU64(a[1 * NUM_COLUMNS]), 0X1310720153E0ABE4);
    ASSERT_EQ(Goldilocks::toU64(a[2 * NUM_COLUMNS]), 0X20446D2EA50E8F96);
    ASSERT_EQ(Goldilocks::toU64(a[3 * NUM_COLUMNS]), 0XEAB91008C3444102);
    ASSERT_EQ(Goldilocks::toU64(a[4 * NUM_COLUMNS]), 0X68523AC1294A2);
    ASSERT_EQ(Goldilocks::toU64(a[5 * NUM_COLUMNS]), 0X8A0BB8A3EBA8260A);
    ASSERT_EQ(Goldilocks::toU64(a[6 * NUM_COLUMNS]), 0X515CEC478A438B2);
    ASSERT_EQ(Goldilocks::toU64(a[7 * NUM_COLUMNS]), 0XA087431602851263);
    ASSERT_EQ(Goldilocks::toU64(a[8 * NUM_COLUMNS]), 0XF09629139EA12C82);
    ASSERT_EQ(Goldilocks::toU64(a[9 * NUM_COLUMNS]), 0X175DC5A131392734);
    ASSERT_EQ(Goldilocks::toU64(a[10 * NUM_COLUMNS]), 0X72991CA43B50D824);
    ASSERT_EQ(Goldilocks::toU64(a[11 * NUM_COLUMNS]), 0XDE85A385ABE2A817);
    ASSERT_EQ(Goldilocks::toU64(a[12 * NUM_COLUMNS]), 0X281F1BF7178650C);
    ASSERT_EQ(Goldilocks::toU64(a[13 * NUM_COLUMNS]), 0XA0C663876DFF41A7);
    ASSERT_EQ(Goldilocks::toU64(a[14 * NUM_COLUMNS]), 0XD49C07EA43D3806C);
    ASSERT_EQ(Goldilocks::toU64(a[15 * NUM_COLUMNS]), 0XBCEB714F2E6B299A);
    ASSERT_EQ(Goldilocks::toU64(a[16 * NUM_COLUMNS]), 0XC46EE848F93207D8);
    ASSERT_EQ(Goldilocks::toU64(a[17 * NUM_COLUMNS]), 0XF70EC69883DEE2A);
    ASSERT_EQ(Goldilocks::toU64(a[18 * NUM_COLUMNS]), 0XEE28CDAF6C30F9D9);
    ASSERT_EQ(Goldilocks::toU64(a[19 * NUM_COLUMNS]), 0X6356B93C02C259B3);
    ASSERT_EQ(Goldilocks::toU64(a[20 * NUM_COLUMNS]), 0XD19A89639BC31A16);
    ASSERT_EQ(Goldilocks::toU64(a[21 * NUM_COLUMNS]), 0XB097AE217FC93344);
    ASSERT_EQ(Goldilocks::toU64(a[22 * NUM_COLUMNS]), 0X29BB681AF743F8F6);
    ASSERT_EQ(Goldilocks::toU64(a[23 * NUM_COLUMNS]), 0X8E874011A158B00B);
    ASSERT_EQ(Goldilocks::toU64(a[24 * NUM_COLUMNS]), 0XC95F0B718235B6D7);
    ASSERT_EQ(Goldilocks::toU64(a[25 * NUM_COLUMNS]), 0XFE51B4A575AFECA0);
    ASSERT_EQ(Goldilocks::toU64(a[26 * NUM_COLUMNS]), 0XC68CF305A6F17F4F);
    ASSERT_EQ(Goldilocks::toU64(a[27 * NUM_COLUMNS]), 0XC7912AE75E2DD36D);
    ASSERT_EQ(Goldilocks::toU64(a[28 * NUM_COLUMNS]), 0X6EFC40795CF38959);
    ASSERT_EQ(Goldilocks::toU64(a[29 * NUM_COLUMNS]), 0X6BD4745D238824D9);
    ASSERT_EQ(Goldilocks::toU64(a[30 * NUM_COLUMNS]), 0XB4FF76AAC16372AA);
    ASSERT_EQ(Goldilocks::toU64(a[31 * NUM_COLUMNS]), 0XA0705C72DD9F9A2F);

    free(a);
    free(b);
}


TEST(GOLDILOCKS_TEST, LDE_correctness)
{
    struct Case { uint64_t N, NExt, ncols; };
    const Case cases[] = {
        {16,  64,  1},
        {16,  32,  4},
        {64,  256, 8},
    };

    for (const auto &c : cases) {
        const uint64_t N = c.N, NExt = c.NExt, ncols = c.ncols;

        // log2(NExt): NExt is always a power of 2
        uint64_t log2NExt = 0;
        for (uint64_t tmp = NExt; tmp > 1; tmp >>= 1) ++log2NExt;

        // Polynomial coefficients coeff[k*ncols + col] for degree-N polynomial p
        std::vector<Goldilocks::Element> coeff(N * ncols);
        for (uint64_t k = 0; k < N; ++k)
            for (uint64_t col = 0; col < ncols; ++col)
                coeff[k * ncols + col] =
                    Goldilocks::fromU64((k * ncols + col + 1) * 1000003ULL);

        // input = NTT(coeff, N) = evaluations of p at plain N-th roots {omega_N^j}
        // LDE internally applies the coset shift via INTT(extend=true)
        NTT_Goldilocks ntt_N(N);
        std::vector<Goldilocks::Element> input(N * ncols);
        ntt_N.NTT(input.data(), coeff.data(), N, ncols);

        // LDE: evaluations of p at the NExt-point coset {shift * omega_NExt^j}
        std::vector<Goldilocks::Element> output(NExt * ncols, Goldilocks::zero());
        ntt_N.LDE(output.data(), input.data(), NExt, N, ncols);

        // Reference: Horner evaluation of p(shift * omega_NExt^j) for each j
        // p(x) = coeff[0] + coeff[1]*x + ... + coeff[N-1]*x^{N-1}
        Goldilocks::Element omega = Goldilocks::w(log2NExt);  // primitive NExt-th root
        Goldilocks::Element shift = Goldilocks::shift();
        Goldilocks::Element omega_j = Goldilocks::one();
        for (uint64_t j = 0; j < NExt; ++j) {
            Goldilocks::Element x;  // shift * omega_NExt^j
            Goldilocks::mul(x, shift, omega_j);

            for (uint64_t col = 0; col < ncols; ++col) {
                Goldilocks::Element val = coeff[(N - 1) * ncols + col];
                for (int64_t k = (int64_t)N - 2; k >= 0; --k) {
                    Goldilocks::mul(val, val, x);
                    Goldilocks::add(val, val, coeff[k * ncols + col]);
                }
                ASSERT_EQ(Goldilocks::toU64(output[j * ncols + col]),
                          Goldilocks::toU64(val))
                    << "Mismatch at j=" << j << " col=" << col
                    << " (N=" << N << " NExt=" << NExt << " ncols=" << ncols << ")";
            }
            Goldilocks::mul(omega_j, omega_j, omega);
        }
    }
}

// Standalone INTT: verify INTT(NTT(x)) = x explicitly
TEST(GOLDILOCKS_TEST, intt_standalone_roundtrip)
{
    const uint64_t N = 16;
    const uint64_t ncols = 3;
    NTT_Goldilocks gntt(N);

    std::vector<Goldilocks::Element> orig(N * ncols), a(N * ncols), b(N * ncols);
    for (uint64_t i = 0; i < N * ncols; ++i)
        orig[i] = Goldilocks::fromU64(i * 31 + 7);

    std::memcpy(a.data(), orig.data(), N * ncols * sizeof(Goldilocks::Element));

    // Forward NTT
    gntt.NTT(b.data(), a.data(), N, ncols);
    // Inverse NTT
    gntt.INTT(a.data(), b.data(), N, ncols);

    for (uint64_t i = 0; i < N * ncols; ++i)
        ASSERT_EQ(Goldilocks::toU64(a[i]), Goldilocks::toU64(orig[i]))
            << "INTT(NTT(x)) ≠ x at i=" << i;
}
