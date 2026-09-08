// NTT_AltBn128 tests, ported from the pil-fflonk repository
// (0xPolygonHermez/pilfflonk, test/ntt_altbn128.test.cpp). They cover the
// multi-column blocked NTT and LDE, which sppark's single-column GPU shim and
// ffiasm's FFT do not provide.
#include <gtest/gtest.h>
#include <iostream>

#include <alt_bn128.hpp>
#include "../src/ntt/ntt_bn128.hpp"
#include <immintrin.h>

#define FFT_SIZE (1 << 4)
#define NUM_REPS 5
#define BLOWUP_FACTOR 1
#define NUM_COLUMNS 8
#define NPHASES 4

static AltBn128::Engine &E = AltBn128::Engine::engine;

TEST(BN128_TEST, ntt)
{
    AltBn128::FrElement *a = (AltBn128::FrElement *)malloc(FFT_SIZE * sizeof(AltBn128::FrElement));
    AltBn128::FrElement *initial = (AltBn128::FrElement *)malloc(FFT_SIZE * sizeof(AltBn128::FrElement));
    NTT_AltBn128 bn128ntt(E, FFT_SIZE);

    a[0] = E.fr.one();
    a[1] = E.fr.one();
    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        a[i] = E.fr.mul(a[i - 1], a[i - 2]);
    }

    memcpy(initial, a, FFT_SIZE * sizeof(AltBn128::FrElement));

    for (int i = 0; i < NUM_REPS; i++)
    {
        bn128ntt.NTT(a, a, FFT_SIZE);
        bn128ntt.INTT(a, a, FFT_SIZE);
    }

    for (int i = 0; i < FFT_SIZE; i++)
    {
        ASSERT_TRUE(E.fr.eq(a[i], initial[i]));
    }
    free(a);
    free(initial);
}

TEST(BN128_TEST, ntt_block)
{
    AltBn128::FrElement *a = (AltBn128::FrElement *)malloc(FFT_SIZE * NUM_COLUMNS * sizeof(AltBn128::FrElement));
    AltBn128::FrElement *initial = (AltBn128::FrElement *)malloc(FFT_SIZE * NUM_COLUMNS * sizeof(AltBn128::FrElement));
    NTT_AltBn128 bn128ntt(E, FFT_SIZE);

    for (uint i = 0; i < 2; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            E.fr.add(a[i * NUM_COLUMNS + j], E.fr.one(), E.fr.set(j));
        }
    }

    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            a[i * NUM_COLUMNS + j] = E.fr.add(a[NUM_COLUMNS * (i - 1) + j], a[NUM_COLUMNS * (i - 2) + j]);
        }
    }

    memcpy(initial, a, FFT_SIZE * NUM_COLUMNS * sizeof(AltBn128::FrElement));

    // Option 1: dst is a NULL pointer
    for (int i = 0; i < NUM_REPS; i++)
    {
        bn128ntt.NTT(NULL, a, FFT_SIZE, NUM_COLUMNS);
        bn128ntt.INTT(NULL, a, FFT_SIZE, NUM_COLUMNS);
    }

    for (int i = 0; i < FFT_SIZE * NUM_COLUMNS; i++)
    {
        ASSERT_TRUE(E.fr.eq(a[i], initial[i]));
    }

    // Option 2: dst = src
    for (int i = 0; i < NUM_REPS; i++)
    {
        bn128ntt.NTT(a, a, FFT_SIZE, NUM_COLUMNS);
        bn128ntt.INTT(a, a, FFT_SIZE, NUM_COLUMNS);
    }

    for (int i = 0; i < FFT_SIZE * NUM_COLUMNS; i++)
    {
        ASSERT_TRUE(E.fr.eq(a[i], initial[i]));
    }

    // Option 3: dst != src
    AltBn128::FrElement *dst = (AltBn128::FrElement *)malloc(FFT_SIZE * NUM_COLUMNS * sizeof(AltBn128::FrElement));
    for (int i = 0; i < NUM_REPS; i++)
    {
        bn128ntt.NTT(dst, a, FFT_SIZE, NUM_COLUMNS);
        for (uint64_t k = 0; k < FFT_SIZE * NUM_COLUMNS; ++k)
            a[k] = E.fr.zero();
        bn128ntt.INTT(a, dst, FFT_SIZE, NUM_COLUMNS);
    }

    for (int i = 0; i < FFT_SIZE * NUM_COLUMNS; i++)
    {
        ASSERT_TRUE(E.fr.eq(a[i], initial[i]));
    }

    // Option 4: different configurations of phases and blocks
    for (int i = 0; i < NUM_REPS; i++)
    {
        bn128ntt.NTT(a, a, FFT_SIZE, NUM_COLUMNS, NULL, 3, 5);
        bn128ntt.INTT(a, a, FFT_SIZE, NUM_COLUMNS, NULL, 4, 3);
    }

    for (int i = 0; i < FFT_SIZE * NUM_COLUMNS; i++)
    {
        ASSERT_TRUE(E.fr.eq(a[i], initial[i]));
    }

    // Option 5: out of range parameters
    for (int i = 0; i < NUM_REPS; i++)
    {
        bn128ntt.NTT(a, a, FFT_SIZE, NUM_COLUMNS, NULL, 3, 3000);
        bn128ntt.INTT(a, a, FFT_SIZE, NUM_COLUMNS, NULL, 4, -1);
    }

    for (int i = 0; i < FFT_SIZE * NUM_COLUMNS; i++)
    {
        ASSERT_TRUE(E.fr.eq(a[i], initial[i]));
    }
    free(a);
    free(initial);

    // Edge case:Try to call ntt with FFT_SIZE = 1 ncols=3
    uint64_t fft_size = 1;
    uint64_t ncols = 3;
    AltBn128::FrElement a1[3] = {E.fr.set(1), E.fr.set(2), E.fr.set(3)};
    AltBn128::FrElement b1[3];

    bn128ntt.NTT(b1, a1, fft_size, ncols);
    bn128ntt.INTT(a1, b1, fft_size, ncols);
    ASSERT_TRUE(E.fr.eq(a1[0], E.fr.set(1)));
    ASSERT_TRUE(E.fr.eq(a1[1], E.fr.set(2)));
    ASSERT_TRUE(E.fr.eq(a1[2], E.fr.set(3)));

    // Edge case:Try to call ntt with FFT_SIZE = 2 ncols=3
    fft_size = 2;
    ncols = 3;
    AltBn128::FrElement a2[6] = {E.fr.set(1), E.fr.set(2), E.fr.set(3), E.fr.set(4), E.fr.set(5), E.fr.set(6)};
    AltBn128::FrElement b2[6];

    bn128ntt.NTT(b2, a2, fft_size, ncols);
    bn128ntt.INTT(a2, b2, fft_size, ncols);

    ASSERT_TRUE(E.fr.eq(a2[0], E.fr.set(1)));
    ASSERT_TRUE(E.fr.eq(a2[1], E.fr.set(2)));
    ASSERT_TRUE(E.fr.eq(a2[2], E.fr.set(3)));
    ASSERT_TRUE(E.fr.eq(a2[3], E.fr.set(4)));
    ASSERT_TRUE(E.fr.eq(a2[4], E.fr.set(5)));
    ASSERT_TRUE(E.fr.eq(a2[5], E.fr.set(6)));

    // Edge case: It does not crash with size==0 or ncols==0
    fft_size = 0;
    ncols = 3;
    bn128ntt.NTT(b2, a2, fft_size, ncols);
    bn128ntt.INTT(a2, b2, fft_size, ncols);
    fft_size = 1;
    ncols = 0;
    bn128ntt.NTT(b2, a2, fft_size, ncols);
    bn128ntt.INTT(a2, b2, fft_size, ncols);
}
TEST(BN128_TEST, LDE)
{
    AltBn128::FrElement *a = (AltBn128::FrElement *)malloc((FFT_SIZE << BLOWUP_FACTOR) * sizeof(AltBn128::FrElement));
    NTT_AltBn128 bn128ntt(E, FFT_SIZE);
    NTT_AltBn128 gntt_extension(E, FFT_SIZE << BLOWUP_FACTOR);

    AltBn128::FrElement *zeros_array = (AltBn128::FrElement *)malloc(((FFT_SIZE << BLOWUP_FACTOR) - FFT_SIZE) * sizeof(AltBn128::FrElement));
#pragma omp parallel for
    for (uint i = 0; i < ((FFT_SIZE << BLOWUP_FACTOR) - FFT_SIZE); i++)
    {
        zeros_array[i] = E.fr.zero();
    }

    a[0] = E.fr.one();
    a[1] = E.fr.one();
    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        a[i] = E.fr.add(a[i - 1], a[i - 2]);
    }

    bn128ntt.INTT(a, a, FFT_SIZE);

    memcpy(&a[FFT_SIZE], zeros_array, ((FFT_SIZE << BLOWUP_FACTOR) - FFT_SIZE) * sizeof(AltBn128::FrElement));

    gntt_extension.NTT(a, a, (FFT_SIZE << BLOWUP_FACTOR));

    ASSERT_TRUE(E.fr.toString(a[0]) == "1");
    ASSERT_TRUE(E.fr.toString(a[1]) == "12199744136350215368893260239437927912690930103873125685655434110751806988426");
    ASSERT_TRUE(E.fr.toString(a[2]) == "1");
    ASSERT_TRUE(E.fr.toString(a[3]) == "13413063301420227361219474278892191680737530348053495782569314555289601361708");
    ASSERT_TRUE(E.fr.toString(a[4]) == "2");
    ASSERT_TRUE(E.fr.toString(a[5]) == "8641579953475746055265511917813648377992510994588574752020338167013607187603");
    ASSERT_TRUE(E.fr.toString(a[6]) == "3");
    ASSERT_TRUE(E.fr.toString(a[7]) == "6543734385231615553058210196989850624548993706027877887352631640240466658323");
    ASSERT_TRUE(E.fr.toString(a[8]) == "5");
    ASSERT_TRUE(E.fr.toString(a[9]) == "1250920374157260975352227630228177503808885970238776755589787667992397801644");
    ASSERT_TRUE(E.fr.toString(a[10]) == "8");
    ASSERT_TRUE(E.fr.toString(a[11]) == "9268750356330487353851625658018196733360444785989629409262459104413224659042");
    ASSERT_TRUE(E.fr.toString(a[12]) == "13");
    ASSERT_TRUE(E.fr.toString(a[13]) == "13795834510122633581296910149995637405625304893786359380617673775485765557649");
    ASSERT_TRUE(E.fr.toString(a[14]) == "21");
    ASSERT_TRUE(E.fr.toString(a[15]) == "91255014789166369597717561768512284961282717243299459087117686166970727286");
    ASSERT_TRUE(E.fr.toString(a[16]) == "34");
    ASSERT_TRUE(E.fr.toString(a[17]) == "19044458151952122174081266977819687909017942697553052334181622020172075989186");
    ASSERT_TRUE(E.fr.toString(a[18]) == "55");
    ASSERT_TRUE(E.fr.toString(a[19]) == "9081567416765402784526259883775816694088053926072873148470047536601399007151");
    ASSERT_TRUE(E.fr.toString(a[20]) == "89");
    ASSERT_TRUE(E.fr.toString(a[21]) == "7397102711100226531080959533356240960762606717855378675464928075327512109709");
    ASSERT_TRUE(E.fr.toString(a[22]) == "144");
    ASSERT_TRUE(E.fr.toString(a[23]) == "8769577039180267737118834959879535719373864238664784823189101445667032495675");
    ASSERT_TRUE(E.fr.toString(a[24]) == "233");
    ASSERT_TRUE(E.fr.toString(a[25]) == "18313534207715988974079866502695335562558788800641796079515729557776845196598");
    ASSERT_TRUE(E.fr.toString(a[26]) == "377");
    ASSERT_TRUE(E.fr.toString(a[27]) == "3307998818443504547147080446242603579171985572374516159168133167731509728340");
    ASSERT_TRUE(E.fr.toString(a[28]) == "610");
    ASSERT_TRUE(E.fr.toString(a[29]) == "6909797442482908571147555484888694775566591863006449210593380497691743098662");
    ASSERT_TRUE(E.fr.toString(a[30]) == "987");
    ASSERT_TRUE(E.fr.toString(a[31]) == "15188782283357152618008078794998867895572833466942250863149730297708700904900");

    free(a);
    free(zeros_array);
}
TEST(BN128_TEST, LDE_block)
{
    AltBn128::FrElement *a = (AltBn128::FrElement *)malloc((FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(AltBn128::FrElement));
    NTT_AltBn128 bn128ntt(E, FFT_SIZE);
    NTT_AltBn128 gntt_extension(E, FFT_SIZE << BLOWUP_FACTOR);

    for (uint i = 0; i < 2; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            a[i * NUM_COLUMNS + j] = E.fr.add(E.fr.one(), E.fr.set(j));
        }
    }

    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            a[i * NUM_COLUMNS + j] = E.fr.add(a[NUM_COLUMNS * (i - 1) + j], a[NUM_COLUMNS * (i - 2) + j]);
        }
    }

    bn128ntt.INTT(a, a, FFT_SIZE, NUM_COLUMNS, NULL, NPHASES);

#pragma omp parallel for schedule(static)
    for (uint i = FFT_SIZE * NUM_COLUMNS; i < (FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS; i++)
    {
        a[i] = E.fr.zero();
    }

    gntt_extension.NTT(a, a, (FFT_SIZE << BLOWUP_FACTOR), NUM_COLUMNS, NULL, NUM_PHASES);


    ASSERT_TRUE(E.fr.toString(a[0]) == "1");
    ASSERT_TRUE(E.fr.toString(a[1]) == "2");
    ASSERT_TRUE(E.fr.toString(a[2]) == "3");
    ASSERT_TRUE(E.fr.toString(a[3]) == "4");
    ASSERT_TRUE(E.fr.toString(a[4]) == "5");
    ASSERT_TRUE(E.fr.toString(a[5]) == "6");
    ASSERT_TRUE(E.fr.toString(a[6]) == "7");
    ASSERT_TRUE(E.fr.toString(a[7]) == "8");
    ASSERT_TRUE(E.fr.toString(a[8]) == "12199744136350215368893260239437927912690930103873125685655434110751806988426");
    ASSERT_TRUE(E.fr.toString(a[9]) == "2511245400861155515540114733618580736833495807330217027612664034927805481235");
    ASSERT_TRUE(E.fr.toString(a[10]) == "14710989537211370884433374973056508649524425911203342713268098145679612469661");
    ASSERT_TRUE(E.fr.toString(a[11]) == "5022490801722311031080229467237161473666991614660434055225328069855610962470");
    ASSERT_TRUE(E.fr.toString(a[12]) == "17222234938072526399973489706675089386357921718533559740880762180607417950896");
    ASSERT_TRUE(E.fr.toString(a[13]) == "7533736202583466546620344200855742210500487421990651082837992104783416443705");
    ASSERT_TRUE(E.fr.toString(a[14]) == "19733480338933681915513604440293670123191417525863776768493426215535223432131");
    ASSERT_TRUE(E.fr.toString(a[15]) == "10044981603444622062160458934474322947333983229320868110450656139711221924940");
    ASSERT_TRUE(E.fr.toString(a[240]) == "987");
    ASSERT_TRUE(E.fr.toString(a[241]) == "1974");
    ASSERT_TRUE(E.fr.toString(a[242]) == "2961");
    ASSERT_TRUE(E.fr.toString(a[243]) == "3948");
    ASSERT_TRUE(E.fr.toString(a[244]) == "4935");
    ASSERT_TRUE(E.fr.toString(a[245]) == "5922");
    ASSERT_TRUE(E.fr.toString(a[246]) == "6909");
    ASSERT_TRUE(E.fr.toString(a[247]) == "7896");
    ASSERT_TRUE(E.fr.toString(a[248]) == "15188782283357152618008078794998867895572833466942250863149730297708700904900");
    ASSERT_TRUE(E.fr.toString(a[249]) == "8489321694875030013769751844740460702597302533468467382601256408841593314183");
    ASSERT_TRUE(E.fr.toString(a[250]) == "1789861106392907409531424894482053509621771599994683902052782519974485723466");
    ASSERT_TRUE(E.fr.toString(a[251]) == "16978643389750060027539503689480921405194605066936934765202512817683186628366");
    ASSERT_TRUE(E.fr.toString(a[252]) == "10279182801267937423301176739222514212219074133463151284654038928816079037649");
    ASSERT_TRUE(E.fr.toString(a[253]) == "3579722212785814819062849788964107019243543199989367804105565039948971446932");
    ASSERT_TRUE(E.fr.toString(a[254]) == "18768504496142967437070928583962974914816376666931618667255295337657672351832");
    ASSERT_TRUE(E.fr.toString(a[255]) == "12069043907660844832832601633704567721840845733457835186706821448790564761115");

    free(a);
}
TEST(BN128_TEST, extendePol)
{

    AltBn128::FrElement *a = (AltBn128::FrElement *)malloc((FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(AltBn128::FrElement));
    AltBn128::FrElement *b = (AltBn128::FrElement *)malloc((FFT_SIZE << BLOWUP_FACTOR) * NUM_COLUMNS * sizeof(AltBn128::FrElement));

    NTT_AltBn128 ntt(E,FFT_SIZE);

    for (uint i = 0; i < 2; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            a[i * NUM_COLUMNS + j] = E.fr.add(E.fr.one(), E.fr.set(j));
        }
    }

    for (uint64_t i = 2; i < FFT_SIZE; i++)
    {
        for (uint j = 0; j < NUM_COLUMNS; j++)
        {
            a[i * NUM_COLUMNS + j] = E.fr.add(a[NUM_COLUMNS * (i - 1) + j], a[NUM_COLUMNS * (i - 2) + j]);
        }
    }

    ntt.extendPol(a, a, FFT_SIZE << BLOWUP_FACTOR, FFT_SIZE, NUM_COLUMNS, b);

    ASSERT_TRUE(E.fr.toString(a[0]) == "1");
    ASSERT_TRUE(E.fr.toString(a[1]) == "2");
    ASSERT_TRUE(E.fr.toString(a[2]) == "3");
    ASSERT_TRUE(E.fr.toString(a[3]) == "4");
    ASSERT_TRUE(E.fr.toString(a[4]) == "5");
    ASSERT_TRUE(E.fr.toString(a[5]) == "6");
    ASSERT_TRUE(E.fr.toString(a[6]) == "7");
    ASSERT_TRUE(E.fr.toString(a[7]) == "8");
    ASSERT_TRUE(E.fr.toString(a[8]) == "12199744136350215368893260239437927912690930103873125685655434110751806988426");
    ASSERT_TRUE(E.fr.toString(a[9]) == "2511245400861155515540114733618580736833495807330217027612664034927805481235");
    ASSERT_TRUE(E.fr.toString(a[10]) == "14710989537211370884433374973056508649524425911203342713268098145679612469661");
    ASSERT_TRUE(E.fr.toString(a[11]) == "5022490801722311031080229467237161473666991614660434055225328069855610962470");
    ASSERT_TRUE(E.fr.toString(a[12]) == "17222234938072526399973489706675089386357921718533559740880762180607417950896");
    ASSERT_TRUE(E.fr.toString(a[13]) == "7533736202583466546620344200855742210500487421990651082837992104783416443705");
    ASSERT_TRUE(E.fr.toString(a[14]) == "19733480338933681915513604440293670123191417525863776768493426215535223432131");
    ASSERT_TRUE(E.fr.toString(a[15]) == "10044981603444622062160458934474322947333983229320868110450656139711221924940");
    ASSERT_TRUE(E.fr.toString(a[240]) == "987");
    ASSERT_TRUE(E.fr.toString(a[241]) == "1974");
    ASSERT_TRUE(E.fr.toString(a[242]) == "2961");
    ASSERT_TRUE(E.fr.toString(a[243]) == "3948");
    ASSERT_TRUE(E.fr.toString(a[244]) == "4935");
    ASSERT_TRUE(E.fr.toString(a[245]) == "5922");
    ASSERT_TRUE(E.fr.toString(a[246]) == "6909");
    ASSERT_TRUE(E.fr.toString(a[247]) == "7896");
    ASSERT_TRUE(E.fr.toString(a[248]) == "15188782283357152618008078794998867895572833466942250863149730297708700904900");
    ASSERT_TRUE(E.fr.toString(a[249]) == "8489321694875030013769751844740460702597302533468467382601256408841593314183");
    ASSERT_TRUE(E.fr.toString(a[250]) == "1789861106392907409531424894482053509621771599994683902052782519974485723466");
    ASSERT_TRUE(E.fr.toString(a[251]) == "16978643389750060027539503689480921405194605066936934765202512817683186628366");
    ASSERT_TRUE(E.fr.toString(a[252]) == "10279182801267937423301176739222514212219074133463151284654038928816079037649");
    ASSERT_TRUE(E.fr.toString(a[253]) == "3579722212785814819062849788964107019243543199989367804105565039948971446932");
    ASSERT_TRUE(E.fr.toString(a[254]) == "18768504496142967437070928583962974914816376666931618667255295337657672351832");
    ASSERT_TRUE(E.fr.toString(a[255]) == "12069043907660844832832601633704567721840845733457835186706821448790564761115");

    free(a);
    free(b);
}