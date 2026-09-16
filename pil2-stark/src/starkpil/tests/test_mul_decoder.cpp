// Decoder self-check tests: each case builds a table's fixed column the way its PIL does, then
// requires decoder(value(r)) == r for every row. The negative cases matter as much as the positive
// ones -- that property is what turns a stale bias into a setup-time error.
//
// Build and run:
//   g++ -O2 -std=c++17 -DMUL_DECODER_TEST_MAIN -o /tmp/test_mul_decoder \
//       src/starkpil/tests/test_mul_decoder.cpp && /tmp/test_mul_decoder
//
// The MUL_DECODER_TEST_MAIN guard matters: the Makefile globs every .cpp under ./src into libstarks,
// and a stray `main` in a static library collides with the host binary's.
#include "../multiplicity_decoders.hpp"
#include <cstdio>
#include <vector>
#include <string>

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

#ifdef MUL_DECODER_TEST_MAIN
int main() {
    std::string err;

    // ---- predefined range [0, 2^16-1]: row r holds value r, bias 0 --------------------------
    {
        const uint64_t N = 65536;
        std::vector<uint64_t> col(N);
        for (uint64_t r = 0; r < N; ++r) col[r] = r;

        MulDecoder d{}; d.table_id = 104; d.n_rows = N;
        check(mul_decoder_selfcheck(d, col.data(), N, err), "range [0,2^16) accepted");

        MulDecoder bad = d; bad.bias = 1;
        check(!mul_decoder_selfcheck(bad, col.data(), N, err), "wrong bias rejected");
    }

    // ---- specified range [min, max] with min > 0: row r holds min + r, bias -min ------------
    {
        const uint64_t N = 1024;
        const int64_t  MIN = 4096;
        std::vector<uint64_t> col(N);
        for (uint64_t r = 0; r < N; ++r) col[r] = (uint64_t)MIN + r;

        MulDecoder d{}; d.table_id = 330; d.n_rows = N; d.bias = -MIN;
        check(mul_decoder_selfcheck(d, col.data(), N, err), "specified range, bias = -min accepted");

        MulDecoder bad = d; bad.bias = MIN;
        check(!mul_decoder_selfcheck(bad, col.data(), N, err), "bias of the wrong sign rejected");
    }

    // ---- negative min: the first rows are field elements just below p, so the add must wrap --
    {
        const uint64_t N = 8;
        std::vector<uint64_t> col(N);
        for (uint64_t r = 0; r < N; ++r) col[r] = (r < 3) ? MUL_P - (3 - r) : r - 3;

        MulDecoder d{}; d.table_id = 331; d.n_rows = N; d.bias = 3;
        check(mul_decoder_selfcheck(d, col.data(), N, err), "negative-min range accepted");
    }

    // ---- unreduced input: the evaluator can hand back p + x --------------------------------
    {
        const uint64_t N = 256;
        std::vector<uint64_t> col(N);
        for (uint64_t r = 0; r < N; ++r) col[r] = MUL_P + r;   // deliberately not canonical

        MulDecoder d{}; d.table_id = 106; d.n_rows = N;
        check(mul_decoder_selfcheck(d, col.data(), N, err), "non-canonical values canonicalised");
    }

    // ---- table 125's shape: indexed-base row map, two selector fields + two strides ---------
    {
        MulDecoder d{};
        uint64_t sel[]    = {0, 0, 3, 1, 4, 1};   // (col, shift, mask) x 2
        uint64_t base[]   = {0, 65536, 131072, 196608, 262144, 327680,
                             MUL_DIGIT_INVALID, MUL_DIGIT_INVALID};
        uint64_t stride[] = {2, 1, 3, 256};        // (col, stride) x 2
        mulSetIndexedBase(d, sel, 2, base, 8, stride, 2);

        uint64_t row = 0;
        uint64_t key1[] = {1, 1u << 4, 5, 2};
        check(mulResolveRow(key1, 4, 0, nullptr, row, 0, nullptr, &d) && row == 196608u + 5u + 512u,
              "indexed-base resolves row from selector + strides");

        uint64_t key2[] = {3, 0, 0, 0};            // idx 6 is INVALID
        check(!mulResolveRow(key2, 4, 0, nullptr, row, 0, nullptr, &d),
              "indexed-base misses outside the table");
    }

    printf("\ntest_mul_decoder: %s\n", failures ? "FAILURES" : "all passed");
    return failures ? 1 : 0;
}
#endif
