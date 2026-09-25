// Decoder self-check tests: build a table's fixed column as its PIL does, then require
// decoder(value(r)) == r for every row; negative cases check a wrong bias is rejected.
//
// Build and run:
//   g++ -O2 -std=c++17 -DMUL_DECODER_TEST_MAIN -o /tmp/test_mul_decoder \
//       src/starkpil/tests/test_mul_decoder.cpp && /tmp/test_mul_decoder
//
// The guard keeps `main` out of libstarks (the Makefile globs every .cpp under ./src).
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

// Each row of the table must decode back to its own index: a stale bias fails here rather than
// silently miscounting.
inline bool mul_decoder_selfcheck(const MulDecoder& d, const uint64_t* column,
                                  uint64_t n_rows, std::string& err) {
    for (uint64_t r = 0; r < n_rows; ++r) {
        const uint64_t got = mul_decode(d, column[r]);
        if (got != r) {
            err = "decoder for table " + std::to_string(d.table_id) + " maps row "
                + std::to_string(r) + " to " + std::to_string(got);
            return false;
        }
    }
    return true;
}

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

    printf("\ntest_mul_decoder: %s\n", failures ? "FAILURES" : "all passed");
    return failures ? 1 : 0;
}
#endif
