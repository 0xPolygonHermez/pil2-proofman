// The GPU expression interpreter's binary ops (dim1 x dim1, dim3 x dim1, dim3 x dim3) on every edge-valued
// operand, against exact CPU arithmetic. Regression for a ptxas 12.8 (-O1+) miscompile that made a
// multiply by a non-canonical operand (>= p) read a stale register: on such a build this test fails.
#include <gtest/gtest.h>
#include <vector>
#include <unistd.h>
#include "expressions_gpu.cuh"
#include "starks_gpu.cuh"
#include "expressions_codegen.cuh"
#include "cuda_utils.cuh"

namespace {

constexpr uint64_t kP = 0xFFFFFFFF00000001ull;
constexpr uint32_t ROWS = 128;  // one block of nRowsPack rows
constexpr uint32_t BASE = 4;    // bufferCommitSize: the challenges buffer is type BASE + 7

uint64_t addp(uint64_t a, uint64_t b) { return (uint64_t)(((unsigned __int128)a + b) % kP); }
uint64_t subp(uint64_t a, uint64_t b) { return addp(a, kP - b); }
uint64_t mulp(uint64_t a, uint64_t b) { return (uint64_t)(((unsigned __int128)a * b) % kP); }

// GF(p^3) with x^3 = x + 1, canonical inputs.
void mul3(const uint64_t a[3], const uint64_t b[3], uint64_t r[3]) {
    uint64_t c[5] = {};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) c[i + j] = addp(c[i + j], mulp(a[i], b[j]));
    r[0] = addp(c[0], c[3]);
    r[1] = addp(addp(c[1], c[3]), c[4]);
    r[2] = addp(c[2], c[4]);
}

// Bytecode case: 0 = dim1 x dim1, 1 = dim3 x dim1, 2 = dim3 x dim3. A dim1 operand is its component 0.
void expected(uint8_t kind, uint64_t op, const uint64_t a[3], const uint64_t b[3], uint64_t r[3]) {
    uint64_t ca[3] = {}, cb[3] = {};
    for (int i = 0; i < 3; i++) {
        ca[i] = kind == 0 && i > 0 ? 0 : a[i] % kP;
        cb[i] = kind != 2 && i > 0 ? 0 : b[i] % kP;
    }
    if (op == 2) { mul3(ca, cb, r); return; }
    for (int i = 0; i < 3; i++)
        r[i] = op == 0 ? addp(ca[i], cb[i]) : op == 1 ? subp(ca[i], cb[i]) : subp(cb[i], ca[i]);
}

struct Interp {
    StepsParams *dParams = nullptr;
    DeviceArguments *dDevArgs = nullptr;
    ExpsArguments *dExpsArgs = nullptr;
    DestParamsGPU *dDest = nullptr;
    uint8_t *dOps = nullptr;
    uint16_t *dArgs = nullptr;
    uint64_t *dChallenges = nullptr, *dAux = nullptr, *dOut = nullptr;

    Interp() {
        CHECKCUDAERR(cudaMalloc(&dChallenges, 6 * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&dAux, 3 * ROWS * sizeof(uint64_t)));  // destVals of the single block
        CHECKCUDAERR(cudaMalloc(&dOut, 3 * ROWS * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMalloc(&dOps, 1));
        CHECKCUDAERR(cudaMalloc(&dArgs, 8 * sizeof(uint16_t)));

        StepsParams h = {};
        h.aux_trace = (Goldilocks::Element *)dAux;
        h.challenges = (Goldilocks::Element *)dChallenges;
        CHECKCUDAERR(cudaMalloc(&dParams, sizeof(StepsParams)));
        CHECKCUDAERR(cudaMemcpy(dParams, &h, sizeof(h), cudaMemcpyHostToDevice));

        DeviceArguments d = {};
        d.N = ROWS;
        d.bufferCommitSize = BASE;
        d.ops = dOps;
        d.args = dArgs;
        CHECKCUDAERR(cudaMalloc(&dDevArgs, sizeof(DeviceArguments)));
        CHECKCUDAERR(cudaMemcpy(dDevArgs, &d, sizeof(d), cudaMemcpyHostToDevice));

        ExpsArguments e = {};
        e.nRowsPack = ROWS;
        e.k_min = 0;
        e.k_max = ROWS;
        e.domainSize = ROWS;
        e.dest_gpu = (Goldilocks::Element *)dOut;
        e.dest_domainSize = ROWS;
        e.dest_dim = 3;
        e.dest_nParams = 1;
        e.dest_expr = true;
        CHECKCUDAERR(cudaMalloc(&dExpsArgs, sizeof(ExpsArguments)));
        CHECKCUDAERR(cudaMemcpy(dExpsArgs, &e, sizeof(e), cudaMemcpyHostToDevice));

        DestParamsGPU p = {};
        p.dim = 3;
        p.op = opType::tmp;
        p.nOps = 1;
        p.nArgs = 8;
        CHECKCUDAERR(cudaMalloc(&dDest, sizeof(DestParamsGPU)));
        CHECKCUDAERR(cudaMemcpy(dDest, &p, sizeof(p), cudaMemcpyHostToDevice));

        h_e = e;
        h_p = p;
    }

    ExpsArguments h_e = {};
    DestParamsGPU h_p = {};

    ~Interp() {
        for (void *ptr : {(void *)dParams, (void *)dDevArgs, (void *)dExpsArgs, (void *)dDest, (void *)dOps, (void *)dArgs,
                          (void *)dChallenges, (void *)dAux, (void *)dOut})
            cudaFree(ptr);
    }

    // dest = challenge[0..] <op> challenge[3..]; every row computes the same value.
    void run(uint8_t kind, uint64_t op, const uint64_t a[3], const uint64_t b[3], uint64_t out[3]) {
        const uint64_t dim = kind == 0 ? 1 : 3;
        h_e.dest_dim = dim;
        h_p.dim = dim;
        CHECKCUDAERR(cudaMemcpy(dExpsArgs, &h_e, sizeof(h_e), cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(dDest, &h_p, sizeof(h_p), cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(dOps, &kind, 1, cudaMemcpyHostToDevice));
        const uint16_t args[8] = {(uint16_t)op, 0, BASE + 7, 0, 0, BASE + 7, 3, 0};
        const uint64_t ch[6] = {a[0], a[1], a[2], b[0], b[1], b[2]};
        CHECKCUDAERR(cudaMemcpy(dArgs, args, sizeof(args), cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(dChallenges, ch, sizeof(ch), cudaMemcpyHostToDevice));
        computeExpressions_<<<1, ROWS, 32 * sizeof(uint64_t)>>>(dParams, dDevArgs, dExpsArgs, dDest, false);
        CHECKCUDAERR(cudaGetLastError());
        std::vector<uint64_t> h(3 * ROWS);
        CHECKCUDAERR(cudaMemcpy(h.data(), dOut, h.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        for (uint64_t i = 0; i < 3; i++) out[i] = i < dim ? h[i] % kP : 0;
        for (uint32_t r = 1; r < ROWS; r++)
            for (uint64_t i = 0; i < dim; i++) ASSERT_EQ(h[dim * r + i] % kP, out[i]) << "row " << r << " differs from row 0";
    }
};

}  // namespace

TEST(InterpreterGpu, BinaryOpsOnEdgeOperands) {
    const uint64_t E[] = {0, 1, 8, kP - 1, kP, kP + 1, kP + 8, 0xFFFFFFFFFFFFFFFFull};
    const uint64_t Bs[][3] = {{0xaad77f593372ee0eull, 0x6d27c3d0eaee5af7ull, 0x7be680498cf8f5aaull},
                              {kP, 1, kP + 8},
                              {0xFFFFFFFFFFFFFFFFull, 0, 0x123456789ull}};
    Interp interp;
    int failures = 0;
    for (uint8_t kind = 0; kind < 3; kind++)
    for (uint64_t op = 0; op < 4; op++)
        for (const auto &b : Bs)
            for (uint64_t x : E)
                for (uint64_t y : E)
                    for (uint64_t z : E) {
                        const uint64_t a[3] = {x, y, z};
                        uint64_t got[3], want[3];
                        interp.run(kind, op, a, b, got);
                        expected(kind, op, a, b, want);
                        if (got[0] != want[0] || got[1] != want[1] || got[2] != want[2]) {
                            if (++failures <= 5)
                                ADD_FAILURE() << "case " << (int)kind << " op " << op << " a=(" << std::hex << x << "," << y << "," << z << ") b=(" << b[0] << ","
                                              << b[1] << "," << b[2] << "): got (" << got[0] << "," << got[1] << "," << got[2]
                                              << ") want (" << want[0] << "," << want[1] << "," << want[2] << ")";
                        }
                    }
    EXPECT_EQ(failures, 0);
}

// ---- Domain zerofiers on the GPU: the tables, the interpreter's Zi read, the .exps.so layout guard.

namespace {

Boundary everyRowB() { Boundary b; b.name = "everyRow"; return b; }

Boundary domainB(uint64_t cycleBits, std::vector<uint64_t> offsets, bool complement) {
    Boundary b;
    b.name = "domain";
    b.cycleBits = cycleBits;
    b.offsets = offsets;
    b.complement = complement;
    return b;
}

// everyRow, FIRST, LAST, !FIRST, LIMITS_16, ODD, !ODD, !QUARTER.
StarkInfo zerofierStarkInfo(uint64_t nBits, uint64_t nBitsExt) {
    StarkInfo si;
    si.starkStruct.nBits = nBits;
    si.starkStruct.nBitsExt = nBitsExt;
    const uint64_t last = (1ull << nBits) - 1;
    si.boundaries = {everyRowB(), domainB(nBits, {0}, false), domainB(nBits, {last}, false), domainB(nBits, {0}, true),
                     domainB(4, {0, 15}, false), domainB(1, {1}, false), domainB(1, {1}, true), domainB(2, {0}, true)};
    si.ziSize = ziLayout(si.boundaries, nBits, nBitsExt, si.ziOffsets, si.ziMasks);
    return si;
}

std::vector<uint64_t> gpuZerofier(StarkInfo &si) {
    Goldilocks::Element *d_zi = nullptr;
    CHECKCUDAERR(cudaMalloc(&d_zi, si.ziSize * sizeof(uint64_t)));
    computeZerofier(d_zi, si, 0);
    CHECKCUDAERR(cudaDeviceSynchronize());
    std::vector<uint64_t> h(si.ziSize);
    CHECKCUDAERR(cudaMemcpy(h.data(), d_zi, h.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    cudaFree(d_zi);
    return h;
}

// Loads exps_fixtures/<name>.exps.so (built next to this binary from exps_fixture.c) through the prover's loader.
ExpsKernel openFixture(StarkInfo &si, const std::string &name) {
    char exe[4096] = {};
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    std::string dir = std::string(exe, n > 0 ? n : 0);
    dir = dir.substr(0, dir.rfind('/'));
    ExpressionsBin bin;
    bin.binFile = dir + "/exps_fixtures/" + name + ".bin";
    SetupCtx sc(si, bin);
    return expsOpenForAir(sc);
}

}  // namespace

TEST(DomainZerofierGpu, GpuZerofierMatchesCpu) {
    for (auto [nBits, nBitsExt] : {std::pair<uint64_t, uint64_t>{6, 7}, {10, 11}, {10, 12}}) {
        StarkInfo si = zerofierStarkInfo(nBits, nBitsExt);
        ProverHelpers cpu;
        cpu.computeZerofier(si);
        std::vector<uint64_t> gpu = gpuZerofier(si);
        for (uint64_t i = 0; i < si.ziSize; i++) {
            // the CPU table may hold a non-canonical representative (x + p); the GPU stores canonical
            ASSERT_EQ(gpu[i], Goldilocks::toU64(cpu.zi[i])) << "nBits " << nBits << " nBitsExt " << nBitsExt << " entry " << i;
        }
    }
}

// Zi{b} * ch[0] via the interpreter on every extended row vs the CPU read zi[ziOffsets[b] + (row & ziMasks[b])].
TEST(DomainZerofierGpu, InterpreterReadsDomainZi) {
    const uint64_t nBits = 6, nBitsExt = 7, NExt = 1ull << nBitsExt;
    static_assert(ROWS == 128, "one block spans the whole extended domain");
    StarkInfo si = zerofierStarkInfo(nBits, nBitsExt);
    ProverHelpers cpu;
    cpu.computeZerofier(si);
    const uint64_t ch0 = 0x1234567890abcdefull % kP;
    const uint32_t base = 6;  // bufferCommitSize with nStages = 2, no custom commits: type 4 is Zi

    uint64_t *dAux, *dCh, *dOut, *dZiOff, *dZiMask;
    int64_t *dStrides;
    uint8_t *dOps;
    uint16_t *dArgs;
    const uint64_t ziAt = 3 * ROWS;  // past the block's destVals
    CHECKCUDAERR(cudaMalloc(&dAux, (ziAt + si.ziSize) * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&dCh, 3 * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&dOut, NExt * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&dStrides, sizeof(int64_t)));
    CHECKCUDAERR(cudaMalloc(&dOps, 1));
    CHECKCUDAERR(cudaMalloc(&dArgs, 8 * sizeof(uint16_t)));
    CHECKCUDAERR(cudaMalloc(&dZiOff, si.boundaries.size() * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMalloc(&dZiMask, si.boundaries.size() * sizeof(uint64_t)));
    CHECKCUDAERR(cudaMemcpy(dAux + ziAt, cpu.zi, si.ziSize * sizeof(uint64_t), cudaMemcpyHostToDevice));
    const uint64_t ch[3] = {ch0, 0, 0};
    CHECKCUDAERR(cudaMemcpy(dCh, ch, sizeof(ch), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemset(dStrides, 0, sizeof(int64_t)));
    CHECKCUDAERR(cudaMemset(dOps, 0, 1));
    CHECKCUDAERR(cudaMemcpy(dZiOff, si.ziOffsets.data(), si.boundaries.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(dZiMask, si.ziMasks.data(), si.boundaries.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    StepsParams h = {};
    h.aux_trace = (Goldilocks::Element *)dAux;
    h.challenges = (Goldilocks::Element *)dCh;
    StepsParams *dParams;
    CHECKCUDAERR(cudaMalloc(&dParams, sizeof(StepsParams)));
    CHECKCUDAERR(cudaMemcpy(dParams, &h, sizeof(h), cudaMemcpyHostToDevice));

    DeviceArguments d = {};
    d.N = 1ull << nBits;
    d.NExtended = NExt;
    d.nStages = 2;
    d.bufferCommitSize = base;
    d.zi_offset = ziAt;
    d.ziOffsets = dZiOff;
    d.ziMasks = dZiMask;
    d.ops = dOps;
    d.args = dArgs;
    DeviceArguments *dDevArgs;
    CHECKCUDAERR(cudaMalloc(&dDevArgs, sizeof(d)));
    CHECKCUDAERR(cudaMemcpy(dDevArgs, &d, sizeof(d), cudaMemcpyHostToDevice));

    ExpsArguments e = {};
    e.nRowsPack = ROWS;
    e.k_min = 0;
    e.k_max = NExt;
    e.domainSize = NExt;
    e.domainExtended = true;
    e.nextStridesExps = (uint64_t *)dStrides;
    e.dest_gpu = (Goldilocks::Element *)dOut;
    e.dest_domainSize = NExt;
    e.dest_dim = 1;
    e.dest_nParams = 1;
    e.dest_expr = true;
    ExpsArguments *dExpsArgs;
    CHECKCUDAERR(cudaMalloc(&dExpsArgs, sizeof(e)));
    CHECKCUDAERR(cudaMemcpy(dExpsArgs, &e, sizeof(e), cudaMemcpyHostToDevice));

    DestParamsGPU p = {};
    p.dim = 1;
    p.op = opType::tmp;
    p.nOps = 1;
    p.nArgs = 8;
    DestParamsGPU *dDest;
    CHECKCUDAERR(cudaMalloc(&dDest, sizeof(p)));
    CHECKCUDAERR(cudaMemcpy(dDest, &p, sizeof(p), cudaMemcpyHostToDevice));

    for (uint16_t b = 0; b < si.boundaries.size(); b++) {
        // mul: a = Zi{b} (type 4, argIdx b + 1), b = challenge component 0
        const uint16_t args[8] = {2, 0, 4, (uint16_t)(b + 1), 0, (uint16_t)(base + 7), 0, 0};
        CHECKCUDAERR(cudaMemcpy(dArgs, args, sizeof(args), cudaMemcpyHostToDevice));
        computeExpressions_<<<1, ROWS, 32 * sizeof(uint64_t)>>>(dParams, dDevArgs, dExpsArgs, dDest, false);
        CHECKCUDAERR(cudaGetLastError());
        std::vector<uint64_t> got(NExt);
        CHECKCUDAERR(cudaMemcpy(got.data(), dOut, NExt * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        for (uint64_t row = 0; row < NExt; row++) {
            Goldilocks::Element want = cpu.zi[si.ziOffsets[b] + (row & si.ziMasks[b])] * Goldilocks::fromU64(ch0);
            ASSERT_EQ(got[row] % kP, Goldilocks::toU64(want)) << "boundary " << b << " row " << row;
        }
    }
    for (void *ptr : {(void *)dAux, (void *)dCh, (void *)dOut, (void *)dStrides, (void *)dOps, (void *)dArgs, (void *)dZiOff,
                      (void *)dZiMask, (void *)dParams, (void *)dDevArgs, (void *)dExpsArgs, (void *)dDest})
        cudaFree(ptr);
}

// The hash baked into exps_fixtures/match.exps.so (Makefile); the known answers are in test_domain_zerofier.cpp.
TEST(DomainZerofierGpu, FixtureLayoutHash) {
    StarkInfo si = zerofierStarkInfo(6, 7);
    EXPECT_EQ(boundaryLayoutHash(si.boundaries, 6, 7), 0xfb1d2cc6ab8792b6ull);
}

TEST(DomainZerofierGpu, StaleExpsSoRejected) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    StarkInfo si = zerofierStarkInfo(6, 7);
    ExpsKernel ok = openFixture(si, "match");
    EXPECT_NE(ok.lib, nullptr);
    EXPECT_NE(ok.qLaunch, nullptr);
    expsClose(ok.lib);
    EXPECT_EXIT(openFixture(si, "stale"), ::testing::ExitedWithCode(255), "stale .exps.so: boundary layout changed, rerun gen-exps");
    // a library from before the guard has no hash at all
    EXPECT_EXIT(openFixture(si, "nohash"), ::testing::ExitedWithCode(255), "nohash.exps.so: stale .exps.so");
}
