// The GPU expression interpreter's binary ops (dim1 x dim1, dim3 x dim1, dim3 x dim3) on every edge-valued
// operand, against exact CPU arithmetic. Regression for a ptxas 12.8 (-O1+) miscompile that made a
// multiply by a non-canonical operand (>= p) read a stale register: on such a build this test fails.
#include <gtest/gtest.h>
#include <vector>
#include "expressions_gpu.cuh"
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
