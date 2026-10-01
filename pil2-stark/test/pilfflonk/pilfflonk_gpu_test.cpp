// Tests of the GPU path's entry points (pilfflonk/docs/performance.md#selection-memory-and-errors),
// in every build: pilfflonk_gpu_available, and pilfflonk_ctx_new_on and ProvingKey::load, which
// refuse the GPU, saying why, before they read anything, in a library built without it
// (make pilfflonk_test) and on a machine without one (make pilfflonk_gpu_test there). Where there is
// one, the tests of each module compare its GPU path with its CPU one (gpuUnderTest):
// pilfflonk_lde_test.cpp the transforms, pilfflonk_commit_test.cpp the MSM,
// pilfflonk_prover_test.cpp whole proofs.
#include "pilfflonk_test.hpp"
#include "pilfflonk_test_ptau.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "pilfflonk_api.hpp"
#include "pilfflonk_gpu.hpp"
#include "pilfflonk_proving_key.hpp"

namespace PilFflonkTest {

namespace {

bool contains(const char *s, const char *part) { return std::strstr(s, part) != nullptr; }

// What a refused GPU says: there is none, and why.
#ifdef __USE_CUDA__
constexpr const char *NO_GPU = "no GPU: CUDA sees no device";
#else
constexpr const char *NO_GPU = "no GPU: this library was built without it";
#endif

void testAvailability() {
#ifdef __USE_CUDA__
    assert(pilfflonk_gpu_available() == (PilFflonk::Gpu::available() ? 1 : 0));
#else
    assert(pilfflonk_gpu_available() == 0);
#endif
    assert(pilfflonk_gpu_available() == (PilFflonk::gpuAvailable() ? 1 : 0));
    assert(pilfflonk_last_status() == PILFFLONK_OK && pilfflonk_last_error()[0] == '\0');
}

void testCtxNewOn() {
    TestDir dir;
    const std::string missing = dir.path() + "/nothing";

    assert(pilfflonk_ctx_new_on(missing.c_str(), 2) == nullptr);
    assert(pilfflonk_last_status() == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(contains(pilfflonk_last_error(), "pilfflonk_ctx_new_on: device 2 is neither the CPU (0) nor the GPU (1)"));
    assert(pilfflonk_ctx_new_on(nullptr, PILFFLONK_DEVICE_CPU) == nullptr);
    assert(pilfflonk_last_status() == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(contains(pilfflonk_last_error(), "proving_key_dir is NULL"));

    // The CPU reads the key, which is not there; so does pilfflonk_ctx_new, the same ctx.
    assert(pilfflonk_ctx_new_on(missing.c_str(), PILFFLONK_DEVICE_CPU) == nullptr);
    assert(pilfflonk_last_status() == PILFFLONK_ERR_IO && contains(pilfflonk_last_error(), "cannot open"));

    if (pilfflonk_gpu_available()) {
        // A GPU there: it reads the key too.
        assert(pilfflonk_ctx_new_on(missing.c_str(), PILFFLONK_DEVICE_GPU) == nullptr);
        assert(pilfflonk_last_status() == PILFFLONK_ERR_IO);
        return;
    }
    // No GPU: refused before anything is read, so not an IoError, and the reason given.
    assert(pilfflonk_ctx_new_on(missing.c_str(), PILFFLONK_DEVICE_GPU) == nullptr);
    assert(pilfflonk_last_status() == PILFFLONK_ERR_INVALID_ARGUMENT);
    assert(contains(pilfflonk_last_error(), "pilfflonk_ctx_new_on: ProvingKey::load: "));
    assert(contains(pilfflonk_last_error(), NO_GPU));
    try {
        PilFflonk::ProvingKey::load(missing, PilFflonk::Device::Gpu);
        assert(!"expected std::invalid_argument");
    } catch (const std::invalid_argument &e) {
        assert(contains(e.what(), NO_GPU));
    }
}

} // namespace

bool gpuUnderTest(const char *what) {
    if (pilfflonk_gpu_available() == 1) {
        return true;
    }
    const char *required = std::getenv("PILFFLONK_GPU");
    if (required != nullptr && std::strcmp(required, "1") == 0) {
        std::fprintf(stderr, "pilfflonk_test: PILFFLONK_GPU=1, and there is no GPU for %s\n", what);
        assert(!"PILFFLONK_GPU=1 requires a GPU");
    }
    std::printf("pilfflonk_test: no GPU, %s skipped\n", what);
    return false;
}

void runGpuTests() {
    testAvailability();
    testCtxNewOn();
}

} // namespace PilFflonkTest
