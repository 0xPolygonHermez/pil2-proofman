#include "../src/platform.hpp"

#include <gtest/gtest.h>

#if PIL2_HAS_METAL

#include "../src/metal/metal_context.hpp"

#include <cstdint>

TEST(MetalKernel, NoopDispatchReturnsFortyTwo) {
    auto* ctx = pil2::metal::get_context();
    ASSERT_NE(ctx, nullptr);
    const uint32_t got = pil2::metal::run_noop_test(ctx);
    EXPECT_EQ(got, 42u);
}

TEST(MetalKernel, NoopDispatchIsRepeatable) {
    // The helper recompiles the library + PSO every call; running twice
    // validates that there's no one-shot state that breaks reuse.
    auto* ctx = pil2::metal::get_context();
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(pil2::metal::run_noop_test(ctx), 42u);
    EXPECT_EQ(pil2::metal::run_noop_test(ctx), 42u);
}

#else

TEST(MetalKernel, SkippedBuildFlag) {
    GTEST_SKIP() << "PIL2_HAS_METAL=0 at compile time";
}

#endif // PIL2_HAS_METAL
