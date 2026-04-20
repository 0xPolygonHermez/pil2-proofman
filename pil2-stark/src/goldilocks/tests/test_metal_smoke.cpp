#include "../src/platform.hpp"

#include <gtest/gtest.h>

#if PIL2_HAS_METAL

#include "../src/metal/metal_context.hpp"

#include <iostream>

TEST(Metal, DeviceOpens) {
    auto* ctx = pil2::metal::get_context();
    ASSERT_NE(ctx, nullptr);
    auto name = pil2::metal::device_name(ctx);
    EXPECT_FALSE(name.empty());
    std::cout << "[   METAL  ] device: " << name << std::endl;
}

TEST(Metal, SingletonIsStable) {
    auto* a = pil2::metal::get_context();
    auto* b = pil2::metal::get_context();
    EXPECT_EQ(a, b);
}

#else

TEST(Metal, SkippedBuildFlag) {
    GTEST_SKIP() << "PIL2_HAS_METAL=0 at compile time";
}

#endif // PIL2_HAS_METAL
