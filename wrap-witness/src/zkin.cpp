// The zkin of the final circuit as its witness calculator takes it: final.so's getWitness
// (setup/final_snark_circom/main.cpp) reads a nlohmann::json, which in the PLONK and FFLONK wraps is
// the recursivef proof libstarks returns. A zkin read from a file is made into one here, with the
// same nlohmann/json final.so is compiled against (build.rs). See src/zkin.rs.

#include <cstddef>
#include <utility>

#include <nlohmann/json.hpp>

// The JSON document in text[0 .. len), on the heap, or null if it is not one or memory runs out.
// Nothing escapes: a C++ exception must not unwind into Rust.
extern "C" void *pilfflonk_wrap_zkin_parse(const char *text, size_t len) noexcept {
    try {
        nlohmann::json zkin = nlohmann::json::parse(text, text + len, nullptr, /*allow_exceptions=*/false);
        if (zkin.is_discarded()) {
            return nullptr;
        }
        return new nlohmann::json(std::move(zkin));
    } catch (...) {
        return nullptr;
    }
}

// Frees what pilfflonk_wrap_zkin_parse returned.
extern "C" void pilfflonk_wrap_zkin_free(void *zkin) noexcept {
    delete static_cast<nlohmann::json *>(zkin);
}
