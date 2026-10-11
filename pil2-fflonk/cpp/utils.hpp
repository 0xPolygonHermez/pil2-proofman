#ifndef PIL2_FFLONK_UTILS_HPP
#define PIL2_FFLONK_UTILS_HPP

#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include <string>

#include <nlohmann/json.hpp>

// A shim so pil-fflonk's sources compile here unedited; see zklog.hpp.
//
// The original utils.hpp carried memory probes, call-stack dumps and file
// mapping alongside this. Only file2json is reached from the sources moved
// here, and pil2-stark's own utils.hpp reaches Goldilocks, so just this is
// provided.

inline void file2json(const std::string &fileName, nlohmann::json &j) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        throw std::runtime_error("file2json: cannot open " + fileName);
    }
    file >> j;
}

inline void file2json(const std::string &fileName, nlohmann::ordered_json &j) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        throw std::runtime_error("file2json: cannot open " + fileName);
    }
    file >> j;
}

/// Read a whole file into a fresh buffer of `size` bytes.
///
/// The original mapped the file; this copies, which is all the callers need and
/// avoids carrying the unmap bookkeeping too.
inline void *copyFile(const std::string &fileName, uint64_t size) {
    std::ifstream file(fileName, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("copyFile: cannot open " + fileName);
    }

    void *buffer = malloc(size);
    if (buffer == nullptr) {
        throw std::runtime_error("copyFile: cannot allocate " + std::to_string(size) + " bytes");
    }

    file.read((char *)buffer, size);
    if ((uint64_t)file.gcount() != size) {
        free(buffer);
        throw std::runtime_error("copyFile: " + fileName + " is shorter than the " + std::to_string(size) +
                                 " bytes expected");
    }
    return buffer;
}

#endif // PIL2_FFLONK_UTILS_HPP
