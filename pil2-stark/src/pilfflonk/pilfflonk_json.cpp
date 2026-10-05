#include "pilfflonk_json.hpp"

#include <fstream>
#include <iterator>

#include <nlohmann/json.hpp>

#include "pilfflonk_error.hpp"

namespace PilFflonk {

std::string readText(const std::string &path, const std::string &cannotOpen, const std::string &cannotRead) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw IoError(cannotOpen);
    }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (file.bad()) {
        throw IoError(cannotRead);
    }
    return text;
}

void JsonReader::fail(const std::string &where, const std::string &what) const {
    throw FormatError(std::string(file) + ": " + (where.empty() ? std::string("the file") : where) + ": " + what);
}

uint64_t JsonReader::u64(const nlohmann::json &value, const std::string &where) const {
    if (!value.is_number_unsigned()) {
        fail(where, "must be an unsigned integer");
    }
    return value.get<uint64_t>();
}

std::string JsonReader::str(const nlohmann::json &value, const std::string &where) const {
    if (!value.is_string()) {
        fail(where, "must be a string");
    }
    return value.get<std::string>();
}

const nlohmann::json &JsonReader::array(const nlohmann::json &value, const std::string &where) const {
    if (!value.is_array()) {
        fail(where, "must be an array");
    }
    return value;
}

} // namespace PilFflonk
