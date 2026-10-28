#ifndef PILFFLONK_JSON_HPP
#define PILFFLONK_JSON_HPP

#include <cstdint>
#include <string>

#include <nlohmann/json_fwd.hpp>

namespace PilFflonk {

// The text of the file at `path`, whole. Throws IoError with the message `cannotOpen` if it cannot be
// opened, and `cannotRead` if it cannot be read.
std::string readText(const std::string &path, const std::string &cannotOpen, const std::string &cannotRead);

// The values of a JSON file of the provingKey/ (the globalInfo, an AIR's pilfflonkinfo), each read as
// the type it must have. A value refused is a FormatError "<file>: <where>: <what>", `file` what the
// errors call the file and `where` the field, as a path from the top-level object ("" for that
// object itself, which the errors call "the file").
class JsonReader {
public:
    constexpr explicit JsonReader(const char *_file) : file(_file) {}

    [[noreturn]] void fail(const std::string &where, const std::string &what) const;

    uint64_t u64(const nlohmann::json &value, const std::string &where) const;
    std::string str(const nlohmann::json &value, const std::string &where) const;
    const nlohmann::json &array(const nlohmann::json &value, const std::string &where) const;

private:
    const char *file;
};

} // namespace PilFflonk

#endif
