#ifndef PILFFLONK_ERROR_HPP
#define PILFFLONK_ERROR_HPP

#include <stdexcept>
#include <string>

namespace PilFflonk {

// The std::invalid_argument of an argument refused by a function of `scope`, a class ("Srs::"), or
// "" for free functions, which a module keeps as its `invalid`: invalid(function, message) is
// "<scope><function>: <message>".
class InvalidArgument {
public:
    constexpr explicit InvalidArgument(const char *_scope) : scope(_scope) {}

    std::invalid_argument operator()(const char *function, const std::string &message) const {
        return std::invalid_argument(std::string(scope) + function + ": " + message);
    }

private:
    const char *scope;
};

// The exceptions the pilfflonk modules throw besides std::invalid_argument (an argument refused
// before anything is done) and std::bad_alloc. The C API turns each into its own status code.

// A file cannot be opened, read or written: PILFFLONK_ERR_IO.
class IoError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// A file is not in the format expected: the wrong type, a section missing or of the wrong size, a
// value out of range. PILFFLONK_ERR_FORMAT.
class FormatError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The witness does not satisfy the AIR's constraints: the constraint polynomial Q computed from it
// is not a polynomial of its degree bound (pilfflonk/docs/protocol.md#degrees).
// PILFFLONK_ERR_UNSATISFIED.
class UnsatisfiedError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

} // namespace PilFflonk

#endif
