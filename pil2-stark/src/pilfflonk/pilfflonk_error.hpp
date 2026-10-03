#ifndef PILFFLONK_ERROR_HPP
#define PILFFLONK_ERROR_HPP

#include <stdexcept>

namespace PilFflonk {

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
