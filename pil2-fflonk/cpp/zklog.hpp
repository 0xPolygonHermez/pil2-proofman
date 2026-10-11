#ifndef PIL2_FFLONK_ZKLOG_HPP
#define PIL2_FFLONK_ZKLOG_HPP

#include <iostream>
#include <string>

// A shim so pil-fflonk's sources compile here unedited.
//
// They log through a `zklog` global with info/warning/error. pil2-stark has a
// class of that name, but its header reaches Goldilocks and the STARK C API,
// which this BN254-only tree deliberately does not depend on -- the same reason
// cpp/timer.hpp exists.
//
// Keeping the interface rather than rewriting the call sites means the moved
// files stay byte-identical to the originals, so they can be diffed against
// them later.

namespace PilFflonk {

class ZkLog {
    std::string prefix;

public:
    void setPrefix(const std::string &p) { prefix = p; }
    void info(const std::string &message) { std::cout << prefix << message << std::endl; }
    void warning(const std::string &message) { std::cerr << prefix << "WARNING: " << message << std::endl; }
    void error(const std::string &message) { std::cerr << prefix << "ERROR: " << message << std::endl; }
};

} // namespace PilFflonk

// The sources refer to it unqualified, as a global.
inline PilFflonk::ZkLog zklog;

#endif // PIL2_FFLONK_ZKLOG_HPP
