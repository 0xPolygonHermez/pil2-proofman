#ifndef PIL2_FFLONK_EXIT_PROCESS_HPP
#define PIL2_FFLONK_EXIT_PROCESS_HPP

#include <cstdlib>

#include "zklog.hpp"

// A shim so pil-fflonk's sources compile here unedited; see zklog.hpp.
//
// The original dumped a backtrace before aborting. Nothing here runs as a
// standalone process -- this is a library reached over FFI -- so it exits
// rather than pretending to be one. Call sites that reach it are reporting a
// malformed key or info file, which is a caller error either way.
inline void exitProcess() {
    zklog.error("pil-fflonk: unrecoverable error, exiting");
    exit(-1);
}

#endif // PIL2_FFLONK_EXIT_PROCESS_HPP
