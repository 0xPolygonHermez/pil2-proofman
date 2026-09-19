#ifndef PIL2_FFLONK_ZKASSERT_HPP
#define PIL2_FFLONK_ZKASSERT_HPP

#include <cassert>

// A shim so pil-fflonk's sources compile here unedited; see zklog.hpp.
#define zkassert(a) assert(a)

#endif // PIL2_FFLONK_ZKASSERT_HPP
