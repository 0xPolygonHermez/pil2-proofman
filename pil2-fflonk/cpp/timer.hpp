#ifndef PIL2_FFLONK_TIMER_HPP
#define PIL2_FFLONK_TIMER_HPP

#include <sys/time.h>
#include <string>
#include "logger.hpp"

// Same macro surface as pil2-stark/src/utils/timer.hpp, but built on the snark
// side's own logger.
//
// pil2-stark's timer reaches zklog.hpp, which includes utils.hpp (Goldilocks)
// and starks_api.hpp (the whole STARK C API). pil2-fflonk is a BN254 SNARK and
// has no business pulling either in, so it uses rapidsnark's logger -- the same
// one fflonk_prover and plonk_prover use -- which depends on nothing outside
// itself.

using namespace CPlusPlusLogging;

#define TimeDiff(start, stop) ((double)(stop.tv_sec - start.tv_sec) * 1000000 + (double)(stop.tv_usec - start.tv_usec))

// Logger's string overloads take std::string& (non-const), so every message is
// bound to a named local before it is passed.
#define PilFflonkTrace(msg) do { std::string _t = (msg); LOG_TRACE(_t); } while (0)
#define PilFflonkInfo(msg)  do { std::string _i = (msg); LOG_INFO(_i);  } while (0)

#define TimerStart(name) \
    struct timeval name##_start; \
    gettimeofday(&name##_start, NULL); \
    PilFflonkTrace("--> " + std::string(#name) + " starting...")

#define TimerStop(name) \
    struct timeval name##_stop; \
    gettimeofday(&name##_stop, NULL); \
    PilFflonkTrace("<-- " + std::string(#name) + " done")

#define TimerLog(name) PilFflonkTrace("<-- " + std::string(#name))

#define TimerStopAndLog(name) \
    struct timeval name##_stop; \
    gettimeofday(&name##_stop, NULL); \
    PilFflonkTrace("<-- " + std::string(#name) + " done: " + \
                   std::to_string(TimeDiff(name##_start, name##_stop) / 1000000) + " s")

#endif
