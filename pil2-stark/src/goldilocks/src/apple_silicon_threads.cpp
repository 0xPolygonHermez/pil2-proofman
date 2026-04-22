// Apple Silicon OMP thread tuning. Caller (Rust-side proofman CLI startup)
// invokes `pil2_apple_silicon_tune_threads()` once before any OMP region.
// Two effects: (1) set main-thread QoS to USER_INITIATED so OMP workers
// inherit it and the kernel biases them onto P-cores; (2) cap OMP thread
// count at the P-core count reported by `hw.perflevel0.logicalcpu` so the
// ~3× slower E-cores don't become stragglers under schedule(static).
// Measured: fibonacci-square INNER_PROOFS median 11078 → 10672 ms
// (-3.7%) and spread 613 → 168 ms on M4 Pro. Respects an explicit
// OMP_NUM_THREADS override. Empty implementation on non-Apple-Silicon.

#include "platform.hpp"

extern "C" void pil2_apple_silicon_tune_threads();

#if PIL2_ARCH_ARM64 && defined(__APPLE__)

#include <cstdlib>
#include <sys/sysctl.h>
#include <pthread.h>
#include <dispatch/dispatch.h>

#if defined(_OPENMP)
  #include <omp.h>
#endif

extern "C" void pil2_apple_silicon_tune_threads() {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#if defined(_OPENMP)
    if (std::getenv("OMP_NUM_THREADS")) return;
    int p_cores = 0;
    size_t size = sizeof(p_cores);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &p_cores, &size, nullptr, 0) != 0
        || p_cores <= 0) return;
    int current = omp_get_max_threads();
    omp_set_num_threads(p_cores < current ? p_cores : current);
#endif
}

#else

extern "C" void pil2_apple_silicon_tune_threads() {}

#endif
