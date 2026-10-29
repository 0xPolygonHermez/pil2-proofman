// A stand-in .exps.so for the loader tests (StaleExpsSoRejected): the symbols expsOpenForAir resolves, no kernels.
#include <stdint.h>
void exps_launch(void) {}
unsigned long long exps_min_scratch(void) { return 0; }
#ifdef LAYOUT_HASH
uint64_t boundary_layout_hash(void) { return LAYOUT_HASH; }
#endif
