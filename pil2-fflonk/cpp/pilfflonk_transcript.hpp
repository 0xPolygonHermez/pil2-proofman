#ifndef PIL2_FFLONK_TRANSCRIPT_HPP
#define PIL2_FFLONK_TRANSCRIPT_HPP

#include <alt_bn128.hpp>
#include <keccak_256_transcript.hpp>

// pil-fflonk shipped its own `PilFflonkTranscript`, which diffing shows to be a
// de-templated copy of rapidsnark's Keccak256Transcript with the Keccak call
// inlined instead of going through keccak_wrapper. Rather than move a second
// copy of the same Fiat-Shamir encoding, the name is bound to the original --
// so the sources that use it compile here unedited.
typedef Keccak256Transcript<AltBn128::Engine> PilFflonkTranscript;

#endif // PIL2_FFLONK_TRANSCRIPT_HPP
