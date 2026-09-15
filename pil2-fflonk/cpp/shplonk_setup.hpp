#ifndef SHPLONK_SETUP_HPP
#define SHPLONK_SETUP_HPP

#include <string>
#include <map>
#include <gmp.h>
#include <alt_bn128.hpp>

// The data model SHPLONK opens against: which committed polynomials are packed
// into each combined polynomial f_i, at which opening points, and to what
// degree.
//
// pil-fflonk carried this inside PilFflonkZkey, a reader for the pil1 `.zkey`
// binary. pil2 derives the same information from StarkInfo (cm_pols_map,
// opening_points, n_stages, q_deg), so only the model is kept here -- the pil1
// file format and its read/write routines are deliberately not ported.
namespace PilFflonk
{
    using FrElement = typename AltBn128::Engine::FrElement;
    using G1PointAffine = typename AltBn128::Engine::G1PointAffine;

    // One committed polynomial's slot inside a stage.
    struct ShPlonkStagePol
    {
        std::string name;
        u_int64_t degree;
    };

    // The polynomials a given stage contributes to an f_i.
    struct ShPlonkStage
    {
        u_int32_t stage;
        u_int32_t nPols;
        ShPlonkStagePol *pols;
    };

    // A combined polynomial f_i: the set of committed polynomials packed behind
    // one commitment, and the points it is opened at.
    struct ShPlonkPol
    {
        uint32_t index;
        u_int64_t degree;
        uint32_t nOpeningPoints;
        uint32_t *openingPoints;
        uint32_t nPols;
        std::string *pols;
        uint32_t nStages;
        ShPlonkStage *stages;
    };

    // A precomputed commitment to an f_i built only from constant polynomials,
    // together with the polynomial itself.
    struct ShPlonkCommitment
    {
        std::string name;
        G1PointAffine commit;
        uint64_t lenPol;
        FrElement *pol;
    };

    // Everything the SHPLONK opening needs that is fixed at setup time.
    class ShPlonkSetup
    {
    public:
        u_int32_t power = 0;
        u_int32_t powerW = 0;
        u_int32_t maxQDegree = 0;
        u_int32_t nPublics = 0;

        void *X2 = nullptr;

        std::map<u_int32_t, ShPlonkPol *> f;

        std::map<u_int32_t, std::map<u_int32_t, std::string> *> polsNamesStage;

        std::map<std::string, FrElement> omegas;

        std::map<std::string, ShPlonkCommitment *> fCommitments;
    };
}

#endif
