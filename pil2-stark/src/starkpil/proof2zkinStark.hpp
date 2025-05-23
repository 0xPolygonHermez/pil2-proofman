#ifndef PROOF2ZKIN__STARK_HPP
#define PROOF2ZKIN__STARK_HPP

#include <nlohmann/json.hpp>
#include "proof_stark.hpp"

using json = nlohmann::json;

json pointer2json(uint64_t *pointer, StarkInfo& starkInfo);

json joinzkin(json &zkin1, json &zkin2, json &verKey, StarkInfo &starkInfo);
json joinzkinfinal(json& globalInfo, Goldilocks::Element* publics, Goldilocks::Element *proofValues, Goldilocks::Element* challenges, void **zkin_vec, void **starkInfo_vec);
json joinzkinrecursive2(json& globalInfo, uint64_t airgroupId, Goldilocks::Element* publics, Goldilocks::Element* proofValues, Goldilocks::Element* challenges, json &zkin1, json &zkin2, StarkInfo &starkInfo);

json publics2zkin(json &zkin, uint64_t nPublics, Goldilocks::Element* publics, json& globalInfo, uint64_t airgroupId);
json addRecursive2VerKey(json &zkin, Goldilocks::Element* recursive2VerKey);

#endif