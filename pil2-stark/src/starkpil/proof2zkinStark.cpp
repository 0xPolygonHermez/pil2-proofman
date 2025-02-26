

#include <string>
#include <iostream>
#include "proof2zkinStark.hpp"
using namespace std;

json pointer2json(uint64_t *pointer, StarkInfo& starkInfo) {
    json j = json::object();
    
    uint64_t p = 0;

    if(starkInfo.airgroupValuesMap.size() > 0) {
        j["airgroupvalues"] = json::array();
        for(uint64_t i = 0; i < starkInfo.airgroupValuesMap.size(); i++) {
            j["airgroupvalues"][i] = json::array();
            for (uint64_t k = 0; k < FIELD_EXTENSION; k++)
            {
                j["airgroupvalues"][i][k] = std::to_string(pointer[p++]);
            }
        }
    }


    if(starkInfo.airValuesMap.size() > 0) {
        j["airvalues"] = json::array();
        for (uint i = 0; i < starkInfo.airValuesMap.size(); i++)
        {
            j["airvalues"][i] = json::array();
            for (uint k = 0; k < FIELD_EXTENSION; k++)
            {
                j["airvalues"][i][k] = std::to_string(pointer[p++]);
            }
        }
    }

    for(uint64_t i = 0; i < starkInfo.nStages + 1; i++) {
         j["root" + to_string(i + 1)] = json::array();
        for (uint64_t k = 0; k < 4; k++)
        {
            j["root" + to_string(i + 1)][k] = std::to_string(pointer[p++]);
        }
    }

    j["evals"] = json::array();
    for (uint i = 0; i < starkInfo.evMap.size(); i++)
    {
        j["evals"][i] = json::array();
        for (uint k = 0; k < FIELD_EXTENSION; k++)
        {
            j["evals"][i][k] = std::to_string(pointer[p++]);
        }
    }
    
    j["s0_valsC"] = json::array();
    for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++) {
        j["s0_valsC"][i] = json::array();
        for(uint64_t l = 0; l < starkInfo.nConstants; l++) {
            j["s0_valsC"][i][l] = std::to_string(pointer[p++]);
        }
    }

    j["s0_siblingsC"] = json::array();
    for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++) {
        j["s0_siblingsC"][i] = json::array();
        for(uint64_t l = 0; l < starkInfo.starkStruct.steps[0].nBits; ++l) {
            j["s0_siblingsC"][i][l] = json::array();
            for(uint64_t k = 0; k < 4; ++k) {
                j["s0_siblingsC"][i][l][k] = std::to_string(pointer[p++]);
            }
        }
    }

    for(uint64_t i = 0; i < starkInfo.customCommits.size(); ++i) {
        j["s0_siblings_" + starkInfo.customCommits[i].name + "_0"] = json::array();
        j["s0_vals_" + starkInfo.customCommits[i].name + "_0"] = json::array();
    }

    for(uint64_t c = 0; c < starkInfo.customCommits.size(); ++c) {
        for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++) {
            for(uint64_t l = 0; l < starkInfo.mapSectionsN[starkInfo.customCommits[c].name + "0"]; l++) {
                j["s0_vals_" + starkInfo.customCommits[c].name + "_0"][i][l] = std::to_string(pointer[p++]);
            }
        }
        for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++) {
            for(uint64_t l = 0; l < starkInfo.starkStruct.steps[0].nBits; ++l) {
                for(uint64_t k = 0; k < 4; ++k) {
                    j["s0_siblings_" + starkInfo.customCommits[c].name + "_0"][i][l][k] = std::to_string(pointer[p++]);
                }
            }
        }
    }

    for(uint64_t i = 0; i < starkInfo.nStages + 1; ++i) {
        uint64_t stage = i + 1;
        j["s0_siblings" + to_string(stage)] = json::array();
        j["s0_vals" + to_string(stage)] = json::array();
    }

    for (uint64_t s = 0; s < starkInfo.nStages + 1; ++s) {
        uint64_t stage = s + 1;
        for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++) {
            for(uint64_t l = 0; l < starkInfo.mapSectionsN["cm" + to_string(stage)]; l++) {
                j["s0_vals" + to_string(stage)][i][l] = std::to_string(pointer[p++]);
            }
        }

        for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++) {
            for(uint64_t l = 0; l < starkInfo.starkStruct.steps[0].nBits; ++l) {
                for(uint64_t k = 0; k < 4; ++k) {
                    j["s0_siblings" + to_string(stage)][i][l][k] = std::to_string(pointer[p++]);
                }
            }
        }
    }

    for(uint64_t step = 1; step < starkInfo.starkStruct.steps.size(); ++step) {
        j["s" + std::to_string(step) + "_root"] = json::array();
        for(uint64_t i = 0; i < 4; i++) {
            j["s" + std::to_string(step) + "_root"][i] = std::to_string(pointer[p++]);
        }
    }

    for(uint64_t step = 1; step < starkInfo.starkStruct.steps.size(); ++step) {
        for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++) {
            j["s" + std::to_string(step) + "_vals"][i] = json::array();
            for(uint64_t l = 0; l < uint64_t(1 << (starkInfo.starkStruct.steps[step - 1].nBits - starkInfo.starkStruct.steps[step].nBits)) * FIELD_EXTENSION; l++) {
                j["s" + std::to_string(step) + "_vals"][i][l] = std::to_string(pointer[p++]);
            }
        }

        for (uint64_t i = 0; i < starkInfo.starkStruct.nQueries; i++) {
            j["s" + std::to_string(step) + "_siblings"][i] = json::array();
            for(uint64_t l = 0; l < starkInfo.starkStruct.steps[step].nBits; ++l) {
                for(uint64_t k = 0; k < 4; ++k) {
                    j["s" + std::to_string(step) + "_siblings"][i][l][k] = std::to_string(pointer[p++]);
                }
            }
        }
    }

    j["finalPol"] = json::array();
    for (uint64_t i = 0; i < uint64_t (1 << (starkInfo.starkStruct.steps[starkInfo.starkStruct.steps.size() - 1].nBits)); i++)
    {
        j["finalPol"][i] = json::array();
        for(uint64_t l = 0; l < FIELD_EXTENSION; l++) {
            j["finalPol"][i][l] = std::to_string(pointer[p++]);
        }
    }
    return j;
}


json joinzkin(json &zkin1, json &zkin2, json &verKey, StarkInfo &starkInfo)
{

    uint64_t friSteps = starkInfo.starkStruct.steps.size();
    uint64_t nStages = starkInfo.nStages;

    string valsQ = "s0_vals" + to_string(nStages + 1);
    string siblingsQ = "s0_siblings" + to_string(nStages + 1);
    string rootQ = "root" + to_string(nStages + 1);

    json zkinOut = json::object();

    // Load oldStateRoot
    for (int i = 0; i < 8; i++)
    {
        zkinOut["publics"][i] = zkin1["publics"][i];
    }

    // Load oldAccInputHash0
    for (int i = 0; i < 8; i++)
    {
        zkinOut["publics"][i + 8] = zkin1["publics"][8 + i];
    }

    zkinOut["publics"][16] = zkin1["publics"][16]; // oldBatchNum

    zkinOut["publics"][17] = zkin1["publics"][17]; // chainId

    zkinOut["publics"][18] = zkin1["publics"][18]; // forkid

    // newStateRoot
    for (int i = 0; i < 8; i++)
    {
        zkinOut["publics"][19 + i] = zkin2["publics"][19 + i];
    }
    // newAccInputHash0
    for (int i = 0; i < 8; i++)
    {
        zkinOut["publics"][27 + i] = zkin2["publics"][27 + i];
    }
    // newLocalExitRoot
    for (int i = 0; i < 8; i++)
    {
        zkinOut["publics"][35 + i] = zkin2["publics"][35 + i];
    }

    zkinOut["publics"][43] = zkin2["publics"][43]; // oldBatchNum

    zkinOut["a_publics"] = zkin1["publics"];

    for(uint64_t stage = 1; stage <= nStages; stage++) {
        zkinOut["a_root" + to_string(stage)] = zkin1["root" + to_string(stage)];
    }
    zkinOut["a_" + rootQ] = zkin1[rootQ];

    zkinOut["a_evals"] = zkin1["evals"];
    zkinOut["a_s0_valsC"] = zkin1["s0_valsC"];
    zkinOut["a_s0_siblingsC"] = zkin1["s0_siblingsC"];
    for(uint64_t stage = 1; stage <= nStages; ++stage) {
        if(starkInfo.mapSectionsN["cm" + to_string(stage)] > 0) {
            zkinOut["a_s0_vals" + to_string(stage)] = zkin1["s0_vals" + to_string(stage)];
            zkinOut["a_s0_siblings" + to_string(stage)] = zkin1["s0_siblings" + to_string(stage)];
        }
    }
    zkinOut["a_" + siblingsQ] = zkin1[siblingsQ];
    zkinOut["a_" + valsQ] = zkin1[valsQ];

    for (uint64_t i = 1; i < friSteps; i++)
    {
        zkinOut["a_s" + std::to_string(i) + "_root"] = zkin1["s" + std::to_string(i) + "_root"];
        zkinOut["a_s" + std::to_string(i) + "_siblings"] = zkin1["s" + std::to_string(i) + "_siblings"];
        zkinOut["a_s" + std::to_string(i) + "_vals"] = zkin1["s" + std::to_string(i) + "_vals"];
    }
    zkinOut["a_finalPol"] = zkin1["finalPol"];

    zkinOut["b_publics"] = zkin2["publics"];
    for(uint64_t stage = 1; stage <= nStages; stage++) {
        zkinOut["b_root" + to_string(stage)] = zkin2["root" + to_string(stage)];
    }
    zkinOut["b_" + rootQ] = zkin2[rootQ];

    zkinOut["b_evals"] = zkin2["evals"];
    zkinOut["b_s0_valsC"] = zkin2["s0_valsC"];
    zkinOut["b_s0_siblingsC"] = zkin2["s0_siblingsC"];
    for(uint64_t stage = 1; stage <= nStages; ++stage) {
        if(starkInfo.mapSectionsN["cm" + to_string(stage)] > 0) {
            zkinOut["b_s0_vals" + to_string(stage)] = zkin2["s0_vals" + to_string(stage)];
            zkinOut["b_s0_siblings" + to_string(stage)] = zkin2["s0_siblings" + to_string(stage)];
        }
    }
    zkinOut["b_" + siblingsQ] = zkin2[siblingsQ];
    zkinOut["b_" + valsQ] = zkin2[valsQ];

    for (uint64_t i = 1; i < friSteps; i++)
    {
        zkinOut["b_s" + std::to_string(i) + "_root"] = zkin2["s" + std::to_string(i) + "_root"];
        zkinOut["b_s" + std::to_string(i) + "_siblings"] = zkin2["s" + std::to_string(i) + "_siblings"];
        zkinOut["b_s" + std::to_string(i) + "_vals"] = zkin2["s" + std::to_string(i) + "_vals"];
    }
    zkinOut["b_finalPol"] = zkin2["finalPol"];

    zkinOut["rootC"] = json::array();
    for (int i = 0; i < 4; i++)
    {
        zkinOut["rootC"][i] = to_string(verKey["constRoot"][i]);
    }

    return zkinOut;
}

json publics2zkin(json &zkin_, uint64_t nPublics, Goldilocks::Element* publics, json& globalInfo, uint64_t airgroupId) {
    json zkin = json::object();
    zkin = zkin_;

    uint64_t p = 0;
    zkin["sv_circuitType"] = Goldilocks::toString(publics[p++]);
    if(globalInfo["aggTypes"][airgroupId].size() > 0) {
        zkin["sv_aggregationTypes"] = json::array();
        for(uint64_t i = 0; i < globalInfo["aggTypes"][airgroupId].size(); ++i) {
            zkin["sv_aggregationTypes"][i] = Goldilocks::toString(publics[p++]);
        }

        zkin["sv_airgroupvalues"] = json::array();
        for(uint64_t i = 0; i < globalInfo["aggTypes"][airgroupId].size(); ++i) {
            zkin["sv_airgroupvalues"][i] = json::array();
            for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
                zkin["sv_airgroupvalues"][i][k] = Goldilocks::toString(publics[p++]);
            }
        }
    }

    zkin["sv_stage1Hash"] = json::array();
    for(uint64_t j = 0; j < 4; ++j) {
        zkin["sv_stage1Hash"][j] = Goldilocks::toString(publics[p++]);
    }
    
    if(uint64_t(globalInfo["nPublics"]) > 0) {
        zkin["publics"] = json::array();
        for(uint64_t i = 0; i < uint64_t(globalInfo["nPublics"]); ++i) {
            zkin["publics"][i] = Goldilocks::toString(publics[p++]);
        }
    }

    if(globalInfo["proofValuesMap"].size() > 0) {
        zkin["proofValues"] = json::array();
        for(uint64_t i = 0; i < globalInfo["proofValuesMap"].size(); ++i) {
            zkin["proofValues"][i] = json::array();
            cout << i << endl;
            for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
                            cout << k << endl;

                zkin["proofValues"][i][k] = Goldilocks::toString(publics[p++]);
            }
        }
    }

    zkin["globalChallenge"] = json::array();
    for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
        zkin["globalChallenge"][k] = Goldilocks::toString(publics[p++]);
    }

    cout << p << " vs " << nPublics << endl;

    return zkin;
}

json addRecursive2VerKey(json &zkin, Goldilocks::Element* recursive2VerKey) {
    json zkinUpdated = json::object();
    zkinUpdated = zkin;
    zkinUpdated["rootCAgg"] = json::array();
    for(uint64_t i = 0; i < 4; ++i) {
        zkinUpdated["rootCAgg"][i] = Goldilocks::toString(recursive2VerKey[i]);
    }

    return zkinUpdated;
}

json joinzkinfinal(json& globalInfo, Goldilocks::Element* publics, Goldilocks::Element* proofValues, Goldilocks::Element* globalChallenge, void **zkin_vec, void **starkInfo_vec) {
    json zkinFinal = json::object();
    
    if(globalInfo["nPublics"] > 0) {
        for (uint64_t i = 0; i < globalInfo["nPublics"]; i++)
        {
            zkinFinal["publics"][i] = Goldilocks::toString(publics[i]);
        }
    }

    if(globalInfo["proofValuesMap"].size() > 0) {
        uint64_t p = 0;
        for (uint64_t i = 0; i < globalInfo["proofValuesMap"].size(); i++)
        {
            if(globalInfo["proofValuesMap"][i]["stage"] == 1) {
                zkinFinal["proofValues"][i][0] = Goldilocks::toString(proofValues[p++]);
                zkinFinal["proofValues"][i][1] = "0";
                zkinFinal["proofValues"][i][2] = "0";
            } else {
                zkinFinal["proofValues"][i][0] = Goldilocks::toString(proofValues[p++]);
                zkinFinal["proofValues"][i][1] = Goldilocks::toString(proofValues[p++]);
                zkinFinal["proofValues"][i][2] = Goldilocks::toString(proofValues[p++]);
            }
        }
    }

    zkinFinal["globalChallenge"] = json::array();
    for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
        zkinFinal["globalChallenge"][k] = Goldilocks::toString(globalChallenge[k]);
    }

    for(uint64_t i = 0; i < globalInfo["air_groups"].size(); ++i) {
        json zkin = *(json *)zkin_vec[i];
        StarkInfo &starkInfo = *(StarkInfo *)starkInfo_vec[i];

        uint64_t nStages = starkInfo.nStages + 1;

        for(uint64_t stage = 1; stage <= nStages; stage++) {
            zkinFinal["s" + to_string(i) + "_root" + to_string(stage)] = zkin["root" + to_string(stage)];
        }

        for(uint64_t stage = 1; stage <= nStages; stage++) {
            if(starkInfo.mapSectionsN["cm" + to_string(stage)] > 0) {
                zkinFinal["s" + to_string(i) + "_s0_vals" + to_string(stage)] = zkin["s0_vals" + to_string(stage)];
                zkinFinal["s" + to_string(i) + "_s0_siblings" + to_string(stage)] = zkin["s0_siblings" + to_string(stage)];
            }
        }
        
        zkinFinal["s" + to_string(i) + "_s0_valsC"] = zkin["s0_valsC"];
        zkinFinal["s" + to_string(i) + "_s0_siblingsC"] = zkin["s0_siblingsC"];

        zkinFinal["s" + to_string(i) + "_evals"] = zkin["evals"];

        for(uint64_t s = 1; s < starkInfo.starkStruct.steps.size(); ++s) {
            zkinFinal["s" + to_string(i) + "_s" + to_string(s) + "_root"] = zkin["s" + to_string(s) + "_root"];
            zkinFinal["s" + to_string(i) + "_s" + to_string(s) + "_vals"] = zkin["s" + to_string(s) + "_vals"];
            zkinFinal["s" + to_string(i) + "_s" + to_string(s) + "_siblings"] = zkin["s" + to_string(s) + "_siblings"];
        }
        
        zkinFinal["s" + to_string(i) + "_finalPol"] = zkin["finalPol"];

        zkinFinal["s" + to_string(i) + "_sv_circuitType"] = zkin["sv_circuitType"];

        if(globalInfo["aggTypes"][i].size() > 0) {
            zkinFinal["s" + to_string(i) + "_sv_aggregationTypes"] = zkin["sv_aggregationTypes"];
            zkinFinal["s" + to_string(i) + "_sv_airgroupvalues"] = zkin["sv_airgroupvalues"];
        }

        zkinFinal["s" + to_string(i) + "_sv_stage1Hash"] = zkin["sv_stage1Hash"];
    }

    return zkinFinal;
}

json joinzkinrecursive2(json& globalInfo, uint64_t airgroupId, Goldilocks::Element* publics, Goldilocks::Element* proofValues, Goldilocks::Element* globalChallenge, json &zkin1, json &zkin2, StarkInfo &starkInfo) {
    json zkinRecursive2 = json::object();

    uint64_t nStages = starkInfo.nStages + 1;

    for (uint64_t i = 0; i < globalInfo["nPublics"]; i++)
    {
        zkinRecursive2["publics"][i] = Goldilocks::toString(publics[i]);
    }

    uint64_t p = 0;
    for (uint64_t i = 0; i < globalInfo["proofValuesMap"].size(); i++)
    {
        if(globalInfo["proofValuesMap"][i]["stage"] == 1) {
            zkinRecursive2["proofValues"][i][0] = Goldilocks::toString(proofValues[p++]);
            zkinRecursive2["proofValues"][i][1] = "0";
            zkinRecursive2["proofValues"][i][2] = "0";
        } else {
            zkinRecursive2["proofValues"][i][0] = Goldilocks::toString(proofValues[p++]);
            zkinRecursive2["proofValues"][i][1] = Goldilocks::toString(proofValues[p++]);
            zkinRecursive2["proofValues"][i][2] = Goldilocks::toString(proofValues[p++]);
        }
    }

    zkinRecursive2["globalChallenge"] = json::array();
    for(uint64_t k = 0; k < FIELD_EXTENSION; ++k) {
        zkinRecursive2["globalChallenge"][k] = Goldilocks::toString(globalChallenge[k]);
    }

    for(uint64_t stage = 1; stage <= nStages; stage++) {
        zkinRecursive2["a_root" + to_string(stage)] = zkin1["root" + to_string(stage)];
        zkinRecursive2["b_root" + to_string(stage)] = zkin2["root" + to_string(stage)];
    }

    for(uint64_t stage = 1; stage <= nStages; stage++) {
        if(starkInfo.mapSectionsN["cm" + to_string(stage)] > 0) {
            zkinRecursive2["a_s0_vals" + to_string(stage)] = zkin1["s0_vals" + to_string(stage)];
            zkinRecursive2["a_s0_siblings" + to_string(stage)] = zkin1["s0_siblings" + to_string(stage)];
            zkinRecursive2["b_s0_vals" + to_string(stage)] = zkin2["s0_vals" + to_string(stage)];
            zkinRecursive2["b_s0_siblings" + to_string(stage)] = zkin2["s0_siblings" + to_string(stage)];
        }
    }
    
    zkinRecursive2["a_s0_valsC"] = zkin1["s0_valsC"];
    zkinRecursive2["b_s0_valsC"] = zkin2["s0_valsC"];

    zkinRecursive2["a_s0_siblingsC"] = zkin1["s0_siblingsC"];
    zkinRecursive2["b_s0_siblingsC"] = zkin2["s0_siblingsC"];
    
    zkinRecursive2["a_evals"] = zkin1["evals"];
    zkinRecursive2["b_evals"] = zkin2["evals"];


    for(uint64_t s = 1; s < starkInfo.starkStruct.steps.size(); ++s) {
        zkinRecursive2["a_s" + to_string(s) + "_root"] = zkin1["s" + to_string(s) + "_root"];
        zkinRecursive2["a_s" + to_string(s) + "_vals"] = zkin1["s" + to_string(s) + "_vals"];
        zkinRecursive2["a_s" + to_string(s) + "_siblings"] = zkin1["s" + to_string(s) + "_siblings"];

        zkinRecursive2["b_s" + to_string(s) + "_root"] = zkin2["s" + to_string(s) + "_root"];
        zkinRecursive2["b_s" + to_string(s) + "_vals"] = zkin2["s" + to_string(s) + "_vals"];
        zkinRecursive2["b_s" + to_string(s) + "_siblings"] = zkin2["s" + to_string(s) + "_siblings"];
    }
    
    zkinRecursive2["a_finalPol"] = zkin1["finalPol"];
    zkinRecursive2["b_finalPol"] = zkin2["finalPol"];

    zkinRecursive2["a_sv_circuitType"] = zkin1["sv_circuitType"];
    zkinRecursive2["b_sv_circuitType"] = zkin2["sv_circuitType"];
    
    if(globalInfo["aggTypes"][airgroupId].size() > 0) {
        zkinRecursive2["aggregationTypes"] = zkin2["sv_aggregationTypes"];
        for(uint64_t a = 0; a < globalInfo["aggTypes"][airgroupId].size(); ++a) {
            assert(zkin2["sv_aggregationTypes"][a] == zkin1["sv_aggregationTypes"][a]);
        }

        zkinRecursive2["a_sv_airgroupvalues"] = zkin1["sv_airgroupvalues"];
        zkinRecursive2["b_sv_airgroupvalues"] = zkin2["sv_airgroupvalues"];
    }

    zkinRecursive2["a_sv_stage1Hash"] = zkin1["sv_stage1Hash"];
    zkinRecursive2["b_sv_stage1Hash"] = zkin2["sv_stage1Hash"];

    return zkinRecursive2;
}