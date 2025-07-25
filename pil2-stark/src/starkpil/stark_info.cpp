#include "stark_info.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "expressions_pack.hpp"

StarkInfo::StarkInfo(string file, bool recursive_, bool verify_constraints_, bool verify_, bool gpu_, bool preallocate_)
{

    recursive = recursive_;
    verify_constraints = verify_constraints_;
    verify = verify_;
    gpu = gpu_;
    preallocate = preallocate_;

    // Load contents from json file
    json starkInfoJson;
    file2json(file, starkInfoJson);
    load(starkInfoJson);
}

void StarkInfo::load(json j)
{   
    starkStruct.nBits = j["starkStruct"]["nBits"];
    starkStruct.nBitsExt = j["starkStruct"]["nBitsExt"];
    starkStruct.nQueries = j["starkStruct"]["nQueries"];
    starkStruct.verificationHashType = j["starkStruct"]["verificationHashType"];
    if(starkStruct.verificationHashType == "BN128") {
        if(j["starkStruct"].contains("merkleTreeArity")) {
            starkStruct.merkleTreeArity = j["starkStruct"]["merkleTreeArity"];
        } else {
            starkStruct.merkleTreeArity = 16;
        }
        if(j["starkStruct"].contains("merkleTreeCustom")) {
            starkStruct.merkleTreeCustom = j["starkStruct"]["merkleTreeCustom"];
        } else {
            starkStruct.merkleTreeCustom = false;
        }
    } else {
        starkStruct.merkleTreeArity = 3;
        starkStruct.merkleTreeCustom = true;
    }
    if(j["starkStruct"].contains("hashCommits")) {
        starkStruct.hashCommits = j["starkStruct"]["hashCommits"];
    } else {
        starkStruct.hashCommits = false;
    }

    for (uint64_t i = 0; i < j["starkStruct"]["steps"].size(); i++)
    {
        StepStruct step;
        step.nBits = j["starkStruct"]["steps"][i]["nBits"];
        starkStruct.steps.push_back(step);
    }

    nPublics = j["nPublics"];
    nConstants = j["nConstants"];

    nStages = j["nStages"];

    qDeg = j["qDeg"];
    qDim = j["qDim"];

    friExpId = j["friExpId"];
    cExpId = j["cExpId"];


    for(uint64_t i = 0; i < j["customCommits"].size(); i++) {
        CustomCommits c;
        c.name = j["customCommits"][i]["name"];
        for(uint64_t k = 0; k < j["customCommits"][i]["publicValues"].size(); k++) {
            c.publicValues.push_back(j["customCommits"][i]["publicValues"][k]["idx"]);
        }
        for(uint64_t k = 0; k < j["customCommits"][i]["stageWidths"].size(); k++) {
            c.stageWidths.push_back(j["customCommits"][i]["stageWidths"][k]);
        }
        customCommits.push_back(c);
    }

    for(uint64_t i = 0; i < j["openingPoints"].size(); i++) {
        openingPoints.push_back(j["openingPoints"][i]);
    }

    for(uint64_t i = 0; i < j["boundaries"].size(); i++) {
        Boundary b;
        b.name = j["boundaries"][i]["name"];
        if(b.name == string("everyFrame")) {
            b.offsetMin = j["boundaries"][i]["offsetMin"];
            b.offsetMax = j["boundaries"][i]["offsetMax"];
        }
        boundaries.push_back(b);
    }

    for (uint64_t i = 0; i < j["challengesMap"].size(); i++) 
    {
        PolMap map;
        map.stage = j["challengesMap"][i]["stage"];
        map.name = j["challengesMap"][i]["name"];
        map.dim = j["challengesMap"][i]["dim"];
        map.stageId = j["challengesMap"][i]["stageId"];
        challengesMap.push_back(map);
    }

    for (uint64_t i = 0; i < j["publicsMap"].size(); i++) 
    {
        PolMap map;
        map.name = j["publicsMap"][i]["name"];
        if(j["publicsMap"][i].contains("lengths")) {
            for (uint64_t l = 0; l < j["publicsMap"][i]["lengths"].size(); l++) {
                map.lengths.push_back(j["publicsMap"][i]["lengths"][l]);
            } 
        }
        publicsMap.push_back(map);
    }

    airgroupValuesSize = 0;
    for (uint64_t i = 0; i < j["airgroupValuesMap"].size(); i++) 
    {
        PolMap map;
        map.name = j["airgroupValuesMap"][i]["name"];
        map.stage = j["airgroupValuesMap"][i]["stage"];
        airgroupValuesMap.push_back(map);
        if(map.stage == 1) {
            airgroupValuesSize += 1;
        } else {
            airgroupValuesSize += FIELD_EXTENSION;
        }
    }

    airValuesSize = 0;
    for (uint64_t i = 0; i < j["airValuesMap"].size(); i++) 
    {
        PolMap map;
        map.name = j["airValuesMap"][i]["name"];
        map.stage = j["airValuesMap"][i]["stage"];
        airValuesMap.push_back(map);
        if(map.stage == 1) {
            airValuesSize += 1;
        } else {
            airValuesSize += FIELD_EXTENSION;
        }
    }

    proofValuesSize = 0;
    for (uint64_t i = 0; i < j["proofValuesMap"].size(); i++) 
    {
        PolMap map;
        map.name = j["proofValuesMap"][i]["name"];
        map.stage = j["proofValuesMap"][i]["stage"];
        proofValuesMap.push_back(map);
        if(map.stage == 1) {
            proofValuesSize += 1;
        } else {
            proofValuesSize += FIELD_EXTENSION;
        }
    }

    for (uint64_t i = 0; i < j["cmPolsMap"].size(); i++) 
    {
        PolMap map;
        map.stage = j["cmPolsMap"][i]["stage"];
        map.name = j["cmPolsMap"][i]["name"];
        map.dim = j["cmPolsMap"][i]["dim"];
        map.imPol = j["cmPolsMap"][i].contains("imPol") ? true : false;
        map.stagePos = j["cmPolsMap"][i]["stagePos"];
        map.stageId = j["cmPolsMap"][i]["stageId"];
        if(j["cmPolsMap"][i].contains("expId")) {
            map.expId = j["cmPolsMap"][i]["expId"];
        }
        if(j["cmPolsMap"][i].contains("lengths")) {
            for (uint64_t k = 0; k < j["cmPolsMap"][i]["lengths"].size(); k++) {
                map.lengths.push_back(j["cmPolsMap"][i]["lengths"][k]);
            } 
        }
        map.polsMapId = j["cmPolsMap"][i]["polsMapId"];
        cmPolsMap.push_back(map);
    }

    for (uint64_t i = 0; i < j["customCommitsMap"].size(); i++) 
    {
        vector<PolMap> custPolsMap(j["customCommitsMap"][i].size());
        for(uint64_t k = 0; k < j["customCommitsMap"][i].size(); ++k) {
            PolMap map;
            map.stage = j["customCommitsMap"][i][k]["stage"];
            map.name = j["customCommitsMap"][i][k]["name"];
            map.dim = j["customCommitsMap"][i][k]["dim"];
            map.stagePos = j["customCommitsMap"][i][k]["stagePos"];
            map.stageId = j["customCommitsMap"][i][k]["stageId"];
            map.commitId = i;
            if(j["customCommitsMap"][i][k].contains("expId")) {
                map.expId = j["customCommitsMap"][i][k]["expId"];
            }
            if(j["customCommitsMap"][i].contains("lengths")) {
                for (uint64_t l = 0; l < j["customCommitsMap"][i][k]["lengths"].size(); l++) {
                    map.lengths.push_back(j["customCommitsMap"][i][k]["lengths"][l]);
                } 
            }
            map.polsMapId = j["customCommitsMap"][i][k]["polsMapId"];
            custPolsMap[k] = map;
        }
        customCommitsMap.push_back(custPolsMap);
    }


    for (uint64_t i = 0; i < j["constPolsMap"].size(); i++) 
    {
        PolMap map;
        map.stage = j["constPolsMap"][i]["stage"];
        map.name = j["constPolsMap"][i]["name"];
        map.dim = j["constPolsMap"][i]["dim"];
        map.imPol = false;
        map.stagePos = j["constPolsMap"][i]["stageId"];
        map.stageId = j["constPolsMap"][i]["stageId"];
        if(j["constPolsMap"][i].contains("lengths")) {
            for (uint64_t k = 0; k < j["constPolsMap"][i]["lengths"].size(); k++) {
                map.lengths.push_back(j["constPolsMap"][i]["lengths"][k]);
            } 
        }
        map.polsMapId = j["constPolsMap"][i]["polsMapId"];
        constPolsMap.push_back(map);
    }

    for (uint64_t i = 0; i < j["evMap"].size(); i++)
    {
        EvMap map;
        map.setType(j["evMap"][i]["type"]);
        if(j["evMap"][i]["type"] == "custom") {
            map.commitId = j["evMap"][i]["commitId"];
        }
        map.id = j["evMap"][i]["id"];
        map.prime = j["evMap"][i]["prime"];
        if(j["evMap"][i].contains("openingPos")) {
            map.openingPos = j["evMap"][i]["openingPos"];
        } else {
            int64_t prime = map.prime;
            auto openingPoint = std::find_if(openingPoints.begin(), openingPoints.end(), [prime](int p) { return p == prime; });
            if(openingPoint == openingPoints.end()) {
                zklog.error("Opening point not found");
                exitProcess();
                exit(-1);
            }
            map.openingPos = std::distance(openingPoints.begin(), openingPoint);
        }
        evMap.push_back(map);
    }

    for (auto it = j["mapSectionsN"].begin(); it != j["mapSectionsN"].end(); it++)  
    {
        mapSectionsN[it.key()] = it.value();
    }

    getProofSize();

    if(verify) {
        gpu = false;
        mapTotalN = 0;
        mapTotalNCustomCommitsFixed = 0;
        mapOffsets[std::make_pair("const", false)] = 0;
        for(uint64_t stage = 1; stage <= nStages + 1; ++stage) {
            mapOffsets[std::make_pair("cm" + to_string(stage), false)] = mapTotalN;
            mapTotalN += mapSectionsN["cm" + to_string(stage)] * starkStruct.nQueries;
        }

        // Set offsets for custom commits fixed
        for(uint64_t i = 0; i < customCommits.size(); ++i) {
            if(customCommits[i].stageWidths[0] > 0) {
                mapOffsets[std::make_pair(customCommits[i].name + "0", false)] = mapTotalNCustomCommitsFixed;
                mapTotalNCustomCommitsFixed += customCommits[i].stageWidths[0] * starkStruct.nQueries;
            }
        }
    } else if(verify_constraints) {
        uint64_t N = (1 << starkStruct.nBits);
        uint64_t NExtended = (1 << starkStruct.nBitsExt);
        mapTotalN = 0;

        mapOffsets[std::make_pair("const", false)] = 0;

        mapTotalNCustomCommitsFixed = 0;

        // Set offsets for custom commits fixed
        for(uint64_t i = 0; i < customCommits.size(); ++i) {
            if(customCommits[i].stageWidths[0] > 0) {
                mapOffsets[std::make_pair(customCommits[i].name + "0", false)] = mapTotalNCustomCommitsFixed;
                mapTotalNCustomCommitsFixed += customCommits[i].stageWidths[0] * N;
                mapOffsets[std::make_pair(customCommits[i].name + "0", true)] = mapTotalNCustomCommitsFixed;
                mapTotalNCustomCommitsFixed += customCommits[i].stageWidths[0] * NExtended + getNumNodesMT(NExtended);
            }
        }

        for(uint64_t stage = 1; stage <= nStages; stage++) {
            mapOffsets[std::make_pair("cm" + to_string(stage), false)] = mapTotalN;
            mapTotalN += N * mapSectionsN["cm" + to_string(stage)];
        }
        mapOffsets[std::make_pair("q", true)] = mapTotalN;
        mapTotalN += NExtended * FIELD_EXTENSION;
        mapOffsets[std::make_pair("mem_exps", false)] = mapTotalN;
    } else {
        setMapOffsets();
    }
}

void StarkInfo::getProofSize() {
    proofSize = 0;
    proofSize += airgroupValuesMap.size() * FIELD_EXTENSION;
    proofSize += airValuesMap.size() * FIELD_EXTENSION;

    proofSize += (nStages + 1) * 4; // Roots

    proofSize += evMap.size() * FIELD_EXTENSION; // Evals

    uint64_t nSiblings = std::ceil(starkStruct.steps[0].nBits / std::log2(starkStruct.merkleTreeArity));
    uint64_t nSiblingsPerLevel = (starkStruct.merkleTreeArity - 1) * 4;

    proofSize += starkStruct.nQueries * nConstants; // Constants Values
    proofSize += starkStruct.nQueries * nSiblings * nSiblingsPerLevel; // Siblings Constants Values

    for(uint64_t i = 0; i < customCommits.size(); ++i) {
        proofSize += starkStruct.nQueries * mapSectionsN[customCommits[i].name + "0"]; // Custom Commits Values
        proofSize += starkStruct.nQueries * nSiblings * nSiblingsPerLevel; // Siblings Custom Commits Siblings
    }

    for(uint64_t i = 0; i < nStages + 1; ++i) {
        proofSize += starkStruct.nQueries * mapSectionsN["cm" + to_string(i+1)];
        proofSize += starkStruct.nQueries * nSiblings * nSiblingsPerLevel;
    }

    proofSize += (starkStruct.steps.size() - 1) * 4; // Roots

    for(uint64_t i = 1; i < starkStruct.steps.size(); ++i) {
        uint64_t nSiblings = std::ceil(starkStruct.steps[i].nBits / std::log2(starkStruct.merkleTreeArity));
        uint64_t nSiblingsPerLevel = (starkStruct.merkleTreeArity - 1) * 4;
        proofSize += starkStruct.nQueries * (1 << (starkStruct.steps[i-1].nBits - starkStruct.steps[i].nBits))*FIELD_EXTENSION;
        proofSize += starkStruct.nQueries * nSiblings * nSiblingsPerLevel;
    }

    proofSize += (1 << starkStruct.steps[starkStruct.steps.size()-1].nBits) * FIELD_EXTENSION;
}

void StarkInfo::setMapOffsets() {
    uint64_t N = (1 << starkStruct.nBits);
    uint64_t NExtended = (1 << starkStruct.nBitsExt);

    // Set offsets for constants
    mapOffsets[std::make_pair("const", false)] = 0;
    mapOffsets[std::make_pair("const", true)] = 0;
    mapOffsets[std::make_pair("cm1", false)] = 0;

    mapTotalNCustomCommitsFixed = 0;

    // Set offsets for custom commits fixed
    for(uint64_t i = 0; i < customCommits.size(); ++i) {
        if(customCommits[i].stageWidths[0] > 0) {
            mapOffsets[std::make_pair(customCommits[i].name + "0", false)] = mapTotalNCustomCommitsFixed;
            mapTotalNCustomCommitsFixed += customCommits[i].stageWidths[0] * N;
            mapOffsets[std::make_pair(customCommits[i].name + "0", true)] = mapTotalNCustomCommitsFixed;
            mapTotalNCustomCommitsFixed += customCommits[i].stageWidths[0] * NExtended + getNumNodesMT(NExtended);
        }
    }

    mapTotalN = 0;

    uint64_t numNodes = getNumNodesMT(NExtended);

    if(!preallocate) {
        mapOffsets[std::make_pair("const", true)] = mapTotalN;
        MerkleTreeGL mt(starkStruct.merkleTreeArity, true, NExtended, nConstants);
        uint64_t constTreeSize = (2 + (NExtended * nConstants) + numNodes);
        mapTotalN += constTreeSize;

        mapOffsets[std::make_pair("const", false)] = mapTotalN;
        mapTotalN += N * nConstants;
    }

    if(gpu) {
        mapOffsets[std::make_pair("custom_fixed", false)] = mapTotalN;
        mapTotalN += mapTotalNCustomCommitsFixed;

        mapOffsets[std::make_pair("publics", false)] = mapTotalN;
        mapTotalN += nPublics;

        mapOffsets[std::make_pair("airgroupvalues", false)] = mapTotalN;
        mapTotalN += airgroupValuesSize;

        mapOffsets[std::make_pair("airvalues", false)] = mapTotalN;
        mapTotalN += airValuesSize;

        mapOffsets[std::make_pair("proofvalues", false)] = mapTotalN;
        mapTotalN += proofValuesSize;

        mapOffsets[std::make_pair("evals", false)] = mapTotalN;
        mapTotalN += evMap.size() * FIELD_EXTENSION;

        mapOffsets[std::make_pair("challenges", false)] = mapTotalN;
        mapTotalN += challengesMap.size() * FIELD_EXTENSION;

        mapOffsets[std::make_pair("xdivxsub", false)] = mapTotalN;
        mapTotalN += openingPoints.size() * FIELD_EXTENSION;

        mapOffsets[std::make_pair("fri_queries", false)] = mapTotalN;
        mapTotalN += starkStruct.nQueries;

        mapOffsets[std::make_pair("challenge", false)] = mapTotalN;
        mapTotalN += HASH_SIZE;

        maxTreeWidth = 0;
        for (auto it = mapSectionsN.begin(); it != mapSectionsN.end(); it++) 
        {
            uint64_t treeWidth = it->second;
            if(treeWidth > maxTreeWidth) {
                maxTreeWidth = treeWidth;
            }
        }
        for(uint64_t i = 0; i < starkStruct.steps.size() - 1; ++i) {
            uint64_t nGroups = 1 << starkStruct.steps[i + 1].nBits;
            uint64_t groupSize = (1 << starkStruct.steps[i].nBits) / nGroups;
            uint64_t treeWidth = groupSize * FIELD_EXTENSION;
            if(treeWidth > maxTreeWidth) {
                maxTreeWidth = treeWidth;
            }
        }

        maxProofSize = ceil(log10(1 << starkStruct.nBitsExt) / log10(starkStruct.merkleTreeArity)) * (starkStruct.merkleTreeArity - 1) * HASH_SIZE;

        maxProofBuffSize = maxTreeWidth + maxProofSize;
        uint64_t nTrees = 1 + (nStages + 1) + customCommits.size();
        uint64_t nTreesFRI = starkStruct.steps.size() - 1;
    
        uint64_t queriesProofSize = (nTrees + nTreesFRI) * maxProofBuffSize * starkStruct.nQueries;

        mapOffsets[std::make_pair("proof_queries", false)] = mapTotalN;
        mapTotalN += queriesProofSize;
        
        // TODO: ADD EXPRESSIONS MEM
    }

    
    assert(nStages <= 2);

    uint64_t maxTotalN = 0;
    // Set offsets for all stages in the extended field (cm1, cm2, ..., cmN)
    for(uint64_t stage = 1; stage <= nStages + 1; stage++) {
        mapOffsets[std::make_pair("cm" + to_string(stage), true)] = mapTotalN;
        mapTotalN += NExtended * mapSectionsN["cm" + to_string(stage)];
        if(starkStruct.verificationHashType == "GL") {
            mapOffsets[std::make_pair("mt" + to_string(stage), true)] = mapTotalN;
            mapTotalN += numNodes;
        }
    }

    if(!gpu || recursive) {
        for(uint64_t stage = 1; stage <= nStages + 1; stage++) {
            mapOffsets[std::make_pair("cm" + to_string(stage), false)] = mapOffsets[std::make_pair("cm" + to_string(stage), true)];
        }
    } else {
        uint64_t offsetTraces = mapOffsets[std::make_pair("cm2", true)];
        for(uint64_t stage = nStages; stage >= 1; stage--) {
            mapOffsets[std::make_pair("cm" + to_string(stage), false)] = offsetTraces;
            offsetTraces += N * mapSectionsN["cm" + to_string(stage)]; 
        }

        mapTotalN = std::max(mapTotalN, offsetTraces);
    }

    if(!gpu) {
        mapOffsets[std::make_pair("evals", true)] = mapTotalN;
        mapTotalN += evMap.size() * omp_get_max_threads() * FIELD_EXTENSION;
    }

    if(recursive) {
        uint64_t maxSizeHelper = 0;
        if(gpu) {
            maxSizeHelper = (boundaries.size() + 1) * NExtended;
            mapOffsets[std::make_pair("zi", true)] = mapTotalN;
            mapOffsets[std::make_pair("x_n", false)] = mapTotalN;
            mapOffsets[std::make_pair("x", true)] = mapTotalN + boundaries.size() * NExtended;
            mapTotalN += maxSizeHelper;
        }
        mapOffsets[std::make_pair("f", true)] = mapTotalN;
        mapOffsets[std::make_pair("q", true)] = mapTotalN;
        mapTotalN += NExtended * FIELD_EXTENSION;
        mapOffsets[std::make_pair("mem_exps", false)] = mapTotalN;
    } else {
        mapOffsets[std::make_pair("f", true)] = mapTotalN;
        mapOffsets[std::make_pair("q", true)] = mapTotalN;
        mapTotalN += NExtended * FIELD_EXTENSION;

        uint64_t maxSizeHelper = 0;
        if(gpu) {
            maxSizeHelper += boundaries.size() * NExtended;
            mapOffsets[std::make_pair("zi", true)] = mapTotalN;
            mapOffsets[std::make_pair("x", true)] = mapTotalN;
        }
        
        maxTotalN = std::max(maxTotalN, mapTotalN + maxSizeHelper);
        mapOffsets[std::make_pair("mem_exps", false)] = mapTotalN + maxSizeHelper;
    }

    uint64_t LEvSize = mapOffsets[std::make_pair("f", true)];
    mapOffsets[std::make_pair("lev", false)] = LEvSize;
    uint64_t maxOpenings = std::min(openingPoints.size(), uint64_t(4));
    LEvSize += maxOpenings * N * FIELD_EXTENSION;
    if(!gpu) {
        mapOffsets[std::make_pair("buff_helper_fft_lev", false)] = LEvSize;
        LEvSize += maxOpenings * N * FIELD_EXTENSION;
    } else {    
        mapOffsets[std::make_pair("extra_helper_fft_lev", false)] = LEvSize;
        LEvSize += FIELD_EXTENSION * N + openingPoints.size() * FIELD_EXTENSION;
    }

    maxTotalN = std::max(maxTotalN, LEvSize);

    mapOffsets[std::make_pair("buff_helper", false)] = mapTotalN;
    mapTotalN += NExtended * FIELD_EXTENSION;
    if(starkStruct.steps.size() > 1) {
        mapTotalN += (1 << (starkStruct.steps[0].nBits - starkStruct.steps[1].nBits)) >> 1;
    }

    if (!gpu) {
        for(uint64_t stage = 1; stage <= nStages; stage++) {
            uint64_t maxTotalNStage = mapOffsets[std::make_pair("mt" + to_string(stage), true)];
            mapOffsets[std::make_pair("buff_helper_fft_" + to_string(stage), false)] = maxTotalNStage;
            maxTotalNStage += NExtended * mapSectionsN["cm" + to_string(stage)];
            maxTotalN = std::max(maxTotalN, maxTotalNStage);
        }

        uint64_t maxTotalNStageQ = mapOffsets[std::make_pair("q", true)] + NExtended * FIELD_EXTENSION;
        mapOffsets[std::make_pair("buff_helper_fft_" + to_string(nStages + 1), false)] = maxTotalNStageQ;
        maxTotalNStageQ += NExtended * mapSectionsN["cm" + to_string(nStages + 1)];
        maxTotalN = std::max(maxTotalN, maxTotalNStageQ);
    } else {
        uint64_t maxTotalNStageQ = mapOffsets[std::make_pair("q", true)] + NExtended * FIELD_EXTENSION;
        mapOffsets[std::make_pair("extra_helper_fft", false)] = maxTotalNStageQ;
        maxTotalNStageQ += NExtended * FIELD_EXTENSION + qDeg;
        maxTotalN = std::max(maxTotalN, maxTotalNStageQ);
    }    
    

 
    for(uint64_t step = 0; step < starkStruct.steps.size() - 1; ++step) {
        uint64_t height = 1 << starkStruct.steps[step + 1].nBits;
        uint64_t width = ((1 << starkStruct.steps[step].nBits) / height) * FIELD_EXTENSION;
        mapOffsets[std::make_pair("fri_" + to_string(step + 1), true)] = mapTotalN;
        mapTotalN += height * width;
        if(starkStruct.verificationHashType == "GL") {
            uint64_t numNodes = getNumNodesMT(height);
            mapOffsets[std::make_pair("mt_fri_" + to_string(step + 1), true)] = mapTotalN;
            mapTotalN += numNodes;
        }
    }

    mapTotalN = std::max(mapTotalN, maxTotalN);
}

void StarkInfo::setMemoryExpressions(uint64_t nTmp1, uint64_t nTmp3) {
    uint64_t mapBuffHelper;
    if(verify) {
        maxNBlocks = 1;
        nrowsPack = starkStruct.nQueries;
        mapBuffHelper = mapTotalN;
    } else {
        mapBuffHelper =  mapOffsets[std::make_pair("mem_exps", false)];
        if(!gpu) {
            nrowsPack = NROWS_PACK;
            maxNBlocks = omp_get_max_threads();
        } else {
            // TODO: SHOULD NOT BE HARDCODED
            if(recursive) {
                nrowsPack = 64;
                maxNBlocks = 4096;
            } else {
                nrowsPack = 128;
                maxNBlocks = 2048;
            }
        }
    }
    
    
    uint64_t memoryTmp1 = nTmp1 * nrowsPack * maxNBlocks;
    mapOffsets[std::make_pair("tmp1", false)] = mapBuffHelper;
    mapBuffHelper += memoryTmp1;

    uint64_t memoryTmp3 = nTmp3 * FIELD_EXTENSION * nrowsPack * maxNBlocks;
    mapOffsets[std::make_pair("tmp3", false)] = mapBuffHelper;
    mapBuffHelper += memoryTmp3;

    if(!gpu) {
        uint64_t values = 3 * FIELD_EXTENSION * nrowsPack * maxNBlocks;
        mapOffsets[std::make_pair("values", false)] = mapBuffHelper;
        mapBuffHelper += values;
    } else {
        uint64_t destVals = 2 * FIELD_EXTENSION * nrowsPack * maxNBlocks;
        mapOffsets[std::make_pair("destVals", false)] = mapBuffHelper;
        mapBuffHelper += destVals;
    }

    if(mapBuffHelper > mapTotalN) {
        mapTotalN = mapBuffHelper;
    }
}

uint64_t StarkInfo::getNumNodesMT(uint64_t height) {
    uint64_t numNodes = height;
    uint64_t nodesLevel = height;
    
    while (nodesLevel > 1) {
        uint64_t extraZeros = (starkStruct.merkleTreeArity - (nodesLevel % starkStruct.merkleTreeArity)) % starkStruct.merkleTreeArity;
        numNodes += extraZeros;
        uint64_t nextN = (nodesLevel + (starkStruct.merkleTreeArity - 1))/starkStruct.merkleTreeArity;        
        numNodes += nextN;
        nodesLevel = nextN;
    }

    return numNodes * HASH_SIZE;
}

uint64_t StarkInfo::getTraceOffset(string type, PolMap &polInfo, bool domainExtended)
{
    std::string stage = type == "cm" ? "cm" + to_string(polInfo.stage) : type == "custom" ? customCommits[polInfo.commitId].name + "0"
                                                                                          : "const";
    uint64_t offset = mapOffsets[std::make_pair(stage, domainExtended)];
    offset += polInfo.stagePos;
    return offset;
}

uint64_t StarkInfo::getTraceNColsSection(string type, PolMap &polInfo, bool domainExtended)
{
    std::string stage = type == "cm" ? "cm" + to_string(polInfo.stage) : type == "custom" ? customCommits[polInfo.commitId].name + "0"
                                                                                          : "const";
    return mapSectionsN[stage];
}

opType string2opType(const string s) 
{
    if(s == "const") 
        return const_;
    if(s == "cm")
        return cm;
    if(s == "tmp")
        return tmp;
    if(s == "public")
        return public_;
    if(s == "airgroupvalue")
        return airgroupvalue;
    if(s == "challenge")
        return challenge;
    if(s == "number")
        return number;
    if(s == "string") 
        return string_;
    if(s == "airvalue") 
        return airvalue;
    if(s == "custom") 
        return custom;
    if(s == "proofvalue") 
        return proofvalue;
    zklog.error("string2opType() found invalid string=" + s);
    exitProcess();
    exit(-1);
}