#include "stark_info.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "expressions_pack.hpp"

StarkInfo::StarkInfo(string file, bool verify_constraints, bool verify, bool gpu)
{
    // Load contents from json file
    json starkInfoJson;
    file2json(file, starkInfoJson);
    load(starkInfoJson, verify_constraints, verify, gpu);
}

void StarkInfo::load(json j, bool verify_constraints_, bool verify_,  bool gpu_)
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

    for (uint64_t i = 0; i < j["airgroupValuesMap"].size(); i++) 
    {
        PolMap map;
        map.name = j["airgroupValuesMap"][i]["name"];
        map.stage = j["airgroupValuesMap"][i]["stage"];
        airgroupValuesMap.push_back(map);
    }

    for (uint64_t i = 0; i < j["airValuesMap"].size(); i++) 
    {
        PolMap map;
        map.name = j["airValuesMap"][i]["name"];
        map.stage = j["airValuesMap"][i]["stage"];
        airValuesMap.push_back(map);
    }

    for (uint64_t i = 0; i < j["proofValuesMap"].size(); i++) 
    {
        PolMap map;
        map.name = j["proofValuesMap"][i]["name"];
        map.stage = j["proofValuesMap"][i]["stage"];
        proofValuesMap.push_back(map);
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

    if(verify_) {
        verify = verify_;
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
    } else {
        verify_constraints = verify_constraints_;
        gpu = gpu_;
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

    assert(nStages <= 2);

    uint64_t maxTotalN = 0;
    // Set offsets for all stages in the extended field (cm1, cm2, ..., cmN)
    for(uint64_t stage = 1; stage <= nStages + 1; stage++) {
        mapOffsets[std::make_pair("cm" + to_string(stage), false)] = mapTotalN;
        mapOffsets[std::make_pair("cm" + to_string(stage), true)] = mapTotalN;
        mapTotalN += NExtended * mapSectionsN["cm" + to_string(stage)];
        if(starkStruct.verificationHashType == "GL") {
            mapOffsets[std::make_pair("mt" + to_string(stage), true)] = mapTotalN;
            mapTotalN += numNodes;
        }
    }

    mapOffsets[std::make_pair("evals", true)] = mapTotalN;
    mapTotalN += evMap.size() * omp_get_max_threads() * FIELD_EXTENSION;

    mapOffsets[std::make_pair("f", true)] = mapTotalN;
    mapOffsets[std::make_pair("q", true)] = mapTotalN;
    mapTotalN += NExtended * FIELD_EXTENSION;

    uint64_t LEvSize = mapOffsets[std::make_pair("f", true)];
    mapOffsets[std::make_pair("lev", false)] = LEvSize;
    LEvSize += openingPoints.size() * N * FIELD_EXTENSION;

    maxTotalN = std::max(maxTotalN, LEvSize);

    if(!gpu) {
        for(uint64_t stage = 1; stage <= nStages; stage++) {
            mapOffsets[std::make_pair("buff_helper_fft_" + to_string(stage), false)] = mapOffsets[std::make_pair("mt" + to_string(stage), true)];
            maxTotalN = std::max(maxTotalN, mapOffsets[std::make_pair("mt" + to_string(stage), true)] + NExtended * mapSectionsN["cm" + to_string(stage)]);
        }

        mapOffsets[std::make_pair("buff_helper_fft_" + to_string(nStages + 1), false)] = mapOffsets[std::make_pair("q", true)] + NExtended * FIELD_EXTENSION;
        maxTotalN = std::max(maxTotalN, mapOffsets[std::make_pair("q", true)] + NExtended * FIELD_EXTENSION + NExtended * mapSectionsN["cm" + to_string(nStages + 1)]);
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
    uint64_t NExtended = (1 << starkStruct.nBitsExt);
    uint64_t maxNBlocks, nrowsPack, mapBuffHelper;
    if(verify) {
        maxNBlocks = 1;
        nrowsPack = starkStruct.nQueries;
        mapBuffHelper = mapTotalN;
    } else {
        mapBuffHelper = mapOffsets[std::make_pair("f", true)] + NExtended * FIELD_EXTENSION;
        if(!gpu) {
            nrowsPack = NROWS_PACK;
            maxNBlocks = omp_get_max_threads();
        } else {
            nrowsPack = 128; // TODO: SHOULD NOT BE HARDCODED
            maxNBlocks = 2048; // TODO: SHOULD NOT BE HARDCODED
        }
    }
    
    
    uint64_t memoryTmp1 = nTmp1 * nrowsPack * maxNBlocks;
    mapOffsets[std::make_pair("tmp1", false)] = mapBuffHelper;
    mapBuffHelper += memoryTmp1;

    uint64_t memoryTmp3 = nTmp3 * FIELD_EXTENSION * nrowsPack * maxNBlocks;
    mapOffsets[std::make_pair("tmp3", false)] = mapBuffHelper;
    mapBuffHelper += memoryTmp3;

    uint64_t values = 3 * FIELD_EXTENSION * nrowsPack * maxNBlocks;
    mapOffsets[std::make_pair("values", false)] = mapBuffHelper;
    mapBuffHelper += values;

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