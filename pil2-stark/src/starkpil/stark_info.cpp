#include "stark_info.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "expressions_pack.hpp"

StarkInfo::StarkInfo(string file, bool final_, bool recursive_, bool verify_constraints_, bool verify_, bool gpu_, bool preallocate_)
{

    recursive = recursive_;
    verify_constraints = verify_constraints_;
    verify = verify_;
    gpu = gpu_;
    recursive_final = final_;
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
        starkStruct.merkleTreeArity = j["starkStruct"]["merkleTreeArity"];
        starkStruct.merkleTreeCustom = j["starkStruct"]["merkleTreeCustom"];
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

uint64_t StarkInfo::getPinnedProofSize() {
    uint64_t pinnedProofSize = 0;

    pinnedProofSize += (nStages + 1) * 4; // Roots
    pinnedProofSize += customCommits.size() * 4; // Custom commits roots
    pinnedProofSize += (starkStruct.steps.size() - 1) * 4; // Steps roots

    uint64_t maxTreeWidth = 0;
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

    uint64_t maxProofSize = ceil(log10(1 << starkStruct.nBitsExt) / log10(starkStruct.merkleTreeArity)) * (starkStruct.merkleTreeArity - 1) * Poseidon2GoldilocksCommit::HASH_SIZE;

    uint64_t maxProofBuffSize = maxTreeWidth + maxProofSize;

    uint64_t nTrees = nStages + customCommits.size() + 2;
    uint64_t nTreesFRI = starkStruct.steps.size() - 1;
    uint64_t queriesProofSize = (nTrees + nTreesFRI) * maxProofBuffSize * starkStruct.nQueries;

    pinnedProofSize += queriesProofSize;

    pinnedProofSize += evMap.size() * FIELD_EXTENSION; // Evals

    pinnedProofSize += airgroupValuesSize;
    pinnedProofSize += airValuesSize;

    uint64_t finalPolDegree = 1 << starkStruct.steps[starkStruct.steps.size() - 1].nBits;
    pinnedProofSize += finalPolDegree * FIELD_EXTENSION; // Final polynomial values
    return pinnedProofSize;
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

    if(!preallocate && gpu && !recursive_final) {    
        mapOffsets[std::make_pair("const", true)] = mapTotalN;
        MerkleTreeGL mt(starkStruct.merkleTreeArity, true, NExtended, nConstants);
        uint64_t constTreeSize = (NExtended * nConstants) + numNodes;
        mapTotalN += constTreeSize;

        if (!recursive &&  (N * nConstants * 8.0 / (1024 * 1024)) >= 256) {
            calculateFixedExtended = true;
        }
        
        // NOTE: We could overwrite fixed but we would need to remove height and width from fixed file
        mapOffsets[std::make_pair("const", false)] = mapTotalN;
        mapTotalN += N * nConstants;
        
    }

    if(gpu && !recursive_final) {
        mapOffsets[std::make_pair("custom_fixed", false)] = mapTotalN;
        mapTotalN += mapTotalNCustomCommitsFixed;

        mapOffsets[std::make_pair("publics", false)] = mapTotalN;
        mapTotalN += nPublics;

        mapOffsets[std::make_pair("proofvalues", false)] = mapTotalN;
        mapTotalN += proofValuesSize;

        mapOffsets[std::make_pair("airgroupvalues", false)] = mapTotalN;
        mapTotalN += airgroupValuesSize;

        mapOffsets[std::make_pair("airvalues", false)] = mapTotalN;
        mapTotalN += airValuesSize;

        mapOffsets[std::make_pair("challenge", false)] = mapTotalN;
        mapTotalN += Poseidon2GoldilocksCommit::HASH_SIZE;

        mapOffsets[std::make_pair("evals", false)] = mapTotalN;
        mapTotalN += evMap.size() * FIELD_EXTENSION;

        mapOffsets[std::make_pair("challenges", false)] = mapTotalN;
        mapTotalN += challengesMap.size() * FIELD_EXTENSION;

        mapOffsets[std::make_pair("xdivxsub", false)] = mapTotalN;
        mapTotalN += openingPoints.size() * FIELD_EXTENSION;

        mapOffsets[std::make_pair("fri_queries", false)] = mapTotalN;
        mapTotalN += starkStruct.nQueries;

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

        maxProofSize = ceil(log10(1 << starkStruct.nBitsExt) / log10(starkStruct.merkleTreeArity)) * (starkStruct.merkleTreeArity - 1) * Poseidon2GoldilocksCommit::HASH_SIZE;

        maxProofBuffSize = maxTreeWidth + maxProofSize;
        uint64_t nTrees = 1 + (nStages + 1) + customCommits.size();
        uint64_t nTreesFRI = starkStruct.steps.size() - 1;
    
        uint64_t queriesProofSize = (nTrees + nTreesFRI) * maxProofBuffSize * starkStruct.nQueries;

        mapOffsets[std::make_pair("proof_queries", false)] = mapTotalN;
        mapTotalN += queriesProofSize;
        
        // TODO: ADD EXPRESSIONS MEM
    }

    assert(nStages == 2);

    uint64_t maxTotalN = 0;
    
    mapOffsets[std::make_pair("cm1", true)] = mapTotalN;
    mapTotalN += NExtended * mapSectionsN["cm1"];
    mapOffsets[std::make_pair("mt1", true)] = mapTotalN;
    mapTotalN += numNodes;

    mapOffsets[std::make_pair("cm1", false)] = mapTotalN;

    mapOffsets[std::make_pair("cm2", true)] = mapTotalN;
    mapTotalN += NExtended * mapSectionsN["cm2"];
    mapOffsets[std::make_pair("mt2", true)] = mapTotalN;
    mapTotalN += numNodes;
    mapTotalN = std::max(mapOffsets[std::make_pair("cm1", false)] + N * mapSectionsN["cm1"], mapTotalN);

    mapOffsets[std::make_pair("cm2", false)] = mapTotalN;
    
    mapOffsets[std::make_pair("cm3", true)] = mapTotalN;
    mapTotalN += NExtended * mapSectionsN["cm3"];
    mapOffsets[std::make_pair("mt3", true)] = mapTotalN;
    mapTotalN += numNodes;

    if(!gpu || (gpu && recursive_final)) {
        mapOffsets[std::make_pair("evals", true)] = mapTotalN;
        mapTotalN += evMap.size() * omp_get_max_threads() * FIELD_EXTENSION;
    }

    mapTotalN = std::max(mapOffsets[std::make_pair("cm2", false)] + N * mapSectionsN["cm2"], mapTotalN);
    mapOffsets[std::make_pair("f", true)] = mapTotalN;
    mapOffsets[std::make_pair("q", true)] = mapTotalN;
    mapTotalN += NExtended * FIELD_EXTENSION;

    uint64_t maxSizeHelper = 0;
    if(gpu && !recursive_final) {
        maxSizeHelper += boundaries.size() * NExtended;
        mapOffsets[std::make_pair("zi", true)] = mapTotalN;
        mapOffsets[std::make_pair("x", true)] = mapTotalN;
    }
    
    maxTotalN = std::max(maxTotalN, mapTotalN + maxSizeHelper);
    mapOffsets[std::make_pair("mem_exps", false)] = mapTotalN + maxSizeHelper;   

    uint64_t LEvSize = mapOffsets[std::make_pair("f", true)];
    mapOffsets[std::make_pair("lev", false)] = LEvSize;
    uint64_t maxOpenings = std::min(uint64_t(openingPoints.size()), uint64_t(4));
    LEvSize += maxOpenings * N * FIELD_EXTENSION;
    if(!gpu || (gpu && recursive_final)) {
        mapOffsets[std::make_pair("buff_helper_fft_lev", false)] = LEvSize;
        LEvSize += maxOpenings * N * FIELD_EXTENSION;
    } else {    
        mapOffsets[std::make_pair("extra_helper_fft_lev", false)] = LEvSize;
        LEvSize += FIELD_EXTENSION * N + openingPoints.size() * FIELD_EXTENSION;
    }

    maxTotalN = std::max(maxTotalN, LEvSize);

    mapOffsets[std::make_pair("buff_helper", false)] = mapTotalN;
    mapTotalN += NExtended * FIELD_EXTENSION;

    if (!gpu || (gpu && recursive_final)) {
        uint64_t maxTotalNStage2 = mapOffsets[std::make_pair("cm2", false)] + N * mapSectionsN["cm2"];
        mapOffsets[std::make_pair("buff_helper_fft_2", false)] = maxTotalNStage2;
        maxTotalNStage2 += NExtended * mapSectionsN["cm2"];
        maxTotalN = std::max(maxTotalN, maxTotalNStage2);
        
        uint64_t maxTotalNStage1 = mapOffsets[std::make_pair("cm1", false)] + N * mapSectionsN["cm1"];
        mapOffsets[std::make_pair("buff_helper_fft_1", false)] = maxTotalNStage1;
        maxTotalNStage1 += NExtended * mapSectionsN["cm1"];
        maxTotalN = std::max(maxTotalN, maxTotalNStage1);

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
        if(!gpu || (gpu && recursive_final)) {
            nrowsPack = NROWS_PACK;
            maxNBlocks = omp_get_max_threads();
        } else {
            if(recursive) {
                uint64_t NExtended = (1 << starkStruct.nBitsExt);
                nrowsPack = 32;
                maxNBlocks = NExtended / nrowsPack;
            } else {
                nrowsPack = 256;
                maxNBlocks = 4096;
            }
        }
    }
    
    
    uint64_t memoryTmp1 = nTmp1 * nrowsPack * maxNBlocks;
    mapOffsets[std::make_pair("tmp1", false)] = mapBuffHelper;
    mapBuffHelper += memoryTmp1;

    uint64_t memoryTmp3 = nTmp3 * FIELD_EXTENSION * nrowsPack * maxNBlocks;
    mapOffsets[std::make_pair("tmp3", false)] = mapBuffHelper;
    mapBuffHelper += memoryTmp3;

    if(!gpu || (gpu && recursive_final)) {
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

    return numNodes * Poseidon2GoldilocksCommit::HASH_SIZE;
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
    if(s == "Zi")
        return Zi;
    if(s == "eval")
        return eval;
    if(s == "xDivXSubXi") 
        return xDivXSubXi;
    if(s == "q") 
        return q;
    if(s == "f") 
        return f;
    if(s == "proofvalue")
        return proofvalue;
    zklog.error("string2opType() found invalid string=" + s);
    exitProcess();
    exit(-1);
}


string opType2string(const opType op) 
{
    if(op == opType::const_) 
        return "const";
    if(op == opType::cm)
        return "cm";
    if(op == opType::tmp)
        return "tmp";
    if(op == opType::public_)
        return "public";
    if(op == opType::airgroupvalue)
        return "airgroupvalue";
    if(op == opType::challenge)
        return "challenge";
    if(op == opType::number)
        return "number";
    if(op == opType::string_) 
        return "string";
    if(op == opType::airvalue) 
        return "airvalue";
    if(op == opType::custom) 
        return "custom";
    if(op == opType::Zi)
        return "Zi";
    if(op == opType::eval)
        return "eval";
    if(op == opType::xDivXSubXi) 
        return "xDivXSubXi";
    if(op == opType::q)
        return "q";
    if(op == opType::f)
        return "f";
    if(op == opType::eval)
        return "eval";
    if(op == opType::proofvalue)
        return "proofvalue";

    zklog.error("string2opType() found invalid operation");
    exitProcess();
    exit(-1);
}
