#include "stark_info.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "expressions_pack.hpp"
#include "grinding_launch.hpp"
#include "stir/stir_layout.hpp"

StarkInfo::StarkInfo(string file, bool final_, bool recursive_, bool verify_constraints_, bool verify_, bool gpu_)
{

    recursive = recursive_;
    verify_constraints = verify_constraints_;
    verify = verify_;
    gpu = gpu_;
    inplaceStageCommit = !final_ && !recursive_ && !verify_constraints_ && !verify_;

    // Load contents from json file
    json starkInfoJson;
    file2json(file, starkInfoJson);
    load(starkInfoJson);
}

void StarkInfo::load(json j)
{   
    starkStruct.nBits = j["starkStruct"]["nBits"];
    starkStruct.nBitsExt = j["starkStruct"]["nBitsExt"];
    starkStruct.verificationHashType = j["starkStruct"]["verificationHashType"];
    if(starkStruct.verificationHashType == "BN128") {
        if(j["starkStruct"].contains("merkleTreeArity")) {
            starkStruct.merkleTreeArity = j["starkStruct"]["merkleTreeArity"];
            starkStruct.transcriptArity = j["starkStruct"]["transcriptArity"];
        } else {
            starkStruct.merkleTreeArity = 16;
            starkStruct.transcriptArity = 16;
        }
        if(j["starkStruct"].contains("merkleTreeCustom")) {
            starkStruct.merkleTreeCustom = j["starkStruct"]["merkleTreeCustom"];
        } else {
            starkStruct.merkleTreeCustom = false;
        }
        if(j["starkStruct"].contains("lastLevelVerification")) {
            starkStruct.lastLevelVerification = j["starkStruct"]["lastLevelVerification"];
        } else {
            starkStruct.lastLevelVerification = 0;
        }
    } else {
        starkStruct.merkleTreeArity = j["starkStruct"]["merkleTreeArity"];
        starkStruct.transcriptArity = j["starkStruct"]["transcriptArity"];
        starkStruct.merkleTreeCustom = j["starkStruct"]["merkleTreeCustom"];
        starkStruct.lastLevelVerification = j["starkStruct"]["lastLevelVerification"];
    }
    if(j["starkStruct"].contains("hashCommits")) {
        starkStruct.hashCommits = j["starkStruct"]["hashCommits"];
    } else {
        starkStruct.hashCommits = false;
    }

    std::string lowDegreeTest = j["starkStruct"].contains("lowDegreeTest") ? j["starkStruct"]["lowDegreeTest"].get<std::string>() : "FRI";
    if (lowDegreeTest == "STIR") {
        starkStruct.lowDegreeTest = LowDegreeTestKind::STIR;
        if (starkStruct.verificationHashType != "GL") {
            zklog.error("StarkInfo::load() STIR supports GL verification hashes only, got " + starkStruct.verificationHashType);
            exitProcess();
        }
        json stir = j["starkStruct"];
        starkStruct.stir.foldingFactors = stir["foldingFactors"].get<vector<uint64_t>>();
        starkStruct.stir.logDegrees = stir["logDegrees"].get<vector<uint64_t>>();
        starkStruct.stir.logDomainSizes = stir["logDomainSizes"].get<vector<uint64_t>>();
        starkStruct.stir.numQueries = stir.contains("numQueries") ? stir["numQueries"].get<vector<uint64_t>>() : vector<uint64_t>();
        starkStruct.stir.grindingBitsQueries = stir.contains("grindingBitsQueries") ? stir["grindingBitsQueries"].get<vector<uint64_t>>() : vector<uint64_t>();
        uint64_t M = starkStruct.stir.numIterations();
        if (starkStruct.stir.logDegrees.size() != M + 1 || starkStruct.stir.logDomainSizes.size() != M + 1 ||
            starkStruct.stir.numQueries.size() != M || starkStruct.stir.grindingBitsQueries.size() != M) {
            zklog.error("StarkInfo::load() STIR schedule lengths are inconsistent (M=" + to_string(M) + ")");
            exitProcess();
        }
        // Every quotient round i = 1..M-1 divides by |G_i| = t_{i-1} + s points, so it needs
        // |G_i| < d_i — the setup enforces this, but a stark info is untrusted input here.
        for (uint64_t i = 1; i < M; i++) {
            uint64_t nG = starkStruct.stir.numQueries[i - 1] + starkStruct.stir.numOodSamples;
            if (nG >= (uint64_t(1) << starkStruct.stir.logDegrees[i])) {
                zklog.error("StarkInfo::load() invalid STIR schedule: |G_" + to_string(i) + "| = " + to_string(nG) +
                            " is not below d_" + to_string(i) + " = 2^" + to_string(starkStruct.stir.logDegrees[i]));
                exitProcess();
            }
        }
        // The stage/constant/custom trees are opened at the round-1 shift queries, t_0 of them —
        // the STIR counterpart of FRI's nQueries, and what every stage-level consumer keys on.
        starkStruct.nQueries = starkStruct.stir.numQueries.empty() ? 0 : starkStruct.stir.numQueries[0];
    } else if (lowDegreeTest == "FRI") {
        starkStruct.lowDegreeTest = LowDegreeTestKind::FRI;
        starkStruct.nQueries = j["starkStruct"]["numQueries"];
        starkStruct.logDomainSizes = j["starkStruct"]["logDomainSizes"].get<vector<uint64_t>>();
        // The same key holds FRI's scalar grinding budget and STIR's per-round vector,
        // so it can only be read after the lowDegreeTest dispatch.
        starkStruct.grindingBitsQueries =
            j["starkStruct"].contains("grindingBitsQueries") ? j["starkStruct"]["grindingBitsQueries"].get<uint64_t>() : 0;
    } else {
        zklog.error("StarkInfo::load() unknown lowDegreeTest: " + lowDegreeTest);
        exitProcess();
    }

    nPublics = j["nPublics"];
    nConstants = j["nConstants"];

    hasCircuitType = j.contains("circuitType");
    if (hasCircuitType) circuitType = j["circuitType"];

    nStages = j["nStages"];

    qDeg = j["qDeg"];
    qDim = j["qDim"];

    nConstraints = j["nConstraints"];

    deepExpId = j["deepExpId"];
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

        if(gpu) {
            mapOffsets[std::make_pair("const", false)] = 0;
            mapTotalN += N * nConstants;
        }

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

        if(gpu) {
            mapOffsets[std::make_pair("constraints", false)] = mapTotalN;
            mapTotalN += N * 3;

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
            mapTotalN += 2 * FIELD_EXTENSION;
        }
        
        for(uint64_t stage = 1; stage <= nStages; stage++) {
            mapOffsets[std::make_pair("cm" + to_string(stage), false)] = mapTotalN;
            mapTotalN += N * mapSectionsN["cm" + to_string(stage)];
        }
        mapOffsets[std::make_pair("q", true)] = mapTotalN;
        mapTotalN += N * FIELD_EXTENSION;
        mapOffsets[std::make_pair("mem_exps", false)] = mapTotalN;
    } else {
        setMapOffsets();
    }
}

// The trees the low-degree test commits: FRI's folded oracles (one per step after the first),
// STIR's T_0..T_{M−1}. Leaf `i` holds one folding coset, so its width is 2^{k_i} cubic elements.
uint64_t StarkInfo::numLowDegreeTestTrees() const {
    if (starkStruct.lowDegreeTest == LowDegreeTestKind::STIR) return starkStruct.stir.numIterations();
    return starkStruct.logDomainSizes.empty() ? 0 : starkStruct.logDomainSizes.size() - 1;
}

uint64_t StarkInfo::lowDegreeTestTreeWidth(uint64_t i) const {
    if (starkStruct.lowDegreeTest == LowDegreeTestKind::STIR) {
        return (uint64_t(1) << starkStruct.stir.foldingFactors[i]) * FIELD_EXTENSION;
    }
    uint64_t nGroups = 1 << starkStruct.logDomainSizes[i + 1];
    uint64_t groupSize = (1 << starkStruct.logDomainSizes[i]) / nGroups;
    return groupSize * FIELD_EXTENSION;
}

// The u64 the flat proof buffer spends on the STIR section — exactly what `Proofs::proof2pointer`
// writes for it. Kept here so the proof-size accounting and the writer share one formula.
uint64_t StarkInfo::stirProofSectionSize() const {
    uint64_t M = starkStruct.stir.numIterations();
    bool bn128 = starkStruct.verificationHashType == std::string("BN128");
    uint64_t nFieldElements = bn128 ? 1 : HASH_SIZE;
    uint64_t nSiblingsPerLevel = (starkStruct.merkleTreeArity - 1) * nFieldElements;
    uint64_t numNodesLevel = starkStruct.lastLevelVerification == 0 ? 0 : (uint64_t)std::pow(starkStruct.merkleTreeArity, starkStruct.lastLevelVerification);

    uint64_t size = 0;
    size += M * nFieldElements;                                       // roots of T_0..T_{M−1}
    for (uint64_t i = 0; i < M; ++i) {
        uint64_t k = uint64_t(1) << starkStruct.stir.foldingFactors[i];
        uint64_t logLeaves = starkStruct.stir.logDomainSizes[i] - starkStruct.stir.foldingFactors[i];
        uint64_t nSiblings = merkleProofLevels(logLeaves, starkStruct.merkleTreeArity, starkStruct.lastLevelVerification, bn128);
        size += starkStruct.stir.numQueries[i] * k * FIELD_EXTENSION;           // opened cosets
        size += starkStruct.stir.numQueries[i] * nSiblings * nSiblingsPerLevel; // their Merkle paths
        size += numNodesLevel * nFieldElements;                                 // published last level
    }
    size += (M - 1) * starkStruct.stir.numOodSamples * FIELD_EXTENSION;         // β
    size += (uint64_t(1) << starkStruct.stir.logDegrees[M]) * FIELD_EXTENSION;  // p, in coefficients
    size += M;                                                                  // one nonce per query message
    for (uint64_t i = 1; i < M; ++i) {                                          // Âns and shake coefficients, zero-padded
        size += 2 * (starkStruct.stir.numOodSamples + starkStruct.stir.numQueries[i - 1]) * FIELD_EXTENSION;
    }
    return size;
}

void StarkInfo::getProofSize() {
    proofSize = 0;
    proofSize += airgroupValuesMap.size() * FIELD_EXTENSION;
    proofSize += airValuesMap.size() * FIELD_EXTENSION;

    proofSize += (nStages + 1) * 4; // Roots

    proofSize += evMap.size() * FIELD_EXTENSION; // Evals

    uint64_t nSiblings = merkleProofLevels(starkStruct.nBitsExt, starkStruct.merkleTreeArity, starkStruct.lastLevelVerification, starkStruct.verificationHashType == std::string("BN128"));
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

    if(starkStruct.lowDegreeTest == LowDegreeTestKind::STIR) {
        if(starkStruct.lastLevelVerification > 0) {
            uint64_t numNodesLevel = std::pow(starkStruct.merkleTreeArity, starkStruct.lastLevelVerification);
            proofSize += (nStages + 2 + customCommits.size()) * numNodesLevel * 4;
        }
        proofSize += stirProofSectionSize();
        return;
    }

    proofSize += (starkStruct.logDomainSizes.size() - 1) * 4; // Roots

    if(starkStruct.lastLevelVerification > 0) {
        uint64_t numNodesLevel = std::pow(starkStruct.merkleTreeArity, starkStruct.lastLevelVerification);
        proofSize += (starkStruct.logDomainSizes.size() - 1) * numNodesLevel * 4;
        proofSize += (nStages + 2 + customCommits.size()) * numNodesLevel * 4;
    }

    for(uint64_t i = 1; i < starkStruct.logDomainSizes.size(); ++i) {
        uint64_t nSiblings = merkleProofLevels(starkStruct.logDomainSizes[i], starkStruct.merkleTreeArity, starkStruct.lastLevelVerification, starkStruct.verificationHashType == std::string("BN128"));
        uint64_t nSiblingsPerLevel = (starkStruct.merkleTreeArity - 1) * 4;
        proofSize += starkStruct.nQueries * (1 << (starkStruct.logDomainSizes[i-1] - starkStruct.logDomainSizes[i]))*FIELD_EXTENSION;
        proofSize += starkStruct.nQueries * nSiblings * nSiblingsPerLevel;
    }

    proofSize += (1 << starkStruct.logDomainSizes[starkStruct.logDomainSizes.size()-1]) * FIELD_EXTENSION;
    proofSize += 1; // Nonce
}

uint64_t StarkInfo::getPinnedProofSize() {
    const bool stirTest = starkStruct.lowDegreeTest == LowDegreeTestKind::STIR;
    const uint64_t nLdtTrees = numLowDegreeTestTrees();

    uint64_t pinnedProofSize = 0;

    pinnedProofSize += (nStages + 1) * 4; // Roots
    pinnedProofSize += customCommits.size() * 4; // Custom commits roots
    pinnedProofSize += nLdtTrees * 4; // Low-degree-test roots

    if(starkStruct.lastLevelVerification > 0) {
        uint64_t numNodesLevel = std::pow(starkStruct.merkleTreeArity, starkStruct.lastLevelVerification);
        pinnedProofSize += (nStages + 2 + customCommits.size()) * numNodesLevel * 4;
        pinnedProofSize += nLdtTrees * numNodesLevel * 4;
    }
    
    uint64_t maxTreeWidth = 0;
    for (auto it = mapSectionsN.begin(); it != mapSectionsN.end(); it++) 
    {
        uint64_t treeWidth = it->second;
        if(treeWidth > maxTreeWidth) {
            maxTreeWidth = treeWidth;
        }
    }
    for(uint64_t i = 0; i < nLdtTrees; ++i) {
        uint64_t treeWidth = lowDegreeTestTreeWidth(i);
        if(treeWidth > maxTreeWidth) {
            maxTreeWidth = treeWidth;
        }
    }

    uint64_t nSiblings = merkleProofLevels(starkStruct.nBitsExt, starkStruct.merkleTreeArity, starkStruct.lastLevelVerification, starkStruct.verificationHashType == std::string("BN128"));
    uint64_t nSiblingsPerLevel = (starkStruct.merkleTreeArity - 1) * HASH_SIZE;
    uint64_t maxProofSize = nSiblings * nSiblingsPerLevel;

    uint64_t maxProofBuffSize = maxTreeWidth + maxProofSize;

    // FRI opens every tree at the same number of queries; STIR's counts shrink per iteration, so
    // the first (t_0 = nQueries) is already the largest.
    uint64_t nTrees = nStages + customCommits.size() + 2;
    uint64_t queriesProofSize = (nTrees + nLdtTrees) * maxProofBuffSize * starkStruct.nQueries;

    pinnedProofSize += queriesProofSize;

    pinnedProofSize += evMap.size() * FIELD_EXTENSION; // Evals

    pinnedProofSize += airgroupValuesSize;
    pinnedProofSize += airValuesSize;

    if(stirTest) {
        pinnedProofSize += stirLayout().proofSize; // the device proof block, copied as one
    } else {
        uint64_t finalPolDegree = 1 << starkStruct.logDomainSizes[starkStruct.logDomainSizes.size() - 1];
        pinnedProofSize += finalPolDegree * FIELD_EXTENSION; // Final polynomial values
        pinnedProofSize += 1; // Nonce
    }
    return pinnedProofSize;
}

// Every aux_trace region starts on 256 bytes (32 elements), so every column in it does too: columns start
// at a multiple of the region's row count. Only the reservations whose size is not such a multiple need it.
static uint64_t alignRegion(uint64_t n) { return (n + 31) & ~uint64_t(31); }

StirLayout StarkInfo::stirLayout() const {
    StarkInfo *self = const_cast<StarkInfo *>(this); // getNumNodesMT is non-const
    return makeStirLayout(stirParamsFromStarkInfo(*this), stirBase, [self](uint64_t h) { return self->getNumNodesMT(h); });
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

    // Level 0 must lie below the last levels, and levels >= 2 must fit the build's leaf scratch.
    const uint64_t arity = starkStruct.merkleTreeArity;
    dropLeafLevel = gpu && starkStruct.verificationHashType == "GL" &&
                    NExtended > (uint64_t)std::pow(arity, starkStruct.lastLevelVerification) &&
                    NExtended >= arity * arity * arity;
    uint64_t numNodes = getNumNodesMTCommit(NExtended);

    if(gpu) {
        mapOffsets[std::make_pair("const", true)] = mapTotalN;
        MerkleTreeGL mt(starkStruct.merkleTreeArity, starkStruct.lastLevelVerification, starkStruct.merkleTreeCustom, NExtended, nConstants);
        uint64_t constTreeSize = (NExtended * nConstants) + numNodes;
        mapTotalN += alignRegion(constTreeSize);

        // This air's const tree is rebuilt on device (unpack + extend + merkelize from the
        // resident packed pols). No consttree file is ever read on GPU, so this is false only
        // for airs with no constants at all.
        if (nConstants > 0) {
            calculateFixedExtended = true;
        }
    }

    if (gpu) {
        // Const pols and const tree are always distinct regions. They used to be allowed to
        // overlap for single-use airs -- valid only because the merkelize was then the last
        // reader of the small domain, an ordering a pre-proof rebuild cannot honour.
        mapOffsets[std::make_pair("const", false)] = mapTotalN;
        mapTotalN += N * nConstants;
    }

    if(gpu) {
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
        mapTotalN += HASH_SIZE;

        mapOffsets[std::make_pair("nonce", false)] = mapTotalN;
        mapTotalN += 1;

        mapOffsets[std::make_pair("nonce_blocks", false)] = mapTotalN;
        mapTotalN += GRIND_NONCE_BLOCKS_MAX;

        // Align to 4 Goldilocks elements (32 bytes) for BN128 FrElement alignment
        if (starkStruct.verificationHashType == "BN128") {
            mapTotalN = ((mapTotalN + 3) / 4) * 4;  // Round up to next multiple of 4
        }
        mapOffsets[std::make_pair("input_hash_nonce", false)] = mapTotalN;
        if (starkStruct.verificationHashType == "BN128") {
            mapTotalN += 12;  // 3 BN128 FrElements = 12 Goldilocks elements
        } else {
            mapTotalN += HASH_SIZE;
        }

        mapOffsets[std::make_pair("evals", false)] = mapTotalN;
        mapTotalN += evMap.size() * FIELD_EXTENSION;

        mapOffsets[std::make_pair("challenges", false)] = mapTotalN;
        mapTotalN += challengesMap.size() * FIELD_EXTENSION;

        // FRI constants (deep_expression.cuh): vf2^c per polynomial (at most one per eval), a_o and K_o per opening.
        mapOffsets[std::make_pair("fri_constants", false)] = mapTotalN;
        mapTotalN += (evMap.size() + 2 * openingPoints.size()) * FIELD_EXTENSION;

        mapOffsets[std::make_pair("fri_queries", false)] = mapTotalN;
        mapTotalN += starkStruct.nQueries;

        uint64_t permBits = starkStruct.nBitsExt;
        uint64_t permNFields = (starkStruct.nQueries * permBits + 62) / 63;
        mapOffsets[std::make_pair("fri_queries_perm", false)] = mapTotalN;
        mapTotalN += permNFields;        

        maxTreeWidth = 0;
        for (auto it = mapSectionsN.begin(); it != mapSectionsN.end(); it++) 
        {
            uint64_t treeWidth = it->second;
            if(treeWidth > maxTreeWidth) {
                maxTreeWidth = treeWidth;
            }
        }
        for(uint64_t i = 0; i < numLowDegreeTestTrees(); ++i) {
            uint64_t treeWidth = lowDegreeTestTreeWidth(i);
            if(treeWidth > maxTreeWidth) {
                maxTreeWidth = treeWidth;
            }
        }

        uint64_t nSiblings = merkleProofLevels(starkStruct.nBitsExt, starkStruct.merkleTreeArity, starkStruct.lastLevelVerification, starkStruct.verificationHashType == std::string("BN128"));
        uint64_t nSiblingsPerLevel;
        if (starkStruct.verificationHashType == "BN128") {
            // BN128: all arity siblings per level, each is 4 uint64_t (32 bytes)
            nSiblingsPerLevel = starkStruct.merkleTreeArity * 4;
        } else {
            nSiblingsPerLevel = (starkStruct.merkleTreeArity - 1) * HASH_SIZE;
        }
        maxProofSize = nSiblings * nSiblingsPerLevel;

        maxProofBuffSize = maxTreeWidth + maxProofSize;
        uint64_t nTrees = 1 + (nStages + 1) + customCommits.size();
        uint64_t nTreesFRI = numLowDegreeTestTrees();
    
        uint64_t queriesProofSize = (nTrees + nTreesFRI) * maxProofBuffSize * starkStruct.nQueries;

        mapOffsets[std::make_pair("proof_queries", false)] = mapTotalN;
        mapTotalN += queriesProofSize;

        mapOffsets[std::make_pair("query_rows_scratch", false)] = mapTotalN;
        if (dropLeafLevel) mapTotalN += starkStruct.nQueries * (starkStruct.merkleTreeArity - 1) * (maxTreeWidth + HASH_SIZE);
        
        // TODO: ADD EXPRESSIONS MEM
    }

    assert(nStages == 2);

    uint64_t maxTotalN = 0;

    mapTotalN = alignRegion(mapTotalN);
    mapOffsets[std::make_pair("cm1", true)] = mapTotalN;
    mapTotalN += NExtended * mapSectionsN["cm1"];
    mapOffsets[std::make_pair("mt1", true)] = mapTotalN;
    mapTotalN += alignRegion(numNodes);

    if (inplaceStageCommit) {
        mapOffsets[std::make_pair("cm1", false)] = mapOffsets[std::make_pair("cm1", true)];
        mapOffsets[std::make_pair("cm2", true)] = mapTotalN;
        mapOffsets[std::make_pair("cm2", false)] = mapTotalN;
        mapTotalN += NExtended * mapSectionsN["cm2"];
        mapOffsets[std::make_pair("mt2", true)] = mapTotalN;
        mapTotalN += alignRegion(numNodes);
        // On GPU cm3/mt3 go after q's FRI zone (end of this function) so the expression scratch spans them.
        if (!gpu) {
            mapOffsets[std::make_pair("cm3", true)] = mapTotalN;
            mapTotalN += NExtended * mapSectionsN["cm3"];
            mapOffsets[std::make_pair("mt3", true)] = mapTotalN;
            mapTotalN += alignRegion(numNodes);
        }
    } else {
    mapOffsets[std::make_pair("cm1", false)] = mapTotalN;

    mapOffsets[std::make_pair("cm2", true)] = mapTotalN;
    mapTotalN += NExtended * mapSectionsN["cm2"];
    mapOffsets[std::make_pair("mt2", true)] = mapTotalN;
    mapTotalN += alignRegion(numNodes);
    mapTotalN = std::max(mapOffsets[std::make_pair("cm1", false)] + N * mapSectionsN["cm1"], mapTotalN);

    mapOffsets[std::make_pair("cm2", false)] = mapTotalN;

    mapOffsets[std::make_pair("cm3", true)] = mapTotalN;
    mapTotalN += NExtended * mapSectionsN["cm3"];
    mapOffsets[std::make_pair("mt3", true)] = mapTotalN;
    mapTotalN += alignRegion(numNodes);


    mapTotalN = std::max(mapOffsets[std::make_pair("cm2", false)] + N * mapSectionsN["cm2"], mapTotalN);
    }
    mapOffsets[std::make_pair("f", true)] = mapTotalN;
    mapOffsets[std::make_pair("q", true)] = mapTotalN;
    mapTotalN += NExtended * FIELD_EXTENSION;

    mapOffsets[std::make_pair("constraints", false)] = mapOffsets[std::make_pair("q", true)];

    uint64_t maxSizeHelper = 0;
    if(gpu) {
        maxSizeHelper += boundaries.size() * NExtended;
        mapOffsets[std::make_pair("zi", true)] = mapTotalN;
    }
    
    maxTotalN = std::max(maxTotalN, mapTotalN + maxSizeHelper);
    mapOffsets[std::make_pair("mem_exps", false)] = mapTotalN + maxSizeHelper;   

    uint64_t LEvSize = mapOffsets[std::make_pair("f", true)];
    // Opening 0's Lagrange vector serves every opening point (evmap).
    mapOffsets[std::make_pair("lev", false)] = LEvSize;
    LEvSize += N * FIELD_EXTENSION;
    if(gpu) {
        // Opening 0's shifted point and factor (computeLEv_inplace), then the stripe partials.
        mapOffsets[std::make_pair("lev_helper", false)] = LEvSize;
        LEvSize += 2 * FIELD_EXTENSION;
        mapOffsets[std::make_pair("evals_partials", false)] = LEvSize;
        LEvSize += evMap.size() * EVALS_HELPER_CHUNKS * FIELD_EXTENSION;
    }

    maxTotalN = std::max(maxTotalN, LEvSize);

    // FRI layers follow q directly: zi and the expression tmps are dead once folding starts.

    if (!gpu) {
        for (uint64_t stage = 1; stage <= nStages; ++stage) {
            std::string section = "cm" + to_string(stage);
            uint64_t helper = inplaceStageCommit ? mapOffsets[std::make_pair("cm" + to_string(nStages + 1), true)]
                                                 : mapOffsets[std::make_pair(section, false)] + N * mapSectionsN[section];
            mapOffsets[std::make_pair("buff_helper_fft_" + to_string(stage), false)] = helper;
            maxTotalN = std::max(maxTotalN, helper + NTT_Goldilocks::LDEBlockedScratchSize(NExtended, mapSectionsN[section]));
        }

        uint64_t maxTotalNStageQ = mapOffsets[std::make_pair("q", true)] + NExtended * FIELD_EXTENSION;
        mapOffsets[std::make_pair("buff_helper_fft_" + to_string(nStages + 1), false)] = maxTotalNStageQ;
        maxTotalNStageQ += NExtended * mapSectionsN["cm" + to_string(nStages + 1)];
        maxTotalN = std::max(maxTotalN, maxTotalNStageQ);
    } else {
        uint64_t maxTotalNStageQ = mapOffsets[std::make_pair("q", true)] + NExtended * FIELD_EXTENSION;
        mapOffsets[std::make_pair("extra_helper_fft", false)] = maxTotalNStageQ;
        // The FRI polynomial's 1/(x - xi) planes: zi and the Q helper are dead by then, and the first fold
        // (fri_1, placed right after q) overwrites them only once f is complete.
        mapOffsets[std::make_pair("fri_denominators", true)] = maxTotalNStageQ;
        maxTotalNStageQ += NExtended * FIELD_EXTENSION + qDeg;
        maxTotalN = std::max(maxTotalN, maxTotalNStageQ);
    }
 
    // FRI's per-step folded-oracle buffers (fri_i / mt_fri_i). STIR's prover owns its own oracles,
    // so it needs none of these.
    for(uint64_t step = 0; starkStruct.lowDegreeTest == LowDegreeTestKind::FRI && step < starkStruct.logDomainSizes.size() - 1; ++step) {
        uint64_t height = 1 << starkStruct.logDomainSizes[step + 1];
        uint64_t width = ((1 << starkStruct.logDomainSizes[step]) / height) * FIELD_EXTENSION;
        mapOffsets[std::make_pair("fri_" + to_string(step + 1), true)] = mapTotalN;
        mapTotalN += height * width;
        if(starkStruct.verificationHashType == "GL") {
            uint64_t numNodes = getNumNodesMT(height);
            mapOffsets[std::make_pair("mt_fri_" + to_string(step + 1), true)] = mapTotalN;
            mapTotalN += alignRegion(numNodes);
        }
    }

    if (gpu && starkStruct.lowDegreeTest == LowDegreeTestKind::STIR) {
        if (starkStruct.verificationHashType != "GL") {
            zklog.error("setMapOffsets: STIR supports GL verification hashes only");
            exitProcess();
        }
        if (starkStruct.stir.logDomainSizes[0] != starkStruct.nBitsExt) {
            zklog.error("setMapOffsets: GPU STIR expects |L_0| = 2^nBitsExt");
            exitProcess();
        }
        stirBase = alignRegion(mapTotalN);
        StirLayout l = stirLayout();
        mapTotalN = l.total;
        // Rounds reuse fri_queries (t_0 raw indices) and fri_queries_perm; both are sized for t_0·nBitsExt.
        for (uint64_t i = 0; i < starkStruct.stir.numIterations(); i++) {
            assert(starkStruct.stir.numQueries[i] * starkStruct.stir.logDomainSizes[i] <= starkStruct.nQueries * starkStruct.nBitsExt);
        }
    }

    // cm3/mt3 are first written by the quotient commit, so until then the expression scratch
    // (from mem_exps) runs through them; FRI/STIR only need q and the zone after it.
    if (gpu && inplaceStageCommit) {
        mapTotalN = alignRegion(std::max(mapTotalN, maxTotalN));
        mapOffsets[std::make_pair("cm3", true)] = mapTotalN;
        mapTotalN += NExtended * mapSectionsN["cm3"];
        mapOffsets[std::make_pair("mt3", true)] = mapTotalN;
        mapTotalN += alignRegion(numNodes);
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
            nrowsPack = 256;
            maxNBlocks = 512;

            uint64_t tmpsUsed = nTmp1 + (nTmp3 + 2) * FIELD_EXTENSION;
            // The interpreter's grid is sized as if cm3/mt3 still sat below q: same grid, never more memory.
            uint64_t NExtended = 1ull << starkStruct.nBitsExt;
            uint64_t cm3Above = inplaceStageCommit ? NExtended * mapSectionsN["cm3"] + alignRegion(getNumNodesMTCommit(NExtended)) : 0;
            uint64_t maxTotal = (recursive ? mapTotalN + (1 << 26) : mapTotalN + (1 << 25)) - cm3Above;
            while((mapBuffHelper + tmpsUsed * nrowsPack * maxNBlocks) > maxTotal) {
                if (nrowsPack > 128) {
                    nrowsPack /= 2;
                } else {
                    maxNBlocks /= 2;
                }
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

    // Everything past mem_exps is dead while the quotient runs: the generated Q kernel may use all of it.
    if(gpu && !verify) {
        mapOffsets[std::make_pair("mem_exps_end", false)] = std::max(mapBuffHelper, mapTotalN);
    }

    if(mapBuffHelper > mapTotalN) {
        mapTotalN = mapBuffHelper;
    }
}

// In place the small cm1 owns the section's lower half, so the raw trace lands in the upper one.
uint64_t StarkInfo::getTraceLandingOffset() {
    uint64_t base = mapOffsets[std::make_pair("cm1", true)];
    return inplaceStageCommit ? base + (1ull << starkStruct.nBits) * mapSectionsN["cm1"] : base;
}

uint64_t StarkInfo::getNumNodesMTCommit(uint64_t height) {
    uint64_t full = getNumNodesMT(height);
    if (!dropLeafLevel) return full;
    uint64_t a = starkStruct.merkleTreeArity;
    return full - (height + (a - height % a) % a) * HASH_SIZE;
}

uint64_t StarkInfo::getLastLevelOffset(uint64_t height, uint64_t &nNodes) {
    uint64_t a = starkStruct.merkleTreeArity, offset = 0;
    nNodes = dropLeafLevel ? (height + a - 1) / a : height;
    while (nNodes > std::pow(a, starkStruct.lastLevelVerification)) {
        nNodes = (nNodes + a - 1) / a;
        offset += nNodes * a * HASH_SIZE;
    }
    return offset;
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
