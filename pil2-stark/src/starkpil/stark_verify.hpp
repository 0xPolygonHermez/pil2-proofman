#include "expressions_ctx.hpp"
#include "stir/stir.hpp"
#include "stark_info.hpp"
#include "merkleTreeGL.hpp"
#include "merkleTreeBN128.hpp"

template <typename ElementType>
ElementType fromString(const std::string& element);

template<>
inline Goldilocks::Element fromString(const std::string& element) {
    return Goldilocks::fromString(element);
}

template<>
inline RawFr::Element fromString(const std::string& element) {
    RawFr::Element r;
    RawFr::field.fromString(r, element, 10);
    return r;
}

// Read the STIR section of a proof out of its JSON form (the keys pointer2json / proof2json
// write: "s{i+1}_root/_vals/_siblings/_last_levels", "betas", "finalPol", "nonces").
template <typename ElementType>
StirProof<ElementType> stirProofFromJson(json &jproof, const StirParams &params, uint64_t nFieldElements)
{
    StirProof<ElementType> proof = STIR<ElementType>::makeProof(params);
    uint64_t M = params.M();
    bool bn128 = !std::is_same<ElementType, Goldilocks::Element>::value;
    uint64_t nSiblingsPerLevel = bn128 ? params.merkleTreeArity : (params.merkleTreeArity - 1) * nFieldElements;
    uint64_t numNodesLevel = params.lastLevelVerification == 0 ? 0 : std::pow(params.merkleTreeArity, params.lastLevelVerification);

    for (uint64_t i = 0; i < M; ++i)
    {
        std::string prefix = "s" + std::to_string(i + 1);
        if (nFieldElements == 1) {
            proof.trees[i].root[0] = fromString<ElementType>(jproof[prefix + "_root"]);
        } else {
            for (uint64_t l = 0; l < nFieldElements; l++) {
                proof.trees[i].root[l] = fromString<ElementType>(jproof[prefix + "_root"][l]);
            }
        }
        for (uint64_t l = 0; l < numNodesLevel; ++l) {
            for (uint64_t c = 0; c < nFieldElements; ++c) {
                proof.trees[i].last_levels[l * nFieldElements + c] = nFieldElements == 1
                    ? fromString<ElementType>(jproof[prefix + "_last_levels"][l])
                    : fromString<ElementType>(jproof[prefix + "_last_levels"][l][c]);
            }
        }

        uint64_t k = uint64_t(1) << params.logFoldingFactors[i];
        uint64_t logLeaves = params.logDomainSizes[i] - params.logFoldingFactors[i];
        uint64_t nSiblings = merkleProofLevels(logLeaves, params.merkleTreeArity, params.lastLevelVerification, bn128);

        // MerkleProof is built from a flat buffer: the leaf values (Goldilocks), then the siblings.
        for (uint64_t q = 0; q < params.numQueries[i]; q++)
        {
            std::vector<Goldilocks::Element> buff(k * FIELD_EXTENSION + (nSiblings * nSiblingsPerLevel * sizeof(ElementType)) / sizeof(Goldilocks::Element));
            for (uint64_t l = 0; l < k * FIELD_EXTENSION; l++) {
                buff[l] = Goldilocks::fromString(jproof[prefix + "_vals"][q][l]);
            }
            ElementType *mp = (ElementType *)&buff[k * FIELD_EXTENSION];
            for (uint64_t l = 0; l < nSiblings; ++l) {
                for (uint64_t c = 0; c < nSiblingsPerLevel; ++c) {
                    mp[l * nSiblingsPerLevel + c] = fromString<ElementType>(jproof[prefix + "_siblings"][q][l][c]);
                }
            }
            proof.trees[i].polQueries[q] = std::vector<MerkleProof<ElementType>>{MerkleProof<ElementType>(k * FIELD_EXTENSION, nSiblings, nSiblingsPerLevel, buff.data())};
        }
    }

    for (uint64_t i = 0; i + 1 < M; ++i) {
        for (uint64_t l = 0; l < proof.betas[i].size(); l++) {
            proof.betas[i][l] = Goldilocks::fromString(jproof["betas"][i][l]);
        }
    }
    for (uint64_t l = 0; l < proof.finalPol.size(); l++) {
        proof.finalPol[l] = Goldilocks::fromString(jproof["finalPol"][l]);
    }
    for (uint64_t i = 0; i < M; ++i) {
        proof.nonces[i] = std::stoull(jproof["nonces"][i].get<std::string>());
    }
    if (jproof.contains("ansCoeffs")) {
        for (uint64_t i = 0; i + 1 < M; ++i) {
            for (uint64_t l = 0; l < proof.ansCoeffs[i].size(); l++) {
                proof.ansCoeffs[i][l] = Goldilocks::fromString(jproof["ansCoeffs"][i][l]);
            }
        }
    }
    return proof;
}

template <typename ElementType>
bool starkVerify(json jproof, StarkInfo& starkInfo, ExpressionsBin& expressionsBin, string verkeyFile, Goldilocks::Element *publics, Goldilocks::Element *proofValues, bool challengesVadcop, Goldilocks::Element* globalChallenge) {

    json verkeyJson;
    file2json(verkeyFile, verkeyJson);

    using TranscriptType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, TranscriptGL, TranscriptBN128>;

    using MerkleTreeType = std::conditional_t<std::is_same<ElementType, Goldilocks::Element>::value, MerkleTreeGL, MerkleTreeBN128>;

    uint64_t nFieldElements = starkInfo.starkStruct.verificationHashType == std::string("BN128") ? 1 : HASH_SIZE;

    ElementType verkey[nFieldElements];
    if(starkInfo.starkStruct.verificationHashType == "GL") {
        for(uint64_t i = 0; i < nFieldElements; i++) {
            verkey[i] = fromString<ElementType>(verkeyJson[i].dump());
        }
    } else {
        verkey[0] = fromString<ElementType>(verkeyJson);
    }

    uint64_t queriesL0[starkInfo.starkStruct.nQueries];

    Goldilocks::Element evals[starkInfo.evMap.size()  * FIELD_EXTENSION];
    for(uint64_t i = 0; i < starkInfo.evMap.size(); ++i) {
        for(uint64_t j = 0; j < FIELD_EXTENSION; ++j) {
            evals[i*FIELD_EXTENSION + j] = Goldilocks::fromString(jproof["evals"][i][j]);
        }
    }

    Goldilocks::Element airgroupValues[starkInfo.airgroupValuesSize];
    for(uint64_t i = 0; i < starkInfo.airgroupValuesMap.size() ; ++i) {
        for(uint64_t j = 0; j < FIELD_EXTENSION; ++j) {
            airgroupValues[i*FIELD_EXTENSION + j] = Goldilocks::fromString(jproof["airgroupvalues"][i][j]);
        }
    }

    Goldilocks::Element airValues[starkInfo.airValuesSize];
    uint64_t a = 0;
    for(uint64_t i = 0; i < starkInfo.airValuesMap.size(); ++i) {
        if(starkInfo.airValuesMap[i].stage == 1) {
            airValues[a++] = Goldilocks::fromString(jproof["airvalues"][i][0]);
        } else {
            airValues[a++] = Goldilocks::fromString(jproof["airvalues"][i][0]);
            airValues[a++] = Goldilocks::fromString(jproof["airvalues"][i][1]);
            airValues[a++] = Goldilocks::fromString(jproof["airvalues"][i][2]);
        }
    }

    // Stage challenges, then for FRI one r^fold per fold (M = domains − 1) and the query seed.
    Goldilocks::Element challenges[(starkInfo.challengesMap.size() + starkInfo.starkStruct.logDomainSizes.size()) * FIELD_EXTENSION];

    TranscriptType transcript(starkInfo.starkStruct.transcriptArity, starkInfo.starkStruct.merkleTreeCustom);
    if(!challengesVadcop) {
        transcript.put(&verkey[0], nFieldElements);
        if(starkInfo.nPublics > 0) {
            if(!starkInfo.starkStruct.hashCommits) {
                transcript.put(&publics[0], starkInfo.nPublics);
            } else {
                ElementType hash[nFieldElements];
                TranscriptType transcriptHash(starkInfo.starkStruct.transcriptArity, starkInfo.starkStruct.merkleTreeCustom);
                transcriptHash.put(&publics[0], starkInfo.nPublics);
                transcriptHash.getState(hash);
                transcript.put(hash, nFieldElements);
            }
        }

        ElementType root[nFieldElements];
        if(nFieldElements == 1) {
            root[0] = fromString<ElementType>(jproof["root1"]);
        } else {
            for(uint64_t i = 0; i < nFieldElements; ++i) {
                root[i] = fromString<ElementType>(jproof["root1"][i]);
            }
        }
        transcript.put(&root[0], nFieldElements);
    } else {
        transcript.put(globalChallenge, FIELD_EXTENSION);
    }

    uint64_t c = 0;
    for(uint64_t s = 2; s <= starkInfo.nStages + 1; ++s) {
        uint64_t nChallenges = std::count_if(starkInfo.challengesMap.begin(), starkInfo.challengesMap.end(),[s](const PolMap& c) { return c.stage == s; });
        for(uint64_t i = 0; i < nChallenges; ++i) {
            transcript.getField((uint64_t *)&challenges[c*FIELD_EXTENSION]);
            c++;
        }
        ElementType root[nFieldElements];
        if(nFieldElements == 1) {
            root[0] = fromString<ElementType>(jproof["root" + to_string(s)]);
        } else {
            for(uint64_t i = 0; i < nFieldElements; ++i) {
                root[i] = fromString<ElementType>(jproof["root" + to_string(s)][i]);
            }
        }

        transcript.put(&root[0], nFieldElements);

        uint64_t p = 0;
        for(uint64_t i = 0; i < starkInfo.airValuesMap.size(); i++) {
            if(starkInfo.airValuesMap[i].stage == 1) {
                p++;
            } else {
                if(starkInfo.airValuesMap[i].stage == s) {
                    transcript.put(&airValues[p], FIELD_EXTENSION);
                }
                p += 3;
            }
        }

        // TODO: ADD PROOF VALUES ??
    }

    // Evals challenge
    transcript.getField((uint64_t *)&challenges[c*FIELD_EXTENSION]);
    c++;

    if(!starkInfo.starkStruct.hashCommits) {
        transcript.put(&evals[0], starkInfo.evMap.size()  * FIELD_EXTENSION);
    } else {
        ElementType hash[nFieldElements];
        TranscriptType transcriptHash(starkInfo.starkStruct.transcriptArity, starkInfo.starkStruct.merkleTreeCustom);
        transcriptHash.put(&evals[0], starkInfo.evMap.size()  * FIELD_EXTENSION);
        transcriptHash.getState(hash);
        transcript.put(hash, nFieldElements);
    }

    // FRI challenges
    transcript.getField((uint64_t *)&challenges[c*FIELD_EXTENSION]);
    c++;
    transcript.getField((uint64_t *)&challenges[c*FIELD_EXTENSION]);
    c++;

    // The recorded f_0 claims of a STIR proof: T_0's leaf value at each round-1 query, compared
    // below against the recomputed DEEP polynomial at the same rows (the STIR counterpart of
    // FRI's s1_vals consistency check).
    std::vector<Goldilocks::Element> stirF0Claims;

    if (starkInfo.starkStruct.lowDegreeTest == LowDegreeTestKind::STIR) {
    if constexpr (std::is_same<ElementType, Goldilocks::Element>::value) {
        // Everything from here down to the query indices is STIR's own round structure: hand the
        // transcript to STIR::verify, which replays it message by message (roots, out-of-domain
        // samples, β's, per-round grinding) and re-derives every query index. The f_0 claims are
        // recorded rather than checked here, and the indices come back in `queriesL0`, so the
        // shared machinery below (trace loading, evals check, DEEP recomputation, stage Merkle
        // trees) runs unchanged.
        StirParams stirParams = stirParamsFromStarkInfo(starkInfo);
        StirProof<ElementType> stirProof = stirProofFromJson<ElementType>(jproof, stirParams, nFieldElements);

        stirF0Claims.resize(starkInfo.starkStruct.nQueries * FIELD_EXTENSION);
        auto checkF0 = [&](uint64_t q, uint64_t idxL0, const stir::E3 &committed) {
            queriesL0[q] = idxL0;
            std::memcpy(&stirF0Claims[q * FIELD_EXTENSION], &committed[0], FIELD_EXTENSION * sizeof(Goldilocks::Element));
            return true;
        };
        auto checkGrinding = [](const uint64_t *c, uint64_t nonce, uint32_t grindingBits) {
            if (grindingBits == 0) return true;
            constexpr uint32_t W = 8;
            Goldilocks::Element result[W];
            Goldilocks::Element x[W] = {};
            x[0].fe = c[0]; x[1].fe = c[1]; x[2].fe = c[2];
            x[3] = Goldilocks::fromU64(nonce);
            runGrindingPermute(result, x);
            return Goldilocks::toU64(result[0]) < (1ULL << (64 - grindingBits));
        };

        std::string failure;
        if (!STIR<ElementType>::verify(stirProof, stirParams, transcript, checkF0, checkGrinding, &failure)) {
            zklog.error("STIR verification failed: " + failure);
            return false;
        }
    } else {
        zklog.error("starkVerify: STIR is not supported for BN128 proofs");
        return false;
    }
    } else {
    // r^fold_{step−1} is drawn after the root of f_{step−1}; nothing precedes f_0's root.
    for (uint64_t step=0; step<starkInfo.starkStruct.logDomainSizes.size(); step++) {
        if (step > 0) {
            transcript.getField((uint64_t *)&challenges[c*FIELD_EXTENSION]);
            c++;
        }
        if (step < starkInfo.starkStruct.logDomainSizes.size() - 1) {
            ElementType root[nFieldElements];
            if(nFieldElements == 1) {
                root[0] = fromString<ElementType>(jproof["s" + std::to_string(step + 1) + "_root"]);
            } else {
                for(uint64_t i = 0; i < nFieldElements; ++i) {
                    root[i] = fromString<ElementType>(jproof["s" + std::to_string(step + 1) + "_root"][i]);
                }
            }
            
            transcript.put(&root[0], nFieldElements);
        } else {
            uint64_t finalPolSize = (1<< starkInfo.starkStruct.logDomainSizes[step]);
            Goldilocks::Element finalPol[finalPolSize * FIELD_EXTENSION];
            for(uint64_t i = 0; i < finalPolSize; ++i) {
                for(uint64_t j = 0; j < FIELD_EXTENSION; ++j) {
                    finalPol[i*FIELD_EXTENSION + j] = Goldilocks::fromString(jproof["finalPol"][i][j]);
                }
            }

            if(!starkInfo.starkStruct.hashCommits) {
                transcript.put(&finalPol[0],finalPolSize*FIELD_EXTENSION);
            } else {
                ElementType hash[nFieldElements];
                TranscriptType transcriptHash(starkInfo.starkStruct.transcriptArity, starkInfo.starkStruct.merkleTreeCustom);
                transcriptHash.put(&finalPol[0], finalPolSize*FIELD_EXTENSION);
                transcriptHash.getState(hash);
                transcript.put(hash, nFieldElements);
            }
        }
    }
    transcript.getField((uint64_t *)&challenges[c*FIELD_EXTENSION]);
    c++;
    assert(c == (starkInfo.challengesMap.size() + starkInfo.starkStruct.logDomainSizes.size()));

    Goldilocks::Element *challenge = &challenges[(starkInfo.challengesMap.size() + starkInfo.starkStruct.logDomainSizes.size() - 1) * FIELD_EXTENSION];

    Goldilocks::Element nonce = Goldilocks::fromString(jproof["nonce"]);
    if constexpr (std::is_same<ElementType, Goldilocks::Element>::value) {
        // STARK grinding contract: x[0..2] = FIELD_EXTENSION challenge,
        // x[3] = nonce, x[4..W-1] = 0.
        constexpr uint32_t W = 8;
        static_assert(PoseidonGoldilocksGrinding::SPONGE_WIDTH == W && Poseidon2GoldilocksGrinding::SPONGE_WIDTH == W,
                      "stark_verify: grinding SPONGE_WIDTH must be 8 for both families");
        Goldilocks::Element result[W];
        Goldilocks::Element x[W] = {};
        x[0] = challenge[0]; x[1] = challenge[1]; x[2] = challenge[2];
        x[3] = nonce;
        runGrindingPermute(result, x);
        if (Goldilocks::toU64(result[0]) >= (1ULL << (64 - starkInfo.starkStruct.grindingBitsQueries))) {
            zklog.error("starkVerify: PoW verification failed");
            return false;
        }
    } else {
        // TODO
    }

    TranscriptType transcriptPermutation(starkInfo.starkStruct.transcriptArity, starkInfo.starkStruct.merkleTreeCustom);
    transcriptPermutation.put(challenge, FIELD_EXTENSION);    
    transcriptPermutation.put(&nonce, 1);
    transcriptPermutation.getPermutations(queriesL0, starkInfo.starkStruct.nQueries, starkInfo.starkStruct.logDomainSizes[0]);

    }

    Goldilocks::Element constPolsVals[starkInfo.nConstants * starkInfo.starkStruct.nQueries];
#pragma omp parallel for
    for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
        for(uint64_t i = 0; i < starkInfo.nConstants; ++i) {
            constPolsVals[q*starkInfo.nConstants + i] = Goldilocks::fromString(jproof["s0_valsC"][q][i]);
        }
    }
    
    Goldilocks::Element xiChallenge[FIELD_EXTENSION];

    for (uint64_t i = 0; i < starkInfo.challengesMap.size(); i++)
    {
        if(starkInfo.challengesMap[i].stage == starkInfo.nStages + 2) {
            if(starkInfo.challengesMap[i].stageId == 0) {
                std::memcpy(&xiChallenge[0], &challenges[i*FIELD_EXTENSION], FIELD_EXTENSION * sizeof(Goldilocks::Element));
            }
        }
    }

    ProverHelpers proverHelpers(starkInfo, xiChallenge);

    SetupCtx setupCtx(starkInfo, expressionsBin);

    Goldilocks::Element *xDivXSub = new Goldilocks::Element[starkInfo.openingPoints.size() * FIELD_EXTENSION * starkInfo.starkStruct.nQueries];
    for(uint64_t i = 0; i < starkInfo.starkStruct.nQueries; ++i) {
        uint64_t query = queriesL0[i];
        Goldilocks::Element x = Goldilocks::shift() * Goldilocks::exp(Goldilocks::w(starkInfo.starkStruct.nBitsExt), query);
        for(uint64_t o = 0; o < starkInfo.openingPoints.size(); ++o) {
            Goldilocks::Element w = Goldilocks::one();

            for(uint64_t j = 0; j < uint64_t(std::abs(starkInfo.openingPoints[o])); ++j) {
                w = w * Goldilocks::w(starkInfo.starkStruct.nBits);
            }
            if(starkInfo.openingPoints[o] < 0) {
                w = Goldilocks::inv(w);
            }
            
            Goldilocks::Element x_ext[FIELD_EXTENSION] = { x, Goldilocks::zero(), Goldilocks::zero() };
            Goldilocks::Element aux[FIELD_EXTENSION];
            Goldilocks3::mul((Goldilocks3::Element &)aux[0], (Goldilocks3::Element &)xiChallenge[0], w);
            Goldilocks3::sub((Goldilocks3::Element &)aux[0], (Goldilocks3::Element &)x_ext[0], (Goldilocks3::Element &)aux[0]);
            Goldilocks3::inv((Goldilocks3::Element *)aux, (Goldilocks3::Element *)aux);
            std::memcpy(&xDivXSub[(i*starkInfo.openingPoints.size() + o)*FIELD_EXTENSION], &aux[0], FIELD_EXTENSION * sizeof(Goldilocks::Element));
        }
    }

    Goldilocks::Element *trace = new Goldilocks::Element[starkInfo.mapSectionsN["cm1"]*starkInfo.starkStruct.nQueries];
    Goldilocks::Element *aux_trace = new Goldilocks::Element[starkInfo.mapTotalN];
    Goldilocks::Element *trace_custom_commits_fixed = new Goldilocks::Element[starkInfo.mapTotalNCustomCommitsFixed];
#pragma omp parallel for
    for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
        for(uint64_t i = 0; i < starkInfo.cmPolsMap.size(); ++i) {
            uint64_t stage = starkInfo.cmPolsMap[i].stage;
            uint64_t stagePos = starkInfo.cmPolsMap[i].stagePos;
            uint64_t offset = starkInfo.mapOffsets[std::make_pair("cm" + to_string(stage), false)];
            uint64_t nPols = starkInfo.mapSectionsN["cm" + to_string(stage)];
            Goldilocks::Element *pols = stage == 1 ? trace : aux_trace;
            if(starkInfo.cmPolsMap[i].dim == 1) {
                pols[offset + q*nPols + stagePos] = Goldilocks::fromString(jproof["s0_vals" + to_string(stage)][q][stagePos]);
            } else {
                pols[offset + q*nPols + stagePos] = Goldilocks::fromString(jproof["s0_vals" + to_string(stage)][q][stagePos]);
                pols[offset + q*nPols + stagePos + 1] = Goldilocks::fromString(jproof["s0_vals" + to_string(stage)][q][stagePos + 1]);
                pols[offset + q*nPols + stagePos + 2] = Goldilocks::fromString(jproof["s0_vals" + to_string(stage)][q][stagePos + 2]);
            }
        }
    }
    
#pragma omp parallel for
    for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
        for(uint64_t c = 0; c < starkInfo.customCommits.size(); ++c) {
            for(uint64_t i = 0; i < starkInfo.customCommitsMap[c].size(); ++i) {
                uint64_t stagePos = starkInfo.customCommitsMap[c][i].stagePos;
                uint64_t offset = starkInfo.mapOffsets[std::make_pair(starkInfo.customCommits[c].name + "0", false)];
                uint64_t nPols = starkInfo.mapSectionsN[starkInfo.customCommits[c].name + "0"];
                trace_custom_commits_fixed[offset + q*nPols + stagePos] = Goldilocks::fromString(jproof["s0_vals_" + starkInfo.customCommits[c].name + "_0"][q][stagePos]);
            }
        }   
    }

    StepsParams params = {
        .trace = trace,
        .aux_trace = aux_trace,
        .publicInputs = publics,
        .proofValues = proofValues,
        .challenges = challenges,
        .airgroupValues = airgroupValues,
        .airValues = airValues,
        .evals = evals,
        .xDivXSub = xDivXSub,
        .pConstPolsAddress = constPolsVals,
        .pConstPolsExtendedTreeAddress = nullptr,
        .pCustomCommitsFixed = trace_custom_commits_fixed,
    };

    bool isValid = true;

    zklog.trace("Verifying evaluations");
    ExpressionsPack expressionsPack(setupCtx, &proverHelpers, 1);
    
    Goldilocks::Element buff[FIELD_EXTENSION];
    Dest dest(buff, 1, 0);
    dest.addParams(starkInfo.cExpId, setupCtx.expressionsBin.expressionsInfo[starkInfo.cExpId].destDim);
    
    expressionsPack.calculateExpressions(params, dest, 1, false, false);

    Goldilocks::Element xN[3] = {Goldilocks::one(), Goldilocks::zero(), Goldilocks::zero()};
    for(uint64_t i = 0; i < uint64_t(1 << starkInfo.starkStruct.nBits); ++i) {
        Goldilocks3::mul((Goldilocks3::Element *)xN, (Goldilocks3::Element *)xN, (Goldilocks3::Element *)xiChallenge);
    }

    Goldilocks::Element xAcc[3] = { Goldilocks::one(), Goldilocks::zero(), Goldilocks::zero() };
    Goldilocks::Element q[3] = { Goldilocks::zero(), Goldilocks::zero(), Goldilocks::zero() };
    uint64_t qStage = starkInfo.nStages + 1;
    uint64_t qIndex = std::find_if(starkInfo.cmPolsMap.begin(), starkInfo.cmPolsMap.end(), [qStage](const PolMap& p) {
        return p.stage == qStage && p.stageId == 0;
    }) - starkInfo.cmPolsMap.begin();

    for(uint64_t i = 0; i < starkInfo.qDeg; ++i) {
        uint64_t index = qIndex + i;
        uint64_t evId = std::find_if(starkInfo.evMap.begin(), starkInfo.evMap.end(), [index](const EvMap& e) {
           return e.type == EvMap::eType::cm && e.id == index;
        }) - starkInfo.evMap.begin();
        Goldilocks::Element aux[3];
        Goldilocks3::mul((Goldilocks3::Element &)aux[0], (Goldilocks3::Element &)xAcc[0], (Goldilocks3::Element &)evals[evId * FIELD_EXTENSION]);
        Goldilocks3::add((Goldilocks3::Element &)q, (Goldilocks3::Element &)q, (Goldilocks3::Element &)aux[0]);
        Goldilocks3::mul((Goldilocks3::Element &)xAcc[0], (Goldilocks3::Element &)xAcc[0], (Goldilocks3::Element &)xN);
    }

    Goldilocks::Element res[3] = { q[0] - buff[0], q[1] - buff[1], q[2] - buff[2]};
    if(!Goldilocks::isZero(res[0]) || !Goldilocks::isZero(res[1]) || !Goldilocks::isZero(res[2])) {
        zklog.error("Invalid evaluations");
        isValid = false;
    }

    zklog.trace("Verifying FRI queries consistency");
    Goldilocks::Element buffQueries[FIELD_EXTENSION*starkInfo.starkStruct.nQueries];
    Dest destQueries(buffQueries, starkInfo.starkStruct.nQueries, 0);
    destQueries.addParams(starkInfo.deepExpId, setupCtx.expressionsBin.expressionsInfo[starkInfo.deepExpId].destDim);
    expressionsPack.calculateExpressions(params, destQueries, starkInfo.starkStruct.nQueries, false, false);
    bool isValidFRIConsistency = true;
#pragma omp parallel for
    for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
        if(starkInfo.starkStruct.lowDegreeTest == LowDegreeTestKind::STIR) {
            // The recomputed DEEP polynomial must equal what T_0's opened leaf claimed for f_0 at
            // this row (recorded by checkF0 during STIR::verify).
            if(!Goldilocks::isZero(stirF0Claims[q * FIELD_EXTENSION] - buffQueries[q*FIELD_EXTENSION])
                || !Goldilocks::isZero(stirF0Claims[q * FIELD_EXTENSION + 1] - buffQueries[q*FIELD_EXTENSION + 1])
                || !Goldilocks::isZero(stirF0Claims[q * FIELD_EXTENSION + 2] - buffQueries[q*FIELD_EXTENSION + 2])) {
                isValidFRIConsistency = false;
            }
            continue;
        }
        uint64_t idx = queriesL0[q] % (1 << starkInfo.starkStruct.logDomainSizes[0]);
        if(starkInfo.starkStruct.logDomainSizes.size() > 1) {
            uint64_t nextNGroups = 1 << starkInfo.starkStruct.logDomainSizes[1];
            uint64_t groupIdx = idx / nextNGroups;
            if(!Goldilocks::isZero(Goldilocks::fromString(jproof["s1_vals"][q][groupIdx * FIELD_EXTENSION]) - buffQueries[q*FIELD_EXTENSION]) 
                || !Goldilocks::isZero(Goldilocks::fromString(jproof["s1_vals"][q][groupIdx * FIELD_EXTENSION + 1]) - buffQueries[q*FIELD_EXTENSION + 1]) 
                || !Goldilocks::isZero(Goldilocks::fromString(jproof["s1_vals"][q][groupIdx * FIELD_EXTENSION + 2]) - buffQueries[q*FIELD_EXTENSION + 2])) {
                isValidFRIConsistency = false;
            }
        } else {
            if(!Goldilocks::isZero(Goldilocks::fromString(jproof["finalPol"][idx][0]) - buffQueries[q*FIELD_EXTENSION]) 
                || !Goldilocks::isZero(Goldilocks::fromString(jproof["finalPol"][idx][1]) - buffQueries[q*FIELD_EXTENSION + 1]) 
                || !Goldilocks::isZero(Goldilocks::fromString(jproof["finalPol"][idx][2]) - buffQueries[q*FIELD_EXTENSION + 2])) {
                isValidFRIConsistency = false;
            }
        }
    }
    if(!isValidFRIConsistency) {
        isValid = false;
        zklog.error("Verify FRI query consistency failed");
    }

    uint64_t numNodesLevel = starkInfo.starkStruct.lastLevelVerification == 0 ? 0 : std::pow(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification);
    for(uint64_t s = 0; s < starkInfo.nStages + 1; ++s) {
        zklog.trace("Verifying stage " +  to_string(s + 1) + " Merkle tree");
        std::string section = "cm" + to_string(s + 1);
        uint64_t nCols = starkInfo.mapSectionsN[section];
        MerkleTreeType tree(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification, starkInfo.starkStruct.merkleTreeCustom, 1 << starkInfo.starkStruct.nBitsExt, nCols);
        ElementType root[nFieldElements];
        ElementType level[nFieldElements * numNodesLevel];
        if(nFieldElements == 1) {
            root[0] = fromString<ElementType>(jproof["root" + to_string(s + 1)]);
            for(uint64_t i = 0; i < numNodesLevel; ++i) {
                level[i] = fromString<ElementType>(jproof["s0_last_levels" + to_string(s + 1)][i]);
            }
        } else {
            for(uint64_t j = 0; j < nFieldElements; ++j) {
                root[j] = fromString<ElementType>(jproof["root" + to_string(s + 1)][j]);
            }
            for(uint64_t i = 0; i < numNodesLevel; ++i) {
                for (uint64_t j = 0; j < nFieldElements; ++j) {
                    level[i * nFieldElements + j] = fromString<ElementType>(jproof["s0_last_levels" + to_string(s + 1)][i][j]);
                }
            }
        }
        
        if (starkInfo.starkStruct.lastLevelVerification > 0) {   
            bool isValidRoot = MerkleTreeType::verifyMerkleRoot(root, level, 1 << starkInfo.starkStruct.nBitsExt, starkInfo.starkStruct.lastLevelVerification, starkInfo.starkStruct.merkleTreeArity, nFieldElements);

            if (!isValidRoot) {
                zklog.error("Stage " + to_string(s + 1) + " Merkle Tree root verification failed");
                isValid = false;
            }
        }

        bool isValidStageMT = true;
    #pragma omp parallel for
        for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
            std::vector<Goldilocks::Element> values(nCols);
            for (uint64_t i = 0; i < nCols; ++i) {
                values[i] = Goldilocks::fromString(jproof["s0_vals" + to_string(s + 1)][q][i]);
            }

            uint64_t nSiblings = merkleProofLevels(starkInfo.starkStruct.nBitsExt, starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification, starkInfo.starkStruct.verificationHashType == std::string("BN128"));
            uint64_t nSiblingsPerLevel = starkInfo.starkStruct.verificationHashType == std::string("BN128") ? starkInfo.starkStruct.merkleTreeArity : (starkInfo.starkStruct.merkleTreeArity - 1) * nFieldElements;

            std::vector<std::vector<ElementType>> siblings(
                nSiblings, 
                std::vector<ElementType>(nSiblingsPerLevel)
            );

            for (uint64_t i = 0; i < nSiblings; ++i) {
                for (uint64_t j = 0; j < nSiblingsPerLevel; ++j) {
                    siblings[i][j] = fromString<ElementType>(jproof["s0_siblings" + to_string(s + 1)][q][i][j]);
                }
            }

            bool res = tree.verifyGroupProof(root, level, siblings, queriesL0[q], values);
            if(!res) {
                isValidStageMT = false;
            }
        }
        if(!isValidStageMT) {
            zklog.error("Stage " + to_string(s + 1) + " Merkle Tree verification failed");
            isValid = false;
        }
    }

    zklog.trace("Verifying constant Merkle tree");
    MerkleTreeType treeC(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification, starkInfo.starkStruct.merkleTreeCustom, 1 << starkInfo.starkStruct.nBitsExt, starkInfo.nConstants);

    ElementType levelC[nFieldElements * numNodesLevel];
    if(nFieldElements == 1) {
        for(uint64_t i = 0; i < numNodesLevel; ++i) {
            levelC[i] = fromString<ElementType>(jproof["s0_last_levelsC"][i]);
        }
    } else {
        for(uint64_t i = 0; i < numNodesLevel; ++i) {
            for (uint64_t j = 0; j < nFieldElements; ++j) {
                levelC[i * nFieldElements + j] = fromString<ElementType>(jproof["s0_last_levelsC"][i][j]);
            }
        }
    }

    if (starkInfo.starkStruct.lastLevelVerification > 0) {   
        bool isValidRootC = MerkleTreeType::verifyMerkleRoot(verkey, levelC, 1 << starkInfo.starkStruct.nBitsExt, starkInfo.starkStruct.lastLevelVerification, starkInfo.starkStruct.merkleTreeArity, nFieldElements);

        if (!isValidRootC) {
            zklog.error("Constant Merkle Tree root verification failed");
            isValid = false;
        }
    }

    bool isValidConstantMT = true;
#pragma omp parallel for
    for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
        std::vector<Goldilocks::Element> values(starkInfo.nConstants);
        for (uint64_t i = 0; i < starkInfo.nConstants; ++i) {
            values[i] = Goldilocks::fromString(jproof["s0_valsC"][q][i]);
        }

        uint64_t nSiblings = merkleProofLevels(starkInfo.starkStruct.nBitsExt, starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification, starkInfo.starkStruct.verificationHashType == std::string("BN128"));
        uint64_t nSiblingsPerLevel = starkInfo.starkStruct.verificationHashType == std::string("BN128") ? starkInfo.starkStruct.merkleTreeArity : (starkInfo.starkStruct.merkleTreeArity - 1) * nFieldElements;

        std::vector<std::vector<ElementType>> siblings(
            nSiblings, 
            std::vector<ElementType>(nSiblingsPerLevel)
        );

        for (uint64_t i = 0; i < nSiblings; ++i) {
            for (uint64_t j = 0; j < nSiblingsPerLevel; ++j) {
                siblings[i][j] = fromString<ElementType>(jproof["s0_siblingsC"][q][i][j]);
            }
        }

        bool res = treeC.verifyGroupProof(verkey, levelC, siblings, queriesL0[q], values);
        if(!res) {
            isValidConstantMT = false;
        }
    }
    if(!isValidConstantMT) {
        zklog.error("Constant Merkle Tree verification failed");
        isValid = false;
    }

    for(uint64_t c = 0; c < starkInfo.customCommits.size(); ++c) {
        std::string section = starkInfo.customCommits[c].name + "0";
        zklog.trace("Verifying custom commit " + section + " Merkle tree root");
        uint64_t nCols = starkInfo.mapSectionsN[section];
        MerkleTreeType tree(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification, starkInfo.starkStruct.merkleTreeCustom, 1 << starkInfo.starkStruct.nBitsExt, nCols);
        ElementType root[nFieldElements];
        ElementType level[nFieldElements * numNodesLevel];
        for(uint64_t j = 0; j < nFieldElements; ++j) {
            root[j] = fromString<ElementType>(Goldilocks::toString(publics[starkInfo.customCommits[c].publicValues[j]]));
        }
        if(nFieldElements == 1) {
            for(uint64_t i = 0; i < numNodesLevel; ++i) {
                level[i] = fromString<ElementType>(jproof["s0_last_levels_" + starkInfo.customCommits[c].name + "_0"][i]);
            }
        } else {
            for(uint64_t i = 0; i < numNodesLevel; ++i) {
                for (uint64_t j = 0; j < nFieldElements; ++j) {
                    level[i * nFieldElements + j] = fromString<ElementType>(jproof["s0_last_levels_" + starkInfo.customCommits[c].name + "_0"][i][j]);
                }
            }
        }

        if (starkInfo.starkStruct.lastLevelVerification > 0) {   
            bool isValidRoot = MerkleTreeType::verifyMerkleRoot(root, level, 1 << starkInfo.starkStruct.nBitsExt, starkInfo.starkStruct.lastLevelVerification, starkInfo.starkStruct.merkleTreeArity, nFieldElements);

            if (!isValidRoot) {
                zklog.error("Custom commit " + starkInfo.customCommits[c].name + " Merkle Tree root verification failed");
                isValid = false;
            }
        }
        
        bool isValidCustomCommitsMT = true;
    #pragma omp parallel for
        for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
            std::vector<Goldilocks::Element> values(nCols);
            for (uint64_t i = 0; i < nCols; ++i) {
                values[i] = Goldilocks::fromString(jproof["s0_vals_" + starkInfo.customCommits[c].name + "_0"][q][i]);
            }

            uint64_t nSiblings = merkleProofLevels(starkInfo.starkStruct.nBitsExt, starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification, starkInfo.starkStruct.verificationHashType == std::string("BN128"));
            uint64_t nSiblingsPerLevel = starkInfo.starkStruct.verificationHashType == std::string("BN128") ? starkInfo.starkStruct.merkleTreeArity : (starkInfo.starkStruct.merkleTreeArity - 1) * nFieldElements;
            
            std::vector<std::vector<ElementType>> siblings(
                nSiblings, 
                std::vector<ElementType>(nSiblingsPerLevel)
            );

            for (uint64_t i = 0; i < nSiblings; ++i) {
                for (uint64_t j = 0; j < nSiblingsPerLevel; ++j) {
                    siblings[i][j] = fromString<ElementType>(jproof["s0_siblings_" + starkInfo.customCommits[c].name + "_0"][q][i][j]);
                }
            }
            bool res = tree.verifyGroupProof(root, level, siblings, queriesL0[q], values);
            if(!res) {
                isValidCustomCommitsMT = false;
            }
        }
        if(!isValidCustomCommitsMT) {
            zklog.error("Custom Commit " + starkInfo.customCommits[c].name + " Merkle Tree verification failed");
            isValid = false;
        }
    }


    if (starkInfo.starkStruct.lowDegreeTest == LowDegreeTestKind::FRI) {
        zklog.trace("Verifying FRI foldings Merkle Trees");
        for (uint64_t step=1; step< starkInfo.starkStruct.logDomainSizes.size(); step++) {
            uint64_t nGroups = 1 << starkInfo.starkStruct.logDomainSizes[step];
            uint64_t groupSize = (1 << starkInfo.starkStruct.logDomainSizes[step - 1]) / nGroups;
            MerkleTreeType treeFRI(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification, starkInfo.starkStruct.merkleTreeCustom, nGroups, groupSize * FIELD_EXTENSION);
            ElementType root[nFieldElements];
            ElementType level[nFieldElements * numNodesLevel];
            if (nFieldElements == 1) {
                root[0] = fromString<ElementType>(jproof["s" + std::to_string(step) + "_root"]);
                for(uint64_t i = 0; i < numNodesLevel; ++i) {
                    level[i] = fromString<ElementType>(jproof["s" + std::to_string(step) + "_last_levels"][i]);
                }
            } else {
                for(uint64_t j = 0; j < nFieldElements; ++j) {
                    root[j] = fromString<ElementType>(jproof["s" + std::to_string(step) + "_root"][j]);
                }
                for(uint64_t i = 0; i < numNodesLevel; ++i) {
                    for (uint64_t j = 0; j < nFieldElements; ++j) {
                        level[i * nFieldElements + j] = fromString<ElementType>(jproof["s" + std::to_string(step) + "_last_levels"][i][j]);
                    }
                }
            }

            if (starkInfo.starkStruct.lastLevelVerification > 0) {
                bool isValidRoot = MerkleTreeType::verifyMerkleRoot(root, level, nGroups, starkInfo.starkStruct.lastLevelVerification, starkInfo.starkStruct.merkleTreeArity, nFieldElements);

                if (!isValidRoot) {
                    zklog.error("Step " + to_string(step) + " FRI folding Merkle Tree root verification failed");
                    isValid = false;
                }
            }

            bool isValidFoldingMT = true;
        #pragma omp parallel for
            for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
                uint64_t n_values = (1 << (starkInfo.starkStruct.logDomainSizes[step-1] - starkInfo.starkStruct.logDomainSizes[step]))*FIELD_EXTENSION;
                std::vector<Goldilocks::Element> values(n_values);
                for (uint64_t i = 0; i < n_values; ++i) {
                    values[i] = Goldilocks::fromString(jproof["s" + std::to_string(step) + "_vals"][q][i]);
                }

                uint64_t nSiblings = merkleProofLevels(starkInfo.starkStruct.logDomainSizes[step], starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.lastLevelVerification, starkInfo.starkStruct.verificationHashType == std::string("BN128"));
                uint64_t nSiblingsPerLevel = starkInfo.starkStruct.verificationHashType == std::string("BN128") ? starkInfo.starkStruct.merkleTreeArity : (starkInfo.starkStruct.merkleTreeArity - 1) * nFieldElements;
            
                std::vector<std::vector<ElementType>> siblings(
                    nSiblings, 
                    std::vector<ElementType>(nSiblingsPerLevel)
                );

                for (uint64_t i = 0; i < nSiblings; ++i) {
                    for (uint64_t j = 0; j < nSiblingsPerLevel; ++j) {
                        siblings[i][j] = fromString<ElementType>(jproof["s" + std::to_string(step) + "_siblings"][q][i][j]);
                    }
                }
                bool res = treeFRI.verifyGroupProof(root, level, siblings, queriesL0[q] % (1 << starkInfo.starkStruct.logDomainSizes[step]), values);
                if(!res) {
                    isValidFoldingMT = false;
                }
            }
            if(!isValidFoldingMT) {
                zklog.error("FRI folding Merkle Tree verification failed");
                isValid = false;
            }
        }

        zklog.trace("Verifying FRI foldings");
        for (uint64_t step=1; step < starkInfo.starkStruct.logDomainSizes.size(); step++) {
            bool isValidFolding = true;
        #pragma omp parallel for
            for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
                uint64_t idx = queriesL0[q] % (1 << starkInfo.starkStruct.logDomainSizes[step]);     
                Goldilocks::Element value[3];
                uint64_t n_values = (1 << (starkInfo.starkStruct.logDomainSizes[step-1] - starkInfo.starkStruct.logDomainSizes[step]))*FIELD_EXTENSION;
                std::vector<Goldilocks::Element> values(n_values);
                for (uint64_t i = 0; i < n_values; ++i) {
                    values[i] = Goldilocks::fromString(jproof["s" + std::to_string(step) + "_vals"][q][i]);
                }
                FRI<ElementType>::verify_fold(
                    value,
                    step, 
                    starkInfo.starkStruct.nBitsExt, 
                    starkInfo.starkStruct.logDomainSizes[step], 
                    starkInfo.starkStruct.logDomainSizes[step - 1],
                    &challenges[(starkInfo.challengesMap.size() + step - 1)*FIELD_EXTENSION], // r^fold_{step−1}
                    idx,
                    values
                );
                if (step < starkInfo.starkStruct.logDomainSizes.size() - 1) {
                    uint64_t groupIdx = idx / (1 << starkInfo.starkStruct.logDomainSizes[step + 1]);
                    for(uint64_t i = 0; i < FIELD_EXTENSION; ++i) {
                        if(!Goldilocks::isZero(value[i] - Goldilocks::fromString(jproof["s" + to_string(step + 1) + "_vals"][q][groupIdx * FIELD_EXTENSION + i]))) {
                            isValidFolding = false;
                        }
                    }
                } else {
                    for(uint64_t i = 0; i < FIELD_EXTENSION; ++i) {
                        if(!Goldilocks::isZero(value[i] - Goldilocks::fromString(jproof["finalPol"][idx][i]))) {
                            isValidFolding = false;
                        }
                    }
                }
            }
            if(!isValidFolding) {
                zklog.error("FRI folding verification failed");
                isValid = false;
            }
        }

        zklog.trace("Verifying final pol");
        uint64_t finalPolSize = ( 1<< starkInfo.starkStruct.logDomainSizes[starkInfo.starkStruct.logDomainSizes.size() - 1]);
        NTT_Goldilocks ntt(finalPolSize, 1);
        Goldilocks::Element finalPol[finalPolSize * FIELD_EXTENSION];
        for(uint64_t i = 0; i < finalPolSize; ++i) {
            for(uint64_t j = 0; j < FIELD_EXTENSION; ++j) {
                finalPol[i*FIELD_EXTENSION + j] = Goldilocks::fromString(jproof["finalPol"][i][j]);
            }
        }
        ntt.INTT(finalPol, finalPol, finalPolSize, FIELD_EXTENSION);
        uint64_t lastStep = starkInfo.starkStruct.logDomainSizes[starkInfo.starkStruct.logDomainSizes.size() - 1];
        uint64_t blowupFactor = starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits;
        uint64_t init = blowupFactor > lastStep ? 0 : 1 << (lastStep - blowupFactor);
        for(uint64_t i = init; i < finalPolSize; ++i) {
            for(uint64_t j = 0; j < FIELD_EXTENSION; ++j) {
                if (!Goldilocks::isZero(finalPol[i*FIELD_EXTENSION + j])) {
                    zklog.error("Final polynomial is not zero at position " + std::to_string(i));
                    isValid = false;
                }
            }
        }
        delete[] xDivXSub;
        delete[] trace;
        delete[] aux_trace;
        delete[] trace_custom_commits_fixed;
    }

    return isValid;
}


