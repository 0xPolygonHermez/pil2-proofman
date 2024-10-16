#include "expressions_ctx.hpp"
#include "stark_info.hpp"
#include "merkleTreeGL.hpp"

bool starkVerify(FRIProof<Goldilocks::Element> &fproof, StarkInfo& starkInfo, ExpressionsBin& expressionsBin, Goldilocks::Element *verkey, Goldilocks::Element *publics, bool challengesVadcop, Goldilocks::Element* challenges_) {

    uint64_t friQueries[starkInfo.starkStruct.nQueries];

    Goldilocks::Element evals[starkInfo.evMap.size()  * FIELD_EXTENSION];
    for(uint64_t i = 0; i < starkInfo.evMap.size(); ++i) {
        memcpy(&evals[i * FIELD_EXTENSION], fproof.proof.evals[i].data(), FIELD_EXTENSION * sizeof(Goldilocks::Element));
    }

    Goldilocks::Element subproofValues[starkInfo.subproofValuesMap.size()  * FIELD_EXTENSION];
    for(uint64_t i = 0; i < starkInfo.subproofValuesMap.size() ; ++i) {
        memcpy(&subproofValues[i * FIELD_EXTENSION], fproof.proof.subproofValues[i].data(), FIELD_EXTENSION * sizeof(Goldilocks::Element));
    }

    Goldilocks::Element challenges[(starkInfo.challengesMap.size() + starkInfo.starkStruct.steps.size() + 1) * FIELD_EXTENSION];
    if(!challengesVadcop) {
        uint64_t c = 0;
        TranscriptGL transcript(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom);
        transcript.put(&verkey[0], 4);

        if(starkInfo.nPublics > 0) {
            if(!starkInfo.starkStruct.hashCommits) {
                transcript.put(&publics[0], starkInfo.nPublics);
            } else {
                Goldilocks::Element hash[4];
                TranscriptGL transcriptHash(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom);
                transcriptHash.put(&publics[0], starkInfo.nPublics);
                transcriptHash.getState(hash);
                transcript.put(hash, 4);
            }
        }

        for(uint64_t s = 1; s <= starkInfo.nStages + 1; ++s) {
            uint64_t nChallenges = std::count_if(starkInfo.challengesMap.begin(), starkInfo.challengesMap.end(),[s](const PolMap& c) { return c.stage == s; });
            for(uint64_t i = 0; i < nChallenges; ++i) {
                transcript.getField((uint64_t *)&challenges[c*FIELD_EXTENSION]);
                c++;
            }
            transcript.put(&fproof.proof.roots[s - 1][0], 4);
        }

        // Evals challenge
        transcript.getField((uint64_t *)&challenges[c*FIELD_EXTENSION]);
        c++;

        if(!starkInfo.starkStruct.hashCommits) {
            transcript.put(&evals[0], starkInfo.evMap.size()  * FIELD_EXTENSION);
        } else {
            Goldilocks::Element hash[4];
            TranscriptGL transcriptHash(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom);
            transcriptHash.put(&evals[0], starkInfo.evMap.size()  * FIELD_EXTENSION);
            transcriptHash.getState(hash);
            transcript.put(hash, 4);
        }

        // FRI challenges
        transcript.getField((uint64_t *)&challenges[c*FIELD_EXTENSION]);
        c++;
        transcript.getField((uint64_t *)&challenges[c*FIELD_EXTENSION]);
        c++;


        for (uint64_t step=0; step<starkInfo.starkStruct.steps.size(); step++) {
            transcript.getField((uint64_t *)&challenges[c*FIELD_EXTENSION]);
            c++;
            if (step < starkInfo.starkStruct.steps.size() - 1) {
                transcript.put(fproof.proof.fri.treesFRI[step].root.data(), 4);
            } else {
                uint64_t finalPolSize = (1<< starkInfo.starkStruct.steps[step].nBits);
                Goldilocks::Element finalPol[finalPolSize * FIELD_EXTENSION];
                for(uint64_t i = 0; i < finalPolSize; ++i) {
                    memcpy(&finalPol[i * FIELD_EXTENSION], fproof.proof.fri.pol[i].data(), FIELD_EXTENSION * sizeof(Goldilocks::Element));
                }

                if(!starkInfo.starkStruct.hashCommits) {
                    transcript.put(&finalPol[0],finalPolSize*FIELD_EXTENSION);
                } else {
                    Goldilocks::Element hash[4];
                    TranscriptGL transcriptHash(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom);
                    transcriptHash.put(&finalPol[0], finalPolSize*FIELD_EXTENSION);
                    transcriptHash.getState(hash);
                    transcript.put(hash, 4);
                }
            }
        }
        transcript.getField((uint64_t *)&challenges[c*FIELD_EXTENSION]);
        c++;
        assert(c == (starkInfo.challengesMap.size() + starkInfo.starkStruct.steps.size() + 1));
        
    } else {
        std::memcpy(challenges, challenges_, ((starkInfo.challengesMap.size() + starkInfo.starkStruct.steps.size() + 1) * FIELD_EXTENSION) * sizeof(Goldilocks::Element));
    }

    Goldilocks::Element *challenge = &challenges[(starkInfo.challengesMap.size() + starkInfo.starkStruct.steps.size()) * FIELD_EXTENSION];

    TranscriptGL transcriptPermutation(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom);
    transcriptPermutation.put(challenge, FIELD_EXTENSION);
    transcriptPermutation.getPermutations(friQueries, starkInfo.starkStruct.nQueries, starkInfo.starkStruct.steps[0].nBits);

    Goldilocks::Element constPolsVals[starkInfo.nConstants * starkInfo.starkStruct.nQueries];
    for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
        for(uint64_t i = 0; i < starkInfo.nConstants; ++i) {
            constPolsVals[q*starkInfo.nConstants + i] = fproof.proof.fri.trees.polQueries[q][starkInfo.nStages + 1].v[i][0];
        }
    }
    
    Goldilocks::Element xiChallenge[FIELD_EXTENSION];

    for (uint64_t i = 0; i < starkInfo.challengesMap.size(); i++)
    {
        if(starkInfo.challengesMap[i].stage == starkInfo.nStages + 2) {
            if(starkInfo.challengesMap[i].stageId == 0) {
                std::memcpy(&xiChallenge[0], &challenges[i*FIELD_EXTENSION], FIELD_EXTENSION * sizeof(Goldilocks::Element));
                break;
            }
        }
    }

    ConstPols constPols(starkInfo, xiChallenge, constPolsVals);

    SetupCtx setupCtx(starkInfo, expressionsBin, constPols);

    Goldilocks::Element xDivXSub[starkInfo.openingPoints.size() * FIELD_EXTENSION * starkInfo.starkStruct.nQueries];
    for(uint64_t i = 0; i < starkInfo.starkStruct.nQueries; ++i) {
        uint64_t query = friQueries[i];
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
            Goldilocks3::mul((Goldilocks3::Element &)aux[0], (Goldilocks3::Element &)aux[0], (Goldilocks3::Element &)x_ext[0]);
            std::memcpy(&xDivXSub[(i + o*starkInfo.starkStruct.nQueries)*FIELD_EXTENSION], &aux[0], FIELD_EXTENSION * sizeof(Goldilocks::Element));
        }
    }

    Goldilocks::Element pols[starkInfo.mapTotalN];
    for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
        for(uint64_t i = 0; i < starkInfo.cmPolsMap.size(); ++i) {
            uint64_t stage = starkInfo.cmPolsMap[i].stage;
            uint64_t stagePos = starkInfo.cmPolsMap[i].stagePos;
            uint64_t offset = starkInfo.mapOffsets[std::make_pair("cm" + to_string(stage), false)];
            uint64_t nPols = starkInfo.mapSectionsN["cm" + to_string(stage)];
            if(starkInfo.cmPolsMap[i].dim == 1) {
                std::memcpy(&pols[offset + q*nPols + stagePos], fproof.proof.fri.trees.polQueries[q][stage - 1].v[stagePos].data(), sizeof(Goldilocks::Element));
            } else {
                std::memcpy(&pols[offset + q*nPols + stagePos], fproof.proof.fri.trees.polQueries[q][stage - 1].v[stagePos].data(), sizeof(Goldilocks::Element));
                std::memcpy(&pols[offset + q*nPols + stagePos + 1], fproof.proof.fri.trees.polQueries[q][stage - 1].v[stagePos + 1].data(), sizeof(Goldilocks::Element));
                std::memcpy(&pols[offset + q*nPols + stagePos + 2], fproof.proof.fri.trees.polQueries[q][stage - 1].v[stagePos + 2].data(), sizeof(Goldilocks::Element));
            }
        }
    }

    StepsParams params = {
        pols : pols,
        publicInputs : publics,
        challenges : challenges,
        subproofValues : subproofValues,
        evals : evals,
        xDivXSub : xDivXSub,
    };

    for(uint64_t i = 0; i < starkInfo.nStages + 1; ++i) {
        zklog.trace("Verifying stage " +  to_string(i + 1) + " Merkle tree");
        std::string section = "cm" + to_string(i + 1);
        uint64_t nCols = starkInfo.mapSectionsN[section];
        MerkleTreeGL tree(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom, 1 << starkInfo.starkStruct.nBitsExt, nCols, NULL, false);
        for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
            bool res = tree.verifyGroupProof(&fproof.proof.roots[i][0], fproof.proof.fri.trees.polQueries[q][i].mp, friQueries[q], fproof.proof.fri.trees.polQueries[q][i].v);
            if(!res) {
                zklog.error("Stage " + to_string(i + 1) + " Merkle Tree verification failed");
                return false;
            }
        }
    }

    zklog.trace("Verifying constant Merkle tree");
    MerkleTreeGL treeC(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom, 1 << starkInfo.starkStruct.nBitsExt, starkInfo.nConstants, NULL, false);
    for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
        bool res = treeC.verifyGroupProof(verkey, fproof.proof.fri.trees.polQueries[q][starkInfo.nStages + 1].mp, friQueries[q], fproof.proof.fri.trees.polQueries[q][starkInfo.nStages + 1].v);
        if(!res) {
            zklog.error("Constant Merkle Tree verification failed");
            return false;
        }
    }

    zklog.trace("Verifying evaluations");
    ExpressionsPack expressionsPack(setupCtx, 1);
    
    Goldilocks::Element buff[FIELD_EXTENSION];
    Dest dest(buff);
    dest.addParams(setupCtx.expressionsBin.expressionsInfo[starkInfo.cExpId]);
    std::vector<Dest> dests = {dest};
    
    expressionsPack.calculateExpressions(params, setupCtx.expressionsBin.expressionsBinArgsExpressions, dests, 1);

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
        return false;
    }

    zklog.trace("Verifying FRI foldings Merkle Trees");
    for (uint64_t step=1; step< starkInfo.starkStruct.steps.size(); step++) {
        uint64_t nGroups = 1 << starkInfo.starkStruct.steps[step].nBits;
        uint64_t groupSize = (1 << starkInfo.starkStruct.steps[step - 1].nBits) / nGroups;
        MerkleTreeGL treeFRI(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom, nGroups, groupSize * FIELD_EXTENSION, NULL);
        for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
            bool res = treeFRI.verifyGroupProof(fproof.proof.fri.treesFRI[step - 1].root.data(), fproof.proof.fri.treesFRI[step - 1].polQueries[q][0].mp, friQueries[q], fproof.proof.fri.treesFRI[step - 1].polQueries[q][0].v);
            if(!res) {
                zklog.error("Merkle tree FRI verification for FRI failed");
                return false;
            }
        }
    }

    zklog.trace("Verifying FRI queries consistency");
    ExpressionsPack expressionsPackQueries(setupCtx);

    Goldilocks::Element buffQueries[FIELD_EXTENSION*starkInfo.starkStruct.nQueries];
    Dest destQueries(buffQueries);
    destQueries.addParams(setupCtx.expressionsBin.expressionsInfo[starkInfo.friExpId]);
    std::vector<Dest> destsQueries = {destQueries};
    expressionsPackQueries.calculateExpressions(params, setupCtx.expressionsBin.expressionsBinArgsExpressions, destsQueries, starkInfo.starkStruct.nQueries);
    for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
        uint64_t idx = friQueries[q] % (1 << starkInfo.starkStruct.steps[0].nBits);
        if(starkInfo.starkStruct.steps.size() > 1) {
            uint64_t nextNGroups = 1 << starkInfo.starkStruct.steps[1].nBits;
            uint64_t groupIdx = idx / nextNGroups;
            for(uint64_t d = 0; d < FIELD_EXTENSION; ++d) {
                if(!Goldilocks::isZero(fproof.proof.fri.treesFRI[0].polQueries[q][0].v[groupIdx * FIELD_EXTENSION + d][0] - buffQueries[q*FIELD_EXTENSION + d])) {
                    zklog.error("Verify FRI query consistency failed");
                    return false;
                }
            }
        } else {
            for(uint64_t d = 0; d < FIELD_EXTENSION; ++d) {
                if(!Goldilocks::isZero(fproof.proof.fri.pol[idx][d] - buffQueries[q*FIELD_EXTENSION + d])) {
                    zklog.error("Verify FRI query consistency failed");
                    return false;
                }
            }
        }
        
    }

    zklog.trace("Verifying FRI foldings");
    for (uint64_t step=1; step < starkInfo.starkStruct.steps.size(); step++) {
        for(uint64_t q = 0; q < starkInfo.starkStruct.nQueries; ++q) {
            uint64_t idx = friQueries[q] % (1 << starkInfo.starkStruct.steps[step].nBits);     
            Goldilocks::Element value[3];
            FRI<Goldilocks::Element>::verify_fold(
                value,
                step, 
                starkInfo.starkStruct.nBitsExt, 
                starkInfo.starkStruct.steps[step].nBits, 
                starkInfo.starkStruct.steps[step - 1].nBits,
                &challenges[(starkInfo.challengesMap.size() + step)*FIELD_EXTENSION],
                idx,
                fproof.proof.fri.treesFRI[step - 1].polQueries[q][0].v
            );
            if (step < starkInfo.starkStruct.steps.size() - 1) {
                uint64_t groupIdx = idx / (1 << starkInfo.starkStruct.steps[step + 1].nBits);
                for(uint64_t i = 0; i < FIELD_EXTENSION; ++i) {
                    if(Goldilocks::toU64(value[i]) != Goldilocks::toU64(fproof.proof.fri.treesFRI[step].polQueries[q][0].v[groupIdx * FIELD_EXTENSION + i][0])) {
                        return false;
                    }
                }
            } else {
                for(uint64_t i = 0; i < FIELD_EXTENSION; ++i) {
                    if(Goldilocks::toU64(value[i]) != Goldilocks::toU64(fproof.proof.fri.pol[idx][i])) {
                        return false;
                    }
                }
            }
        }
    }
    
    return true;
}



