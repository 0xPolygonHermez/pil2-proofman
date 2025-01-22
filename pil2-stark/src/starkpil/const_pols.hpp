#ifndef CONST_POLS_STARKS_HPP
#define CONST_POLS_STARKS_HPP

#include <cstdint>
#include "goldilocks_base_field.hpp"
#include "zkassert.hpp"
#include "stark_info.hpp"
#include "zklog.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "ntt_goldilocks.hpp"
#include "merkleTreeBN128.hpp"
#include "merkleTreeGL.hpp"

class ConstTree {
public:
    ConstTree () {};

    uint64_t getConstTreeSizeBytesBN128(StarkInfo& starkInfo)
    {   
        uint64_t NExtended = 1 << starkInfo.starkStruct.nBitsExt;
        MerkleTreeBN128 mt(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom, NExtended, starkInfo.nConstants);
        return 16 + (NExtended * starkInfo.nConstants) * sizeof(Goldilocks::Element) + mt.numNodes * sizeof(RawFr::Element);
    }

    uint64_t getConstTreeSizeBytesGL(StarkInfo& starkInfo)
    {   
        uint64_t NExtended = 1 << starkInfo.starkStruct.nBitsExt;
        MerkleTreeGL mt(2, true, NExtended, starkInfo.nConstants);
        return (2 + (NExtended * starkInfo.nConstants) + mt.numNodes) * sizeof(Goldilocks::Element);
    }

    void calculateConstTreeGL(StarkInfo& starkInfo, Goldilocks::Element *pConstPolsAddress, void *treeAddress) {
        uint64_t N = 1 << starkInfo.starkStruct.nBits;
        uint64_t NExtended = 1 << starkInfo.starkStruct.nBitsExt;
        NTT_Goldilocks ntt(N);
        Goldilocks::Element *treeAddressGL = (Goldilocks::Element *)treeAddress;
        ntt.extendPol(&treeAddressGL[2], pConstPolsAddress, NExtended, N, starkInfo.nConstants);
        MerkleTreeGL mt(2, true, NExtended, starkInfo.nConstants);
        
        mt.setSource(&treeAddressGL[2]);
        mt.setNodes(&treeAddressGL[2 + starkInfo.nConstants * NExtended]);
        mt.merkelize();

        treeAddressGL[0] = Goldilocks::fromU64(starkInfo.nConstants);  
        treeAddressGL[1] = Goldilocks::fromU64(NExtended);
    }

    void writeConstTreeFileGL(StarkInfo& starkInfo, void *treeAddress, std::string constTreeFile) {
        TimerStart(WRITING_TREE_FILE);
        MerkleTreeGL mt(2, true, (Goldilocks::Element *)treeAddress);
        mt.writeFile(constTreeFile);
        TimerStopAndLog(WRITING_TREE_FILE);
    }

    void calculateConstTreeBN128(StarkInfo& starkInfo, Goldilocks::Element *pConstPolsAddress, void *treeAddress) {
        uint64_t N = 1 << starkInfo.starkStruct.nBits;
        uint64_t NExtended = 1 << starkInfo.starkStruct.nBitsExt;
        NTT_Goldilocks ntt(N);
        Goldilocks::Element *treeAddressGL = (Goldilocks::Element *)treeAddress;
        ntt.extendPol(&treeAddressGL[2], pConstPolsAddress, NExtended, N, starkInfo.nConstants);
        MerkleTreeBN128 mt(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom, NExtended, starkInfo.nConstants);
        mt.setSource(&treeAddressGL[2]);
        mt.setNodes((RawFr::Element *)(&treeAddressGL[2 + starkInfo.nConstants * NExtended]));
        mt.merkelize();

        treeAddressGL[0] = Goldilocks::fromU64(starkInfo.nConstants);  
        treeAddressGL[1] = Goldilocks::fromU64(NExtended);
    }

    void writeConstTreeFileBN128(StarkInfo& starkInfo, void *treeAddress, std::string constTreeFile) {
        TimerStart(WRITING_TREE_FILE);
        MerkleTreeBN128 mt(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom, (Goldilocks::Element *)treeAddress);
        mt.writeFile(constTreeFile);
        TimerStopAndLog(WRITING_TREE_FILE);
    }

    bool loadConstTree(StarkInfo &starkInfo, void *constTreePols, std::string constTreeFile, uint64_t constTreeSize, std::string verkeyFile) {
        bool fileLoaded = loadFileParallel(constTreePols, constTreeFile, constTreeSize, false);
        if(!fileLoaded) {
            return false;
        }
        
        json verkeyJson;
        file2json(verkeyFile, verkeyJson);

        if (starkInfo.starkStruct.verificationHashType == "BN128") {
            MerkleTreeBN128 mt(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom, (Goldilocks::Element *)constTreePols);
            RawFr::Element root[1];
            mt.getRoot(root);
            if(RawFr::field.toString(root[0], 10) != verkeyJson) {
                return false;
            }
        } else {
            MerkleTreeGL mt(2, true, (Goldilocks::Element *)constTreePols);
            Goldilocks::Element root[4];
            mt.getRoot(root);

            if (Goldilocks::toU64(root[0]) != verkeyJson[0] ||
                Goldilocks::toU64(root[1]) != verkeyJson[1] ||
                Goldilocks::toU64(root[2]) != verkeyJson[2] ||
                Goldilocks::toU64(root[3]) != verkeyJson[3]) 
            {
                return false;
            }

        }

        return true;
    }

    void loadConstPols(void *constPols, std::string constPolsFile, uint64_t constPolsSize) {
        loadFileParallel(constPols, constPolsFile, constPolsSize);
    }
};

#endif