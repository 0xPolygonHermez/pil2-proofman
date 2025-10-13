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

    uint64_t getConstTreeSizeBN128(StarkInfo& starkInfo)
    {   
        uint64_t NExtended = 1 << starkInfo.starkStruct.nBitsExt;
        MerkleTreeBN128 mt(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom, NExtended, starkInfo.nConstants);
        return (NExtended * starkInfo.nConstants) + mt.numNodes * (sizeof(RawFr::Element) / sizeof(Goldilocks::Element));
    }

    uint64_t getConstTreeSizeGL(StarkInfo& starkInfo)
    {   
        uint64_t NExtended = 1 << starkInfo.starkStruct.nBitsExt;
        MerkleTreeGL mt(starkInfo.starkStruct.merkleTreeArity, true, NExtended, starkInfo.nConstants);
        return (NExtended * starkInfo.nConstants) + mt.numNodes;
    }

    void calculateConstTreeGL(StarkInfo& starkInfo, Goldilocks::Element *pConstPolsAddress, void *treeAddress) {
        uint64_t N = 1 << starkInfo.starkStruct.nBits;
        uint64_t NExtended = 1 << starkInfo.starkStruct.nBitsExt;
        NTT_Goldilocks ntt(N);
        Goldilocks::Element *treeAddressGL = (Goldilocks::Element *)treeAddress;
        ntt.extendPol(treeAddressGL, pConstPolsAddress, NExtended, N, starkInfo.nConstants);
        MerkleTreeGL mt(starkInfo.starkStruct.merkleTreeArity, true, NExtended, starkInfo.nConstants);
        
        mt.setSource(treeAddressGL);
        mt.setNodes(&treeAddressGL[starkInfo.nConstants * NExtended]);
        mt.merkelize();
    }

    void writeConstTreeFileGL(StarkInfo& starkInfo, void *treeAddress, std::string constTreeFile) {
        TimerStart(WRITING_TREE_FILE);
        MerkleTreeGL mt(starkInfo.starkStruct.merkleTreeArity, true, (Goldilocks::Element *)treeAddress, 1 << starkInfo.starkStruct.nBitsExt, starkInfo.nConstants);
        mt.writeFile(constTreeFile);
        TimerStopAndLog(WRITING_TREE_FILE);
    }

    void calculateConstTreeBN128(StarkInfo& starkInfo, Goldilocks::Element *pConstPolsAddress, void *treeAddress) {
        uint64_t N = 1 << starkInfo.starkStruct.nBits;
        uint64_t NExtended = 1 << starkInfo.starkStruct.nBitsExt;
        NTT_Goldilocks ntt(N);
        Goldilocks::Element *treeAddressGL = (Goldilocks::Element *)treeAddress;
        ntt.extendPol(treeAddressGL, pConstPolsAddress, NExtended, N, starkInfo.nConstants);
        MerkleTreeBN128 mt(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom, NExtended, starkInfo.nConstants);
        mt.setSource(treeAddressGL);
        mt.setNodes((RawFr::Element *)(&treeAddressGL[starkInfo.nConstants * NExtended]));
        mt.merkelize();
    }

    void writeConstTreeFileBN128(StarkInfo& starkInfo, void *treeAddress, std::string constTreeFile) {
        TimerStart(WRITING_TREE_FILE);
        MerkleTreeBN128 mt(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom, (Goldilocks::Element *)treeAddress, 1 << starkInfo.starkStruct.nBitsExt, starkInfo.nConstants);
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
            MerkleTreeBN128 mt(starkInfo.starkStruct.merkleTreeArity, starkInfo.starkStruct.merkleTreeCustom, (Goldilocks::Element *)constTreePols, 1 << starkInfo.starkStruct.nBitsExt, starkInfo.nConstants);
            RawFr::Element root[1];
            mt.getRoot(root);
            if(RawFr::field.toString(root[0], 10) != verkeyJson) {
                return false;
            }
        } else {
            MerkleTreeGL mt(starkInfo.starkStruct.merkleTreeArity, true, (Goldilocks::Element *)constTreePols, 1 << starkInfo.starkStruct.nBitsExt, starkInfo.nConstants);
            Goldilocks::Element root[4];
            mt.getRoot(root);

            cout << " STORED ROOT " << Goldilocks::toU64(root[0]) << " " << Goldilocks::toU64(root[1]) << " " << Goldilocks::toU64(root[2]) << " " << Goldilocks::toU64(root[3]) << endl;

            cout << " VALID ROOT " << verkeyJson[0] << " " << verkeyJson[1] << " " << verkeyJson[2] << " " << verkeyJson[3] << endl;

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

    void loadConstTree(void *constTreePols, std::string constTreeFile, uint64_t constTreeSize) {
        loadFileParallel(constTreePols, constTreeFile, constTreeSize, false);
    }
};

#endif