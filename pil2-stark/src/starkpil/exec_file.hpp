#ifndef EXEC_FILE
#define EXEC_FILE

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include "utils.hpp"

void readExecFile(uint64_t *exec_data, std::string execFile, uint64_t nCommitedPols) {
    uint64_t nAdds;
    uint64_t nSMap;

    std::ifstream file(execFile, std::ios::binary);
    file.read(reinterpret_cast<char *>(&nAdds), sizeof(uint64_t));
    file.read(reinterpret_cast<char *>(&nSMap), sizeof(uint64_t));
    
    loadFileParallel(exec_data, execFile, (2 + nAdds * 4 + nSMap * nCommitedPols) * sizeof(uint64_t));
}

void getCommitedPols(uint64_t *signalValues, uint64_t *witness2SignalList, uint64_t *exec_data, Goldilocks::Element *witness, Goldilocks::Element* publics, uint64_t sizeWitness, uint64_t N, uint64_t nPublics, uint64_t nCommitedPols)  {

    uint64_t nAdds = exec_data[0];
    uint64_t nSMap = exec_data[1];
    uint64_t *p_adds = &exec_data[2];
    uint64_t *p_sMap = &exec_data[2 + nAdds * 4];

    std::vector<Goldilocks::Element> adds_scratch(nAdds);

    // Helper: get a field element by flat witness index.
    // Indices [0, sizeWitness) map through witness2SignalList into signalValues.
    // Indices [sizeWitness, sizeWitness+nAdds) are computed additions stored in adds_scratch.
    auto getVal = [&](uint64_t idx) -> Goldilocks::Element {
        if (idx < sizeWitness)
            return Goldilocks::fromU64(signalValues[witness2SignalList[idx]]);
        else
            return adds_scratch[idx - sizeWitness];
    };

    for(uint64_t i = 0; i < nPublics; ++i) {
        publics[i] = getVal(1 + i);
    }
        
    for (uint64_t i = 0; i < nAdds; i++) {
        uint64_t idx_1 = p_adds[i * 4];
        uint64_t idx_2 = p_adds[i * 4 + 1];

        Goldilocks::Element c = getVal(idx_1) * Goldilocks::fromU64(p_adds[i * 4 + 2]);
        Goldilocks::Element d = getVal(idx_2) * Goldilocks::fromU64(p_adds[i * 4 + 3]);
        adds_scratch[i] = c + d;
    }

    for (uint i = 0; i < N; i++) {
        for (uint j = 0; j < nCommitedPols; j++) {
            if (i < nSMap && p_sMap[nCommitedPols * i + j] != 0) {
                witness[i * nCommitedPols + j] = getVal(p_sMap[nCommitedPols * i + j]);
            } else {
                witness[i * nCommitedPols + j] = Goldilocks::zero();
            }
        }
    } 
}

#endif
