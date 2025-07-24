#include <string>
#include <map>
#include "binfile_utils.hpp"
#include "binfile_writer.hpp"
#include "goldilocks_base_field.hpp"
#include "goldilocks_base_field_avx.hpp"
#include "goldilocks_base_field_avx512.hpp"
#include "goldilocks_base_field_pack.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "goldilocks_cubic_extension_pack.hpp"
#include "goldilocks_cubic_extension_avx.hpp"
#include "goldilocks_cubic_extension_avx512.hpp"
#include "stark_info.hpp"
#include <cassert>

const int FIXED_POLS_SECTION = 1;
const int N_SECTIONS = 1;

struct FixedPolsInfo {
    uint64_t name_size;
    uint8_t *name;
    uint64_t n_lengths;
    uint64_t *lengths;
    Goldilocks::Element *values;
};

void writeFixedColsBin(string binFileName, string airgroupName, string airName, uint64_t N, uint64_t nFixedPols, FixedPolsInfo* fixedPolsInfo) {
    BinFileUtils::BinFileWriter binFile(binFileName, "cnst", 1, N_SECTIONS);

    binFile.startWriteSection(FIXED_POLS_SECTION);

    binFile.writeString(airgroupName);
    binFile.writeString(airName);
    binFile.writeU64LE(N);
    binFile.writeU32LE(nFixedPols);
    for(uint64_t i = 0; i < nFixedPols; ++i) {
        std::string name = std::string((char *)fixedPolsInfo[i].name, fixedPolsInfo[i].name_size);
        binFile.writeString(name);
        binFile.writeU32LE(fixedPolsInfo[i].n_lengths);
        for(uint64_t j = 0; j < fixedPolsInfo[i].n_lengths; ++j) {
            binFile.writeU32LE(fixedPolsInfo[i].lengths[j]);
        }

        binFile.write((void *)fixedPolsInfo[i].values, N * sizeof(Goldilocks::Element));
    }

    binFile.endWriteSection();
}
