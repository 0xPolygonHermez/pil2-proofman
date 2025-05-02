#ifndef BINARY_HPP
#define BINARY_HPP

#include <string>
#include <map>
#include "binfile_utils.hpp"
#include "goldilocks_base_field.hpp"
#include "goldilocks_base_field_avx.hpp"
#include "goldilocks_base_field_avx512.hpp"
#include "goldilocks_base_field_pack.hpp"
#include "goldilocks_cubic_extension.hpp"
#include "goldilocks_cubic_extension_pack.hpp"
#include "goldilocks_cubic_extension_avx.hpp"
#include "goldilocks_cubic_extension_avx512.hpp"
#include "stark_info.hpp"
#include <immintrin.h>
#include <cassert>

const int BINARY_EXPRESSIONS_SECTION = 2;
const int BINARY_CONSTRAINTS_SECTION = 3;
const int BINARY_HINTS_SECTION = 4;

const int GLOBAL_CONSTRAINTS_SECTION = 2;
const int GLOBAL_HINTS_SECTION = 3;

struct HintFieldValue {
    opType operand;
    uint64_t id;
    uint64_t commitId;
    uint64_t rowOffsetIndex;
    uint64_t dim;
    uint64_t value;
    string stringValue;
    std::vector<uint64_t> pos;
};

struct HintField {
    string name;
    std::vector<HintFieldValue> values;
};


struct Hint
{
    std::string name;
    std::vector<HintField> fields;
};

struct ParserParams
{
    uint32_t stage = 0;
    uint32_t expId = 0;
    uint32_t nTemp1 = 0;
    uint32_t nTemp3 = 0;
    uint32_t nOps = 0;
    uint32_t opsOffset = 0;
    uint32_t nArgs = 0;
    uint32_t argsOffset = 0;
    uint32_t firstRow = 0;
    uint32_t lastRow = 0;
    uint32_t destDim = 0;
    uint32_t destId = 0;
    bool imPol = false;
    string line = "";
};

struct ParserArgs 
{
    uint8_t* ops;
    uint16_t* args;
    Goldilocks::Element* numbers;
    uint64_t nNumbers;
};

class ExpressionsBin
{
public:

    uint32_t  nOpsTotal;
    uint32_t  nArgsTotal;
    uint32_t  nOpsDebug;
    uint32_t  nArgsDebug;

    std::map<uint64_t, ParserParams> expressionsInfo;

    std::vector<ParserParams> constraintsInfoDebug;

    std::vector<Hint> hints;

    ParserArgs expressionsBinArgsConstraints;
    
    ParserArgs expressionsBinArgsExpressions;

    uint64_t maxTmp1;
    uint64_t maxTmp3;
    
    ~ExpressionsBin() {
        if (expressionsBinArgsExpressions.ops) delete[] expressionsBinArgsExpressions.ops;
        if (expressionsBinArgsExpressions.args) delete[] expressionsBinArgsExpressions.args;
        if (expressionsBinArgsExpressions.numbers) delete[] expressionsBinArgsExpressions.numbers;

        if (expressionsBinArgsConstraints.ops) delete[] expressionsBinArgsConstraints.ops;
        if (expressionsBinArgsConstraints.args) delete[] expressionsBinArgsConstraints.args;
        if (expressionsBinArgsConstraints.numbers) delete[] expressionsBinArgsConstraints.numbers;
    };

    /* Constructor */
    ExpressionsBin(string file, bool globalBin = false, bool verifierBin = false);

    void loadExpressionsBin(BinFileUtils::BinFile *expressionsBin);

    void loadGlobalBin(BinFileUtils::BinFile *globalBin);

    void loadVerifierBin(BinFileUtils::BinFile *verifierBin);

    uint64_t getNumberHintIdsByName(std::string name);

    void getHintIdsByName(uint64_t* hintIds, std::string name);
};


#endif
