#ifndef BINARY_HPP
#define BINARY_HPP

#include <string>
#include <map>
#include "binfile_utils.hpp"
#include "binfile_writer.hpp"
#include "expressions_info.hpp"
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

const int EXPRESSIONS_SECTION = 1;
const int CONSTRAINTS_SECTION = 2;
const int HINTS_SECTION = 3;
const int N_SECTIONS = 3;

const int GLOBAL_CONSTRAINTS_SECTION = 1;
const int GLOBAL_HINTS_SECTION = 2;
const int N_GLOBAL_SECTIONS = 2;


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
    uint8_t* ops = nullptr;
    uint16_t* args = nullptr;
    Goldilocks::Element* numbers = nullptr;
    uint64_t nNumbers = 0;
};

class ExpressionsBin
{
public:
    
    uint32_t  nOpsTotal = 0;
    uint32_t  nArgsTotal = 0;

    uint32_t  nOpsDebug = 0;
    uint32_t  nArgsDebug = 0;

    bool write = false;

    std::map<uint64_t, ParserParams> expressionsInfo;

    std::vector<ParserParams> constraintsInfoDebug;

    std::vector<Hint> hints;

    ParserArgs expressionsBinArgsConstraints;
    
    ParserArgs expressionsBinArgsExpressions;

    uint64_t maxTmp1 = 0;
    uint64_t maxTmp3 = 0;

    uint64_t maxArgs = 0;
    uint64_t maxOps = 0;

    ~ExpressionsBin() {
        if (!write) {
            if (expressionsBinArgsExpressions.ops) delete[] expressionsBinArgsExpressions.ops;
            if (expressionsBinArgsExpressions.args) delete[] expressionsBinArgsExpressions.args;
            if (expressionsBinArgsExpressions.numbers) delete[] expressionsBinArgsExpressions.numbers;

            if (expressionsBinArgsConstraints.ops) delete[] expressionsBinArgsConstraints.ops;
            if (expressionsBinArgsConstraints.args) delete[] expressionsBinArgsConstraints.args;
            if (expressionsBinArgsConstraints.numbers) delete[] expressionsBinArgsConstraints.numbers;
        }        
    };

    /* Constructor */
    ExpressionsBin(string file, bool globalBin = false, bool verifierBin = false);

    ExpressionsBin(string starkInfoFile, string expressionsInfoFile, string expressionsBinFile, bool globalBin = false, bool verifierBin = false);

    void loadExpressionsBin(BinFileUtils::BinFile *expressionsBin);

    void loadGlobalBin(BinFileUtils::BinFile *globalBin);

    void loadVerifierBin(BinFileUtils::BinFile *verifierBin);

    void writeGlobalExpressionsBin(string binFile, ExpressionsInfo& expsInfo);

    void writeExpressionsBin(string binFile, ExpressionsInfo& expsInfo);

    void writeVerifierBin(string binFile, ExpressionsInfo& expsInfo);

    void writeExpressionsSection(BinFileUtils::BinFileWriter &binFile, int section, std::vector<ExpInfoBin> expressionsInfo, std::vector<uint64_t> numbersExps, uint64_t maxTmp1, uint64_t maxTmp3, uint64_t maxArgs, uint64_t maxOps);
    
    void writeGlobalConstraintsSection(BinFileUtils::BinFileWriter &binFile, int section, std::vector<ExpInfoBin> constraintsInfo, std::vector<uint64_t> numbersConstraints);
    void writeGlobalHintsSection(BinFileUtils::BinFileWriter &binFile, int section, std::vector<HintInfo> hintsInfo);

    void writeConstraintsSection(BinFileUtils::BinFileWriter &binFile, int section, std::vector<ExpInfoBin> constraintsInfo, std::vector<uint64_t> numbersConstraints);

    void writeHintsSection(BinFileUtils::BinFileWriter &binFile, int section, std::vector<HintInfo> hintsInfo);

    uint64_t getNumberHintIdsByName(std::string name);

    void getHintIdsByName(uint64_t* hintIds, std::string name);
};


#endif
