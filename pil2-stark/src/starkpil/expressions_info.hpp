#ifndef EXPRESSIONS_INFO_HPP
#define EXPRESSIONS_INFO_HPP

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include "zkassert.hpp"
#include "goldilocks_base_field.hpp"
#include "stark_info.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"

using json = nlohmann::json;
using namespace std;

struct Operation {
    string dest_type;
    string src0_type;
    string src1_type;
};

class HintValues
{
public:
    opType op;
    uint64_t id;
    uint64_t stageId;
    uint64_t airgroupId;
    uint64_t rowOffsetIndex;
    uint64_t stage;
    uint64_t dim;
    uint64_t commitId;
    vector<uint64_t> pos;
    std::string string_;
    uint64_t value;
};

class HintField_
{
public:
    string name;
    vector<HintValues> values;
};

class HintInfo 
{
public:
    std::string name;
    vector<HintField_> fields;
};

class CodeType
{
public:
    opType type;
    uint64_t id;
    uint64_t prime;
    uint64_t dim;
    uint64_t value;
    uint64_t commitId;
    uint64_t boundaryId;
    uint64_t airgroupId;
};

class CodeOperation
{
public:
    typedef enum
    {
        add = 0,
        sub = 1,
        mul = 2,
        sub_swap = 3,
    } eOperation;

    eOperation op;
    uint32_t dest_dim;
    uint32_t src0_dim;
    uint32_t src1_dim;
    CodeType dest;
    vector<CodeType> src;

    uint64_t operationArg (eOperation op) {
        if (op == eOperation::add) return 0;
        else if (op == eOperation::sub) return 1;
        else if (op == eOperation::mul) return 2;
        else if (op == eOperation::sub_swap) return 3;
        else throw runtime_error("Invalid operation");
    }

    void setOperation (string s)
    {
        if (s == "add") op = add;
        else if (s == "sub") op = sub;
        else if (s == "mul") op = mul;
        else if (s == "sub_swap") op = sub_swap;
        else
        {
            zklog.error("StepOperation::setOperation() found invalid type: " + s);
            exitProcess();
        }
    }
};


class ExpInfo
{
public:
    uint64_t expId;
    uint64_t stage;
    uint64_t tmpUsed;
    CodeType dest;
    vector<CodeOperation> code;
    Boundary boundary;
    uint64_t imPol;
    string line;
};

class ExpInfoBin
{
public:
    uint64_t nTemp1;
    uint64_t nTemp3;
    vector<uint64_t> args;
    vector<uint64_t> ops;
    uint64_t expId;
    uint64_t destDim;
    uint64_t destId;
    uint64_t firstRow;
    uint64_t lastRow;
    uint64_t stage;
    uint64_t imPol;
    string line;
};

class ExpressionsInfo
{
public:
    StarkInfo starkInfo;

    uint64_t maxTmp1 = 0;
    uint64_t maxTmp3 = 0;
    uint64_t maxArgs = 0;
    uint64_t maxOps = 0;

    // Read from expressionsInfo file
    vector<HintInfo> hintsInfo;
    vector<ExpInfo> expressionsCode;
    vector<ExpInfo> constraintsCode;

    vector<ExpInfoBin> constraintsInfo;
    vector<ExpInfoBin> expressionsInfo;
    std::vector<uint64_t> numbersConstraints;
    std::vector<uint64_t> numbersExps;


    ExpressionsInfo(string starkInfoFile, string expressionsInfofile, bool verifier = false);

    ExpressionsInfo(string expressionsInfoFile);

    void load(json j, bool global, bool verifier);

    std::map<std::pair<std::string, std::vector<std::string>>, uint64_t> getAllOperations();

    std::map<std::pair<std::string, std::vector<std::string>>, uint64_t> getGlobalOperations();

    void pushArgs(vector<uint64_t> &args, CodeType &r, vector<int64_t> &ID1D, vector<int64_t> &ID3D, vector<uint64_t> &numbers, bool dest, bool global);

    ExpInfoBin getParserArgs(std::vector<CodeOperation> &code, uint64_t nTmpUsed, std::vector<uint64_t> &numbers, bool global = false);

    void prepareExpressionsBin();

    void prepareVerifierExpressionsBin();

    void prepareGlobalExpressionsBin();


};

#endif