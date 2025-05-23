#ifndef GLOBAL_CONSTRAINTS_HPP
#define GLOBAL_CONSTRAINTS_HPP
#include "goldilocks_base_field.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct GlobalConstraintInfo {
    uint64_t id;
    uint64_t dim;
    bool valid;
    bool skip;
    uint64_t value[3];
};

void calculateGlobalExpression(json& globalInfo, Goldilocks::Element* dest, Goldilocks::Element* publics, Goldilocks::Element* challenges, Goldilocks::Element* proofValues, Goldilocks::Element** airgroupValues, ParserArgs &parserArgs, ParserParams &parserParams) {

    uint8_t* ops = &parserArgs.ops[parserParams.opsOffset];
    uint16_t* args = &parserArgs.args[parserParams.argsOffset];

    uint64_t i_args = 0;

    Goldilocks::Element tmp1[parserParams.nTemp1];
    Goldilocks::Element tmp3[parserParams.nTemp3*FIELD_EXTENSION];

    uint64_t nAirgroupValues = 0;
    for(uint64_t i = 0; i < globalInfo["aggTypes"].size(); ++i) {
        nAirgroupValues += globalInfo["aggTypes"][i].size() * FIELD_EXTENSION;
    }

    Goldilocks::Element airgroupvalues[nAirgroupValues];
    uint64_t c = 0;
    for(uint64_t i = 0; i < globalInfo["aggTypes"].size(); ++i) {
        memcpy(&airgroupvalues[c], airgroupValues[i], globalInfo["aggTypes"][i].size() * FIELD_EXTENSION*sizeof(Goldilocks::Element));
        c += globalInfo["aggTypes"][i].size() * FIELD_EXTENSION;
    }

    Goldilocks::Element* expressions_params[7];
    expressions_params[0] = tmp1;
    expressions_params[1] = publics;
    expressions_params[2] = parserArgs.numbers;
    expressions_params[3] = proofValues;
    expressions_params[4] = tmp3;
    expressions_params[5] = airgroupvalues;
    expressions_params[6] = challenges;

    for (uint64_t kk = 0; kk < parserParams.nOps; ++kk) {
        switch (ops[kk]) {
            case 0: {
                // COPY dim1 to dim1
                Goldilocks::copy_pack(1, &expressions_params[0][args[i_args]], &expressions_params[args[i_args + 1]][args[i_args + 2]]);
                i_args += 3;
                break;
            }
            case 1: {
                // OPERATION WITH DEST: dim1 - SRC0: dim1 - SRC1: dim1
                Goldilocks::op_pack(1, args[i_args], &expressions_params[0][args[i_args + 1]], &expressions_params[args[i_args + 2]][args[i_args + 3]], &expressions_params[args[i_args + 4]][args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 2: {
                // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim1
                Goldilocks3::op_31_pack(1, args[i_args], &expressions_params[4][args[i_args + 1]], &expressions_params[args[i_args + 2]][args[i_args + 3]], &expressions_params[args[i_args + 4]][args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 3: {
                // OPERATION WITH DEST: dim3 - SRC0: dim3 - SRC1: dim3
                Goldilocks3::op_pack(1, args[i_args], &expressions_params[4][args[i_args + 1]], &expressions_params[args[i_args + 2]][args[i_args + 3]], &expressions_params[args[i_args + 4]][args[i_args + 5]]);
                i_args += 6;
                break;
            }
            case 4: {
                // COPY dim3 to dim3
                Goldilocks3::copy_pack(1, &expressions_params[4][args[i_args]], &expressions_params[args[i_args + 1]][args[i_args + 2]]);
                i_args += 3;
                break;
            }
        }
    }

    if (i_args != parserParams.nArgs) std::cout << " " << i_args << " - " << parserParams.nArgs << std::endl;
    assert(i_args == parserParams.nArgs);

    if(parserParams.destDim == 1) {
        std::memcpy(dest, &tmp1[parserParams.destId], sizeof(Goldilocks::Element));
    } else if(parserParams.destDim == 3) {
        std::memcpy(dest, &tmp3[parserParams.destId * FIELD_EXTENSION], parserParams.destDim * sizeof(Goldilocks::Element));
    }
}


void verifyGlobalConstraint(json& globalInfo, uint64_t constraintId, Goldilocks::Element* publics, Goldilocks::Element* challenges, Goldilocks::Element* proofValues, Goldilocks::Element** airgroupValues, ParserArgs &parserArgs, ParserParams &parserParams, GlobalConstraintInfo& globalConstraintInfo) {

    globalConstraintInfo.id = constraintId;
    globalConstraintInfo.valid = true;

    Goldilocks::Element dest[parserParams.destDim];

    calculateGlobalExpression(globalInfo, dest, publics, challenges, proofValues, airgroupValues, parserArgs, parserParams);

    if(parserParams.destDim == 1) {
        globalConstraintInfo.dim = parserParams.destDim;
        globalConstraintInfo.value[0] = Goldilocks::toU64(dest[0]);
        globalConstraintInfo.value[1] = 0;
        globalConstraintInfo.value[2] = 0;
        if(globalConstraintInfo.value[0] != 0) globalConstraintInfo.valid = false;
    } else {
        globalConstraintInfo.dim = parserParams.destDim;
        globalConstraintInfo.value[0] = Goldilocks::toU64(dest[0]);
        globalConstraintInfo.value[1] = Goldilocks::toU64(dest[1]);
        globalConstraintInfo.value[2] = Goldilocks::toU64(dest[2]);
        if(globalConstraintInfo.value[0] != 0 || globalConstraintInfo.value[1] != 0 || globalConstraintInfo.value[2] != 0) globalConstraintInfo.valid = false;
    }
}

uint64_t getNumberGlobalConstraints(ExpressionsBin &globalConstraintsBin) {
    std::vector<ParserParams> globalConstraints = globalConstraintsBin.constraintsInfoDebug;
    return globalConstraints.size();
}

void getGlobalConstraintsLinesSizes(ExpressionsBin &globalConstraintsBin, uint64_t* constraintsLinesSizes) {
    std::vector<ParserParams> globalConstraints = globalConstraintsBin.constraintsInfoDebug;
    for(uint64_t i = 0; i <globalConstraints.size(); ++i) {
        constraintsLinesSizes[i] = globalConstraintsBin.constraintsInfoDebug[i].line.size();
    }
}

void getGlobalConstraintsLines(ExpressionsBin &globalConstraintsBin, uint8_t** constraintsLines) {
    std::vector<ParserParams> globalConstraints = globalConstraintsBin.constraintsInfoDebug;
     for(uint64_t i = 0; i < globalConstraintsBin.constraintsInfoDebug.size(); ++i) {
        std::memcpy(constraintsLines[i], globalConstraints[i].line.data(), globalConstraints[i].line.size());
    }
}
   

void verifyGlobalConstraints(json& globalInfo, ExpressionsBin &globalConstraintsBin, Goldilocks::Element* publicInputs, Goldilocks::Element* challenges, Goldilocks::Element* proofValues, Goldilocks::Element** airgroupValues, GlobalConstraintInfo *globalConstraintsInfo)
{
    std::vector<ParserParams> globalConstraints = globalConstraintsBin.constraintsInfoDebug;

    for(uint64_t i = 0; i < globalConstraints.size(); ++i) {
        if(!globalConstraintsInfo[i].skip) {
            verifyGlobalConstraint(globalInfo, i, publicInputs, challenges, proofValues, airgroupValues, globalConstraintsBin.expressionsBinArgsConstraints, globalConstraints[i], globalConstraintsInfo[i]);
        }
    }
}

std::string getExpressionDebug(json& globalInfo, ExpressionsBin &globalConstraintsBin, uint64_t hintId, std::string hintFieldName, HintFieldValue hintFieldVal) {
    std::string debug = "Hint name " + hintFieldName + " for hint id " + to_string(hintId) + " is ";
    if (hintFieldVal.operand == opType::tmp) {
        if(globalConstraintsBin.expressionsInfo[hintFieldVal.id].line != "") {
            debug += "the expression with id: " + to_string(hintFieldVal.id) + " " + globalConstraintsBin.expressionsInfo[hintFieldVal.id].line;
        }
    } else if (hintFieldVal.operand == opType::public_) {
        debug += "public input " + to_string(globalInfo["publicsMap"][hintFieldVal.id]["name"]);
    } else if (hintFieldVal.operand == opType::number) {
        debug += "number " + to_string(hintFieldVal.value);
    } else if (hintFieldVal.operand == opType::airgroupvalue) {
       debug += "airgroupvalue ";
    } else if (hintFieldVal.operand == opType::proofvalue) {
       debug += "proof value  " + to_string(globalInfo["proofValuesMap"][hintFieldVal.id]["name"]);
    } else if (hintFieldVal.operand == opType::string_) {
       debug += "string " + hintFieldVal.stringValue;
    } else {
        zklog.error("Unknown HintFieldType");
        exitProcess();
        exit(-1);
    }

    return debug;
}

uint64_t getHintFieldGlobalConstraintValues(ExpressionsBin &globalConstraintsBin, uint64_t hintId, std::string hintFieldName) {
    if(globalConstraintsBin.hints.size() == 0) {
        zklog.error("No hints were found.");
        exitProcess();
        exit(-1);
    }

    Hint hint = globalConstraintsBin.hints[hintId];
    
    auto hintField = std::find_if(hint.fields.begin(), hint.fields.end(), [hintFieldName](const HintField& hintField) {
        return hintField.name == hintFieldName;
    });

    if(hintField == hint.fields.end()) {
        zklog.error("Hint field " + hintFieldName + " not found in hint " + hint.name + ".");
        exitProcess();
        exit(-1);
    }

    return hintField->values.size();
}

void getHintFieldGlobalConstraintSizes(json& globalInfo, ExpressionsBin &globalConstraintsBin,  HintFieldInfo *hintFieldValues,  uint64_t hintId, std::string hintFieldName, bool print_expression) {
    if(globalConstraintsBin.hints.size() == 0) {
        zklog.error("No hints were found.");
        exitProcess();
        exit(-1);
    }

    Hint hint = globalConstraintsBin.hints[hintId];
    
    auto hintField = std::find_if(hint.fields.begin(), hint.fields.end(), [hintFieldName](const HintField& hintField) {
        return hintField.name == hintFieldName;
    });

    if(hintField == hint.fields.end()) {
        zklog.error("Hint field " + hintFieldName + " not found in hint " + hint.name + ".");
        exitProcess();
        exit(-1);
    }

    for(uint64_t i = 0; i < hintField->values.size(); ++i) {
        HintFieldValue hintFieldVal = hintField->values[i];

        if(print_expression) {
            std::string expression_line = getExpressionDebug(globalInfo, globalConstraintsBin, hintId, hintFieldName, hintFieldVal);
            hintFieldValues[i].expression_line_size = expression_line.size();
        }

        if (hintFieldVal.operand == opType::tmp) {
            uint64_t dim = globalConstraintsBin.expressionsInfo[hintFieldVal.id].destDim;
            hintFieldValues[i].fieldType = dim == 1 ? HintFieldType::Column : HintFieldType::ColumnExtended;
            hintFieldValues[i].offset = dim;
            hintFieldValues[i].size = dim;
        } else if (hintFieldVal.operand == opType::public_) {
            hintFieldValues[i].size = 1;
            hintFieldValues[i].fieldType = HintFieldType::Field;
            hintFieldValues[i].offset = 1;
        } else if (hintFieldVal.operand == opType::number) {
            hintFieldValues[i].size = 1;
            hintFieldValues[i].fieldType = HintFieldType::Field;
            hintFieldValues[i].offset = 1;
        } else if (hintFieldVal.operand == opType::airgroupvalue) {
            hintFieldValues[i].size = FIELD_EXTENSION;
            hintFieldValues[i].fieldType = HintFieldType::FieldExtended;
            hintFieldValues[i].offset = FIELD_EXTENSION;
        } else if (hintFieldVal.operand == opType::proofvalue) {
            uint64_t dim = globalInfo["proofValuesMap"][hintFieldVal.id]["stage"] == 1 ? 1 : FIELD_EXTENSION;
            hintFieldValues[i].size = dim;
            hintFieldValues[i].fieldType = dim == 1 ? HintFieldType::Field : HintFieldType::FieldExtended;
            hintFieldValues[i].offset = FIELD_EXTENSION;
        } else if (hintFieldVal.operand == opType::string_) {
            hintFieldValues[i].fieldType = HintFieldType::String;
            hintFieldValues[i].size = hintFieldVal.stringValue.size();
            hintFieldValues[i].offset = 0;
        } else {
            zklog.error("Unknown HintFieldType");
            exitProcess();
            exit(-1);
        }

        hintFieldValues[i].matrix_size = hintFieldVal.pos.size();
    }
    
    return;
}

void getHintFieldGlobalConstraint(json& globalInfo, ExpressionsBin &globalConstraintsBin, HintFieldInfo *hintFieldValues, Goldilocks::Element* publicInputs, Goldilocks::Element* challenges, Goldilocks::Element* proofValues, Goldilocks::Element** airgroupValues, uint64_t hintId, std::string hintFieldName, bool print_expression) {
   

    if(globalConstraintsBin.hints.size() == 0) {
        zklog.error("No hints were found.");
        exitProcess();
        exit(-1);
    }

    Hint hint = globalConstraintsBin.hints[hintId];
    
    auto hintField = std::find_if(hint.fields.begin(), hint.fields.end(), [hintFieldName](const HintField& hintField) {
        return hintField.name == hintFieldName;
    });

    if(hintField == hint.fields.end()) {
        zklog.error("Hint field " + hintFieldName + " not found in hint " + hint.name + ".");
        exitProcess();
        exit(-1);
    }

    for(uint64_t i = 0; i < hintField->values.size(); ++i) {
        HintFieldValue hintFieldVal = hintField->values[i];
       
        HintFieldInfo hintFieldInfo = hintFieldValues[i];

        if(print_expression) {
            std::string expression_line = getExpressionDebug(globalInfo, globalConstraintsBin, hintId, hintFieldName, hintFieldVal);
            hintFieldValues[i].expression_line_size = expression_line.size();
        }

        if (hintFieldVal.operand == opType::tmp) {
            calculateGlobalExpression(globalInfo, hintFieldInfo.values, publicInputs, challenges, proofValues, airgroupValues, globalConstraintsBin.expressionsBinArgsExpressions, globalConstraintsBin.expressionsInfo[hintFieldVal.id]);
        } else if (hintFieldVal.operand == opType::public_) {
            hintFieldInfo.values[0] = publicInputs[hintFieldVal.id];
        } else if (hintFieldVal.operand == opType::number) {
            hintFieldInfo.values[0] = Goldilocks::fromU64(hintFieldVal.value);
        } else if (hintFieldVal.operand == opType::airgroupvalue) {
            std::memcpy(hintFieldInfo.values, &airgroupValues[hintFieldVal.dim][FIELD_EXTENSION*hintFieldVal.id], FIELD_EXTENSION * sizeof(Goldilocks::Element));
        } else if (hintFieldVal.operand == opType::proofvalue) {
            std::memcpy(hintFieldInfo.values, &proofValues[FIELD_EXTENSION*hintFieldVal.id], hintFieldInfo.size * sizeof(Goldilocks::Element));
        } else if (hintFieldVal.operand == opType::string_) {
            std::memcpy(hintFieldInfo.stringValue, hintFieldVal.stringValue.data(), hintFieldVal.stringValue.size());
        } else {
            zklog.error("Unknown HintFieldType");
            exitProcess();
            exit(-1);
        }

        for(uint64_t i = 0; i < hintFieldInfo.matrix_size; ++i) {
            hintFieldInfo.pos[i] =  hintFieldVal.pos[i];
        }
    }
    
    return;
}


uint64_t setHintFieldGlobalConstraint(json& globalInfo, ExpressionsBin &globalConstraintsBin, Goldilocks::Element* proofValues, Goldilocks::Element* values, uint64_t hintId, std::string hintFieldName) {
    Hint hint = globalConstraintsBin.hints[hintId];

    auto hintField = std::find_if(hint.fields.begin(), hint.fields.end(), [hintFieldName](const HintField& hintField) {
        return hintField.name == hintFieldName;
    });

    if(hintField == hint.fields.end()) {
        zklog.error("Hint field " + hintFieldName + " not found in hint " + hint.name + ".");
        exitProcess();
        exit(-1);
    }

    if(hintField->values.size() != 1) {
        zklog.error("Hint field " + hintFieldName + " in " + hint.name + "has more than one destination.");
        exitProcess();
        exit(-1);
    }

    auto hintFieldVal = hintField->values[0];
    if(hintFieldVal.operand == opType::proofvalue) {
        std::memcpy(&proofValues[FIELD_EXTENSION*hintFieldVal.id], values, FIELD_EXTENSION * sizeof(Goldilocks::Element));
    } else {
        zklog.error("Only committed pols and airgroupvalues can be set");
        exitProcess();
        exit(-1);  
    }

    return hintFieldVal.id;
}

#endif