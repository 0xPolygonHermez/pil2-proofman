#include "expressions_ctx.hpp"

typedef enum {
    Field = 0,
    FieldExtended = 1,
    Column = 2,
    ColumnExtended = 3,
    String = 4,
} HintFieldType;

struct HintFieldInfo {
    uint64_t size; // Destination size (in Goldilocks elements)
    uint8_t offset;
    HintFieldType fieldType;
    Goldilocks::Element* values;
    uint8_t* stringValue;
    uint64_t matrix_size;
    uint64_t* pos;
};

struct HintFieldValues {
    uint64_t nValues;
    HintFieldInfo* values;
};

struct HintFieldArgs {
    std::string name;
    bool inverse = false;  
};

struct HintFieldOptions {
    bool dest = false;
    bool inverse = false;
    bool print_expression = false;
    bool initialize_zeros = false;
};


void getPolynomial(SetupCtx& setupCtx, Goldilocks::Element *buffer, Goldilocks::Element *dest, bool committed, uint64_t idPol, bool domainExtended) {
    PolMap polInfo = committed ? setupCtx.starkInfo.cmPolsMap[idPol] : setupCtx.starkInfo.constPolsMap[idPol];
    uint64_t deg = domainExtended ? 1 << setupCtx.starkInfo.starkStruct.nBitsExt : 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t dim = polInfo.dim;
    std::string stage = committed ? "cm" + to_string(polInfo.stage) : "const";
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[stage];
    uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair(stage, domainExtended)];
    offset += polInfo.stagePos;
    Goldilocks::Element *pols = committed ? buffer : domainExtended ? setupCtx.constPols.pConstPolsAddressExtended : setupCtx.constPols.pConstPolsAddress;
    Polinomial pol = Polinomial(&pols[offset], deg, dim, nCols, std::to_string(idPol));
#pragma omp parallel for
    for(uint64_t j = 0; j < deg; ++j) {
        std::memcpy(&dest[j*dim], pol[j], dim * sizeof(Goldilocks::Element));
    }
}

void setPolynomial(SetupCtx& setupCtx, Goldilocks::Element *buffer, Goldilocks::Element *values, uint64_t idPol, bool domainExtended) {
    PolMap polInfo = setupCtx.starkInfo.cmPolsMap[idPol];
    uint64_t deg = domainExtended ? 1 << setupCtx.starkInfo.starkStruct.nBitsExt : 1 << setupCtx.starkInfo.starkStruct.nBits;
    uint64_t dim = polInfo.dim;
    std::string stage = "cm" + to_string(polInfo.stage);
    uint64_t nCols = setupCtx.starkInfo.mapSectionsN[stage];
    uint64_t offset = setupCtx.starkInfo.mapOffsets[std::make_pair(stage, domainExtended)];
    offset += polInfo.stagePos;
    Polinomial pol = Polinomial(&buffer[offset], deg, dim, nCols, std::to_string(idPol));
#pragma omp parallel for
    for(uint64_t j = 0; j < deg; ++j) {
        std::memcpy(pol[j], &values[j*dim], dim * sizeof(Goldilocks::Element));
    }
}

void printExpression(Goldilocks::Element* pol, uint64_t dim, uint64_t firstPrintValue = 0, uint64_t lastPrintValue = 0) {        
    cout << "-------------------------------------------------" << endl;
    for(uint64_t i = firstPrintValue; i < lastPrintValue; ++i) {
        if(dim == 3) {
            cout << "Value at " << i << " is: " << " [" << Goldilocks::toString(pol[i*FIELD_EXTENSION]) << ", " << Goldilocks::toString(pol[i*FIELD_EXTENSION + 1]) << ", " << Goldilocks::toString(pol[i*FIELD_EXTENSION + 2]) << " ]" << endl; 
        } else {
            cout << "Value at " << i << " is: " << Goldilocks::toString(pol[i]) << endl;
        }
    }
    cout << "-------------------------------------------------" << endl;
}

void printRow(SetupCtx& setupCtx, Goldilocks::Element* buffer, uint64_t stage, uint64_t row) {
    Goldilocks::Element *pol = &buffer[setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + to_string(stage), false)] + setupCtx.starkInfo.mapSectionsN["cm" + to_string(stage)] * row];
    cout << "Values at row " << row << " = {" << endl;
    bool first = true;
    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); ++i) {
        PolMap cmPol = setupCtx.starkInfo.cmPolsMap[i];
        if(cmPol.stage == stage) {
            if(first) {
                first = false;
            } else {
                cout << endl;
            }
            cout << "    " << cmPol.name;
            if(cmPol.lengths.size() > 0) {
                cout << "[";
                for(uint64_t i = 0; i < cmPol.lengths.size(); ++i) {
                    cout << cmPol.lengths[i];
                    if(i != cmPol.lengths.size() - 1) cout << ", ";
                }
                cout << "]";
            }
            cout << ": ";
            if(cmPol.dim == 1) {
                cout << Goldilocks::toString(pol[cmPol.stagePos]) << ",";
            } else {
                cout << "[" << Goldilocks::toString(pol[cmPol.stagePos]) << ", " << Goldilocks::toString(pol[cmPol.stagePos + 1]) << ", " << Goldilocks::toString(pol[cmPol.stagePos + 2]) << "],";
            }
        }
    }
    cout << endl;
    cout << "}" << endl;
}

void printColById(SetupCtx& setupCtx, Goldilocks::Element* buffer, bool committed, uint64_t polId, uint64_t firstPrintValue = 0, uint64_t lastPrintValue = 0)
{   
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    PolMap polInfo = committed ? setupCtx.starkInfo.cmPolsMap[polId] : setupCtx.starkInfo.constPolsMap[polId];
    Goldilocks::Element *pols = committed ? buffer : setupCtx.constPols.pConstPolsAddress;
    Polinomial p;
    setupCtx.starkInfo.getPolynomial(p, pols, committed, polId, false);

    Polinomial pCol;
    Goldilocks::Element *pBuffCol = new Goldilocks::Element[polInfo.dim * N];
    pCol.potConstruct(pBuffCol, N, polInfo.dim, polInfo.dim);
    Polinomial::copy(pCol, p);

    cout << "--------------------" << endl;
    string type = committed ? "witness" : "fixed";
    cout << "Printing " << type << " column: " << polInfo.name;
    if(polInfo.lengths.size() > 0) {
        cout << "[";
        for(uint64_t i = 0; i < polInfo.lengths.size(); ++i) {
            cout << polInfo.lengths[i];
            if(i != polInfo.lengths.size() - 1) cout << ", ";
        }
        cout << "]";
    }
    cout << " (pol id " << polId << ")" << endl;
    printExpression(pBuffCol, polInfo.dim, firstPrintValue, lastPrintValue);
    delete pBuffCol;
}

HintFieldInfo printByName(SetupCtx& setupCtx, StepsParams& params, string name, uint64_t *lengths, uint64_t firstPrintValue, uint64_t lastPrintValue, bool returnValues) {
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;

    HintFieldInfo hintFieldInfo;
    hintFieldInfo.size = 0;

    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap.size(); ++i) {
        PolMap cmPol = setupCtx.starkInfo.cmPolsMap[i];
        if(cmPol.name != name) continue;
        if(cmPol.lengths.size() > 0) {
            bool lengths_match = true;
            for(uint64_t j = 0; j < cmPol.lengths.size(); ++j) {
                if(cmPol.lengths[j] != lengths[j]) {
                    lengths_match = false;
                    break;
                }
            }
            if(!lengths_match) continue;
        }
        if(cmPol.name == name) {
            printColById(setupCtx, params.pols, true, i, firstPrintValue, lastPrintValue);
            if(returnValues) {
                hintFieldInfo.size = cmPol.dim * N;
                hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
                hintFieldInfo.fieldType = cmPol.dim == 1 ? HintFieldType::Column : HintFieldType::ColumnExtended;
                hintFieldInfo.offset = cmPol.dim;
                getPolynomial(setupCtx, params.pols, hintFieldInfo.values, true, i, false);
            }
            return hintFieldInfo;
        } 
    }

    for(uint64_t i = 0; i < setupCtx.starkInfo.constPolsMap.size(); ++i) {
        PolMap constPol = setupCtx.starkInfo.constPolsMap[i];
        if(constPol.name != name) continue;
        if(constPol.lengths.size() > 0) {
            bool lengths_match = true;
            for(uint64_t j = 0; j < constPol.lengths.size(); ++j) {
                if(constPol.lengths[j] != lengths[j]) {
                    lengths_match = false;
                    break;
                }
            }
            if(!lengths_match) continue;
        }
        if(constPol.name == name) {
            printColById(setupCtx, params.pols, false, i, firstPrintValue, lastPrintValue);
            if(returnValues) {
                hintFieldInfo.size = N;
                hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
                hintFieldInfo.fieldType = HintFieldType::Column;
                hintFieldInfo.offset = 1;
                getPolynomial(setupCtx, params.pols, hintFieldInfo.values, false, i, false);
            }
            return hintFieldInfo;
        } 
    }

    for(uint64_t i = 0; i < setupCtx.starkInfo.challengesMap.size(); ++i) {
        PolMap challenge = setupCtx.starkInfo.challengesMap[i];
        if(challenge.name == name) {
            cout << "Printing challenge: " << name << " (stage " << challenge.stage << " and id " << challenge.stageId << "): ";
            cout << "[" << Goldilocks::toString(params.challenges[i*FIELD_EXTENSION]) << " , " << Goldilocks::toString(params.challenges[i*FIELD_EXTENSION + 1]) << " , " << Goldilocks::toString(params.challenges[i*FIELD_EXTENSION + 2]) << "]" << endl;
            if(returnValues) {
                hintFieldInfo.size = FIELD_EXTENSION;
                hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
                hintFieldInfo.fieldType = HintFieldType::FieldExtended;
                hintFieldInfo.offset = FIELD_EXTENSION;
                std::memcpy(hintFieldInfo.values, &params.challenges[FIELD_EXTENSION*i], FIELD_EXTENSION * sizeof(Goldilocks::Element));
            }
            return hintFieldInfo;
        }
    }

    for(uint64_t i = 0; i < setupCtx.starkInfo.publicsMap.size(); ++i) {
        PolMap publicInput = setupCtx.starkInfo.publicsMap[i];
        if(publicInput.name == name) {
            cout << "Printing public: " << name << ": " << Goldilocks::toString(params.publicInputs[i]) << endl;
            if(returnValues) {
                hintFieldInfo.size = 1;
                hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
                hintFieldInfo.values[0] = params.publicInputs[i];
                hintFieldInfo.fieldType = HintFieldType::Field;
                hintFieldInfo.offset = 1;
            }
            return hintFieldInfo;
        }
    }

    for(uint64_t i = 0; i < setupCtx.starkInfo.airgroupValuesMap.size(); ++i) {
        PolMap airgroupValue = setupCtx.starkInfo.airgroupValuesMap[i];
        if(airgroupValue.name == name) {
            cout << "Printing airgroupValue: " << name << ": ";
            cout << "[" << Goldilocks::toString(params.airgroupValues[i*FIELD_EXTENSION]) << " , " << Goldilocks::toString(params.airgroupValues[i*FIELD_EXTENSION + 1]) << " , " << Goldilocks::toString(params.airgroupValues[i*FIELD_EXTENSION + 2]) << "]" << endl;
            if(returnValues) {
                hintFieldInfo.size = FIELD_EXTENSION;
                hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
                hintFieldInfo.fieldType = HintFieldType::FieldExtended;
                hintFieldInfo.offset = FIELD_EXTENSION;
                std::memcpy(hintFieldInfo.values, &params.airgroupValues[FIELD_EXTENSION*i], FIELD_EXTENSION * sizeof(Goldilocks::Element));
            }
            return hintFieldInfo;
        }
    }

    zklog.info("Unknown name " + name);
    exitProcess();
    exit(-1);
}


HintFieldValues getHintField(
    SetupCtx& setupCtx, 
    StepsParams &params,
    uint64_t hintId, 
    std::string hintFieldName, 
    HintFieldOptions& hintOptions
) {

    uint64_t deg = 1 << setupCtx.starkInfo.starkStruct.nBits;

    if(setupCtx.expressionsBin.hints.size() == 0) {
        zklog.error("No hints were found.");
        exitProcess();
        exit(-1);
    }

    Hint hint = setupCtx.expressionsBin.hints[hintId];
    
    auto hintField = std::find_if(hint.fields.begin(), hint.fields.end(), [hintFieldName](const HintField& hintField) {
        return hintField.name == hintFieldName;
    });

    if(hintField == hint.fields.end()) {
        zklog.error("Hint field " + hintFieldName + " not found in hint " + hint.name + ".");
        exitProcess();
        exit(-1);
    }

    HintFieldValues hintFieldValues;
    hintFieldValues.nValues = hintField->values.size();
    hintFieldValues.values = new HintFieldInfo[hintField->values.size()];

    for(uint64_t i = 0; i < hintField->values.size(); ++i) {
        HintFieldValue hintFieldVal = hintField->values[i];
        if(hintOptions.dest && (hintFieldVal.operand != opType::cm && hintFieldVal.operand != opType::airgroupvalue && hintFieldVal.operand != opType::airvalue)) {
            cout << hintFieldName << " " << hintFieldVal.operand << endl;
            zklog.error("Invalid destination.");
            exitProcess();
            exit(-1);
        }

        HintFieldInfo hintFieldInfo;

        if(hintOptions.print_expression) {
            cout << "--------------------------------------------------------" << endl;
            cout << "Hint name " << hintFieldName << " for hint id " << hintId << " is ";
        }
        if(hintFieldVal.operand == opType::cm) {
            uint64_t dim = setupCtx.starkInfo.cmPolsMap[hintFieldVal.id].dim;
            hintFieldInfo.size = deg*dim;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = dim == 1 ? HintFieldType::Column : HintFieldType::ColumnExtended;
            hintFieldInfo.offset = dim;
            if(hintOptions.print_expression) {
                cout << "witness col " << setupCtx.starkInfo.cmPolsMap[hintFieldVal.id].name;
                if(setupCtx.starkInfo.cmPolsMap[hintFieldVal.id].lengths.size() > 0) {
                    cout << "[";
                    for(uint64_t i = 0; i < setupCtx.starkInfo.cmPolsMap[hintFieldVal.id].lengths.size(); ++i) {
                        cout << setupCtx.starkInfo.cmPolsMap[hintFieldVal.id].lengths[i];
                        if(i != setupCtx.starkInfo.cmPolsMap[hintFieldVal.id].lengths.size() - 1) cout << ", ";
                    }
                    cout << "]";
                }
                cout << endl;
            }
            if(!hintOptions.dest) {
                getPolynomial(setupCtx, params.pols, hintFieldInfo.values, true, hintFieldVal.id, false);
                if(hintOptions.inverse) {
                    zklog.error("Inverse not supported still for polynomials");
                    exitProcess();
                }
            } else if(hintOptions.initialize_zeros) {
                memset((uint8_t *)hintFieldInfo.values, 0, hintFieldInfo.size * sizeof(Goldilocks::Element));
            }
        } else if(hintFieldVal.operand == opType::const_) {
            uint64_t dim = setupCtx.starkInfo.constPolsMap[hintFieldVal.id].dim;
            hintFieldInfo.size = deg*dim;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = dim == 1 ? HintFieldType::Column : HintFieldType::ColumnExtended;
            hintFieldInfo.offset = dim;
            if(hintOptions.print_expression) cout << "fixed col" << setupCtx.starkInfo.constPolsMap[hintFieldVal.id].name;
            if(setupCtx.starkInfo.constPolsMap[hintFieldVal.id].lengths.size() > 0) {
                cout << "[";
                for(uint64_t i = 0; i < setupCtx.starkInfo.constPolsMap[hintFieldVal.id].lengths.size(); ++i) {
                    cout << setupCtx.starkInfo.constPolsMap[hintFieldVal.id].lengths[i];
                    if(i != setupCtx.starkInfo.constPolsMap[hintFieldVal.id].lengths.size() - 1) cout << ", ";
                }
                cout << "]";
            }
            cout << endl;
            getPolynomial(setupCtx, params.pols, hintFieldInfo.values, false, hintFieldVal.id, false);
            if(hintOptions.inverse) {
                zklog.error("Inverse not supported still for polynomials");
                exitProcess();
            }
        } else if (hintFieldVal.operand == opType::tmp) {
            uint64_t dim = setupCtx.expressionsBin.expressionsInfo[hintFieldVal.id].destDim;
            hintFieldInfo.size = deg*dim;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = dim == 1 ? HintFieldType::Column : HintFieldType::ColumnExtended;
            hintFieldInfo.offset = dim;
            if(hintOptions.print_expression && setupCtx.expressionsBin.expressionsInfo[hintFieldVal.id].line != "") {
                cout << "the expression with id: " << hintFieldVal.id << " " << setupCtx.expressionsBin.expressionsInfo[hintFieldVal.id].line << endl;
            }
#ifdef __AVX512__
    ExpressionsAvx512 expressionsCtx(setupCtx);
#elif defined(__AVX2__)
    ExpressionsAvx expressionsCtx(setupCtx);
#else
    ExpressionsPack expressionsCtx(setupCtx);
#endif
            expressionsCtx.calculateExpression(params, hintFieldInfo.values, hintFieldVal.id, hintOptions.inverse);
        } else if (hintFieldVal.operand == opType::public_) {
            hintFieldInfo.size = 1;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.values[0] = hintOptions.inverse ? Goldilocks::inv(params.publicInputs[hintFieldVal.id]) : params.publicInputs[hintFieldVal.id];
            hintFieldInfo.fieldType = HintFieldType::Field;
            hintFieldInfo.offset = 1;
            if(hintOptions.print_expression) cout << "public input " << setupCtx.starkInfo.publicsMap[hintFieldVal.id].name << endl;
        } else if (hintFieldVal.operand == opType::number) {
            hintFieldInfo.size = 1;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.values[0] = hintOptions.inverse ? Goldilocks::inv(Goldilocks::fromU64(hintFieldVal.value)) : Goldilocks::fromU64(hintFieldVal.value);
            hintFieldInfo.fieldType = HintFieldType::Field;
            hintFieldInfo.offset = 1;
            if(hintOptions.print_expression) cout << "number " << hintFieldVal.value << endl;
        } else if (hintFieldVal.operand == opType::airgroupvalue) {
            uint64_t dim = setupCtx.starkInfo.airgroupValuesMap[hintFieldVal.id].stage == 1 ? 1 : FIELD_EXTENSION;
            hintFieldInfo.size = dim;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = dim == 1 ? HintFieldType::Field : HintFieldType::FieldExtended;
            hintFieldInfo.offset = FIELD_EXTENSION;
            if(hintOptions.print_expression) cout << "airgroupValue " << setupCtx.starkInfo.airgroupValuesMap[hintFieldVal.id].name << endl;
            if(!hintOptions.dest) {
                if(hintOptions.inverse)  {
                    Goldilocks3::inv((Goldilocks3::Element *)hintFieldInfo.values, (Goldilocks3::Element *)&params.airgroupValues[FIELD_EXTENSION*hintFieldVal.id]);
                } else {
                    std::memcpy(hintFieldInfo.values, &params.airgroupValues[FIELD_EXTENSION*hintFieldVal.id], FIELD_EXTENSION * sizeof(Goldilocks::Element));
                }
            }
        } else if (hintFieldVal.operand == opType::airvalue) {
            uint64_t dim = setupCtx.starkInfo.airValuesMap[hintFieldVal.id].stage == 1 ? 1 : FIELD_EXTENSION;
            hintFieldInfo.size = dim;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = dim == 1 ? HintFieldType::Field : HintFieldType::FieldExtended;
            hintFieldInfo.offset = FIELD_EXTENSION;
            if(hintOptions.print_expression) cout << "airgroupValue " << setupCtx.starkInfo.airValuesMap[hintFieldVal.id].name << endl;
            if(!hintOptions.dest) {
                if(hintOptions.inverse)  {
                    Goldilocks3::inv((Goldilocks3::Element *)hintFieldInfo.values, (Goldilocks3::Element *)&params.airValues[FIELD_EXTENSION*hintFieldVal.id]);
                } else {
                    std::memcpy(hintFieldInfo.values, &params.airValues[FIELD_EXTENSION*hintFieldVal.id], FIELD_EXTENSION * sizeof(Goldilocks::Element));
                }
            }
        } else if (hintFieldVal.operand == opType::challenge) {
            hintFieldInfo.size = FIELD_EXTENSION;
            hintFieldInfo.values = new Goldilocks::Element[hintFieldInfo.size];
            hintFieldInfo.fieldType = HintFieldType::FieldExtended;
            hintFieldInfo.offset = FIELD_EXTENSION;
            if(hintOptions.print_expression) cout << "challenge " << setupCtx.starkInfo.challengesMap[hintFieldVal.id].name << endl;
            if(hintOptions.inverse) {
                Goldilocks3::inv((Goldilocks3::Element *)hintFieldInfo.values, (Goldilocks3::Element *)&params.challenges[FIELD_EXTENSION*hintFieldVal.id]);
            } else {
                std::memcpy(hintFieldInfo.values, &params.challenges[FIELD_EXTENSION*hintFieldVal.id], FIELD_EXTENSION * sizeof(Goldilocks::Element));
            }
        } else if (hintFieldVal.operand == opType::string_) {
            hintFieldInfo.values = nullptr;
            hintFieldInfo.fieldType = HintFieldType::String;
            hintFieldInfo.size = hintFieldVal.stringValue.size();
            hintFieldInfo.stringValue = new uint8_t[hintFieldVal.stringValue.size()];
            std::memcpy(hintFieldInfo.stringValue, hintFieldVal.stringValue.data(), hintFieldVal.stringValue.size());
            hintFieldInfo.offset = 0;
            if(hintOptions.print_expression) cout << "string " << hintFieldVal.stringValue << endl;
        } else {
            zklog.error("Unknown HintFieldType");
            exitProcess();
            exit(-1);
        }

        if(hintOptions.print_expression) cout << "--------------------------------------------------------" << endl;

        hintFieldInfo.matrix_size = hintFieldVal.pos.size();
        hintFieldInfo.pos = new uint64_t[hintFieldInfo.matrix_size];
        for(uint64_t i = 0; i < hintFieldInfo.matrix_size; ++i) {
            hintFieldInfo.pos[i] =  hintFieldVal.pos[i];
        }
        hintFieldValues.values[i] = hintFieldInfo;
    }
    
    return hintFieldValues;
}

uint64_t setHintField(SetupCtx& setupCtx, StepsParams& params, Goldilocks::Element* values, uint64_t hintId, std::string hintFieldName) {
    Hint hint = setupCtx.expressionsBin.hints[hintId];

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
    if(hintFieldVal.operand == opType::cm) {
        setPolynomial(setupCtx, params.pols, values, hintFieldVal.id, false);
    } else if(hintFieldVal.operand == opType::airgroupvalue) {
        if(setupCtx.starkInfo.airgroupValuesMap[hintFieldVal.id].stage > 1) {
            std::memcpy(&params.airgroupValues[FIELD_EXTENSION*hintFieldVal.id], values, FIELD_EXTENSION * sizeof(Goldilocks::Element));
        } else {
           params.airgroupValues[FIELD_EXTENSION*hintFieldVal.id] = values[0]; 
        }
    } else if(hintFieldVal.operand == opType::airvalue) {
        if(setupCtx.starkInfo.airValuesMap[hintFieldVal.id].stage > 1) {
            std::memcpy(&params.airValues[FIELD_EXTENSION*hintFieldVal.id], values, FIELD_EXTENSION * sizeof(Goldilocks::Element));
        } else {
           params.airValues[FIELD_EXTENSION*hintFieldVal.id] = values[0]; 
        }
    } else {
        zklog.error("Only committed pols and airgroupvalues can be set");
        exitProcess();
        exit(-1);  
    }

    return hintFieldVal.id;
}

void opHintFields(SetupCtx& setupCtx, StepsParams& params, Goldilocks::Element* dest, uint64_t offset, uint64_t hintId, std::string hintFieldName1, std::string hintFieldName2,  HintFieldOptions& hintOptions1, HintFieldOptions& hintOptions2) {
    Hint hint = setupCtx.expressionsBin.hints[hintId];

    Dest destStruct(dest, offset);

    std::vector<std::string> names = {hintFieldName1,hintFieldName2};
    std::vector<bool> inverses = {hintOptions1.inverse, hintOptions2.inverse};
    for(uint64_t i = 0; i < names.size(); ++i) {
        std::string name = names[i];
        auto hintField = std::find_if(hint.fields.begin(), hint.fields.end(), [name](const HintField& hintField) {
            return hintField.name == name;
        });
        HintFieldValue hintFieldVal = hintField->values[0];

        if(hintField == hint.fields.end()) {
            zklog.error("Hint field " + name + " not found in hint " + hint.name + ".");
            exitProcess();
            exit(-1);
        }

        if(hintFieldVal.operand == opType::cm) {
            destStruct.addCmPol(setupCtx.starkInfo.cmPolsMap[hintFieldVal.id], inverses[i]);
        } else if(hintFieldVal.operand == opType::number) {
            destStruct.addNumber(hintFieldVal.value, inverses[i]);
        } else if(hintFieldVal.operand == opType::tmp) {
            destStruct.addParams(setupCtx.expressionsBin.expressionsInfo[hintFieldVal.id], inverses[i]);
        } else {
            exitProcess();
        }
    }

#ifdef __AVX512__
    ExpressionsAvx512 expressionsCtx(setupCtx);
#elif defined(__AVX2__)
    ExpressionsAvx expressionsCtx(setupCtx);
#else
    ExpressionsPack expressionsCtx(setupCtx);
#endif

    expressionsCtx.multiplyExpressions(params, destStruct);
}

uint64_t multiplyHintFields(SetupCtx& setupCtx, StepsParams &params, uint64_t hintId, std::string hintFieldNameDest, std::string hintFieldName1, std::string hintFieldName2,  HintFieldOptions &hintOptions1, HintFieldOptions &hintOptions2) {
    if(setupCtx.expressionsBin.hints.size() == 0) {
        zklog.error("No hints were found.");
        exitProcess();
        exit(-1);
    }

    Hint hint = setupCtx.expressionsBin.hints[hintId];

    auto hintFieldDest = std::find_if(hint.fields.begin(), hint.fields.end(), [hintFieldNameDest](const HintField& hintField) {
        return hintField.name == hintFieldNameDest;
    });
    HintFieldValue hintFieldDestVal = hintFieldDest->values[0];

    uint64_t offset = setupCtx.starkInfo.mapSectionsN["cm" + to_string(setupCtx.starkInfo.cmPolsMap[hintFieldDestVal.id].stage)];
    Goldilocks::Element *buff = &params.pols[setupCtx.starkInfo.mapOffsets[std::make_pair("cm" + to_string(setupCtx.starkInfo.cmPolsMap[hintFieldDestVal.id].stage), false)] + setupCtx.starkInfo.cmPolsMap[hintFieldDestVal.id].stagePos];
    
    opHintFields(setupCtx, params, buff, offset, hintId, hintFieldName1, hintFieldName2, hintOptions1, hintOptions2);

    return hintFieldDestVal.id;
}

VecU64Result accHintField(SetupCtx& setupCtx, StepsParams &params, uint64_t hintId, std::string hintFieldNameDest, std::string hintFieldNameAirgroupVal, std::string hintFieldName) {
    Hint hint = setupCtx.expressionsBin.hints[hintId];

    auto hintFieldDest = std::find_if(hint.fields.begin(), hint.fields.end(), [hintFieldNameDest](const HintField& hintField) {
        return hintField.name == hintFieldNameDest;
    });
    HintFieldValue hintFieldDestVal = hintFieldDest->values[0];
    
    HintFieldOptions hintOptions;
    HintFieldValues hintValues = getHintField(setupCtx, params, hintId, hintFieldName, hintOptions);

    Goldilocks::Element *vals = &hintValues.values[0].values[0];

    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;
    for(uint64_t i = 1; i < N; ++i) {
        Goldilocks3::add((Goldilocks3::Element &)vals[i * FIELD_EXTENSION], (Goldilocks3::Element &)vals[i * FIELD_EXTENSION], (Goldilocks3::Element &)vals[(i - 1) * FIELD_EXTENSION]);
    }

    VecU64Result hintIds;
    hintIds.nElements = 2;
    hintIds.ids = new uint64_t[hintIds.nElements];
    hintIds.ids[0] = setHintField(setupCtx, params, vals, hintId, hintFieldNameDest);
    hintIds.ids[1] = setHintField(setupCtx, params, &vals[(N - 1)*FIELD_EXTENSION], hintId, hintFieldNameAirgroupVal);

    delete[] hintValues.values;

    return hintIds;
}

VecU64Result accMulHintFields(SetupCtx& setupCtx, StepsParams &params, uint64_t hintId, std::string hintFieldNameDest, std::string hintFieldNameAirgroupVal, std::string hintFieldName1, std::string hintFieldName2, HintFieldOptions &hintOptions1, HintFieldOptions &hintOptions2) {
    uint64_t N = 1 << setupCtx.starkInfo.starkStruct.nBits;

    Hint hint = setupCtx.expressionsBin.hints[hintId];

    auto hintFieldDest = std::find_if(hint.fields.begin(), hint.fields.end(), [hintFieldNameDest](const HintField& hintField) {
        return hintField.name == hintFieldNameDest;
    });
    HintFieldValue hintFieldDestVal = hintFieldDest->values[0];

    Goldilocks::Element *vals = new Goldilocks::Element[setupCtx.starkInfo.cmPolsMap[hintFieldDestVal.id].dim * N];
    uint64_t offset = 0;
    opHintFields(setupCtx, params, vals, offset, hintId, hintFieldName1, hintFieldName2, hintOptions1, hintOptions2);

    for(uint64_t i = 1; i < N; ++i) {
        Goldilocks3::mul((Goldilocks3::Element &)vals[i * FIELD_EXTENSION], (Goldilocks3::Element &)vals[i * FIELD_EXTENSION], (Goldilocks3::Element &)vals[(i - 1) * FIELD_EXTENSION]);
    }

    VecU64Result hintIds;
    hintIds.nElements = 2;
    hintIds.ids = new uint64_t[hintIds.nElements];
    hintIds.ids[0] = setHintField(setupCtx, params, vals, hintId, hintFieldNameDest);
    hintIds.ids[1] = setHintField(setupCtx, params, &vals[(N - 1)*FIELD_EXTENSION], hintId, hintFieldNameAirgroupVal);

    delete[] vals;

    return hintIds;
}
