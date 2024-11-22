#include "expressions_info.hpp"

bool isHexadecimal(const std::string& str) {
    if (str.size() < 3 || str[0] != '0' || (str[1] != 'x' && str[1] != 'X')) return false;
    return true;
}

ExpressionsInfo::ExpressionsInfo(string starkInfoFile, string expressionsInfofile) : starkInfo(starkInfoFile)
{   
    // Load contents from json file
    json expressionsInfoJson;
    file2json(expressionsInfofile, expressionsInfoJson);
    load(expressionsInfoJson, false);
    prepareExpressionsBin();
}

// For Global Constraints
ExpressionsInfo::ExpressionsInfo(string expressionsInfofile)
{   
    // Load contents from json file
    json expressionsInfoJson;
    file2json(expressionsInfofile, expressionsInfoJson);
    load(expressionsInfoJson, true);
    prepareGlobalExpressionsBin();
}

void ExpressionsInfo::load(json j, bool global)
{
    string hints = global ? "hints" : "hintsInfo";
    for (uint64_t i = 0; i < j[hints].size(); i++)
    {
        HintInfo hintInfo;
        hintInfo.name = j[hints][i]["name"];
        for(uint64_t k = 0; k < j[hints][i]["fields"].size(); k++) {
            HintField_ field;
            field.name = j[hints][i]["fields"][k]["name"];
            for(uint64_t l = 0; l < j[hints][i]["fields"][k]["values"].size(); ++l) {
                HintValues f;
                f.op = string2opType(j[hints][i]["fields"][k]["values"][l]["op"]);
                if(j[hints][i]["fields"][k]["values"][l].contains("id")) f.id = j[hints][i]["fields"][k]["values"][l]["id"];
                if(j[hints][i]["fields"][k]["values"][l].contains("airgroupId")) f.airgroupId = j[hints][i]["fields"][k]["values"][l]["airgroupId"];
                if(j[hints][i]["fields"][k]["values"][l].contains("stageId")) f.stageId = j[hints][i]["fields"][k]["values"][l]["stageId"];
                if(j[hints][i]["fields"][k]["values"][l].contains("rowOffsetIndex")) f.rowOffsetIndex = j[hints][i]["fields"][k]["values"][l]["rowOffsetIndex"];
                if(j[hints][i]["fields"][k]["values"][l].contains("stage")) f.stage = j[hints][i]["fields"][k]["values"][l]["stage"];
                if(j[hints][i]["fields"][k]["values"][l].contains("dim")) f.dim = j[hints][i]["fields"][k]["values"][l]["dim"];
                if(j[hints][i]["fields"][k]["values"][l].contains("commitId")) f.commitId = j[hints][i]["fields"][k]["values"][l]["commitId"];
                if(j[hints][i]["fields"][k]["values"][l].contains("string")) f.string_ = j[hints][i]["fields"][k]["values"][l]["string"];
                if(j[hints][i]["fields"][k]["values"][l].contains("value")) f.value = std::stoull(j[hints][i]["fields"][k]["values"][l]["value"].get<std::string>());
                if(j[hints][i]["fields"][k]["values"][l].contains("pos")) {
                    for(uint64_t p = 0; p < j[hints][i]["fields"][k]["values"][l]["pos"].size(); ++p) {
                        f.pos.push_back(j[hints][i]["fields"][k]["values"][l]["pos"][p]);
                    }
                }
                field.values.push_back(f);
            }
            hintInfo.fields.push_back(field);
        }
        hintsInfo.push_back(hintInfo);
    }

    if(!global) {
        for (uint64_t i = 0; i < j["expressionsCode"].size(); ++i) {
            ExpInfo expInfo;
            if(j["expressionsCode"][i].contains("expId")) expInfo.expId = j["expressionsCode"][i]["expId"];
            if(j["expressionsCode"][i].contains("stage")) expInfo.stage = j["expressionsCode"][i]["stage"];
            if(j["expressionsCode"][i].contains("line")) expInfo.line = j["expressionsCode"][i]["line"];
            expInfo.tmpUsed = j["expressionsCode"][i]["tmpUsed"];

            for(uint64_t k = 0; k < j["expressionsCode"][i]["symbolsUsed"].size(); ++k) {
                HintValues f;
                f.op = string2opType(j["expressionsCode"][i]["symbolsUsed"][k]["op"]);
                if(j["expressionsCode"][i]["symbolsUsed"][k].contains("id")) f.id = j["expressionsCode"][i]["symbolsUsed"][k]["id"];
                if(j["expressionsCode"][i]["symbolsUsed"][k].contains("stageId")) f.stageId = j["expressionsCode"][i]["symbolsUsed"][k]["stageId"];
                if(j["expressionsCode"][i]["symbolsUsed"][k].contains("stage")) f.stage = j["expressionsCode"][i]["symbolsUsed"][k]["stage"];
                if(j["expressionsCode"][i]["symbolsUsed"][k].contains("commitId")) f.commitId = j["expressionsCode"][i]["symbolsUsed"][k]["commitId"];
                expInfo.symbolsUsed.push_back(f);
            }

            for(uint64_t k = 0; k < j["expressionsCode"][i]["code"].size(); ++k) {
                CodeOperation c;
                c.setOperation(j["expressionsCode"][i]["code"][k]["op"]);
                c.dest.type = string2opType(j["expressionsCode"][i]["code"][k]["dest"]["type"]);

                if(j["expressionsCode"][i]["code"][k]["dest"].contains("id")) c.dest.id = j["expressionsCode"][i]["code"][k]["dest"]["id"];  
                if(j["expressionsCode"][i]["code"][k]["dest"].contains("prime")) c.dest.prime = j["expressionsCode"][i]["code"][k]["dest"]["prime"];  
                if(j["expressionsCode"][i]["code"][k]["dest"].contains("dim")) c.dest.dim = j["expressionsCode"][i]["code"][k]["dest"]["dim"];  
                if(j["expressionsCode"][i]["code"][k]["dest"].contains("commitId")) c.dest.commitId = j["expressionsCode"][i]["code"][k]["dest"]["commitId"];  
                for (uint64_t l = 0; l < j["expressionsCode"][i]["code"][k]["src"].size(); l++) {
                    CodeType src;
                    src.type = string2opType(j["expressionsCode"][i]["code"][k]["src"][l]["type"]);
                    if(j["expressionsCode"][i]["code"][k]["src"][l].contains("id")) src.id = j["expressionsCode"][i]["code"][k]["src"][l]["id"];  
                    if(j["expressionsCode"][i]["code"][k]["src"][l].contains("prime")) src.prime = j["expressionsCode"][i]["code"][k]["src"][l]["prime"];  
                    if(j["expressionsCode"][i]["code"][k]["src"][l].contains("dim")) src.dim = j["expressionsCode"][i]["code"][k]["src"][l]["dim"];  
                    if(j["expressionsCode"][i]["code"][k]["src"][l].contains("value")) src.value = std::stoull(j["expressionsCode"][i]["code"][k]["src"][l]["value"].get<std::string>());
                    if(j["expressionsCode"][i]["code"][k]["src"][l].contains("commitId")) src.commitId = j["expressionsCode"][i]["code"][k]["src"][l]["commitId"];
                    if(j["expressionsCode"][i]["code"][k]["src"][l].contains("boundaryId")) src.boundaryId = j["expressionsCode"][i]["code"][k]["src"][l]["boundaryId"];
                    if(j["expressionsCode"][i]["code"][k]["src"][l].contains("airgroupId")) src.airgroupId = j["expressionsCode"][i]["code"][k]["src"][l]["airgroupId"];
                    c.src.push_back(src);
                }
                expInfo.code.push_back(c);
            }

            if(j["expressionsCode"][i].contains("dest")) {
                expInfo.dest.type = string2opType(j["expressionsCode"][i]["dest"]["op"]);
                if(j["expressionsCode"][i]["dest"].contains("id")) expInfo.dest.id = j["expressionsCode"][i]["dest"]["id"];  
                if(j["expressionsCode"][i]["dest"].contains("prime")) expInfo.dest.prime = j["expressionsCode"][i]["dest"]["prime"];  
                if(j["expressionsCode"][i]["dest"].contains("dim")) expInfo.dest.dim = j["expressionsCode"][i]["dest"]["dim"];  
                if(j["expressionsCode"][i]["dest"].contains("commitId")) expInfo.dest.commitId = j["expressionsCode"][i]["dest"]["commitId"];  
            }

            expressionsCode.push_back(expInfo);
        }
    }

    for (uint64_t i = 0; i < j["constraints"].size(); ++i) {
        ExpInfo constraintInfo;
        if(j["constraints"][i].contains("stage")) constraintInfo.stage = j["constraints"][i]["stage"];
        if(j["constraints"][i].contains("line")) constraintInfo.line = j["constraints"][i]["line"];
        if(j["constraints"][i].contains("imPol")) constraintInfo.imPol = j["constraints"][i]["imPol"];
        if(j["constraints"][i].contains("boundary")) {
            Boundary b;
            b.name = j["constraints"][i]["boundary"];
            if(b.name == string("everyFrame")) {
                b.offsetMin = j["constraints"][i]["offsetMin"];
                b.offsetMax = j["constraints"][i]["offsetMax"];
            }
            constraintInfo.boundary = b;
        }
        constraintInfo.tmpUsed = j["constraints"][i]["tmpUsed"];

        for(uint64_t k = 0; k < j["constraints"][i]["symbolsUsed"].size(); ++k) {
            HintValues f;
            f.op = string2opType(j["constraints"][i]["symbolsUsed"][k]["op"]);
            if(j["constraints"][i]["symbolsUsed"][k].contains("id")) f.id = j["constraints"][i]["symbolsUsed"][k]["id"];
            if(j["constraints"][i]["symbolsUsed"][k].contains("stageId")) f.stageId = j["constraints"][i]["symbolsUsed"][k]["stageId"];
            if(j["constraints"][i]["symbolsUsed"][k].contains("stage")) f.stage = j["constraints"][i]["symbolsUsed"][k]["stage"];
            if(j["constraints"][i]["symbolsUsed"][k].contains("commitId")) f.commitId = j["constraints"][i]["symbolsUsed"][k]["commitId"];
            constraintInfo.symbolsUsed.push_back(f);
        }

        for(uint64_t k = 0; k < j["constraints"][i]["code"].size(); ++k) {
            CodeOperation c;
            c.setOperation(j["constraints"][i]["code"][k]["op"]);
            c.dest.type = string2opType(j["constraints"][i]["code"][k]["dest"]["type"]);
            if(j["constraints"][i]["code"][k]["dest"].contains("id")) c.dest.id = j["constraints"][i]["code"][k]["dest"]["id"];
            if(j["constraints"][i]["code"][k]["dest"].contains("prime")) c.dest.prime = j["constraints"][i]["code"][k]["dest"]["prime"];  
            if(j["constraints"][i]["code"][k]["dest"].contains("dim")) c.dest.dim = j["constraints"][i]["code"][k]["dest"]["dim"];  
            if(j["constraints"][i]["code"][k]["dest"].contains("commitId")) c.dest.commitId = j["constraints"][i]["code"][k]["dest"]["commitId"];  
            for (uint64_t l = 0; l < j["constraints"][i]["code"][k]["src"].size(); l++) {
                CodeType src;
                src.type = string2opType(j["constraints"][i]["code"][k]["src"][l]["type"]);
                if(j["constraints"][i]["code"][k]["src"][l].contains("id")) src.id = j["constraints"][i]["code"][k]["src"][l]["id"];  
                if(j["constraints"][i]["code"][k]["src"][l].contains("prime")) src.prime = j["constraints"][i]["code"][k]["src"][l]["prime"];  
                if(j["constraints"][i]["code"][k]["src"][l].contains("airgroupId")) src.airgroupId = j["constraints"][i]["code"][k]["src"][l]["airgroupId"];
                if(j["constraints"][i]["code"][k]["src"][l].contains("dim")) src.dim = j["constraints"][i]["code"][k]["src"][l]["dim"];  
                if(j["constraints"][i]["code"][k]["src"][l].contains("value")) src.value = std::stoull(j["constraints"][i]["code"][k]["src"][l]["value"].get<std::string>());
                if(j["constraints"][i]["code"][k]["src"][l].contains("commitId")) src.commitId = j["constraints"][i]["code"][k]["src"][l]["commitId"];  
                c.src.push_back(src);
            }
            constraintInfo.code.push_back(c);
        }

        constraintsCode.push_back(constraintInfo);
    }
}

std::map<std::pair<std::string, std::vector<std::string>>, uint64_t> ExpressionsInfo::getGlobalOperations() {
    std::map<std::pair<std::string, std::vector<std::string>>, uint64_t> possibleOps;

    uint64_t operationCounter = 0;  // Used to assign a unique ID to each operation

    // Define possible destinations and sources
    std::vector<std::string> possibleDestinationsDim1 = {"tmp1"};
    std::vector<std::string> possibleDestinationsDim3 = {"tmp3"};
    std::vector<std::string> possibleSrcDim1 = {"tmp1", "public", "number"};
    std::vector<std::string> possibleSrcDim3 = {"tmp3", "airgroupvalue", "proofvalue", "challenge"};

    // Dim1 destinations
    for (const auto& dest_type : possibleDestinationsDim1) {
        for (size_t k = 0; k < possibleSrcDim1.size(); ++k) {
            const auto& src0_type = possibleSrcDim1[k];
            for (size_t l = k; l < possibleSrcDim1.size(); ++l) {
                const auto& src1_type = possibleSrcDim1[l];
                possibleOps[std::make_pair(dest_type, std::vector<std::string>{src0_type, src1_type})] = operationCounter++;
            }
        }
    }

    // Dim3 destinations
    for (const auto& dest_type : possibleDestinationsDim3) {
        for (const auto& src0_type : possibleSrcDim3) {
            for (const auto& src1_type : possibleSrcDim1) {
                possibleOps[std::make_pair(dest_type, std::vector<std::string>{src0_type, src1_type})] = operationCounter++;
            }
        }

        for (size_t k = 0; k < possibleSrcDim3.size(); ++k) {
            const auto& src0_type = possibleSrcDim3[k];
            for (size_t l = k; l < possibleSrcDim3.size(); ++l) {
                const auto& src1_type = possibleSrcDim3[l];
                possibleOps[std::make_pair(dest_type, std::vector<std::string>{src0_type, src1_type})] = operationCounter++; // Store with unique ID
            }
        }
    }

    return possibleOps;
}

std::map<std::pair<std::string, std::vector<std::string>>, uint64_t> ExpressionsInfo::getAllOperations() {
    std::map<std::pair<std::string, std::vector<std::string>>, uint64_t> possibleOps;

    uint64_t operationCounter = 0;  // Used to assign a unique ID to each operation

    // Define possible destinations and sources
    std::vector<std::string> possibleDestinationsDim1 = {"tmp1"};
    std::vector<std::string> possibleDestinationsDim3 = {"tmp3"};
    std::vector<std::string> possibleSrcDim1 = {"commit1", "tmp1", "public", "number", "airvalue1"};
    std::vector<std::string> possibleSrcDim3 = {"commit3", "tmp3", "challenge", "airgroupvalue", "airvalue3"};

    // Dim1 destinations
    for (const auto& dest_type : possibleDestinationsDim1) {
        for (size_t k = 0; k < possibleSrcDim1.size(); ++k) {
            const auto& src0_type = possibleSrcDim1[k];
            possibleOps[std::make_pair(dest_type, std::vector<std::string>{src0_type})] = operationCounter++;

            for (size_t l = k; l < possibleSrcDim1.size(); ++l) {
                const auto& src1_type = possibleSrcDim1[l];
                possibleOps[std::make_pair(dest_type, std::vector<std::string>{src0_type, src1_type})] = operationCounter++;
            }
        }
    }

    // Dim3 destinations
    for (const auto& dest_type : possibleDestinationsDim3) {
        for (const auto& src0_type : possibleSrcDim3) {
            for (const auto& src1_type : possibleSrcDim1) {
                possibleOps[std::make_pair(dest_type, std::vector<std::string>{src0_type, src1_type})] = operationCounter++;
            }
        }

        for (size_t k = 0; k < possibleSrcDim3.size(); ++k) {
            const auto& src0_type = possibleSrcDim3[k];
            if (src0_type == "commit3" || src0_type == "tmp3") {
                possibleOps[std::make_pair(dest_type, std::vector<std::string>{src0_type})] = operationCounter++;
            }
            for (size_t l = k; l < possibleSrcDim3.size(); ++l) {
                const auto& src1_type = possibleSrcDim3[l];
                possibleOps[std::make_pair(dest_type, std::vector<std::string>{src0_type, src1_type})] = operationCounter++; // Store with unique ID
            }
        }
    }

    possibleOps[std::make_pair("tmp3", std::vector<std::string>{"eval"})] = operationCounter++;
    possibleOps[std::make_pair("tmp3", std::vector<std::string>{"challenge", "eval"})] = operationCounter++;
    possibleOps[std::make_pair("tmp3", std::vector<std::string>{"tmp3", "eval"})] = operationCounter++;
    possibleOps[std::make_pair("tmp3", std::vector<std::string>{"eval", "commit1"})] = operationCounter++;
    possibleOps[std::make_pair("tmp3", std::vector<std::string>{"commit3", "eval"})] = operationCounter++;
    possibleOps[std::make_pair("tmp3", std::vector<std::string>{"eval", "eval"})] = operationCounter++;
    possibleOps[std::make_pair("tmp3", std::vector<std::string>{"eval", "public"})] = operationCounter++;
    possibleOps[std::make_pair("tmp3", std::vector<std::string>{"eval", "number"})] = operationCounter++;
    possibleOps[std::make_pair("tmp3", std::vector<std::string>{"airgroupvalue", "eval"})] = operationCounter++;

    return possibleOps;
}


bool isIntersecting(const std::vector<int64_t>& segment1, const std::vector<int64_t>& segment2) {
    return segment2[0] < segment1[1] && segment1[0] < segment2[1];
}

std::vector<std::vector<std::vector<int64_t>>> temporalsSubsets(std::vector<std::vector<int64_t>>& segments) {
    // Sort segments by their ending position
    std::stable_sort(segments.begin(), segments.end(),
              [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) { return a[1] < b[1]; });
   
    std::vector<std::vector<std::vector<int64_t>>> tmpSubsets;

    for (const auto& segment : segments) {
        int closestSubsetIndex = -1; // No closest subset yet
        int64_t minDistance = 10000000;

        for (uint64_t i = 0; i < tmpSubsets.size(); ++i) {
            const auto& subset = tmpSubsets[i];
            const auto& lastSegmentSubset = subset.back();

            if (isIntersecting(segment, lastSegmentSubset)) {
                continue;
            }

            int64_t distance = std::abs(lastSegmentSubset[1] - segment[0]);
            if (distance < minDistance) {
                minDistance = distance;
                closestSubsetIndex = i;
            }
        }

        if (closestSubsetIndex != -1) {
            // Add to the closest subset
            tmpSubsets[closestSubsetIndex].push_back(segment);
        } else {
            // Create a new subset
            tmpSubsets.push_back({segment});
        }
    }

    return tmpSubsets;
}

std::pair<int64_t, int64_t> getIdMaps(uint64_t maxid, std::vector<int64_t>& ID1D, std::vector<int64_t>& ID3D, const std::vector<CodeOperation>& code) {
    std::vector<int64_t> Ini1D(maxid, -1);
    std::vector<int64_t> End1D(maxid, -1);
    std::vector<int64_t> Ini3D(maxid, -1);
    std::vector<int64_t> End3D(maxid, -1);

    // Explore all the code to find the first and last appearance of each tmp
    for (uint64_t j = 0; j < code.size(); ++j) {
        const auto& r = code[j];
        if (r.dest.type == opType::tmp) {
            uint64_t id_ = r.dest.id;
            uint64_t dim_ = r.dest.dim;
            assert(id_ >= 0 && id_ < maxid);

            if (dim_ == 1) {
                if (Ini1D[id_] == -1) {
                    Ini1D[id_] = j;
                    End1D[id_] = j;
                } else {
                    End1D[id_] = j;
                }
            } else {
                assert(dim_ == 3);
                if (Ini3D[id_] == -1) {
                    Ini3D[id_] = j;
                    End3D[id_] = j;
                } else {
                    End3D[id_] = j;
                }
            }
        }

        for(uint64_t k = 0; k < r.src.size(); ++k) {
            if (r.src[k].type == opType::tmp) {
                uint64_t id_ = r.src[k].id;
                uint64_t dim_ = r.src[k].dim;
                assert(id_ >= 0 && id_ < maxid);

                if (dim_ == 1) {
                    if (Ini1D[id_] == -1) {
                        Ini1D[id_] = j;
                        End1D[id_] = j;
                    } else {
                        End1D[id_] = j;
                    }
                } else {
                    assert(dim_ == 3);
                    if (Ini3D[id_] == -1) {
                        Ini3D[id_] = j;
                        End3D[id_] = j;
                    } else {
                        End3D[id_] = j;
                    }
                }
            }
        }
    }

    // Store, for each temporal ID, its first and last appearance in the following form: [first, last, id]
    std::vector<std::vector<int64_t>> segments1D;
    std::vector<std::vector<int64_t>> segments3D;
    for (int64_t j = 0; j < int64_t(maxid); j++) {
        if (Ini1D[j] >= 0) {
            segments1D.push_back({Ini1D[j], End1D[j], j});
        }
        if (Ini3D[j] >= 0) {
            segments3D.push_back({Ini3D[j], End3D[j], j});
        }
    }

    // Create subsets of non-intersecting segments for basefield and extended field temporal variables
    auto subsets1D = temporalsSubsets(segments1D);
    auto subsets3D = temporalsSubsets(segments3D);

    // Assign unique numerical IDs to subsets of segments representing 1D and 3D temporal variables
    uint64_t count1d = 0;
    for (const auto& s : subsets1D) {
        for (const auto& a : s) {
            ID1D[a[2]] = count1d;
        }
        ++count1d;
    }

    uint64_t count3d = 0;
    for (const auto& s : subsets3D) {
        for (const auto& a : s) {
            ID3D[a[2]] = count3d;
        }
        ++count3d;
    }

    return {count1d, count3d};
}

std::string getType(const CodeType& r, bool verify) {
    if (r.type == opType::cm) {
        return "commit" + std::to_string(r.dim);
    } else if (r.type == opType::const_ ||
               (r.type == opType::custom && r.dim == 1) ||
               ((r.type == opType::Zi || r.type == opType::x) && !verify)) {
        return "commit1";
    } else if (r.type == opType::xDivXSubXi ||
               (r.type == opType::custom && r.dim == 3) ||
               ((r.type == opType::Zi || r.type == opType::x) && verify)) {
        return "commit3";
    } else if (r.type == opType::tmp) {
        return "tmp" + std::to_string(r.dim);
    } else if (r.type == opType::airvalue) {
        return "airvalue" + std::to_string(r.dim);
    } else {
        return opType2string(r.type); // Assuming this function converts opType to string
    }
}

CodeOperation getOperation(CodeOperation &r, bool verify = false) {
    std::map<std::string, uint64_t> operationsMap = {
        {"commit1", 1},
        {"x", 1},
        {"Zi", 1},
        {"const", 1},
        {"custom1", 1},
        {"tmp1", 2},
        {"public", 3},
        {"number", 4},
        {"airvalue1", 5},
        {"custom3", 6},
        {"commit3", 6},
        {"xDivXSubXi", 6},
        {"tmp3", 7},
        {"airvalue3", 8},
        {"airgroupvalue", 9},
        {"proofvalue", 10},
        {"challenge", 11},
        {"eval", 12}
    };

    CodeOperation codeOp;
    codeOp.op = r.op;

    codeOp.dest = r.dest;

    // Determine destination type
    if (r.dest.type == opType::cm) {
        codeOp.dest_type = "commit" + std::to_string(r.dest.dim);
    } else if (r.dest.type == opType::tmp) {
        codeOp.dest_type = "tmp" + std::to_string(r.dest.dim);
    } else {
        codeOp.dest_type = r.dest.type;
    }

    if(codeOp.op == 4) {
        codeOp.src = r.src;
    } else {
        CodeType a = r.src[0];
        CodeType b = r.src[1];
        int64_t opA = (a.type == opType::cm)      ? operationsMap["commit" + std::to_string(a.dim)] :
            (a.type == opType::tmp)   ? operationsMap["tmp" + std::to_string(a.dim)] :
            (a.type == opType::airvalue) ? operationsMap["airvalue" + std::to_string(a.dim)] :
            (a.type == opType::custom)   ? operationsMap["custom" + std::to_string(a.dim)] :
            operationsMap[opType2string(a.type)];

        int64_t opB = (b.type == opType::cm)      ? operationsMap["commit" + std::to_string(b.dim)] :
            (b.type == opType::tmp)   ? operationsMap["tmp" + std::to_string(b.dim)] :
            (b.type == opType::airvalue) ? operationsMap["airvalue" + std::to_string(b.dim)] :
            (b.type == opType::custom)   ? operationsMap["custom" + std::to_string(b.dim)] :
            operationsMap[opType2string(b.type)];
        bool swap = (a.dim != b.dim) ? (b.dim > a.dim) : (opA > opB);
        if (swap) {
            codeOp.src.push_back(r.src[1]);
            codeOp.src.push_back(r.src[0]);
            if(codeOp.op == 1) codeOp.setOperation("sub_swap");
        } else {
            codeOp.src = r.src;
        }
    }


    for (size_t i = 0; i < codeOp.src.size(); ++i) {
        codeOp.src_types.push_back(getType(codeOp.src[i], verify));
    }
    return codeOp;
}

void ExpressionsInfo::pushArgs(vector<uint64_t> &args, CodeType &r, vector<int64_t> &ID1D, vector<int64_t> &ID3D, vector<uint64_t> &numbers, bool dest, bool verify, bool global, uint64_t nStages) {
    if(dest && r.type != opType::tmp && r.type != opType::cm) {
        zklog.error("Invalid dest type=" + r.type);
        exitProcess();
        exit(-1);
    }

    if (r.type == opType::tmp) {
        if (r.dim == 1) {
            args.push_back(ID1D[r.id]);
        } else {
            assert(r.dim == 3);
            args.push_back(ID3D[r.id]);
        }
    } 
    else if (r.type == opType::const_) {
        auto primeIndex = std::find(starkInfo.openingPoints.begin(), starkInfo.openingPoints.end(), r.prime);
        if (primeIndex == starkInfo.openingPoints.end()) {
            throw std::runtime_error("Something went wrong");
        }

        uint64_t index = std::distance(starkInfo.openingPoints.begin(), primeIndex);
        if (verify) {
            args.push_back(0);
        } else {
            args.push_back(nStages * index);
        }
        args.push_back(r.id);

    } 
    else if (r.type == opType::custom) {
        auto primeIndex = std::find(starkInfo.openingPoints.begin(), starkInfo.openingPoints.end(), r.prime);
        if (primeIndex == starkInfo.openingPoints.end()) {
            throw std::runtime_error("Something went wrong");
        }

        uint64_t index = std::distance(starkInfo.openingPoints.begin(), primeIndex);
        args.push_back(nStages * index + starkInfo.nStages + 2 + r.commitId);
        args.push_back(r.id);
    } 
    else if (r.type == opType::cm) {
        auto primeIndex = std::find(starkInfo.openingPoints.begin(), starkInfo.openingPoints.end(), r.prime);
        if (primeIndex == starkInfo.openingPoints.end()) {
            throw std::runtime_error("Something went wrong");
        }

        uint64_t index = std::distance(starkInfo.openingPoints.begin(), primeIndex);
        if (verify) {
            args.push_back(starkInfo.cmPolsMap[r.id].stage);
        } else {
            args.push_back(nStages * index + starkInfo.cmPolsMap[r.id].stage);
        }
        args.push_back(starkInfo.cmPolsMap[r.id].stagePos);
    } 
    else if (r.type == opType::number) {
        auto it = std::find(numbers.begin(), numbers.end(), r.value);
        uint64_t numberPos;
        if (it == numbers.end()) {
            numberPos = numbers.size();
            numbers.push_back(r.value);
        } else {
            numberPos = std::distance(numbers.begin(), it);
        }
        args.push_back(numberPos);
    } 
    else if (r.type == opType::public_ || r.type == opType::eval || r.type == opType::proofvalue || r.type == opType::airvalue || r.type == opType::challenge) {
        args.push_back(r.id);
    } 
    else if (r.type == opType::airgroupvalue) {
        if (!global) {
            args.push_back(r.id);
        } else {
            args.push_back(r.airgroupId);
            args.push_back(r.id);
        }
    } 
    else if (r.type == opType::xDivXSubXi) {
        if (verify) {
            args.push_back(nStages);
            args.push_back(3 * r.id);
        } else {
            args.push_back(nStages * starkInfo.openingPoints.size());
            args.push_back(3 * r.id);
        }
    } 
    else if (r.type == opType::Zi) {
        if (verify) {
            args.push_back(nStages);
            args.push_back(3 + 3 * r.boundaryId);
        } else {
            args.push_back(nStages * starkInfo.openingPoints.size());
            args.push_back(1 + r.boundaryId);
        }
    } 
    else if (r.type == opType::x) {
        if (verify) {
            args.push_back(nStages);
            args.push_back(0);
        } else {
            args.push_back(nStages * starkInfo.openingPoints.size());
            args.push_back(0);
        }
    } 
    else {
        throw std::invalid_argument("Unknown type " + r.type);
    }

}

ExpInfoBin ExpressionsInfo::getParserArgs(std::map<std::pair<std::string, std::vector<std::string>>, uint64_t> &operations, std::vector<CodeOperation> &code, std::vector<HintValues> &symbolsUsed, uint64_t nTmpUsed, std::vector<uint64_t> &numbers, bool global, bool verify) {
    ExpInfoBin expInfoBin;

    uint64_t nStages = starkInfo.nStages + 2;
    if(!global) nStages += starkInfo.customCommits.size();

    uint64_t maxid = nTmpUsed;
    std::vector<int64_t> ID1D(maxid, -1);
    std::vector<int64_t> ID3D(maxid, -1);

    std::tie(expInfoBin.nTemp1, expInfoBin.nTemp3) = getIdMaps(maxid, ID1D, ID3D, code);


    for(uint64_t i = 0; i < code.size(); ++i) {
        CodeOperation operation = getOperation(code[i], verify);
        if(operation.op != CodeOperation::eOperation::copy) {
            uint64_t arg = operation.operationArg(operation.op);
            expInfoBin.args.push_back(arg);
        }

        
        pushArgs(expInfoBin.args, operation.dest, ID1D, ID3D, numbers, true, verify, global, nStages);
        for(uint64_t j = 0; j < operation.src.size(); ++j) {
            pushArgs(expInfoBin.args, operation.src[j], ID1D, ID3D, numbers, false, verify, global, nStages);
        }

        auto opsIndex = operations.find(std::make_pair(operation.dest_type, operation.src_types));
        if (opsIndex == operations.end()) {
            throw std::runtime_error("Operation not considered: " + operation.dest_type + " " + operation.src_types[0] + " " + operation.src_types[1]);
        }

        expInfoBin.ops.push_back(opsIndex->second);
    }
    if (!symbolsUsed.empty()) {
        // Filter and sort symbols based on the operation type
        for (const auto& symbol : symbolsUsed) {
            if (symbol.op == opType::const_) {
                expInfoBin.constPolsIds.push_back(symbol.id);
            } else if (symbol.op == opType::cm) {
                expInfoBin.cmPolsIds.push_back(symbol.id);
            } else if (symbol.op == opType::challenge) {
                expInfoBin.challengeIds.push_back(symbol.id);
            } else if (symbol.op == opType::public_) {
                expInfoBin.publicsIds.push_back(symbol.id);
            } else if (symbol.op == opType::airgroupvalue) {
                expInfoBin.airgroupValuesIds.push_back(symbol.id);
            } else if (symbol.op == opType::airvalue) {
                expInfoBin.airValuesIds.push_back(symbol.id);
            }
        }

        // Sort the vectors
        std::sort(expInfoBin.constPolsIds.begin(), expInfoBin.constPolsIds.end());
        std::sort(expInfoBin.cmPolsIds.begin(), expInfoBin.cmPolsIds.end());
        std::sort(expInfoBin.challengeIds.begin(), expInfoBin.challengeIds.end());
        std::sort(expInfoBin.publicsIds.begin(), expInfoBin.publicsIds.end());
        std::sort(expInfoBin.airgroupValuesIds.begin(), expInfoBin.airgroupValuesIds.end());
        std::sort(expInfoBin.airValuesIds.begin(), expInfoBin.airValuesIds.end());

        // Process custom values
        for (uint64_t i = 0; i < starkInfo.customCommits.size(); ++i) {
            std::vector<uint64_t> customIds;
            for (const auto& symbol : symbolsUsed) {
                if (symbol.op == opType::custom && symbol.commitId == i) {
                    customIds.push_back(symbol.id);
                }
            }
            std::sort(customIds.begin(), customIds.end());
            expInfoBin.customValuesIds.push_back(customIds);
        }
    }

    if(code[code.size() - 1].dest.dim == 1) {
        expInfoBin.destDim = 1;
        expInfoBin.destId = ID1D[code[code.size() - 1].dest.id];
    } else {
        assert(code[code.size() - 1].dest.dim == 3);
        expInfoBin.destDim = 3;
        expInfoBin.destId = ID3D[code[code.size() - 1].dest.id];
    }
    
    return expInfoBin;
}

void ExpressionsInfo::prepareGlobalExpressionsBin() {
    std::map<std::pair<std::string, std::vector<std::string>>, uint64_t> operations = getGlobalOperations();

    for(uint64_t j = 0; j < constraintsCode.size(); ++j) {
        ExpInfo constraint = constraintsCode[j];
        ExpInfoBin globalConstraintInfo = getParserArgs(operations, constraint.code, constraint.symbolsUsed, constraint.tmpUsed, numbersConstraints, true, false);
        globalConstraintInfo.line = constraint.line;
        constraintsInfo.push_back(globalConstraintInfo);
    }
}

void ExpressionsInfo::prepareExpressionsBin() {
    uint64_t N = (1 << starkInfo.starkStruct.nBits);

    std::map<std::pair<std::string, std::vector<std::string>>, uint64_t> operations = getAllOperations();
    for(uint64_t j = 0; j < constraintsCode.size(); ++j) {
        ExpInfo constraint = constraintsCode[j];
        uint64_t firstRow;
        uint64_t lastRow;

        if(constraint.boundary.name == "everyRow") {
            firstRow = 0;
            lastRow = N;
        } else if(constraint.boundary.name == "lastRow") {
            firstRow = N - 1;
            lastRow = N;
        } else if(constraint.boundary.name == "firstRow" || constraint.boundary.name == "finalProof") {
            firstRow = 0;
            lastRow = 1;
        } else if(constraint.boundary.name == "everyFrame") {
            firstRow = constraint.boundary.offsetMin;
            lastRow = constraint.boundary.offsetMax;
        } else {
            zklog.error("Invalid boundary=" + constraint.boundary.name);
            exitProcess();
            exit(-1);
        }

        ExpInfoBin constraintInfo = getParserArgs(operations, constraint.code, constraint.symbolsUsed, constraint.tmpUsed, numbersConstraints, false, false);
        constraintInfo.stage = constraint.stage;
        constraintInfo.firstRow = firstRow;
        constraintInfo.lastRow = lastRow;
        constraintInfo.line = constraint.line;
        constraintInfo.imPol = constraint.imPol;
        constraintsInfo.push_back(constraintInfo);
    }

    for(uint64_t i = 0; i < expressionsCode.size(); ++i) {
        ExpInfo expCode = expressionsCode[i];
        bool expr = false;
        for(uint64_t j = 0; j < starkInfo.cmPolsMap.size(); ++j) {
            if(starkInfo.cmPolsMap[j].expId == expCode.expId) {
                expr = true;
                break;
            }
        }

        if(expCode.expId == starkInfo.cExpId || expCode.expId == starkInfo.friExpId || expr) {
            expCode.code[expCode.code.size() - 1].dest.type = opType::tmp;
            expCode.code[expCode.code.size() - 1].dest.id = expCode.tmpUsed++;
        }
        ExpInfoBin expressionInfo = getParserArgs(operations, expCode.code, expCode.symbolsUsed, expCode.tmpUsed, numbersExps, false, false);

        expressionInfo.expId = expCode.expId;
        expressionInfo.stage = expCode.stage;
        expressionInfo.line = expCode.line;
        expressionsInfo.push_back(expressionInfo);
    }
}