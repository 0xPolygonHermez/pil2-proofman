#include "expressions_info.hpp"

bool isHexadecimal(const std::string& str) {
    if (str.size() < 3 || str[0] != '0' || (str[1] != 'x' && str[1] != 'X')) return false;
    return true;
}

ExpressionsInfo::ExpressionsInfo(string starkInfoFile, string expressionsInfofile, bool verifier) : starkInfo(starkInfoFile)
{   
    // Load contents from json file
    json expressionsInfoJson;
    file2json(expressionsInfofile, expressionsInfoJson);
    load(expressionsInfoJson, false, verifier);
    if(verifier) {
        prepareVerifierExpressionsBin();
    } else {
        prepareExpressionsBin();
    }
}

// For Global Constraints
ExpressionsInfo::ExpressionsInfo(string expressionsInfofile)
{   
    // Load contents from json file
    json expressionsInfoJson;
    file2json(expressionsInfofile, expressionsInfoJson);
    load(expressionsInfoJson, true, false);
    prepareGlobalExpressionsBin();
}

void ExpressionsInfo::load(json j, bool global, bool verifier)
{
    if(verifier) {
        cout << "Loading verifier expressions info..." << endl;
        ExpInfo qExpInfo;
        qExpInfo.expId = starkInfo.cExpId;
        qExpInfo.stage = starkInfo.nStages + 1;
        qExpInfo.line = "";
        qExpInfo.tmpUsed = j["qVerifier"]["tmpUsed"];

        for(uint64_t k = 0; k < j["qVerifier"]["code"].size(); ++k) {
            CodeOperation c;
            c.setOperation(j["qVerifier"]["code"][k]["op"]);
            c.dest.type = string2opType(j["qVerifier"]["code"][k]["dest"]["type"]);

            if(j["qVerifier"]["code"][k]["dest"].contains("id")) c.dest.id = j["qVerifier"]["code"][k]["dest"]["id"];  
            if(j["qVerifier"]["code"][k]["dest"].contains("prime")) c.dest.prime = j["qVerifier"]["code"][k]["dest"]["prime"];  
            if(j["qVerifier"]["code"][k]["dest"].contains("dim")) c.dest.dim = j["qVerifier"]["code"][k]["dest"]["dim"];  
            if(j["qVerifier"]["code"][k]["dest"].contains("commitId")) c.dest.commitId = j["qVerifier"]["code"][k]["dest"]["commitId"];  
            for (uint64_t l = 0; l < j["qVerifier"]["code"][k]["src"].size(); l++) {
                CodeType src;
                src.type = string2opType(j["qVerifier"]["code"][k]["src"][l]["type"]);
                if(j["qVerifier"]["code"][k]["src"][l].contains("id")) src.id = j["qVerifier"]["code"][k]["src"][l]["id"];  
                if(j["qVerifier"]["code"][k]["src"][l].contains("prime")) src.prime = j["qVerifier"]["code"][k]["src"][l]["prime"];  
                if(j["qVerifier"]["code"][k]["src"][l].contains("dim")) src.dim = j["qVerifier"]["code"][k]["src"][l]["dim"];  
                if(j["qVerifier"]["code"][k]["src"][l].contains("value")) src.value = std::stoull(j["qVerifier"]["code"][k]["src"][l]["value"].get<std::string>());
                if(j["qVerifier"]["code"][k]["src"][l].contains("commitId")) src.commitId = j["qVerifier"]["code"][k]["src"][l]["commitId"];
                if(j["qVerifier"]["code"][k]["src"][l].contains("boundaryId")) src.boundaryId = j["qVerifier"]["code"][k]["src"][l]["boundaryId"];
                if(j["qVerifier"]["code"][k]["src"][l].contains("airgroupId")) src.airgroupId = j["qVerifier"]["code"][k]["src"][l]["airgroupId"];
                c.src.push_back(src);
            }
            qExpInfo.code.push_back(c);
        }

        if(j["qVerifier"].contains("dest")) {
            qExpInfo.dest.type = string2opType(j["qVerifier"]["dest"]["op"]);
            if(j["qVerifier"]["dest"].contains("id")) qExpInfo.dest.id = j["qVerifier"]["dest"]["id"];  
            if(j["qVerifier"]["dest"].contains("prime")) qExpInfo.dest.prime = j["qVerifier"]["dest"]["prime"];  
            if(j["qVerifier"]["dest"].contains("dim")) qExpInfo.dest.dim = j["qVerifier"]["dest"]["dim"];  
            if(j["qVerifier"]["dest"].contains("commitId")) qExpInfo.dest.commitId = j["qVerifier"]["dest"]["commitId"];  
        }

        expressionsCode.push_back(qExpInfo);

        ExpInfo queryExpInfo;
        queryExpInfo.expId = starkInfo.friExpId;
        queryExpInfo.stage = starkInfo.nStages + 2;
        queryExpInfo.line = "";
        queryExpInfo.tmpUsed = j["queryVerifier"]["tmpUsed"];

        for(uint64_t k = 0; k < j["queryVerifier"]["code"].size(); ++k) {
            CodeOperation c;
            c.setOperation(j["queryVerifier"]["code"][k]["op"]);
            c.dest.type = string2opType(j["queryVerifier"]["code"][k]["dest"]["type"]);

            if(j["queryVerifier"]["code"][k]["dest"].contains("id")) c.dest.id = j["queryVerifier"]["code"][k]["dest"]["id"];  
            if(j["queryVerifier"]["code"][k]["dest"].contains("prime")) c.dest.prime = j["queryVerifier"]["code"][k]["dest"]["prime"];  
            if(j["queryVerifier"]["code"][k]["dest"].contains("dim")) c.dest.dim = j["queryVerifier"]["code"][k]["dest"]["dim"];  
            if(j["queryVerifier"]["code"][k]["dest"].contains("commitId")) c.dest.commitId = j["queryVerifier"]["code"][k]["dest"]["commitId"];  
            for (uint64_t l = 0; l < j["queryVerifier"]["code"][k]["src"].size(); l++) {
                CodeType src;
                src.type = string2opType(j["queryVerifier"]["code"][k]["src"][l]["type"]);
                if(j["queryVerifier"]["code"][k]["src"][l].contains("id")) src.id = j["queryVerifier"]["code"][k]["src"][l]["id"];  
                if(j["queryVerifier"]["code"][k]["src"][l].contains("prime")) src.prime = j["queryVerifier"]["code"][k]["src"][l]["prime"];  
                if(j["queryVerifier"]["code"][k]["src"][l].contains("dim")) src.dim = j["queryVerifier"]["code"][k]["src"][l]["dim"];  
                if(j["queryVerifier"]["code"][k]["src"][l].contains("value")) src.value = std::stoull(j["queryVerifier"]["code"][k]["src"][l]["value"].get<std::string>());
                if(j["queryVerifier"]["code"][k]["src"][l].contains("commitId")) src.commitId = j["queryVerifier"]["code"][k]["src"][l]["commitId"];
                if(j["queryVerifier"]["code"][k]["src"][l].contains("boundaryId")) src.boundaryId = j["queryVerifier"]["code"][k]["src"][l]["boundaryId"];
                if(j["queryVerifier"]["code"][k]["src"][l].contains("airgroupId")) src.airgroupId = j["queryVerifier"]["code"][k]["src"][l]["airgroupId"];
                c.src.push_back(src);
            }
            queryExpInfo.code.push_back(c);
        }

        if(j["queryVerifier"].contains("dest")) {
            queryExpInfo.dest.type = string2opType(j["queryVerifier"]["dest"]["op"]);
            if(j["queryVerifier"]["dest"].contains("id")) queryExpInfo.dest.id = j["queryVerifier"]["dest"]["id"];  
            if(j["queryVerifier"]["dest"].contains("prime")) queryExpInfo.dest.prime = j["queryVerifier"]["dest"]["prime"];  
            if(j["queryVerifier"]["dest"].contains("dim")) queryExpInfo.dest.dim = j["queryVerifier"]["dest"]["dim"];  
            if(j["queryVerifier"]["dest"].contains("commitId")) queryExpInfo.dest.commitId = j["queryVerifier"]["dest"]["commitId"];  
        }

        expressionsCode.push_back(queryExpInfo);
        
        cout << "Loaded qVerifier and queryVerifier expressions" << endl;
        return;
    }
    

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

CodeOperation getOperation(CodeOperation &r) {
    std::map<std::string, uint64_t> operationsMap = {
        {"commit1", 0},
        {"Zi", 0},
        {"const", 0},
        {"custom1", 0},
        {"tmp1", 1},
        {"public", 2},
        {"number", 3},
        {"airvalue1", 4},
        {"proofvalue1",5},
        {"custom3", 6},
        {"commit3", 6},
        {"xDivXSubXi", 6},
        {"tmp3", 7},
        {"airvalue3", 8},
        {"airgroupvalue", 9},
        {"proofvalue", 10},
        {"proofvalue3", 10},
        {"challenge", 11},
        {"eval", 12}
    };

    CodeOperation codeOp;
    codeOp.op = r.op;

    codeOp.dest = r.dest;

    codeOp.dest_dim = r.dest.dim;
    

    
    CodeType a = r.src[0];
    CodeType b = r.src[1];
    int64_t opA = (a.type == opType::cm)      ? operationsMap["commit" + std::to_string(a.dim)] :
        (a.type == opType::tmp)   ? operationsMap["tmp" + std::to_string(a.dim)] :
        (a.type == opType::airvalue) ? operationsMap["airvalue" + std::to_string(a.dim)] :
        (a.type == opType::custom)   ? operationsMap["custom" + std::to_string(a.dim)] :
        (a.type == opType::proofvalue) ? operationsMap["proofvalue" + std::to_string(a.dim)] :
        operationsMap[opType2string(a.type)];

    int64_t opB = (b.type == opType::cm)      ? operationsMap["commit" + std::to_string(b.dim)] :
        (b.type == opType::tmp)   ? operationsMap["tmp" + std::to_string(b.dim)] :
        (b.type == opType::airvalue) ? operationsMap["airvalue" + std::to_string(b.dim)] :
        (b.type == opType::custom)   ? operationsMap["custom" + std::to_string(b.dim)] :
        (b.type == opType::proofvalue) ? operationsMap["proofvalue" + std::to_string(b.dim)] :
        operationsMap[opType2string(b.type)];
    bool swap = (a.dim != b.dim) ? (b.dim > a.dim) : (opA > opB);
    if (swap) {
        codeOp.src.push_back(r.src[1]);
        codeOp.src.push_back(r.src[0]);
        if(codeOp.op == 1) codeOp.setOperation("sub_swap");
    } else {
        codeOp.src = r.src;
    }

    codeOp.src0_dim = codeOp.src[0].dim;
    codeOp.src1_dim = codeOp.src[1].dim;
   
    return codeOp;
}

void ExpressionsInfo::pushArgs(vector<uint64_t> &args, CodeType &r, vector<int64_t> &ID1D, vector<int64_t> &ID3D, vector<uint64_t> &numbers, bool dest, bool global) {
    if(dest && r.type != opType::tmp && r.type != opType::cm) {
        zklog.error("Invalid dest type=" + opType2string(r.type));
        exitProcess();
        exit(-1);
    }

    uint32_t bufferSize = 1 + starkInfo.nStages + 3 + starkInfo.customCommits.size();
    if (r.type == opType::tmp) {
        if (r.dim == 1) {
            if (!dest) {
                if (!global) {
                    args.push_back(bufferSize);
                }    
            }
            args.push_back(ID1D[r.id]);
        } else {
            assert(r.dim == 3);
            if (!dest) {
                if (!global) {
                    args.push_back(bufferSize + 1);
                } else {
                    args.push_back(4);
                }
            }
            args.push_back(3*ID3D[r.id]);
        }
        if(!dest && !global) args.push_back(0);
    } 
    else if (r.type == opType::const_) {
        auto primeIndex = std::find(starkInfo.openingPoints.begin(), starkInfo.openingPoints.end(), r.prime);
        if (primeIndex == starkInfo.openingPoints.end()) {
            throw std::runtime_error("Something went wrong");
        }

        args.push_back(0);
        args.push_back(r.id);
        args.push_back(std::distance(starkInfo.openingPoints.begin(), primeIndex));

    } 
    else if (r.type == opType::custom) {
        auto primeIndex = std::find(starkInfo.openingPoints.begin(), starkInfo.openingPoints.end(), r.prime);
        if (primeIndex == starkInfo.openingPoints.end()) {
            throw std::runtime_error("Something went wrong");
        }

        args.push_back(starkInfo.nStages + 4 + r.commitId);
        args.push_back(r.id);
        args.push_back(std::distance(starkInfo.openingPoints.begin(), primeIndex));
    } 
    else if (r.type == opType::cm) {
        auto primeIndex = std::find(starkInfo.openingPoints.begin(), starkInfo.openingPoints.end(), r.prime);
        if (primeIndex == starkInfo.openingPoints.end()) {
            throw std::runtime_error("Something went wrong");
        }

        args.push_back(starkInfo.cmPolsMap[r.id].stage);
        args.push_back(starkInfo.cmPolsMap[r.id].stagePos);
        args.push_back(std::distance(starkInfo.openingPoints.begin(), primeIndex));
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
        if (!global) {
            args.push_back(bufferSize + 3);
        } else {
            args.push_back(2);
        }
        args.push_back(numberPos);
        if (!global) args.push_back(0);
    } else if (r.type == opType::public_) {
        if (!global) {
            args.push_back(bufferSize + 2);
        } else {
            args.push_back(1);
        }
        args.push_back(r.id);
       if (!global) args.push_back(0);
    } else if (r.type == opType::eval) {
        args.push_back(bufferSize + 8);
        args.push_back(3*r.id);
        args.push_back(0);
    } else if (r.type == opType::airvalue) {
        args.push_back(bufferSize + 4);
        uint64_t airValuePos = 0;
        for(uint64_t i = 0; i < r.id; ++i) {
            airValuePos += starkInfo.airValuesMap[i].stage == 1 ? 1 : 3;
        }
        args.push_back(airValuePos);
        args.push_back(0);
    } else if (r.type == opType::proofvalue) {
        if(!global) {
            args.push_back(bufferSize + 5);
        } else {
            args.push_back(3);
        }
        uint64_t proofValuePos = 0;
        for(uint64_t i = 0; i < r.id; ++i) {
            proofValuePos += starkInfo.proofValuesMap[i].stage == 1 ? 1 : 3;
        }
        args.push_back(proofValuePos);
        if (!global) args.push_back(0);
    } else if (r.type == opType::challenge) {
        if(!global) {
            args.push_back(bufferSize + 7);
        } else {
            args.push_back(6);
        }
        args.push_back(3*r.id);
        if (!global) args.push_back(0);
    } else if (r.type == opType::airgroupvalue) {
        if (!global) {
            args.push_back(bufferSize + 6);
        } else {
            args.push_back(5);
        }
        if (!global) {
            uint64_t airGroupValuePos = 0;
            for(uint64_t i = 0; i < r.id; ++i) {
                airGroupValuePos += starkInfo.airgroupValuesMap[i].stage == 1 ? 1 : 3;
            }
            args.push_back(airGroupValuePos);
            args.push_back(0);
        } else {
            uint64_t offset = 0;
            for(uint64_t i = 0; i < r.airgroupId; ++i) {
                // offset += globalInfo.aggTypes[i].size();
                // TODO!
            }
            args.push_back(offset + 3*r.id);
        }
    } 
    else if (r.type == opType::xDivXSubXi) {
        args.push_back(starkInfo.nStages + 3);
        args.push_back(r.id);
        args.push_back(0);
    } 
    else if (r.type == opType::Zi) {
        args.push_back(starkInfo.nStages + 2);
        args.push_back(1 + r.boundaryId);
        args.push_back(0);
    } 
    else {
        throw std::invalid_argument("Unknown type " + opType2string(r.type));
    }

}

ExpInfoBin ExpressionsInfo::getParserArgs(std::vector<CodeOperation> &code, uint64_t nTmpUsed, std::vector<uint64_t> &numbers, bool global) {
    ExpInfoBin expInfoBin;

    uint64_t maxid = nTmpUsed;
    std::vector<int64_t> ID1D(maxid, -1);
    std::vector<int64_t> ID3D(maxid, -1);

    std::tie(expInfoBin.nTemp1, expInfoBin.nTemp3) = getIdMaps(maxid, ID1D, ID3D, code);


    for(uint64_t i = 0; i < code.size(); ++i) {
        CodeOperation operation = getOperation(code[i]);
        uint64_t arg = operation.operationArg(operation.op);
        expInfoBin.args.push_back(arg);
        
        pushArgs(expInfoBin.args, operation.dest, ID1D, ID3D, numbers, true, global);
        pushArgs(expInfoBin.args, operation.src[0], ID1D, ID3D, numbers, false, global);
        pushArgs(expInfoBin.args, operation.src[1], ID1D, ID3D, numbers, false, global);

        if (operation.dest_dim == 1 && operation.src0_dim == 1 && operation.src1_dim == 1) {
            expInfoBin.ops.push_back(0);
        } else if (operation.dest_dim == 3 && operation.src0_dim == 3 && operation.src1_dim == 1) {
            expInfoBin.ops.push_back(1);
        } else if (operation.dest_dim == 3 && operation.src0_dim == 3 && operation.src1_dim == 3) {
            expInfoBin.ops.push_back(2);
        } else {
            zklog.error("Invalid operation: " + string(std::to_string(operation.dest_dim) + " " + std::to_string(operation.src0_dim) + " " + std::to_string(operation.src1_dim)));
            exitProcess();
            exit(-1);
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
    for(uint64_t j = 0; j < constraintsCode.size(); ++j) {
        ExpInfo constraint = constraintsCode[j];
        ExpInfoBin globalConstraintInfo = getParserArgs(constraint.code, constraint.tmpUsed, numbersConstraints, true);
        globalConstraintInfo.line = constraint.line;
        constraintsInfo.push_back(globalConstraintInfo);
    }
}

void ExpressionsInfo::prepareVerifierExpressionsBin() {
    for(uint64_t i = 0; i < expressionsCode.size(); ++i) {
        ExpInfo expCode = expressionsCode[i];
        ExpInfoBin expressionInfo = getParserArgs(expCode.code, expCode.tmpUsed, numbersExps, false);

        expressionInfo.expId = expCode.expId;
        expressionInfo.stage = expCode.stage;
        expressionInfo.line = expCode.line;
        expressionsInfo.push_back(expressionInfo);

        if(expressionInfo.nTemp1 > maxTmp1) maxTmp1 = expressionInfo.nTemp1;
        if(expressionInfo.nTemp3 > maxTmp3) maxTmp3 = expressionInfo.nTemp3;
        if(expressionInfo.args.size() > maxArgs) maxArgs = expressionInfo.args.size();
        if(expressionInfo.ops.size() > maxOps) maxOps = expressionInfo.ops.size();
    }
}

void ExpressionsInfo::prepareExpressionsBin() {
    uint64_t N = (1 << starkInfo.starkStruct.nBits);

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

        ExpInfoBin constraintInfo = getParserArgs(constraint.code, constraint.tmpUsed, numbersConstraints, false);
        constraintInfo.stage = constraint.stage;
        constraintInfo.firstRow = firstRow;
        constraintInfo.lastRow = lastRow;
        constraintInfo.line = constraint.line;
        constraintInfo.imPol = constraint.imPol;
        constraintsInfo.push_back(constraintInfo);

        if(constraintInfo.nTemp1 > maxTmp1) maxTmp1 = constraintInfo.nTemp1;
        if(constraintInfo.nTemp3 > maxTmp3) maxTmp3 = constraintInfo.nTemp3;
        if(constraintInfo.args.size() > maxArgs) maxArgs = constraintInfo.args.size();
        if(constraintInfo.ops.size() > maxOps) maxOps = constraintInfo.ops.size();
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
        ExpInfoBin expressionInfo = getParserArgs(expCode.code, expCode.tmpUsed, numbersExps, false);

        expressionInfo.expId = expCode.expId;
        expressionInfo.stage = expCode.stage;
        expressionInfo.line = expCode.line;
        expressionsInfo.push_back(expressionInfo);

        if(expressionInfo.nTemp1 > maxTmp1) maxTmp1 = expressionInfo.nTemp1;
        if(expressionInfo.nTemp3 > maxTmp3) maxTmp3 = expressionInfo.nTemp3;
        if(expressionInfo.args.size() > maxArgs) maxArgs = expressionInfo.args.size();
        if(expressionInfo.ops.size() > maxOps) maxOps = expressionInfo.ops.size();
    }
}