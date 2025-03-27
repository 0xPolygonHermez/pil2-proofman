#include "expressions_ctx.hpp"

typedef enum {
    Field = 0,
    FieldExtended = 1,
    Column = 2,
    ColumnExtended = 3,
    String = 4,
} HintFieldType;

struct HintFieldInfo {
    uint64_t size;
    uint64_t string_size;
    uint8_t offset;
    HintFieldType fieldType;
    Goldilocks::Element* values;
    uint8_t* stringValue;
    uint64_t matrix_size;
    uint64_t* pos;
    uint8_t* expression_line;
    uint64_t expression_line_size;
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
    bool compilation_time = false;
};

void getPolynomial(SetupCtx& setupCtx, Goldilocks::Element *buffer, Goldilocks::Element *dest, PolMap& polInfo, uint64_t rowOffsetIndex, string type);

void printRow(SetupCtx& setupCtx, Goldilocks::Element* buffer, uint64_t stage, uint64_t row);

std::string getExpressionDebug(SetupCtx& setupCtx, uint64_t hintId, std::string hintFieldName, HintFieldValue hintFieldVal);
uint64_t getHintFieldValues(SetupCtx& setupCtx, uint64_t hintId, std::string hintFieldName);
void getHintFieldSizes(
    SetupCtx& setupCtx, 
    HintFieldInfo *hintFieldValues,
    uint64_t hintId, 
    std::string hintFieldName,
    HintFieldOptions& hintOptions
);

void getHintField(
    SetupCtx& setupCtx,
    StepsParams &params,
    ExpressionsCtx& expressionsCtx,
    HintFieldInfo *hintFieldValues,
    uint64_t hintId, 
    std::string hintFieldName, 
    HintFieldOptions& hintOptions
);

uint64_t setHintField(SetupCtx& setupCtx, StepsParams& params, Goldilocks::Element* values, uint64_t hintId, std::string hintFieldName, StepsParams * d_params = NULL);

void addHintField(SetupCtx& setupCtx, StepsParams& params, uint64_t hintId, Dest &destStruct, std::string hintFieldName, HintFieldOptions hintFieldOptions);

void multiplyHintFields(SetupCtx& setupCtx, StepsParams &params, ExpressionsCtx& expressionsCtx, uint64_t nHints, uint64_t* hintId, std::string *hintFieldNameDest, std::string* hintFieldName1, std::string* hintFieldName2,  HintFieldOptions *hintOptions1, HintFieldOptions *hintOptions2, void* GPUExpressionsCtx = NULL, StepsParams * d_params = NULL);

void accHintField(SetupCtx& setupCtx, StepsParams &params, ExpressionsCtx& expressionsCtx, uint64_t hintId, std::string hintFieldNameDest, std::string hintFieldNameAirgroupVal, std::string hintFieldName, bool add);

#ifdef __USE_CUDA__
void opHintFieldsGPU(StepsParams* d_params, Dest &dest, uint64_t nRows, bool domainExtended, void* GPUExpressionsCtx);
void allocateDestGPU(Goldilocks::Element**buff, uint64_t size);
void freeDestGPU(Goldilocks::Element* buff);
void setPolynomialGPU(SetupCtx& setupCtx, Goldilocks::Element *buffer, Goldilocks::Element *values, uint64_t idPol);
void copyValueGPU( Goldilocks::Element * target, Goldilocks::Element* src, uint64_t size);
void copyValueHost( Goldilocks::Element * target, Goldilocks::Element* src, uint64_t size);
void copyValueGPUGPU( Goldilocks::Element * target, Goldilocks::Element* src, uint64_t size);
void opAirgroupValueGPU(Goldilocks::Element * airgroupValue,  Goldilocks::Element* val, uint32_t dim, bool add);

#endif

uint64_t getHintId(SetupCtx& setupCtx, uint64_t hintId, std::string name);

void accMulHintFields(SetupCtx& setupCtx, StepsParams &params, ExpressionsCtx &expressionsCtx, uint64_t hintId, std::string hintFieldNameDest, std::string hintFieldNameAirgroupVal, std::string hintFieldName1, std::string hintFieldName2, HintFieldOptions &hintOptions1, HintFieldOptions &hintOptions2, bool add, void* GPUExpressionsCtx = NULL, StepsParams * d_params= NULL);

uint64_t updateAirgroupValue(SetupCtx& setupCtx, StepsParams &params, uint64_t hintId, std::string hintFieldNameAirgroupVal, std::string hintFieldName1, std::string hintFieldName2, HintFieldOptions &hintOptions1, HintFieldOptions &hintOptions2, bool add, void* GPUExpressionsCtx = NULL, StepsParams * d_params= NULL);
