#ifndef MAIN_HPP
#define MAIN_HPP

#include <iostream>
#include <vector>

#include "cookbookHelper.cuh"

using namespace nvinfer1;

static Logger     gLogger(ILogger::Severity::kERROR);

const static Dims32 shape {4, {1, 3, 192, 256}};
// const static Dims32 shape {4, {1, 3, 256, 192}};

class TensorrtInfer {
public:
    TensorrtInfer() : runtime(createInferRuntime(gLogger)), engine(nullptr), context(nullptr) {}
    virtual ~TensorrtInfer();
    bool init(const std::string trtFile);
    void forward();
    void setpData(float* data);
    void output_Data(float* data,int& shape);

private:
    long unsigned int        nIO     = 0;
    long unsigned int        nInput  = 0;
    long unsigned int        nOutput = 0;
    std::vector<std::string> vTensorName;

    IRuntime *runtime;
    ICudaEngine *engine;
    IExecutionContext *context;

    std::vector<void *> vBufferH;
    std::vector<void *> vBufferD;
    std::vector<int> vTensorSize;

    float *pData;
};

#endif
