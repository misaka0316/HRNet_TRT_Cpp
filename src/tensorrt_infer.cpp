
#include "cookbookHelper.cuh"

#include <tensorrt_infer.hpp>

TensorrtInfer::~TensorrtInfer() {
    for (int i = 0; i < nIO; ++i)
    {
        delete[] (char *)vBufferH[i];
        CHECK(cudaFree(vBufferD[i]));
    }
    if (context) {
        delete context;
    }
    if (engine) {
        delete engine;
    }
    if (runtime) {
        delete runtime;
    }
}

bool TensorrtInfer::init(const std::string trtFile) {
    
    if (access(trtFile.c_str(), F_OK) == 0)
    {
        std::ifstream engineFile(trtFile, std::ios::binary);
        long int      fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0)
        {
            std::cout << "Failed getting serialized engine!" << std::endl;
            return false;
        }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr)
        {
            std::cout << "Failed loading engine!" << std::endl;
            return false;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
        
    }else{
        std::cout << "Failed loading engine! No engine file exists!" << std::endl;
        return false;
    }

    nIO = engine->getNbIOTensors();
    
    vBufferH.resize(nIO, nullptr);
    vBufferD.resize(nIO, nullptr);
    vTensorSize.resize(nIO, 0);
    vTensorName.resize(nIO);
    
    for (int i = 0; i < nIO; ++i)
    {
        vTensorName[i] = std::string(engine->getIOTensorName(i));
        nInput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kINPUT);
        nOutput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kOUTPUT);
    }
    
    context = engine->createExecutionContext();

    context->setInputShape(vTensorName[0].c_str(), shape);

    for (int i = 0; i < nIO; ++i)
    {
        std::cout << std::string(i < nInput ? "Input [" : "Output[");
        std::cout << i << std::string("]-> ");
        std::cout << dataTypeToString(engine->getTensorDataType(vTensorName[i].c_str())) << std::string(" ");
        std::cout << shapeToString(engine->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
        std::cout << shapeToString(context->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
        std::cout << vTensorName[i] << std::endl;
    }

    for (int i = 0; i < nIO; ++i)
    {
        Dims32 dim  = context->getTensorShape(vTensorName[i].c_str());
        int    size = 1;
        for (int j = 0; j < dim.nbDims; ++j)
        {
            size *= dim.d[j];
        }
        vTensorSize[i] = size * dataTypeToSize(engine->getTensorDataType(vTensorName[i].c_str()));
        // std::cout << "size: " << size << std::endl;
        // std::cout << "dataTypeToSize: " << dataTypeToSize(engine->getTensorDataType(vTensorName[i].c_str())) << std::endl;
        // std::cout << "Total input size: " << vTensorSize[i] << std::endl;
    }
    
    for (int i = 0; i < nIO; ++i)
    {
        vBufferH[i] = (void *)new char[vTensorSize[i]];
        CHECK(cudaMalloc(&vBufferD[i], vTensorSize[i]));
    }
    pData = (float *)vBufferH[0];
    return true;
}

void TensorrtInfer::forward() {

    for (int i = 0; i < nInput; ++i)
    {
        CHECK(cudaMemcpy(vBufferD[i], vBufferH[i], vTensorSize[i], cudaMemcpyHostToDevice));
    }

    for (int i = 0; i < nIO; ++i)
    {
        context->setTensorAddress(vTensorName[i].c_str(), vBufferD[i]);
    }
    for (int i = 0; i < 10; i++)
    context->enqueueV3(0);
    //等待推理完成
    // cudaDeviceSynchronize();

    for (int i = nInput; i < nIO; ++i)
    {
        CHECK(cudaMemcpy(vBufferH[i], vBufferD[i], vTensorSize[i], cudaMemcpyDeviceToHost));
        // std::cout << "optput_size: " <<vTensorSize[i] << std::endl;
    }
    // cudaDeviceSynchronize();
    std::cout << "Inference done!" << std::endl;

    float* buffer = static_cast<float*>(vBufferH[1]);
    //将输出保存在txt中
    std::ofstream outfile;
    outfile.open("output.txt");
    // outfile << vTensorSize[1] << std::endl;
    
    for (int i = 0; i < vTensorSize[1]; i++) 
    {   
        // std::cout << static_cast<float>(buffer[i]) << " ";
        outfile << buffer[i] <<" ";
        if(i % 46 == 0)
        outfile << std::endl;
        // if(buffer[i] > 0.1)
        // std::cout << buffer[i] << " ";
    }
    std::cout << vTensorSize[1] << std::endl;
}

void TensorrtInfer::setpData(float* data) {
    pData = data;
}
void TensorrtInfer::output_Data(float* data,int& shape){
    data = (float *)vBufferH[1];
    shape = vTensorSize[1];
}
