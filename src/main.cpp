/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.

 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cookbookHelper.cuh"

#include <iostream>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
using namespace nvinfer1;

const std::string trtFile {"../models/HR_net.plan"};
static Logger     gLogger(ILogger::Severity::kERROR);

std::tuple<int, std::vector<int>> shape = {4, {1, 3, 256, 192}};

void run()
{
    ICudaEngine *engine = nullptr;

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
            return;
        }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr)
        {
            std::cout << "Failed loading engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
    }else{
        std::cout << "Failed loading engine!、不存在engine文件" << std::endl;
        return;
    }
    
    long unsigned int        nIO     = engine->getNbIOTensors();
    long unsigned int        nInput  = 0;
    long unsigned int        nOutput = 0;
    std::vector<std::string> vTensorName(nIO);
    for (int i = 0; i < nIO; ++i)
    {
        vTensorName[i] = std::string(engine->getIOTensorName(i));
        nInput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kINPUT);
        nOutput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kOUTPUT);
    }

    IExecutionContext *context = engine->createExecutionContext();
    context->setInputShape(vTensorName[0].c_str(), Dims32 {4, {1, 3, 256, 192}});

    for (int i = 0; i < nIO; ++i)
    {
        std::cout << std::string(i < nInput ? "Input [" : "Output[");
        std::cout << i << std::string("]-> ");
        std::cout << dataTypeToString(engine->getTensorDataType(vTensorName[i].c_str())) << std::string(" ");
        std::cout << shapeToString(engine->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
        std::cout << shapeToString(context->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
        std::cout << vTensorName[i] << std::endl;
    }

    std::vector<int> vTensorSize(nIO, 0);
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
    
    std::vector<void *>
                        vBufferH {nIO, nullptr};
    std::vector<void *> vBufferD {nIO, nullptr};
    for (int i = 0; i < nIO; ++i)
    {
        vBufferH[i] = (void *)new char[vTensorSize[i]];
        CHECK(cudaMalloc(&vBufferD[i], vTensorSize[i]));
    }

    float *pData = (float *)vBufferH[0];

    auto img_path = "../data/arm_leg_data1";

    // 获取路径下的所有图片文件
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(img_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            image_files.push_back(entry.path().string());
        }
    }

    // 对文件进行排序
    std::sort(image_files.begin(), image_files.end());

    // 遍历排序后的文件并进行推理
    for (const auto& img_file : image_files) {

        // std::cout << "Processing image: " << img_file << std::endl;
        cv::Mat inputImage = cv::imread(img_file);
        if (inputImage.empty()) {
            std::cerr << "Could not open or find the image!" << std::endl;
            // return -1;
            continue;
        }
        std::cout << "inputImage.shape: " << inputImage.size() << std::endl;

        cv::Mat crop_img = inputImage(cv::Range(0,1024), cv::Range(50, 818));
        std::cout << "crop_img.shape: " << crop_img.size() << std::endl;

        cv::Mat resize_img;
        cv::resize(crop_img, resize_img, cv::Size(192, 256), 0, 0, cv::INTER_LINEAR);
        std::cout << "resize_img.shape: " << resize_img.size() << std::endl;

        cv::Mat input_img_;
        resize_img.convertTo(input_img_, CV_32F);  
        input_img_ = input_img_ / 255.0;
        
        pData = reinterpret_cast<float*>(input_img_.data);
        // for (int i = 0; i < 10; ++i)
        // {
        //     std::cout << pData[i] << " ";
        // }

        for (int i = 0; i < nInput; ++i)
        {
            CHECK(cudaMemcpy(vBufferD[i], vBufferH[i], vTensorSize[i], cudaMemcpyHostToDevice));
        }

        for (int i = 0; i < nIO; ++i)
        {
            context->setTensorAddress(vTensorName[i].c_str(), vBufferD[i]);
        }

        context->enqueueV3(0);

        for (int i = nInput; i < nIO; ++i)
        {
            CHECK(cudaMemcpy(vBufferH[i], vBufferD[i], vTensorSize[i], cudaMemcpyDeviceToHost));
            std::cout << "optput_size: " <<vTensorSize[i] << std::endl;
        }
        
        // for (int i = 1; i < nIO; ++i)
        // {
        //     printArrayInformation((float *)vBufferH[i], context->getTensorShape(vTensorName[i].c_str()), vTensorName[i], true, true);
        // }
        
    }

    for (int i = 0; i < nIO; ++i)
    {
        delete[] (char *)vBufferH[i];
        CHECK(cudaFree(vBufferD[i]));
    }
    return;
}

int main()
{
    CHECK(cudaSetDevice(0));
    run();
    return 0;
}
