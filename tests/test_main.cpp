#include <vector>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "tensorrt_infer.hpp"

namespace fs = std::filesystem;

int main()
{
    CHECK(cudaSetDevice(0));
    
    // const std::string plan_file = "../models/HR_net_float32.plan";
    const std::string plan_file = "../models/HR_net_float321.plan";

    TensorrtInfer infer;

    if (!infer.init(plan_file)) {
        std::cerr << "Initialization failed" << std::endl;
        return 0;
    }
    
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

        // 访问第一个像素 (0,0)，并分离出 BGR 三个通道的值
        cv::Vec3b pixelValue2 = inputImage.at<cv::Vec3b>(0, 0);
        uchar blue2 = pixelValue2[0];
        uchar green2 = pixelValue2[1];
        uchar red2 = pixelValue2[2];
        std::cout << "The first pixel's BGR values are: "
            << (int)blue2 << ", " << (int)green2 << ", " << (int)red2 << std::endl;

        cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2RGB);

        if (inputImage.empty()) {
            std::cerr << "Could not open or find the image!" << std::endl;
            // return -1;
            continue;
        }
        // std::cout << "inputImage.shape: " << inputImage.size() << std::endl;

        // 访问第一个像素 (0,0)，并分离出 BGR 三个通道的值
        cv::Vec3b pixelValue = inputImage.at<cv::Vec3b>(0, 0);
        uchar blue = pixelValue[0];
        uchar green = pixelValue[1];
        uchar red = pixelValue[2];

        std::cout << "The first pixel's BGR values are: "
            << (int)blue << ", " << (int)green << ", " << (int)red << std::endl;

        cv::Mat crop_img = inputImage(cv::Range(0,1024), cv::Range(50, 818));
        // std::cout << "crop_img.shape: " << crop_img.size() << std::endl;

        // 访问第一个像素 (0,0)，并分离出 BGR 三个通道的值
        cv::Vec3b pixelValue1 = crop_img.at<cv::Vec3b>(0, 0);
        uchar blue1 = pixelValue1[0];
        uchar green1 = pixelValue1[1];
        uchar red1 = pixelValue1[2];

        std::cout << "The first pixel's BGR values are: "
            << (int)blue1 << ", " << (int)green1 << ", " << (int)red1 << std::endl;

        cv::Mat resize_img;
        // cv::resize(crop_img, resize_img, cv::Size(192, 256), 0, 0, cv::INTER_LINEAR);
        cv::resize(crop_img, resize_img, cv::Size(256, 192), 0, 0, cv::INTER_LINEAR);
        std::cout << "resize_img.shape: " << resize_img.size() << std::endl;

        cv::Mat input_img_;
        resize_img.convertTo(input_img_, CV_32F);  
        input_img_ = input_img_ / 255.0;

        //打印input_img_的前20个元素
        for (int i = 0; i < 20; i++) {
            std::cout << input_img_.at<float>(0, i) << " ";
        }
        std::cout << std::endl;

        infer.setpData(reinterpret_cast<float*>(input_img_.data));

        infer.forward();

        float* data;
        int output_shape;

        infer.output_Data(data, output_shape);
        // //打印data的前十个元素
        // for (int i = 0; i < 10; i++) {
        //     std::cout << data[i] << " ";
        // }
        // std::cout << "optput_size: " <<output_shape << std::endl;
    }
    

    return 0;
}