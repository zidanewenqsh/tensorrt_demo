#include <NvInferRuntime.h>
#include <opencv2/highgui.hpp>
#include <unistd.h>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>
#include <fstream>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include "common/logger.h"
#include "common/buffers.h"

#include <opencv2/opencv.hpp>

#define print(x) std::cout << #x << ": " << x << std::endl;
#define oldversion 0
// class Logger : public nvinfer1::ILogger {
//     void log(Severity severity, const char *msg) noexcept override {
//         if (severity != Severity::kINFO) {
//             std::cout << msg << std::endl;
//         }
//     }
// } gLogger;

std::vector<char> load_Data(std::string &file) {
    std::ifstream infile(file, std::ios::binary);
    assert(infile.is_open() && "loadData failed");
    infile.seekg(0, std::ios::end);
    int size = infile.tellg();
    infile.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    infile.read(reinterpret_cast<char *>(data.data()), size);
    infile.close();
    return data;
}

int inference(std::string &trt_file) {
    // 1. 创建推理引擎
    // 2. 读取模型文件
    // 3. 创建上下文
    // 4. 创建执行上下文
    // 5. 执行推理
    // 6. 释放资源
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!runtime) {
        std::cout << "createInferRuntime failed" << std::endl;
        return -1;
    }
    auto engine_data = load_Data(trt_file);
    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine) {
        std::cout << "deserializeCudaEngine failed" << std::endl;
        return -1;
    }
#if oldversion
    if (engine->getNbBindings() != 2) {
#else
    if (engine->getNbIOTensors() != 2) {
#endif
        std::cout << "getNbBindings failed" << std::endl;
        return -1;
    }
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        std::cout << "createExecutionContext failed" << std::endl;
        return -1;
    }
    samplesCommon::BufferManager buffers(engine);
    cv::Mat img = cv::imread("cat01.jpg");
    // cv::imshow("image", img);
    // cv::waitKey(0);

    return 0;
}

int main() {

    std::string trt_file = "yolov5_engine.trtmodel";
    inference(trt_file);
    return 0;
}