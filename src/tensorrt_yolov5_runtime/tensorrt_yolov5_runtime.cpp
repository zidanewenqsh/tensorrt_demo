#include "utils.h"
#include "cuda_utils.h"
#include "config.h"
#include <cuda_runtime_api.h>
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
#include <logger.h>
// #include "common/logger.h"
// #include "common/buffers.h"

#include <opencv2/opencv.hpp>


#include "preprocess.h"
#include "postprocess.h"

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
void printDims(const nvinfer1::Dims& dims) {
    std::cout << "Dimensions: ";
    for (int i = 0; i < dims.nbDims; ++i) {
        std::cout << dims.d[i];
        if (i < dims.nbDims - 1) {
            std::cout << " x ";
        }
    }
    std::cout << std::endl;
}

int inference(std::string &trt_file) {
    // 1. 创建推理引擎
    // 2. 读取模型文件
    // 3. 创建上下文
    // 4. 创建执行上下文
    // 5. 执行推理
    // 6. 释放资源
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
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

    
    // samplesCommon::BufferManager buffers(engine);
    cv::Mat img = cv::imread("bus.jpg");
    // auto input_data_host = std::unique_ptr<float[]>(new float[kInputH * kInputW * 3]);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    float *input_data_host = nullptr;
    float *input_data_device = nullptr;
    // 明确当前推理时，使用的数据输入大小
    auto input_dims = engine->getTensorShape(kInputTensorName);
    int input_batch = input_dims.d[0];

    int input_numel = input_batch * kInputH * kInputW * 3;
    checkRuntime(cudaMallocHost(&input_data_host, sizeof(float) * input_numel));
    checkRuntime(cudaMalloc(&input_data_device, sizeof(float) * input_numel));
    // 预处理
    auto matrix = std::unique_ptr<float[]>(new float[6 * 2]);
    preprocess_cpu(img, input_data_host, (float*)matrix.get(), kInputW, kInputH, false);
    // saveWeight("matrix.data", (float*)matrix.get(), 12);
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

    nvinfer1::Dims output_dims = context->getTensorShape(kOutputTensorName);
    printDims(output_dims);
    int output_batch = output_dims.d[0];
    int output_numbox = output_dims.d[1];
    int output_numprob = output_dims.d[2];
    int num_classes = output_numprob - 5;
    int output_numel = output_batch * output_numbox * output_numprob;
    printf("output_batch: %d, output_numbox: %d, output_numprob: %d, num_classes: %d, output_numel: %d\n", output_batch, output_numbox, output_numprob, num_classes, output_numel); 
    float* output_data_host = nullptr;
    float* output_data_device = nullptr;
    checkRuntime(cudaMallocHost(&output_data_host, sizeof(float) * output_numel));
    checkRuntime(cudaMalloc(&output_data_device, sizeof(float) * output_numel));

    void* bindings[] = {input_data_device, output_data_device};
    bool success = context->executeV2(bindings);
    if (!success) {
        cudaFree(input_data_device);
        cudaFree(output_data_device);
        cudaFreeHost(input_data_host);
        cudaFreeHost(output_data_host);
        cudaStreamDestroy(stream);
        std::cout << "enqueue failed" << std::endl;
        return -1;
    }

    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    saveWeight("yolov5output.data", output_data_host, output_numel);
    std::vector<Box> resboxes = postprocess_cpu(output_data_host, output_batch, output_numbox, output_numprob, kConfThresh, kNmsThresh);
    float *d2i = (float*)matrix.get() + 6;
    cv::Mat img_draw = draw(resboxes, img, d2i);
    cv::imwrite("bus_result.jpg", img_draw);

    cudaFree(input_data_device);
    cudaFree(output_data_device);
    cudaFreeHost(input_data_host);
    cudaFreeHost(output_data_host);
    cudaStreamDestroy(stream);
    return 0;
}

int main() {
    std::string trt_file = "yolov5_engine.trtmodel";
    inference(trt_file);
    return 0;
}