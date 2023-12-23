#include "utils.h"
#include "cuda_utils.h"
#include "config.h"
#include <cuda_runtime_api.h>
#include <NvInferRuntime.h>
#include <opencv2/core/cvdef.h>
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
#include "preprocess_gpu.cuh"
#include "postprocess_gpu.cuh"
// #include "common/logger.h"
// #include "common/buffers.h"

#include <opencv2/opencv.hpp>


#include "preprocess.h"
#include "postprocess.h"

// #define print(x) std::cout << #x << ": " << x << std::endl;
#define oldversion 0
#define opencvcpu 0
#define preprocessongpu 1 
#define postprocessongpu 1 
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

int inference(std::string &trt_file, std::string &img_file) {
    // 1. 创建推理引擎
    // 2. 读取模型文件
    // 3. 创建上下文
    // 4. 创建执行上下文
    // 5. 执行推理
    // 6. 释放资源
    float mean[3] = {0, 0, 0};
    float std[3] = {1, 1, 1};
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
    // 读取图像
    cv::Mat img = cv::imread(img_file);
    if (img.empty()) {
        printf("cv::imread %s failed\n", img_file.c_str());
        return -1;
    }
    // auto input_data_host = std::unique_ptr<float[]>(new float[kInputH * kInputW * 3]);
    // 创建流
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
    
    auto matrix = calculate_matrix(img.cols, img.rows, kInputW, kInputH);
    float *d2i = (float*)matrix.get() + 6;

    float* output_data_host = nullptr;
    float* output_data_device = nullptr;
    // 明确当前推理时，使用的数据输出大小
    nvinfer1::Dims output_dims = context->getTensorShape(kOutputTensorName);
    printDims(output_dims);
    int output_batch = output_dims.d[0];
    int output_numbox = output_dims.d[1];
    int output_numprob = output_dims.d[2];
    int num_classes = output_numprob - 5;
    int output_numel = output_batch * output_numbox * output_numprob;
    printf("output_batch: %d, output_numbox: %d, output_numprob: %d, num_classes: %d, output_numel: %d\n", output_batch, output_numbox, output_numprob, num_classes, output_numel); 
    checkRuntime(cudaMallocHost(&output_data_host, sizeof(float) * output_numel));
    checkRuntime(cudaMalloc(&output_data_device, sizeof(float) * output_numel));

#if preprocessongpu
#if 1
    // 分配和初始化 GPU 内存
    // 将均值和方差拷贝到 GPU 内存
    float *d_mean;
    float *d_std;
    checkRuntime(cudaMalloc(&d_mean, 3 * sizeof(float)));
    checkRuntime(cudaMalloc(&d_std, 3 * sizeof(float)));
    checkRuntime(cudaMemcpyAsync(d_mean, mean, 3 * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaMemcpyAsync(d_std, std, 3 * sizeof(float), cudaMemcpyHostToDevice, stream));
    // 将输入数据拷贝到 GPU 内存
    unsigned char* d_input; // 这是原始图像的输入数据
    checkRuntime(cudaMalloc(&d_input, img.total() * sizeof(unsigned char) * img.channels()));
    checkRuntime(cudaMemcpyAsync(d_input, img.data, img.total() * sizeof(uchar) * img.channels(), cudaMemcpyHostToDevice, stream));
    // 将变换矩阵拷贝到 GPU 内存
    float* d_matrix;
    checkRuntime(cudaMalloc(&d_matrix, 6 * sizeof(float)));
    checkRuntime(cudaMemcpyAsync(d_matrix, d2i, 6 * sizeof(float), cudaMemcpyHostToDevice, stream));
    // 预处理，结果保存在 input_data_device 中， 这是预处理后的输入数据，下一步要送到网络中
    preprocess_gpu(d_input, input_data_device, d_matrix, img.cols, img.rows, kInputW, kInputH, d_mean, d_std);
#else
    // 分配和初始化 GPU 内存
    float *d_mean;
    float *d_std;
    int w = kInputW, h = kInputH;
    cudaMalloc(&d_mean, 3 * sizeof(float));
    cudaMalloc(&d_std, 3 * sizeof(float));
    cudaMemcpy(d_mean, mean, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_std, std, 3 * sizeof(float), cudaMemcpyHostToDevice);
    uchar* d_input;
    float* d_output;
    float* d_matrix;
    uchar* h_input;
    float* h_output;
    cudaMallocHost(&h_input, img.total() * img.channels());
    cudaMallocHost(&h_output, 3 * w * h * sizeof(float));
    // float d = bilinearInterpolateChannel_cpu(img.data, img.cols, img.rows, img.cols-1.5f, img.rows-1.5f, 2, 3);
    // printf("d=%f\n", d);
    int img_total = img.total() * sizeof(unsigned char) * img.channels();
    printf("img_total: %d\n", img_total);
    cudaMalloc(&d_input, img.total() * sizeof(unsigned char) * img.channels());
    cudaMalloc(&d_output, 3 * w * h * sizeof(float));
    cudaMalloc(&d_matrix, 6 * sizeof(float));

    cudaMemcpy(d_input, img.data, img.total() * sizeof(uchar) * img.channels(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix, d2i, 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(h_input, d_input, img.total() * sizeof(uchar) * img.channels(), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) {
        printf("h_input[%d]: %d, img.data[%d]: %d\n", i, h_input[i], i, img.data[i]);
    }
    // // 设置 CUDA 网格和块的大小
    // dim3 blockSize(16, 16);
    // dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    // // 调用 CUDA 内核
    // preprocess_kernel<<<gridSize, blockSize>>>(d_input, d_output, d_matrix, img.cols, img.rows, w, h, mean, std);
    preprocess_gpu(d_input, d_output, d_matrix, img.cols, img.rows, w, h, d_mean, d_std);
#endif
#else 
#if opencvcpu
    // 预处理
    // auto matrix = std::unique_ptr<float[]>(new float[6 * 2]);
    float *i2d = (float*)matrix.get();
    preprocess_opencv_cpu(img, input_data_host, i2d, kInputW, kInputH, false);
#else
    // auto matrix = calculate_invmatrix(img.cols, img.rows, kInputW, kInputH);
    // preprocess_cpu_v2(img, input_data_host, (float*)matrix.get(), kInputW, kInputH, false);
    preprocess_cpu_v2(img, input_data_host, d2i, kInputW, kInputH, false);
#endif
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaStreamSynchronize(stream));
#endif
    // saveWeight("matrix.data", (float*)matrix.get(), 12);
    // 执行推理
    void* bindings[] = {input_data_device, output_data_device};
    bool success = context->executeV2(bindings);
    if (!success) {
#if preprocessongpu
        cudaFree(d_input);
        cudaFree(d_matrix);
        cudaFree(d_mean);
        cudaFree(d_std);
#endif
        cudaFree(input_data_device);
        cudaFree(output_data_device);
        cudaFreeHost(input_data_host);
        cudaFreeHost(output_data_host);
        cudaStreamDestroy(stream);
        std::cout << "enqueue failed" << std::endl;
        return -1;
    }

#if postprocessongpu
// #if 1
    // 分配 GPU 内存以存储过滤后的边界框
    gBox* h_filtered_boxes;
    gBox* d_filtered_boxes;
    int* h_box_count;
    int* d_box_count;
    checkRuntime(cudaMallocHost(&h_filtered_boxes, output_numbox * sizeof(gBox))); 
    checkRuntime(cudaMallocHost(&h_box_count, sizeof(int)));
    checkRuntime(cudaMalloc(&d_filtered_boxes, output_numbox * sizeof(gBox))); 
    checkRuntime(cudaMalloc(&d_box_count, sizeof(int)));
    checkRuntime(cudaMemset(d_box_count, 0, sizeof(int)));
    checkRuntime(cudaMemset(d_filtered_boxes, 0, output_numbox * sizeof(gBox)));

    // 调用封装的函数
    postprocess_cuda(output_data_device, d_filtered_boxes, d_box_count, d_matrix,
            output_numbox, output_numprob, kConfThresh, kNmsThresh);
    
    checkRuntime(cudaMemcpyAsync(h_box_count, d_box_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream)); // 我要用h_box_count，所以要同步
    checkRuntime(cudaMemcpyAsync(h_filtered_boxes, d_filtered_boxes, (*h_box_count) * sizeof(gBox), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    printf("h_box_count: %d\n", *h_box_count);
    cv::Mat img_draw = draw_g(h_filtered_boxes, *h_box_count, img);
    cv::imwrite("result_gpu.jpg", img_draw);
// #else
// #endif
#else
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    // saveWeight("yolov5output.data", output_data_host, output_numel);
    // 后处理
    std::vector<Box> resboxes = postprocess_cpu(output_data_host, output_batch, output_numbox, output_numprob, kConfThresh, kNmsThresh);
    // 绘制结果
    cv::Mat img_draw = draw(resboxes, img, d2i);
    cv::imwrite("result.jpg", img_draw);
#endif
    // 释放资源
#if preprocessongpu
        // cudaFree(d_input);
        cudaFree(d_matrix);
        cudaFree(d_mean);
        cudaFree(d_std);
#endif
#if postprocessongpu
        cudaFree(d_filtered_boxes);
        cudaFree(d_box_count);
#endif
    cudaFree(input_data_device);
    cudaFree(output_data_device);
    cudaFreeHost(input_data_host);
    cudaFreeHost(output_data_host);
    cudaStreamDestroy(stream);
    return 0;
}

int main(int argc, char* argv[]) {
    std::string img_file = "bus.jpg";
    if (argc == 2) {
        img_file = argv[1];
        // printf("Usage: %s image_file\n", argv[0]);
        // return -1;
    }
    std::string trt_file = "yolov5_engine.trtmodel";
    inference(trt_file, img_file);
    printf("Done!\n");
    return 0;
}