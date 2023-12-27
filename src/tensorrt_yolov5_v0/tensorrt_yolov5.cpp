#include <NvInferRuntime.h>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include <memory>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include "config.h"
#include "preprocess_gpu.cuh"
#include "postprocess_gpu.cuh"
#include <cstdlib>
#include <chrono>
#include "preprocess.h"
#include "postprocess.h"
#include "utils.h"
// #include "logger.h"
#include "config.h"
#include "cuda_utils.h"
#define PROCESSONGPU 1
#define USEOPENCV 1

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

// #define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)
// bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
//     if(code != cudaSuccess){    
//         const char* err_name = cudaGetErrorName(code);    
//         const char* err_message = cudaGetErrorString(code);  
//         printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
//         return false;
//     }
//     return true;
// }

// bool exists(const std::string& path){

// #ifdef _WIN32
//     return ::PathFileExistsA(path.c_str());
// #else
//     return access(path.c_str(), F_OK) == 0;
// #endif
// }
static int totalsize = 0;
// static int idx = 0;
class Yolov5 {
private:
    size_t buffersize;
    void *host_buffer;
    void *device_buffer;
    char *host_buffer_now;
    char *device_buffer_now;
    cudaStream_t stream;
    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;
    void *image_data_device = nullptr;
    void *input_data_host = nullptr;
    void *input_data_device = nullptr;
    void *output_data_host = nullptr;
    void *output_data_device = nullptr;
    void *h_mean = nullptr;
    void *h_std = nullptr;
    void *d_mean = nullptr;
    void *d_std = nullptr;
    void *h_matrix = nullptr;
    void *d_matrix = nullptr;
    bool init_finished = false;
    std::string onnx_file;
    std::string trt_file;
    void* h_filtered_boxes = nullptr;
    void* d_filtered_boxes = nullptr;
    void* h_box_count = nullptr;
    void* d_box_count = nullptr;
    int input_batch;
    int input_numel;
    int output_batch;
    int output_numbox;
    int output_numprob;
    int num_classes;
    int output_numel;
    int index = 0;
public:
    Yolov5(std::string &name, int buffer_size):buffersize(buffer_size) {
        std::cout << "Yolov5()" << std::endl;
        checkRuntime(cudaMallocHost(&host_buffer, buffer_size));
        checkRuntime(cudaMalloc(&device_buffer, buffer_size));
        cudaStreamCreate(&stream);
        host_buffer_now = (char*)host_buffer;
        device_buffer_now = (char*)device_buffer;
        onnx_file = name + ".onnx";
        trt_file = name + ".trt";
        std::cout << "trt_file:" << trt_file << std::endl;
        if (init() < 0) {
            std::cout << "init failed" << std::endl;
            exit(-1);
        }
        std::cout << "Yolov5() init finished" << std::endl;
    }
    ~Yolov5() {
        std::cout << "~Yolov5()" << std::endl;
        
    }
    void print() {
        std::cout << "print()" << std::endl;
    }

    int malloc_host(void **ptr, size_t size) {
        if (*ptr != nullptr) {
            std::cout << "malloc_host failed, *ptr is not nullptr" << std::endl;
            return -1;
     }
     if (host_buffer_now + size > (char*)host_buffer + buffersize) {
            std::cout << "malloc_host failed, size is not enough" << std::endl;
            return -1;
        }
        *ptr = host_buffer_now;
        host_buffer_now += size;
        totalsize += size;
        return 0; 
    }
    int malloc_device(void** ptr, size_t size) {
        if (*ptr != nullptr) {
            std::cout << "malloc_device failed, *ptr is not nullptr" << std::endl;
            return -1;
        }
        if (device_buffer_now + size > (char*)device_buffer + buffersize) {
            std::cout << "malloc_device failed, size is not enough" << std::endl;
            return -1;
        }
        *ptr = device_buffer_now;
        device_buffer_now += size;
        return 0;
    }

    int init() {
        // 初始化
        if (build() < 0) {
            std::cout << "build failed" << std::endl;
            return -1;
        }
        if (init_finished) {
            std::cout << "runtime_init already finished" << std::endl;
            return 0;
        }
        float mean[3] = {0, 0, 0};
        float std[3] = {1, 1, 1};
        runtime = nvinfer1::createInferRuntime(gLogger);
        if (!runtime) {
            std::cout << "createInferRuntime failed" << std::endl;
            return -1;
        }
        auto engine_data = loadData(trt_file);
        engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
        if (!engine) {
            std::cout << "deserializeCudaEngine failed" << std::endl;
            return -1;
        }
        if (engine->getNbIOTensors() != 2) {
            std::cout << "getNbBindings failed" << std::endl;
            return -1;
        }
        context = engine->createExecutionContext();
        if (!context) {
            std::cout << "createExecutionContext failed" << std::endl;
            return -1;
        }

        // 明确当前推理时，使用的数据输入大小
        auto input_dims = engine->getTensorShape(kInputTensorName);
        input_batch = input_dims.d[0];
        input_numel = input_batch * kInputH * kInputW * 3;
        if(malloc_host(&input_data_host, sizeof(float) * input_numel)<0) {
            std::cout << "malloc_host input_data_host failed" << std::endl;
            return -1;
        }
        if (malloc_device(&input_data_device, sizeof(float) * input_numel)<0){
            std::cout << "malloc_device input_data_device failed" << std::endl;
            return -1;
        }

        // 明确当前推理时，使用的数据输出大小
        nvinfer1::Dims output_dims = context->getTensorShape(kOutputTensorName);
        printDims(output_dims);
        output_batch = output_dims.d[0];
        output_numbox = output_dims.d[1];
        output_numprob = output_dims.d[2];
        num_classes = output_numprob - 5;
        output_numel = output_batch * output_numbox * output_numprob;
        printf("output_batch: %d, output_numbox: %d, output_numprob: %d, num_classes: %d, output_numel: %d\n", output_batch, output_numbox, output_numprob, num_classes, output_numel); 

        if(malloc_host(&output_data_host, sizeof(float) * output_numel)<0) {
            std::cout << "malloc_host output_data_host failed" << std::endl;
            return -1;
        }
        if (malloc_device(&output_data_device, sizeof(float) * output_numel)<0){
            std::cout << "malloc_device output_data_device failed" << std::endl;
            return -1;
        }

        if (malloc_host(&h_mean, 3 * sizeof(float))<0) {
            std::cout << "malloc_device h_mean failed" << std::endl;
            return -1;
        }
        memcpy(h_mean, mean, 3 * sizeof(float));
        if (malloc_host(&h_std, 3 * sizeof(float))<0) {
            std::cout << "malloc_device h_std failed" << std::endl;
            return -1;
        }
        memcpy(h_std, std, 3 * sizeof(float));
        if (malloc_device(&d_mean, 3 * sizeof(float))<0) {
            std::cout << "malloc_device d_mean failed" << std::endl;
            return -1;
        }
        if (malloc_device(&d_std, 3 * sizeof(float))<0) {
            std::cout << "malloc_device d_std failed" << std::endl;
            return -1;
        }
        checkRuntime(cudaMemcpyAsync(d_mean, mean, 3 * sizeof(float), cudaMemcpyHostToDevice, stream));
        checkRuntime(cudaMemcpyAsync(d_std, std, 3 * sizeof(float), cudaMemcpyHostToDevice, stream));

        if (malloc_host(&h_matrix, 12 * sizeof(float))<0) {
            std::cout << "malloc_device h_matrix failed" << std::endl;
            return -1;
        }

        if (malloc_device(&d_matrix, 12 * sizeof(float))<0) {
            std::cout << "malloc_device d_matrix failed" << std::endl;
            return -1;
        }

        if (malloc_device(&image_data_device, kChannel * kImageHMax * kImageWMax * sizeof(unsigned char))<0) {
            std::cout << "malloc_device image_data_device failed" << std::endl;
            return -1;
        }

        if (malloc_host(&h_filtered_boxes, sizeof(gBox) * output_numbox)) {
            std::cout << "malloc_host h_filtered_boxes failed" << std::endl;
            return -1;
        }
        if (malloc_device(&d_filtered_boxes, sizeof(gBox) * output_numbox)) {
            std::cout << "malloc_device d_filtered_boxes failed" << std::endl;
            return -1;
        }
        if (malloc_host(&h_box_count, sizeof(int) * output_batch)) {
            std::cout << "malloc_host h_box_count failed" << std::endl;
            return -1;
        }
        if (malloc_device(&d_box_count, sizeof(int) * output_batch)) {
            std::cout << "malloc_device d_box_count failed" << std::endl;
            return -1;
        }

        init_finished = true;
        return 0;
    }
    int build() {
        // 生成engine
        /* 这里面为何用智能指针，是因为如果不用构建，这些变量都用不到
        */
        if (exists(trt_file)) {
            std::cout << "trt_file exists" << std::endl;
            return 0;
        }
        if (!exists(onnx_file)) {
            std::cout << "onnx_file not exists" << std::endl;
            return -1;
        }
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
        if (!builder) {
            std::cout << "createInferBuilder failed" << std::endl;
            return -1;
        }
        auto explictBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  // NOLINT   
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explictBatch));
        if (!network) {
            std::cout << "createNetworkV2 failed" << std::endl;
            return -1;
        }
        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
        if (!parser) {
            std::cout << "createParser failed" << std::endl;
            return -1;
        }
        // 创建引擎
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            std::cout << "createBuilderConfig failed" << std::endl;
            return -1;
        }
        // builder->setMaxBatchSize(1);
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        // 设置工作区的大小
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1<<28);
        
        // 创建onnxparser
        auto success= parser->parseFromFile(onnx_file.c_str(), 1);
        if (!success) {
            std::cout << "parseFromFile failed" << std::endl;
            return -1;
        }
        auto input = network->getInput(0);
        nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();

        if (!profile) {
            std::cout << "createOptimizationProfile failed" << std::endl;
            return -1;
        }
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 640, 640});
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, 640, 640});
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 3, 640, 640});
        config->addOptimizationProfile(profile);
        auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        if (!serialized_engine) {
            std::cout << "buildSerializedNetwork failed" << std::endl;
            return -1;
        }
        saveData(trt_file, reinterpret_cast<char*>(serialized_engine->data()), serialized_engine->size());
        printf("Serialize engine success\n");
        return 0;
    }

    int preprocess(cv::Mat &img) {
        // 前处理
        // std::cout << "preprocess" << std::endl;
        calculate_matrix((float*)h_matrix, img.cols, img.rows, kInputW, kInputH);
#if PROCESSONGPU
// #if 0
        float *d2i = (float*)h_matrix + 6;
        checkRuntime(cudaMemcpyAsync(d_matrix, d2i, 6 * sizeof(float), cudaMemcpyHostToDevice, stream));
        checkRuntime(cudaMemset((char*)d_box_count, 0, sizeof(int)));
        checkRuntime(cudaMemset((char*)d_filtered_boxes, 0, output_numbox * sizeof(gBox)));
        checkRuntime(cudaMemcpyAsync(image_data_device, img.data, img.total() * img.channels(), cudaMemcpyHostToDevice, stream));
        checkRuntime(cudaStreamSynchronize(stream));
        int ret = preprocess_gpu((unsigned char*)image_data_device, (float*)input_data_device, 
            img.cols, img.rows, kInputW, kInputH, 
            (float*)d_matrix, (float*)d_mean, (float*)d_std);
        if (ret < 0) {
            std::cout << "preprocess failed" << std::endl;
            return -1;
        }
#else
#if USEOPENCV
        float *i2d = (float*)h_matrix;
        preprocess_opencv_cpu(img, (float*)input_data_host, i2d, kInputW, kInputH, (float *)h_mean, (float*)h_std);
#else
        float *d2i = (float*)h_matrix + 6;
        preprocess_cpu_v2(img, (float*)input_data_host, d2i, kInputW, kInputH, false);

#endif
        checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));
        checkRuntime(cudaStreamSynchronize(stream));
#endif
        return 0;
    }
    int postprocess() {
        // 后处理
        // std::cout << "postprocess()" << std::endl;
#if PROCESSONGPU
        cudaMemcpy(output_data_host, output_data_device, output_numel * sizeof(float), cudaMemcpyDeviceToHost);
        // saveWeight("yolov5output2.data", (float*)output_data_host, output_numel);

        // 调用封装的函数
        int ret = postprocess_cuda((float*)output_data_device, (gBox *)d_filtered_boxes, 
                (int *)d_box_count, (int *)h_box_count, output_batch, output_numbox, output_numprob, 
                kConfThresh, kNmsThresh, (float *)d_matrix);
        if (ret < 0) {
            std::cout << "postprocess failed" << std::endl;
            return -1;
        } 
        // // 在后处理的函数内部，已经完成了h_box_count的同步
        // checkRuntime(cudaMemcpyAsync(h_box_count, d_box_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
        // checkRuntime(cudaStreamSynchronize(stream)); // 我要用h_box_count，所以要同步
        for (int i = 0; i < output_batch; i++) {
            printf("h_box_count[%d]: %d\n", i, *((int*)h_box_count + i));
        }
        checkRuntime(cudaMemcpyAsync(h_filtered_boxes, d_filtered_boxes, (*((int*)h_box_count)) * sizeof(gBox), cudaMemcpyDeviceToHost, stream));
        checkRuntime(cudaStreamSynchronize(stream));
#else
        checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream));
        checkRuntime(cudaStreamSynchronize(stream));
#endif
        return 0;
    }
    void drawimg(cv::Mat &img, const std::string& savepath) {
        // 绘制
        std::cout << "draw()" << std::endl;
#if PROCESSONGPU
        int count = *(int *)h_box_count;
        printf("count: %d\n", count);
        cv::Mat img_draw = draw_gpu((gBox*)h_filtered_boxes, count, img);
        // cv::imwrite("result_gpu_v2.jpg", img_draw);
        // cv::imwrite("result_gpu_" + std::to_string(index) + ".jpg", img_draw);
        cv::imwrite(savepath, img_draw);
        index++;
#else
        float *d2i = (float*)h_matrix + 6;
        std::vector<Box> resboxes = postprocess_cpu((float*)output_data_host, output_batch, output_numbox, output_numprob, kConfThresh, kNmsThresh);
        // 绘制结果
        cv::Mat img_draw = draw_cpu(resboxes, img, d2i);
        cv::imwrite("result_cpu_v2.jpg", img_draw);
#endif
    }
    int inference() {
        // 推理
        // std::cout << "inference()" << std::endl;
        void* bindings[] = {input_data_device, output_data_device};
        bool success = context->executeV2(bindings);
        if (!success) {
            std::cout << "enqueue failed" << std::endl;
            return -1;
        }
        return 0;
    }
    int forward_image(std::string &img_file) {
        // auto imgname = extractFilenameWithoutExtension(img_file);
        auto imgname = extractFilename(img_file);
        printf("imgname:%s\n", imgname.c_str());
        std::string savepath = "result_" + imgname; 
        printf("savepath:%s\n", savepath.c_str());
        // 处理一张图片
        cv::Mat img = cv::imread(img_file);
        if (img.empty()) {
            printf("cv::imread %s failed\n", img_file.c_str());
            return -1;
        }
        int ret;
        auto start = std::chrono::high_resolution_clock::now();
        ret = preprocess(img);
        if (ret < 0) {
            std::cout << "preprocess failed" << std::endl;
            return -1;
        }
        ret = inference();
        if (ret < 0) {
            std::cout << "inference failed" << std::endl;
            return -1;
        }
        ret = postprocess();
        if (ret < 0) {
            std::cout << "postprocess failed" << std::endl;
            return -1;
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
        drawimg(img, savepath);
        return 0;
    }
};

int main() {
    std::string name = "yolov5s";
    Yolov5 yolo(name, 1<<28);
    std::string img_file = "bus.jpg";
    // yolo.forward_image(img_file);
    printf("totalsize:%d\n", totalsize);
    std::string txtfile = "output.txt";
    auto lines = readLinesFromFile(txtfile);
    for (auto line:lines) {
        printf("imgpath:%s\n", line.c_str());
        yolo.forward_image(line);
    }
    return 0;
}