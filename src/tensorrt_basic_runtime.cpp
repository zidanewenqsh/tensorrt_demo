#include <cstdio>
#include <iostream>
#include <fstream>
#include <cassert>
#include <utility>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#define print(x) std::cout << #x << ": " << x << std::endl;
#define oldversion 0

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

// 加载模型
std::vector<char> loadData(std::string &filepath){
    std::ifstream infile(filepath, std::ios::binary);
    assert(infile.is_open() && "loadData failed");
    infile.seekg(0, std::ios::end);
    int size = infile.tellg();
    infile.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    infile.read(reinterpret_cast<char *>(data.data()), size);
    infile.close();
    return data;
}

int main (int argc, char const *argv[]) {
    // std::string serialized_engine_file = "engine.trtmodel";
    // std::string serialized_engine_file = "mlp.engine";
    if (argc < 2) {
        std::cout << "Usage: ./trt_test serialized_engine_file" << std::endl;
        return -1;
    }
    const int input_size = 3;
    const int output_size = 2;
    // 创建一个runtime对象
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    // 反序列化生成engine
    std::string serialized_engine_file = argv[1];
    std::vector<char> engine_data = loadData(serialized_engine_file);
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (engine == nullptr) {
        std::cout << "deserializeCudaEngine failed" << std::endl;
        return -1;
    }
    // 创建执行上下文
#if 0
    nvinfer1::IExecutionContext* context = serialized_engine->createExecutionContext();
#else
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (context == nullptr) {
        std::cout << "createExecutionContext failed" << std::endl;
        return -1;
    }
#endif
    // 输入数据
    // float host_input_data = new float[input_size]{1,2,3};
    std::unique_ptr<float[]> host_input_data(new float[input_size]{2,4,8});
    int host_intput_size = input_size * sizeof(float);
    float *device_input_data = nullptr;
    // 输出数据
    // float host_output_data = new float[output_size]{0};
    std::unique_ptr<float[]> host_output_data(new float[output_size]{0});
    int host_output_size = output_size * sizeof(float);
    float *device_output_data = nullptr;
    
    // 创建 CUDA 流
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // 申请device内存
    cudaMalloc((void **)&device_input_data, host_intput_size);
    cudaMalloc((void**)&device_output_data, host_output_size);
    cudaMemcpyAsync(device_input_data, host_input_data.get(), host_intput_size, cudaMemcpyHostToDevice, stream);


#if 1
    // 准备绑定缓冲区
    void* bindings[] = {device_input_data, device_output_data};
#if oldversion
    bool status = context->enqueueV2((void**)bindings, stream, nullptr);
#else
    bool status = context->executeV2(bindings);
#endif
#else
    // 这一部分没有成功实现
    // // 准备绑定缓冲区
    // void* bindings[] = {device_input_data, device_output_data};
    // // 确保所有异步操作都完成了
    // cudaStreamSynchronize(stream);
    // // 使用 enqueueV3 启动异步推理
    // bool status = context->enqueueV3(stream);
#endif
    if (!status) {
        cudaFree(device_input_data);
        cudaFree(device_output_data);
        std::cout << "enqueue failed" << std::endl;
        return -1;
    }

    // 复制输出数据回主机端
    cudaMemcpyAsync(host_output_data.get(), device_output_data, host_output_size, cudaMemcpyDeviceToHost, stream);

    // 等待 CUDA 流完成
    cudaStreamSynchronize(stream);

    // ... 后处理和资源释放代码 ...

    // 释放设备端内存
    cudaFree(device_input_data);
    cudaFree(device_output_data);

    // 销毁 CUDA 流和执行上下文
    cudaStreamDestroy(stream);
#if 0
    delete context;
#endif
    // 打印结果
    for (int i = 0; i < output_size; i++) {
        std::cout << "output[" << i << "] = " << host_output_data[i] << std::endl;
    }

    
    
    return 0;
    
}