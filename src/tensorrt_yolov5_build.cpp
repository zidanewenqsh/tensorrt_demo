#include <NvInferRuntime.h>
#include <cstdio>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#define print(x) std::cout << #x << ": " << x << std::endl;

void saveData(const std::string &filename, const char *data, const int size) {
    std::ofstream outfile(filename, std::ios::binary);
    assert(outfile.is_open() && "saveData failed");
    // print(size);
    outfile.write(reinterpret_cast<const char *>(data), size);
    outfile.close();
}

std::vector<char> loadData(const std::string &filename) {
    std::ifstream infile(filename, std::ios::binary);
    assert(infile.is_open() && "loadData failed");
    infile.seekg(0, std::ios::end);
    int size = infile.tellg();
    infile.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    infile.read(reinterpret_cast<char *>(data.data()), size);
    infile.close();
    return data;
}

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

bool exists(const std::string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), F_OK) == 0;
#endif
}

int build(std::string onnx_file, std::string trt_file) {
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
    // auto profile = std::unique_ptr<nvinfer1::IOptimizationProfile>(builder->createOptimizationProfile());
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

int main() {

    std::string onnx_file = "yolov5s.onnx";
    std::string trt_file = "yolov5_engine.trtmodel";
    if (build(onnx_file, trt_file) != 0) {
        std::cout << "build failed" << std::endl;
        return -1;
    } 
    std::cout << "build success" << std::endl;
    return 0;
}