// #include <NvInferRuntime.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <NvInfer.h>
#define print(x) std::cout << #x << ": " << x << std::endl;

void saveData(const std::string &filename, const char *data, const int size) {
    std::ofstream outfile(filename, std::ios::binary);
    assert(outfile.is_open() && "saveData failed");
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

void saveWeight(const std::string &filename, const float *data, const int size) {
    std::ofstream outfile(filename, std::ios::binary);
    assert(outfile.is_open() && "saveData failed");
    outfile.write(reinterpret_cast<const char *>(&size), sizeof(int));
    outfile.write(reinterpret_cast<const char *>(data), size * sizeof(float));
    outfile.close();
}

std::vector<float> loadWeight(const std::string &filename) {
    std::ifstream infile(filename, std::ios::binary);
    assert(infile.is_open() && "loadWeight failed");
    int size;
    infile.read(reinterpret_cast<char *>(&size), sizeof(int));
    std::vector<float> data(size);
    infile.read(reinterpret_cast<char *>(data.data()), size * sizeof(float));
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

int main() {
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger);
    auto explictBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  // NOLINT   
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explictBatch);
    const int input_size = 3;
    nvinfer1::ITensor *input = network->addInput("data", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, input_size, 1, 1});
    const float *fc1_weight_data = new float[input_size * 2]{0.1,0.2,0.3,0.4,0.5,0.6};
    const float *fc1_bias_data = new float[2]{0.1,0.5};
    // nvinfer1::Weights fc1_weight{nvinfer1::DataType::kFLOAT, fc1_weight_data, input_size * 2}; 
    // nvinfer1::Weights fc1_bias{nvinfer1::DataType::kFLOAT, fc1_bias_data, 2};
    const int output_size = 2;
    saveWeight("fc1_weight", fc1_weight_data, input_size * output_size);
    saveWeight("fc1_bias", fc1_bias_data, output_size);
    auto fc1_weight_data_vec = loadWeight("fc1_weight");   
    auto fc1_bias_data_vec = loadWeight("fc1_bias");
    print(fc1_bias_data_vec.size());
    print(fc1_weight_data_vec.size());
    nvinfer1::Weights fc1_weight{nvinfer1::DataType::kFLOAT, fc1_weight_data_vec.data(), fc1_weight_data_vec.size()}; 
    nvinfer1::Weights fc1_bias{nvinfer1::DataType::kFLOAT, fc1_bias_data_vec.data(), fc1_bias_data_vec.size()};
    nvinfer1::IFullyConnectedLayer *fc1 = network->addFullyConnected(*input, output_size, fc1_weight, fc1_bias);
    nvinfer1::IActivationLayer *sigmoid = network->addActivation(*fc1->getOutput(0), nvinfer1::ActivationType::kSIGMOID); 
    sigmoid->getOutput(0)->setName("output");
    network->markOutput(*sigmoid->getOutput(0));
    builder->setMaxBatchSize(1);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 28);
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        std::cerr << "build engine failed" << std::endl;
        return -1;
    }
    nvinfer1::IHostMemory *serialized_engine = engine->serialize();
    saveData("engine.trtmodel", reinterpret_cast<char*>(serialized_engine->data()), serialized_engine->size());
    delete engine;
    delete config;
    delete network;
    delete builder; 
    printf("Serialize engine success\n");
    return 0;
}