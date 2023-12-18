// #include <NvInferRuntime.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <NvInfer.h>

#define oldversion 0
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
    const int output_size = 2;
    // 标记输入层和创建输入数据
    nvinfer1::ITensor *input = network->addInput("data", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, input_size, 1, 1});
    const float *fc1_weight_data = new float[input_size * output_size]{0.1,0.2,0.3,0.4,0.5,0.6};
    const float *fc1_bias_data = new float[2]{0.1,0.5};
    // nvinfer1::Weights fc1_weight{nvinfer1::DataType::kFLOAT, fc1_weight_data, input_size * 2}; 
    // nvinfer1::Weights fc1_bias{nvinfer1::DataType::kFLOAT, fc1_bias_data, 2};
    saveWeight("fc1_weight", fc1_weight_data, input_size * output_size);
    saveWeight("fc1_bias", fc1_bias_data, output_size);
    auto fc1_weight_data_vec = loadWeight("fc1_weight");   
    auto fc1_bias_data_vec = loadWeight("fc1_bias");
    nvinfer1::Weights fc1_weight{nvinfer1::DataType::kFLOAT, fc1_weight_data_vec.data(), static_cast<int64_t>(fc1_weight_data_vec.size())}; 
    nvinfer1::Weights fc1_bias{nvinfer1::DataType::kFLOAT, fc1_bias_data_vec.data(), static_cast<int64_t>(fc1_bias_data_vec.size())};
#if oldversion
    // 设定最大batch size
    builder->setMaxBatchSize(1);
    nvinfer1::IFullyConnectedLayer *fc1 = network->addFullyConnected(*input, output_size, fc1_weight, fc1_bias);
    nvinfer1::IActivationLayer *sigmoid = network->addActivation(*fc1->getOutput(0), nvinfer1::ActivationType::kSIGMOID); 
#else
    // 将输入张量转换为2D，以进行矩阵乘法
    nvinfer1::IShuffleLayer* shuffleLayer = network->addShuffle(*input);
    shuffleLayer->setReshapeDimensions(nvinfer1::Dims2{input_size, 1});

    // 创建权重矩阵的常量层
    nvinfer1::IConstantLayer* weightLayer = network->addConstant(nvinfer1::Dims2{output_size, input_size}, fc1_weight);

    // 添加矩阵乘法层
    nvinfer1::IMatrixMultiplyLayer* matMulLayer = network->addMatrixMultiply(
        *weightLayer->getOutput(0), nvinfer1::MatrixOperation::kNONE,
        *shuffleLayer->getOutput(0), nvinfer1::MatrixOperation::kNONE
    );

    // 添加偏置
    nvinfer1::IConstantLayer* biasLayer = network->addConstant(nvinfer1::Dims2{output_size, 1}, fc1_bias);
    nvinfer1::IElementWiseLayer* addBiasLayer = network->addElementWise(
        *matMulLayer->getOutput(0), *biasLayer->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);

    // 添加激活层
    nvinfer1::IActivationLayer* sigmoid = network->addActivation(*addBiasLayer->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
#endif
    // 标记输出层
    sigmoid->getOutput(0)->setName("output");
    network->markOutput(*sigmoid->getOutput(0));
#if oldversion
    builder->setMaxBatchSize(1);
#endif
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
// 设置工作区的大小
#if oldversion
    config->setMaxWorkspaceSize(1 << 28);
#else
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1<<28);
#endif
// 构建并序列化引擎
#if oldversion
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        std::cerr << "build engine failed" << std::endl;
        delete [] fc1_weight_data;
        delete [] fc1_bias_data;
        delete config;
        delete network;
        delete builder; 
        return -1;
    }
    nvinfer1::IHostMemory *serialized_engine = engine->serialize();
    // print(serialized_engine->size());
#else
    // 使用 buildSerializedNetwork 构建并序列化网络
    nvinfer1::IHostMemory* serialized_engine = builder->buildSerializedNetwork(*network, *config);
    if (!serialized_engine) {
        std::cerr << "build engine failed" << std::endl;
        delete [] fc1_weight_data;
        delete [] fc1_bias_data;
        delete config;
        delete network;
        delete builder; 
        return -1;
    }
#endif
    saveData("engine.trtmodel", reinterpret_cast<char*>(serialized_engine->data()), serialized_engine->size());
    delete [] fc1_weight_data;
    delete [] fc1_bias_data;
#if oldversion
    delete engine;
#endif
    delete config;
    delete network;
    delete builder; 
    printf("Serialize engine success\n");
    return 0;
}

// 99716
// 100172