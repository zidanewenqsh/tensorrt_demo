#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

__host__ float bilinearInterpolateChannel_cpu(const unsigned char* img, int width, int height, float x, float y, int channel, int channels, const unsigned char borderValue = 114) {
    if (x < 0 || y < 0 || x >= width - 1 || y >= height - 1) {
        return static_cast<float>(borderValue);
    }

    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    float a = x - x1;
    float b = y - y1;

    int idx1 = (y1 * width + x1) * channels + channel;
    int idx2 = (y1 * width + x2) * channels + channel;
    int idx3 = (y2 * width + x1) * channels + channel;
    int idx4 = (y2 * width + x2) * channels + channel;

    float inter1 = (1 - a) * img[idx1] + a * img[idx2];
    float inter2 = (1 - a) * img[idx3] + a * img[idx4];

    return (1 - b) * inter1 + b * inter2;
}
__device__ float bilinearInterpolateChannel(const unsigned char* img, int width, int height, float x, float y, int channel, int channels, const unsigned char borderValue = 114) {
    if (x < 0 || y < 0 || x >= width - 1 || y >= height - 1) {
        return static_cast<float>(borderValue);
    }

    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    float a = x - x1;
    float b = y - y1;

    int idx1 = (y1 * width + x1) * channels + channel;
    int idx2 = (y1 * width + x2) * channels + channel;
    int idx3 = (y2 * width + x1) * channels + channel;
    int idx4 = (y2 * width + x2) * channels + channel;

    float inter1 = (1 - a) * img[idx1] + a * img[idx2];
    float inter2 = (1 - a) * img[idx3] + a * img[idx4];

    return (1 - b) * inter1 + b * inter2;
}

// __global__ void simpleKernel(int *d_data, int width, int height) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;

//     if (idx < width && idy < height) {
//         printf("idx: %d, idy: %d\n", idx, idy);
//         int index = idy * width + idx;
//         d_data[index] = idx + idy;  // 设置每个元素的值为其索引之和
//     }
// }
__global__ void preprocess_kernel(const unsigned char* input, float* output, const float* matrix, int width, int height, int w, int h, const float* mean, const float* std) {
    
    // printf("blockIdx.x: %d, blockIdx.y: %d, blockDim.x: %d, blockDim.y: %d\n", blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("idx: %d, idy: %d\n", idx, idy);
    // printf("w: %d, h: %d\n", w, h);
    if (idx >= w && idy >= h) return ;
    // if (idx == 0) {
    //     printf("idx: %d, idy: %d\n", idx, idy);
    // }
    float x = matrix[0] * idx + matrix[1] * idy + matrix[2];
    float y = matrix[3] * idx + matrix[4] * idy + matrix[5];

    int channelSize = w * h;
    int index = idy * w + idx;
    // printf("channelSize: %d, index: %d\n", channelSize, index);
    // output[index] = 1; 
    // output[channelSize + index] = 2;
    // output[2 * channelSize + index] = 3;
    // output[index] = bilinearInterpolateChannel(input, width, height, x, y, 2, 3);
    // float a = bilinearInterpolateChannel(input, width, height, x, y, 2, 3) / 255.0f - mean[0];
    // float d = bilinearInterpolateChannel_cpu(img.data, img.cols, img.rows, img.cols-1.5f, img.rows-1.5f, 2, 3);
    // printf("a=%f\n", a);
    output[index] = (bilinearInterpolateChannel(input, width, height, x, y, 2, 3) / 255.0f - mean[0]) / std[0];
    output[channelSize + index] = (bilinearInterpolateChannel(input, width, height, x, y, 1, 3) / 255.0f - mean[1]) / std[1];
    output[2 * channelSize + index] = (bilinearInterpolateChannel(input, width, height, x, y, 0, 3) / 255.0f - mean[2]) / std[2];
}
void preprocess_gpu(const unsigned char* d_input, float* d_output, const float* d_matrix, int original_width, int original_height, int target_width, int target_height, const float* d_mean, const float* d_std) {
    printf("target_width: %d, target_height: %d\n", target_width, target_height);
    // 设置 CUDA 网格和块的大小
    int max_wh = std::max(target_height, target_width);
    if (max_wh > 1024) {
        std::cerr << "Error: max width or height is larger than 1024." << std::endl;
        return;
    } 
    // for (int i = 0 ; i < 10; i++) {
    //     printf("d_input[%d]: %d\n", i, d_input[i]);
    // }
    // int bSize
    dim3 blockSize(16, 16);
    dim3 gridSize((target_width + blockSize.x - 1) / blockSize.x, (target_height + blockSize.y - 1) / blockSize.y);
    // int a = (target_width + blockSize.x - 1) / blockSize.x;
    // int b = (target_height + blockSize.y - 1) / blockSize.y;
    // int c = a * b;
    // printf("gridSize: %d, %d, %d\n", a, b, c);

    // float d = (bilinearInterpolateChannel(d_input, original_width, original_height, target_width, target_height, 2, 3) / 255.0f - mean[0]) / std[0];
    // float d = bilinearInterpolateChannel_cpu(d_input, original_width, original_height, original_width-1.5f, original_height-1.5f, 2, 3);
    // printf("d=%f\n", d);
    // 调用 CUDA 内核
    preprocess_kernel<<<gridSize, blockSize>>>(d_input, d_output, d_matrix, original_width, original_height, target_width, target_height, d_mean, d_std);

    // 检查 CUDA 是否成功执行
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    }

    // 同步 CUDA 设备以确保所有操作都已完成
    cudaDeviceSynchronize();
}


// 假设 preprocess_kernel 和其他相关函数/内核已经定义
std::unique_ptr<float[]> calculate_matrix(int width, int height, int w, int h) {
    float scale = std::min((float)w/width, (float)h/height);
    // float *matrix = new float[12];
    auto matrix = std::unique_ptr<float[]>(new float[12]);
    float *i2d = matrix.get();
    float *d2i = matrix.get() + 6;
    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = (-width * scale + w) / 2;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = (-height * scale + h) / 2;
    d2i[0] = 1 / scale;
    d2i[1] = 0;
    d2i[2] = (width * scale - w) / 2 / scale;
    d2i[3] = 0 ;
    d2i[4] = 1 / scale;
    d2i[5] = (height * scale - h) / 2 / scale;
    return matrix;
}

int main() {
    // float matrix[6] = { /* 初始化变换矩阵 */ };
    // float mean[3] = {0.485, 0.456, 0.406};
    // float std[3] = {0.229, 0.224, 0.225};
    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};
    // 加载图像
    cv::Mat img = cv::imread("bus.jpg", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Image not found." << std::endl;
        return -1;
    }
    printf( "width: %d, height: %d\n", img.cols, img.rows );
    // 目标尺寸
    int w = 640, h = 640;

    auto matrix = calculate_matrix(img.cols, img.rows, w, h);
    for (int i = 0; i < 12; i++) {
        std::cout << matrix[i] << " ";
    }
    std::cout << std::endl;
    // float *i2d = (float*)matrix.get();
    float *d2i = (float*)matrix.get() + 6;
    // 分配cpu内存
    // float *output = new float[3 * w * h];
    
    // 分配和初始化 GPU 内存
    float *d_mean;
    float *d_std;
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
    float d = bilinearInterpolateChannel_cpu(img.data, img.cols, img.rows, img.cols-1.5f, img.rows-1.5f, 2, 3);
    printf("d=%f\n", d);
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
    // 从 GPU 获取结果
    std::vector<float> output(3 * w * h);
    cudaMemcpy(output.data(), d_output, 3 * w * h * sizeof(float), cudaMemcpyDeviceToHost);
    int c = w * h;
    for (int i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
        std::cout << output[i + c] << " ";
        std::cout << output[i + c * 2] << " ";
    }
    std::cout << std::endl;
    // 清理 GPU 内存
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_matrix);
    cudaFree(d_mean);
    cudaFree(d_std);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);

    // 处理 output...
    // ...

    return 0;
}

// nvcc -o my_program preprocess_gpu_demo.cu  -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
// 