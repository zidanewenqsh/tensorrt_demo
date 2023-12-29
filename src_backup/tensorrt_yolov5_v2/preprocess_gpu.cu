#include "preprocess_gpu.cuh"

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

__global__ void preprocess_kernel(const unsigned char* input, float* output, const float* matrix, int width, int height, int w, int h, const float* mean, const float* std) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= w && idy >= h) return ;
    float x = matrix[0] * idx + matrix[1] * idy + matrix[2];
    float y = matrix[3] * idx + matrix[4] * idy + matrix[5];

    int channelSize = w * h;
    int index = idy * w + idx;
    output[index] = (bilinearInterpolateChannel(input, width, height, x, y, 2, 3) / 255.0f - mean[0]) / std[0];
    output[channelSize + index] = (bilinearInterpolateChannel(input, width, height, x, y, 1, 3) / 255.0f - mean[1]) / std[1];
    output[2 * channelSize + index] = (bilinearInterpolateChannel(input, width, height, x, y, 0, 3) / 255.0f - mean[2]) / std[2];
}

int preprocess_gpu(const unsigned char* d_input, float* d_output, int original_width, int original_height, int target_width, int target_height, const float* d_matrix, const float* d_mean, const float* d_std) {
    // 设置 CUDA 网格和块的大小
    dim3 blockSize(16, 16);
    dim3 gridSize((target_width + blockSize.x - 1) / blockSize.x, (target_height + blockSize.y - 1) / blockSize.y);
    // 调用 CUDA 内核
    preprocess_kernel<<<gridSize, blockSize>>>(d_input, d_output, d_matrix, original_width, original_height, target_width, target_height, d_mean, d_std);

    // 检查 CUDA 是否成功执行
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Preprocess CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    // 同步 CUDA 设备以确保所有操作都已完成
    cudaDeviceSynchronize();
    return 0;
}

