#include <iostream>
#include <cuda_runtime.h>

__global__ void simpleKernel(int *d_data, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < width && idy < height) {
        printf("idx: %d, idy: %d\n", idx, idy);
        int index = idy * width + idx;
        d_data[index] = idx + idy;  // 设置每个元素的值为其索引之和
    }
}
void runSimpleKernel(int *d_data, int width, int height) {
    // 设置 CUDA 网格和块的大小
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 调用 CUDA 内核
    simpleKernel<<<gridSize, blockSize>>>(d_data, width, height);

    // 等待 CUDA 内核完成
    cudaDeviceSynchronize();

    // 检查是否有错误发生
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    }
}

int main() {
    const int width = 640;
    const int height = 640;
    const int size = width * height;
    const int bytes = size * sizeof(int);

    // 分配主机内存
    int *h_data = new int[size];

    // 分配设备内存
    int *d_data;
    cudaMalloc(&d_data, bytes);

    // // 设置 CUDA 网格和块的大小
    // dim3 blockSize(4, 4);
    // dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // // 调用 CUDA 内核
    // simpleKernel<<<gridSize, blockSize>>>(d_data, width, height);
    // 使用封装的函数调用 CUDA 内核
    runSimpleKernel(d_data, width, height);
    // 将结果从设备内存复制回主机内存
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    // 打印结果
    // for (int y = 0; y < height; y++) {
    //     for (int x = 0; x < width; x++) {
    //         std::cout << h_data[y * width + x] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // 清理
    cudaFree(d_data);
    delete[] h_data;

    return 0;
}
