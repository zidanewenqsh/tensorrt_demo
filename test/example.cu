#include <iostream>
#include <cuda_runtime.h>

// CUDA 内核函数，每个元素乘以2
__global__ void doubleElements(int* array, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("blockIdx.x: %d, blockIdx.y: %d, blockDim.x: %d, blockDim.y: %d\n", blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);
    printf("blockIdx.x = %d, blockDim.x = %d, threadIdx.x = %d, idx = %d\n", blockIdx.x, blockDim.x, threadIdx.x, idx);
    if (idx < N) {
        array[idx] *= 2;
    }
}
void runKernel(int* deviceArray, int N) {
    // 配置 CUDA 内核的执行参数
    int blockSize = 256;  // 可以根据需要调整
    int numBlocks = (N + blockSize - 1) / blockSize;

    // 执行 CUDA 内核
    doubleElements<<<numBlocks, blockSize>>>(deviceArray, N);
}

int main() {
    const int N = 10;
    int hostArray[N];

    // 初始化数组
    for (int i = 0; i < N; ++i) {
        hostArray[i] = i;
    }

    // 分配设备内存
    int* deviceArray;
    cudaMalloc(&deviceArray, N * sizeof(int));

    // 将数据从主机复制到设备
    cudaMemcpy(deviceArray, hostArray, N * sizeof(int), cudaMemcpyHostToDevice);

    // 执行 CUDA 内核
    // doubleElements<<<1, N>>>(deviceArray, N);
    // 调用函数执行 CUDA 内核
    runKernel(deviceArray, N);
    // 将数据从设备复制回主机
    cudaMemcpy(hostArray, deviceArray, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < N; ++i) {
        std::cout << "hostArray[" << i << "] = " << hostArray[i] << std::endl;
    }

    // 释放设备内存
    cudaFree(deviceArray);

    return 0;
}
