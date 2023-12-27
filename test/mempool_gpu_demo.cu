#include <cstddef>
#include <iostream>
#include <vector>
#include <mutex>
#include <memory>
#include <cuda_runtime.h>
#include <chrono>
// 检查 CUDA 运行时错误的辅助函数
#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)
bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        std::cerr << "runtime error " << file << ":" << line << " " << op << " failed.\n"
                  << "  code = " << err_name << ", message = " << err_message << std::endl;   
        return false;
    }
    return true;
}

class MemoryPool {
public:
    MemoryPool(size_t smallSize=0x100, int smallCount=0x10)
        :smallSize(smallSize), smallPoolSize(smallSize*smallCount) {
        // 初始化小块内存
        addSmallBlock(smallPoolSize);

        // 定义大块内存的潜在大小
        // largeBlockSizes = {2048, 4096, 8192, 16384, 32768}; // 2的倍数大小

        // for (int i = 0; i < 2; ++i) {
        //     if (1<<i < smallSize) continue;
        int i = 0;
        while ((1 << i++) < smallSize);
        // {
        //     i++;
        // }
        printf("i=%d\n", i);
        size_t blockSize = 1 << i;
        largeBlockSizes.push_back(blockSize);
        largePools.push_back({{}, blockSize}); // 为每个大小创建一个空内存池
    }

    ~MemoryPool() {
        // 释放所有小块内存的大块
        for (auto& block : smallPools) {
            checkRuntime(cudaFree(block.memory));
        }
        // checkRuntime(cudaFree(smallPoolMemory));
        // 释放所有大块内存
        for (auto& pool : largePools) {
            for (auto& block : pool.blocks) {
                if (block.memory != nullptr) {
                    checkRuntime(cudaFree(block.memory));
                }
            }
        }
    }

    void* allocate(size_t size) {
        if (size <= smallSize) {
            // 在小块内存中分配
            return allocateSmall(size);
        } else {
            // 在大块内存中分配
            return allocateLarge(size);
        }
    }


    void deallocate(void* block) {
        // 检查是否为大块内存
        for (auto& pool : largePools) {
            for (auto& blockInPool : pool.blocks) {
                if (blockInPool.memory == block) {
                    blockInPool.inUse = false; // 标记为未使用
                    std::cout << "deallocate " << block << std::endl; 
                    return;
                }
            }
        }
    }

private:
    struct LargeMemoryBlock {
        void* memory = nullptr;
        bool inUse = false;
    };

    struct LargePool {
        std::vector<LargeMemoryBlock> blocks;
        size_t blockSize;
    };

    struct SmallMemoryBlock {
        void* memory; // 指向整个大块内存的指针
        char* nextAvailable; // 指向下一个可用小块内存的指针
        size_t freeSize; // 当前已使用的大小
        size_t totalSize; // 整个大块内存的总大小
    };

    // void* smallPoolMemory;
    // char* nextAvailable;
    size_t smallSize; // 小于多少用小块
    size_t maxSize; // 小于多少用小块
    size_t smallPoolSize; // 每个大块内存的大小
    std::vector<LargePool> largePools;
    std::vector<size_t> largeBlockSizes;
    std::vector<SmallMemoryBlock> smallPools;  // 用于跟踪所有小块内存的大块
    void* currentSmallBlock = nullptr; // 当前小块内存的大块
    char* nextAvailable = nullptr; // 指向当前大块内存中下一个可用位置

    void addSmallBlock(size_t size) {
        SmallMemoryBlock newBlock;
        newBlock.totalSize = size;
        newBlock.freeSize = size;
        checkRuntime(cudaMalloc(&newBlock.memory, size));
        newBlock.nextAvailable = static_cast<char*>(newBlock.memory);
        smallPools.push_back(newBlock);
    }

    void* allocateSmall(size_t size) {
        for (auto& block : smallPools) {
            if (block.freeSize >= size) {
                void* allocatedMemory = block.nextAvailable;
                block.nextAvailable += size;
                block.freeSize -= size;
                return allocatedMemory;
            }
        }
        // 所有现有小块内存的大块都已满，分配一个新的大块
        addSmallBlock(smallPools[0].totalSize);
        return allocateSmall(size);
    }

    void* allocateLarge(size_t size) {
        // 如果预设的largeBlockSize不满足要求，扩容
        while (size > largeBlockSizes.back()) {
            largeBlockSizes.push_back(largeBlockSizes.back() * 2);
            largePools.push_back({{}, largeBlockSizes.back()}); // 为每个大小创建一个空内存池
        }
        for (auto& pool : largePools) {
            if (pool.blockSize >= size) {
                for (auto& block : pool.blocks) {
                    if (!block.inUse) {
                        block.inUse = true;
                        return block.memory;
                    }
                }

                LargeMemoryBlock newBlock;
                checkRuntime(cudaMalloc(&newBlock.memory, pool.blockSize));
                newBlock.inUse = true;
                pool.blocks.push_back(newBlock);
                return newBlock.memory;
            }
        }
        throw std::bad_alloc();
    }
};
int main() {
    MemoryPool pool;
    std::vector<void*> smallAll;
    std::vector<void*> largeAll;
#if 1
    for (int i = 0; i < 10; i++) {
        // 分配小块内存
        void* smallMemory = pool.allocate(256);
        std::cout << "Allocated 512 bytes of small memory at " << smallMemory << std::endl;
        smallAll.push_back(smallMemory);
        // 分配大块内存
        void* largeMemory = pool.allocate(4096);
        std::cout << "Allocated 4096 bytes of large memory at " << largeMemory << std::endl;
        largeAll.push_back(largeMemory);
        // pool.deallocate(largeMemory);
        if (i % 2 == 1) {
            // pool.deallocate(largeMemory);
            pool.deallocate(largeAll[0]);
        }
    }
    for (int i = 0; i < 10; i++) {
        pool.deallocate(smallAll[i]);
    }
    std::cout << "---------------" << std::endl;
    for (int i = 0; i < 10; i++) {
        pool.deallocate(largeAll[i]);
    }
#endif
#if 0
    // 分配小块内存
    void* smallMemory = pool.allocate(512);
    std::cout << "Allocated 512 bytes of small memory at " << smallMemory << std::endl;

    // 分配大块内存
    void* largeMemory = pool.allocate(4096);
    std::cout << "Allocated 4096 bytes of large memory at " << largeMemory << std::endl;

    // 释放内存
    pool.deallocate(smallMemory);
    pool.deallocate(largeMemory);
#endif
    return 0;
}
