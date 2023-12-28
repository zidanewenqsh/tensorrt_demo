#include <iostream>
#include <vector>
#include <mutex>
#include <atomic>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)
static bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
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
    MemoryPool(size_t blockSize, size_t blockCount) 
        : blockSize(blockSize), stop(false) {
        // 初始化时预分配一定数量的内存块
        for (size_t i = 0; i < blockCount; ++i) {
            char* block;
            checkRuntime(cudaMallocHost((void**)&block, blockSize));
            freeBlocks.push_back(block);
        }
    }

    ~MemoryPool() {
        // 析构时释放所有内存块
        for (char* block : freeBlocks) {
            checkRuntime(cudaFreeHost(block));
        }
    }

    void* allocate() {
        if (stop.load()) {
            return nullptr; // 如果内存池已停止，直接返回
        }

        std::lock_guard<std::mutex> lock(mutex); // 确保线程安全

        if (freeBlocks.empty()) {
            // throw std::bad_alloc();
            char* block;
            checkRuntime(cudaMallocHost((void**)&block, blockSize));
            return block; 
        }

        char* block = freeBlocks.back();
        freeBlocks.pop_back();
        return block;
    }

    void deallocate(void* block) {
        if (stop.load()) {
            return; // 如果内存池已停止，直接返回
        }

        std::lock_guard<std::mutex> lock(mutex); // 确保线程安全

        freeBlocks.push_back(static_cast<char*>(block));
    }

    void stopPool() {
        stop.store(true);
    }

private:
    size_t blockSize;
    // size_t blockCount;
    std::vector<char*> freeBlocks;
    std::mutex mutex; // 用于线程安全的互斥锁
    std::atomic<bool> stop; // 控制内存池的停止
};

#if 0
// 示例使用
int main() {
    const size_t blockSize = 1024; // 假设每个内存块为1024字节
    const size_t blockCount = 10;  // 假设有10个内存块

    MemoryPool pool(blockSize, blockCount);
    for (int i = 0; i < 100; i++) {
        void* ptr1 = pool.allocate();
        // 使用内存...
        printf("alloc ptr:%p\n", ptr1);
        if (i % 2 == 1) {
            pool.deallocate(ptr1);
            printf("dealloc ptr:%p\n", ptr1);
        }

    }

    // 停止内存池
    pool.stopPool();

    // ...
    return 0;
}
#else
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

void allocateAndDeallocate(MemoryPool& pool, int threadId) {
    for (int i = 0; i < 10; i++) {
        void* ptr = pool.allocate();
        std::cout << "Thread " << threadId << " allocated ptr: " << ptr << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // 可选：释放内存
        pool.deallocate(ptr);
        std::cout << "Thread " << threadId << " deallocated ptr: " << ptr << std::endl;
    }
}

int main() {
    const size_t blockSize = 4096;
    const size_t blockCount = 10;
    MemoryPool pool(blockSize, blockCount);

    std::vector<std::thread> threads;

    // 创建多个线程进行内存分配和释放
    for (int i = 0; i < 5; i++) {
        threads.emplace_back(allocateAndDeallocate, std::ref(pool), i);
    }

    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }

    // 停止内存池
    pool.stopPool();

    return 0;
}

#endif