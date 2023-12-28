#include "mempool.h"
#include <cstring>
int main() {
    MemoryPool pool;
    // std::vector<void*> smallAll;
    // std::vector<void*> largeAll;
#if 1
    int i;
    void* smallMemory = pool.allocate(256);
    std::cout << "Allocated small memory at: " << smallMemory << std::endl;
    memcpy((char*)smallMemory, &i, sizeof(int));

#endif
    return 0;
}