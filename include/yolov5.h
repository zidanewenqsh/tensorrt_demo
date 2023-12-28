#ifndef __YOLOV5_H__
#define __YOLOV5_H__

#include "my_tensorrt.h"
#include "yolov5_config.h"
#include "preprocess_gpu.cuh"
#include "postprocess_gpu.cuh"
#include <cstdlib>
#include <chrono>
#include "preprocess.h"
#include "postprocess.h"
#include "yolov5_utils.h"
#include "mempool.h"
#include "mempool_gpu.h"
class Yolov5:public MyTensorRT {
public:
    Yolov5(std::string& name, int buffer_size);
    ~Yolov5();
    int init() override;
    int preprocess(cv::Mat &img) override;
    int postprocess() override;
    void drawimg(cv::Mat &img, const std::string& savepath);
    int inference() override;
    int forward_image(std::string &img_file);
    int malloc_host(void **ptr, size_t size);
    int malloc_device(void** ptr, size_t size);
    int free_host(void *ptr);
    int free_device(void * ptr);
// private:
    // MemPool mempool;
    // MemPoolGpu mempool_gpu;
private:
    std::unique_ptr<MemoryPool> mempool;
    std::unique_ptr<MemoryPoolGpu> mempool_gpu;
};

#endif