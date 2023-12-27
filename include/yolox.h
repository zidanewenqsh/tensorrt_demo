#ifndef __YOLOX_H__
#define __YOLOX_H__

#include "my_tensorrt.h"
#include "yolox_config.h"
#include "preprocess_gpu.cuh"
#include "postprocess_gpu.cuh"
#include <cstdlib>
#include <chrono>
#include "preprocess.h"
#include "postprocess.h"
#include "yolox_utils.h"


class Yolox:public MyTensorRT {
public:
    Yolox(std::string& name, int buffer_size);
    ~Yolox();
    int init() override;
    int preprocess(cv::Mat &img) override;
    int postprocess() override;
    void drawimg(cv::Mat &img, const std::string& savepath);
    int inference() override;
    int forward_image(std::string &img_file);
};

#endif