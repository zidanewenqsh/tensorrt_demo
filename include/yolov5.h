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
};

#endif