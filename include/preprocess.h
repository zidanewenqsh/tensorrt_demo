#ifndef __PREPROCESS_H__
#define __PREPROCESS_H__
#include <opencv2/opencv.hpp>
#include <vector>
void preprocess_cpu(const cv::Mat& img, float *ret, float*matrix, int w, int h, bool norm=false);
#endif