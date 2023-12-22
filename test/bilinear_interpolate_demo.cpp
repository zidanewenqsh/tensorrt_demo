#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <vector>
#include <fstream>
#include <memory>
#include <cassert>
#include <opencv2/opencv.hpp>

uchar bilinearInterpolateChannel(const cv::Mat& img, float x, float y, int channel, uchar borderValue = 114) {
    // printf("x: %f, y: %f\n", x, y);
    // 检查坐标是否超出图像边界
    if (x < 0 || y < 0 || x >= img.cols - 1 || y >= img.rows - 1) {
        return borderValue;  // 返回常数值
    }

    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    // x1 = std::max(0, std::min(x1, img.cols - 1));
    // x2 = std::max(0, std::min(x2, img.cols - 1));
    // y1 = std::max(0, std::min(y1, img.rows - 1));
    // y2 = std::max(0, std::min(y2, img.rows - 1));

    float a = x - x1;
    float b = y - y1;

    float inter1 = (1 - a) * img.at<cv::Vec3b>(y1, x1)[channel] + a * img.at<cv::Vec3b>(y1, x2)[channel];
    float inter2 = (1 - a) * img.at<cv::Vec3b>(y2, x1)[channel] + a * img.at<cv::Vec3b>(y2, x2)[channel];

    return static_cast<uchar>((1 - b) * inter1 + b * inter2);
}


std::unique_ptr<float[]> calculate_matrix(int width, int height, int w, int h) {
    float scale = std::min((float)w/width, (float)h/height);
    // float *matrix = new float[12];
    auto matrix = std::unique_ptr<float[]>(new float[12]);
    float *i2d = matrix.get();
    float *d2i = matrix.get() + 6;
    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = (-width * scale + w) / 2;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = (-height * scale + h) / 2;
    d2i[0] = 1 / scale;
    d2i[1] = 0;
    d2i[2] = (width * scale - w) / 2 / scale;
    d2i[3] = 0 ;
    d2i[4] = 1 / scale;
    d2i[5] = (height * scale - h) / 2 / scale;
    return matrix;
}

cv::Mat bilinearInterpolateResize(const cv::Mat& img, int w, int h) {
    auto matrix = calculate_matrix(img.cols, img.rows, w, h);
    float *d2i = matrix.get() + 6;
    cv::Mat dst(h, w, CV_8UC3);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            float x = d2i[0] * j + d2i[1] * i + d2i[2];
            float y = d2i[3] * j + d2i[4] * i + d2i[5];
            float b = bilinearInterpolateChannel(img, x, y, 0, 114);
            float g = bilinearInterpolateChannel(img, x, y, 1, 114);
            float r = bilinearInterpolateChannel(img, x, y, 2, 114);
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(b, g, r);
        }
    }
    return dst;
}

int main() {
    // cv::Mat img = cv::imread("cat01.jpg");
    cv::Mat img = cv::imread("bus.jpg");
    int width = img.cols;
    int height = img.rows;
    int w = 640, h = 640;
    auto matrix = calculate_matrix(width, height, w, h);
    // float *i2d = matrix.get();
    float *d2i = matrix.get() + 6;
    for (int i = 0; i < 12; i++) {
        printf("%f ", matrix[i]);
    }
    printf("\n");
    // for (int i = 0; i < 12; i++) {
    //     printf("%f ", d2i[i]);
    // }
    cv::Mat dst(h, w, CV_8UC3);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {

            float x = d2i[0] * j + d2i[1] * i + d2i[2];
            float y = d2i[3] * j + d2i[4] * i + d2i[5];
            // if (i == 320 && j == 320) {
            //     printf("x: %f, y: %f\n", x, y);
            // }
            // printf("%d, %d, %f %f\n", i, j, x, y);
            float b = bilinearInterpolateChannel(img, x, y, 0, 114);
            float g = bilinearInterpolateChannel(img, x, y, 1, 114);
            float r = bilinearInterpolateChannel(img, x, y, 2, 114);
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(b, g, r);
        }
    }
    // cv::imwrite("cat01_warp.jpg", dst);
    cv::imwrite("bus_warp.jpg", dst);
    cv::Mat dst2 = bilinearInterpolateResize(img, w, h);
    // cv::imwrite("cat01_warp2.jpg", dst2);
    cv::imwrite("bus_warp2.jpg", dst2);
    return 0;
}