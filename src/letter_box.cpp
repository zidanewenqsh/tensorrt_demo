#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#define print(x) std::cout << #x << ": " << x << std::endl

cv::Mat letter_box_1(cv::Mat &src, int w, int h) {
    int width = src.cols;
    int height = src.rows;

    float scale = std::min(float(w) / width, float(h) / height);
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(), scale, scale);
    cv::Mat out = cv::Mat::zeros(h, w, CV_8UC3);

    int top = (h - dst.rows) / 2;
    int down = (h - dst.rows + 1) / 2;
    int left = (w - dst.cols) / 2;
    int right = (w - dst.cols + 1) / 2;

    cv::copyMakeBorder(dst, out, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return out;
}

cv::Mat letter_box_2(cv::Mat &src, int w, int h) {
    int width = src.cols;
    int height = src.rows;

    float scale = std::min(float(w) / width, float(h) / height);
    int offsetX = (w - width * scale) / 2;
    int offsetY = (h - height * scale) / 2;
    cv::Point2f srcTri[3];
    cv::Point2f dstTri[3];
    srcTri[0] = cv::Point2f(0, 0);
    srcTri[1] = cv::Point2f(width - 1, 0);
    srcTri[2] = cv::Point2f(0, height - 1);
    dstTri[0] = cv::Point2f(offsetX, offsetY);
    dstTri[1] = cv::Point2f(offsetX + width * scale - 1, offsetY);
    dstTri[2] = cv::Point2f(offsetX, offsetY + height * scale - 1);
    cv::Mat M = cv::getAffineTransform(srcTri, dstTri);
    cv::Mat out = cv::Mat::zeros(h, w, CV_8UC3);
    cv::warpAffine(src, out, M, out.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return out;
}

cv::Mat letter_box_3(cv::Mat &src, int w, int h) {
    int width = src.cols;
    int height = src.rows;

    float scale = std::min(float(w) / width, float(h) / height);
    float i2d[6];
    // float d2i[6];
    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = (-width * scale + w) / 2;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = (-height * scale + h) / 2;
    cv::Mat M(2, 3, CV_32F, i2d);
    cv::Mat out = cv::Mat::zeros(h, w, CV_8UC3);
    cv::warpAffine(src, out, M, out.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));  
    return out;
}

int main() {
    cv::Mat img = cv::imread("cat01.jpg");
    cv::Mat out1 = letter_box_1(img, 640, 640);
    cv::imwrite("cat01_letter_box_1.jpg", out1);
    cv::Mat out2 = letter_box_2(img, 640, 640);
    cv::imwrite("cat01_letter_box_2.jpg", out2);
    cv::Mat out3 = letter_box_3(img, 640, 640);
    cv::imwrite("cat01_letter_box_3.jpg", out3);
    return 0;
}