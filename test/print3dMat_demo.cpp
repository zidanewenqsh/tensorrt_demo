#include <opencv2/opencv.hpp>
#include <iostream>

void print3dMat(const cv::Mat& mat) {
    if (mat.dims != 3) {
        std::cerr << "Error: print3dMat requires a 3-dimensional cv::Mat." << std::endl;
        return;
    }

    // 遍历并打印数据
    for (int i = 0; i < mat.size[0]; ++i) {
        for (int j = 0; j < mat.size[1]; ++j) {
            for (int k = 0; k < mat.size[2]; ++k) {
                std::cout << "mat(" << i << ", " << j << ", " << k << ") = " 
                          << mat.at<float>(i, j, k) << std::endl;
            }
        }
    }
}

int main() {
    // 创建一个形状为 (2, 3, 4) 的三维 cv::Mat 对象
    int sizes[] = {2, 3, 4};
    cv::Mat mat(3, sizes, CV_32F);

    // 填充数据
    int count = 0;
    for (int i = 0; i < mat.size[0]; ++i) {
        for (int j = 0; j < mat.size[1]; ++j) {
            for (int k = 0; k < mat.size[2]; ++k) {
                mat.at<float>(i, j, k) = count++;
            }
        }
    }

    // 使用函数打印数据
    print3dMat(mat);

    return 0;
}
