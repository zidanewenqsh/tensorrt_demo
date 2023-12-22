#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <memory>
// letterbox函数的实现
// cv::Mat letterbox(cv::Mat &src, int w, int h) {
//     int width = src.cols;
//     int height = src.rows;

//     float scale = std::min(float(w) / width, float(h) / height);
//     cv::Mat dst;
//     cv::resize(src, dst, cv::Size(), scale, scale);
//     cv::Mat out = cv::Mat::zeros(h, w, CV_8UC3);

//     int top = (h - dst.rows) / 2;
//     int down = (h - dst.rows + 1) / 2;
//     int left = (w - dst.cols) / 2;
//     int right = (w - dst.cols + 1) / 2;

//     cv::copyMakeBorder(dst, out, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
//     return out;
// }
// static cv::Mat letterbox(cv::Mat &src, int w, int h) {
//     int width = src.cols;
//     int height = src.rows;

//     float scale = std::min(float(w) / width, float(h) / height);
//     float i2d[6];
//     // float d2i[6];
//     i2d[0] = scale;
//     i2d[1] = 0;
//     i2d[2] = (-width * scale + w) / 2;
//     i2d[3] = 0;
//     i2d[4] = scale;
//     i2d[5] = (-height * scale + h) / 2;
//     cv::Mat M(2, 3, CV_32F, i2d);
//     cv::Mat out = cv::Mat::zeros(h, w, CV_8UC3);
//     cv::warpAffine(src, out, M, out.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));  
//     return out;
// }

static cv::Mat letterbox(cv::Mat &src, int w, int h, float* i2d, float *d2i) {
    int width = src.cols;
    int height = src.rows;

    float scale = std::min(float(w) / width, float(h) / height);
    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = (-width * scale + w) / 2;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = (-height * scale + h) / 2;

    d2i[0] = 1 / scale;
    d2i[1] = 0;
    d2i[2] = (width * scale - w) / 2 / scale;
    d2i[3] = 0;
    d2i[4] = 1 / scale;
    d2i[5] = (height * scale - h) / 2 / scale;

    cv::Mat M(2, 3, CV_32F, i2d);
    cv::Mat out = cv::Mat::zeros(h, w, CV_8UC3);
    cv::warpAffine(src, out, M, out.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));  
    return out;
}

void preprocess(const cv::Mat& img, float *ret, float* matrix, int w, int h, bool norm=false) {
    // 创建图像副本
    cv::Mat processed = img.clone();
    float mean[3] = {0};
    float std[3] = {1, 1, 1};
    // 均值和标准差
    if (norm) {
        mean[0] = 0.485;
        mean[1] = 0.456;
        mean[2] = 0.406;
        std[0]  = 0.229;
        std[1]  = 0.224;
        std[2]  = 0.225;
    };
    // 如果图像是BGR格式，则转换为RGB，否则相反
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

    // 应用letterbox变换
    processed = letterbox(processed, w, h, matrix, matrix + 6);

    // 转换为float32
    processed.convertTo(processed, CV_32F, 1.0 / 255.0);
    // processed.convertTo(processed, CV_32F, 1.0);
#if 0
    // 标准化图像
    std::vector<cv::Mat> channels(3);
    cv::split(processed, channels);
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }
    int channelSize = w * h;
    memcpy(ret, channels[0].data, channelSize * sizeof(float));
    memcpy(ret + channelSize, channels[1].data, channelSize * sizeof(float));
    memcpy(ret + channelSize * 2, channels[2].data, channelSize * sizeof(float));
#else
#if 0   
    int channelSize = w * h;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int index = i * w + j;
            ret[index] = (processed.at<cv::Vec3f>(i, j)[0]-mean[0])/std[0];
            ret[index + channelSize] = (processed.at<cv::Vec3f>(i, j)[1]-mean[1])/std[1];
            ret[index + channelSize * 2] = (processed.at<cv::Vec3f>(i, j)[2]-mean[2])/std[2];
        }
    }
#else
    int channelSize = w * h;
    float *c1 = ret, *c2 = ret + channelSize, *c3 = ret + channelSize * 2;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            // int index = i * w + j;
            // *(c1+index) = (processed.at<cv::Vec3f>(i, j)[0]-mean[0])/std[0];
            // *(c2+index) = (processed.at<cv::Vec3f>(i, j)[1]-mean[1])/std[1];
            // *(c3+index) = (processed.at<cv::Vec3f>(i, j)[2]-mean[2])/std[2];
            *(c1++) = (processed.at<cv::Vec3f>(i, j)[0]-mean[0])/std[0];
            *(c2++) = (processed.at<cv::Vec3f>(i, j)[1]-mean[1])/std[1];
            *(c3++) = (processed.at<cv::Vec3f>(i, j)[2]-mean[2])/std[2];
        
        }
    }
#endif


#endif

    return ;
}

int main() {
#if 0
    // 创建一个宽高都为2的三通道 cv::Mat 对象，
    cv::Mat mat(4, 4, CV_8UC3);
    // 填充数据，注意是8UC3格式
    int n = 0;
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            mat.at<cv::Vec3b>(i, j)[0] = n++;
            mat.at<cv::Vec3b>(i, j)[1] = n++;
            mat.at<cv::Vec3b>(i, j)[2] = n++;
        }
    }

    // 打印数据
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            std::cout << "mat(" << i << ", " << j << ") = " 
                      << static_cast<int>(mat.at<cv::Vec3b>(i, j)[0]) << ", "
                      << static_cast<int>(mat.at<cv::Vec3b>(i, j)[1]) << ", "
                      << static_cast<int>(mat.at<cv::Vec3b>(i, j)[2]) << std::endl;
        }
    }


    std::cout << "------" << std::endl;
    // 预处理数据
    auto data = std::unique_ptr<float[]>(new float[2 * 2 * 3]);
    // preprocess(mat, data.get(), 2, 2, false); // 假设图像是BGR格式
    preprocess(mat, data.get(), 2, 2, true); // 假设图像是BGR格式

    // 使用函数打印数据
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 2; ++x) {
                std::cout << "data(" << c << ", " << y << ", " << x << ") = " 
                          << data.get()[c * 2 * 2 + y * 2 + x] << std::endl;
            }
        }
    }
#else
    cv::Mat img = cv::imread("bus.jpg");
    int width = img.cols, height = img.rows;
    std::cout << "width = " << width << ", height = " << height << std::endl;
    int w = 640, h = 640;
    float *data = new float[w * h * 3];
    float *matrix = new float[6 * 2];
    preprocess(img, data, matrix, w, h, false);
    std::cout << "------" << std::endl;
    // 使用函数打印数据
    // for (int c = 0; c < 3; ++c) {
    //     for (int y = 0; y < h; y+=64) {
    //         for (int x = 0; x < w; x+=64) {
    //             std::cout << "data(" << c << ", " << y << ", " << x << ") = " 
    //                       << data[c * w * h + y * w + x] << std::endl;
    //         }
    //     }
    // }
    std::cout << "------" << std::endl;
    for (int i = 0; i < 12; ++i) {
        std::cout << "matrix(" << i << ") = " << matrix[i] << std::endl;
    }
    
    cv::Mat M(2, 3, CV_32F, matrix);
    cv::Mat M_inv(2, 3, CV_32F, matrix + 6);
    cv::Mat out = letterbox(img, w, h, matrix, matrix+6);
    cv::imwrite("out.jpg", out);
    cv::Mat out_inv(height, width, CV_8UC3);
    cv::warpAffine(out, out_inv, M_inv, out_inv.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));  
    cv::imwrite("out_inv.jpg", out_inv);
    delete [] data;
    delete [] matrix;
#endif
    return 0;
}