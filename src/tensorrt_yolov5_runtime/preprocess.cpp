#include <opencv2/opencv.hpp>
#include <vector>

// letterbox函数的实现
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


void preprocess_cpu(const cv::Mat& img, float *ret, float* matrix, int w, int h, bool norm=false) {
    // 创建图像副本
    cv::Mat processed = img.clone();
    // 均值和标准差
    float mean[3] = {0}, std[3] = {1, 1, 1};
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

    // 减去均值并除以标准差, hwc转为chw
    int channelSize = w * h;
    float *c1 = ret, *c2 = ret + channelSize, *c3 = ret + channelSize * 2;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            *(c1++) = (processed.at<cv::Vec3f>(i, j)[0]-mean[0])/std[0];
            *(c2++) = (processed.at<cv::Vec3f>(i, j)[1]-mean[1])/std[1];
            *(c3++) = (processed.at<cv::Vec3f>(i, j)[2]-mean[2])/std[2];
        }
    }
    return ;
}
// int main() {
//     // 加载图像
//     cv::Mat img = cv::imread("path_to_your_image.jpg");

//     // 预处理图像
//     cv::Mat processedImg = preprocess(img, true); // 假设图像是BGR格式

//     // 之后的操作...

//     return 0;
// }
