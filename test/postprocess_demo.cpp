#include <cstdio>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <memory>
#define testiou 0
// static cv::Mat letterbox(cv::Mat &src, int w, int h, float* i2d, float *d2i) {
//     int width = src.cols;
//     int height = src.rows;

//     float scale = std::min(float(w) / width, float(h) / height);
//     // float i2d[6];
//     // float d2i[6];
//     i2d[0] = scale;
//     i2d[1] = 0;
//     i2d[2] = (-width * scale + w) / 2;
//     i2d[3] = 0;
//     i2d[4] = scale;
//     i2d[5] = (-height * scale + h) / 2;

//     d2i[0] = 1 / scale;
//     d2i[1] = 0;
//     d2i[2] = (width * scale - w) / 2 / scale;
//     d2i[3] = 0;
//     d2i[4] = 1 / scale;
//     d2i[5] = (height * scale - h) / 2 / scale;

//     cv::Mat M(2, 3, CV_32F, i2d);
//     cv::Mat out = cv::Mat::zeros(h, w, CV_8UC3);
//     cv::warpAffine(src, out, M, out.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));  
//     return out;
// }    
typedef struct box_s {
    float x1, y1, x2, y2;
    float prob; // 概率
    int label; // 类别
    void print() {
        printf("x1: %f, y1: %f, x2: %f, y2: %f, prob: %f, label: %d\n", x1, y1, x2, y2, prob, label);
    }
}Box;
struct cmp {
    bool operator()(const Box &a, const Box &b) const {
        return a.prob > b.prob;
    }
};
float iou(const Box& a, const Box& b){
    float cross_x1 = std::max(a.x1, b.x1);
    float cross_y1 = std::max(a.y1, b.y1);
    float cross_x2 = std::min(a.x2, b.x2);
    float cross_y2 = std::min(a.y2, b.y2);

    float cross_area = std::max(0.0f, cross_x2 - cross_x1) * std::max(0.0f, cross_y2 - cross_y1);
    float union_area = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1) 
                     + std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1) - cross_area;
    if(cross_area == 0 || union_area == 0) return 0.0f;
    return cross_area / union_area;
};

std::vector<Box> nms(std::vector<Box> &bboxes, const float iou_threshold) {
    std::sort(bboxes.begin(), bboxes.end(), [](const Box &a, const Box &b) {
        return a.prob > b.prob;
    });
    std::vector<Box> res;
    std::vector<bool> flag(bboxes.size(), false);
    for (unsigned int i = 0; i < bboxes.size(); i++) {
        if (flag[i]) {
            continue;
        }
        res.push_back(bboxes[i]);
        for (unsigned int j = i + 1; j < bboxes.size(); j++) {
            if (flag[j]) {
                continue;
            }
            if (iou(bboxes[i], bboxes[j]) > iou_threshold) {
                flag[j] = true;
            }
        }
    }
    return res;
}

// class Yolo {
// public:
//     Yolo() {
//         std::cout << "Yolo()" << std::endl;
//     }

//     ~Yolo() {
//         std::cout << "~Yolo()" << std::endl;

//     }
// };

std::vector<float> loadWeight(const std::string &filename) {
    std::ifstream infile(filename, std::ios::binary);
    assert(infile.is_open() && "loadWeight failed");
    int size;
    infile.read(reinterpret_cast<char *>(&size), sizeof(int));
    std::vector<float> data(size);
    infile.read(reinterpret_cast<char *>(data.data()), size * sizeof(float));
    infile.close();
    return data;
}

int main() {
    std::vector<Box> bboxes;
    float confidence_threshold = 0.25;
    float nms_threshold = 0.5;
    auto data = loadWeight("yolov5output.data");
    // for (int i = 0; i < 10; i++) {
    //     std::cout << data[i] << std::endl;
    // }
    // std::cout << data.size() << std::endl;

    int output_batch = 1;
    int output_numbox = 25200;
    int output_numprob = 85;
    int num_classes = output_numprob - 5;
    int output_numel = output_batch * output_numbox * output_numprob;
    printf("output_numel: %d\n", output_numel);
    for (int i = 0; i < output_batch; i++) {
        float *pBatch = data.data() + i * output_numbox * output_numprob;
        for (int j = 0; j < output_numbox; j++) {
            float *pBox = pBatch + j * output_numprob;
            // float *box_end = box_start + output_numprob;
            float prob = pBox[4];
            // printf("prob: %f\n", prob);
            // break;
            if (prob < confidence_threshold) {
                continue;
            }
            float *pClasses = pBox + 5;
            int label = std::max_element(pClasses, pClasses + num_classes) - pClasses;
            prob *= pClasses[label];
            if (prob < confidence_threshold) {
                continue;
            }
            float x1 = pBox[0] - pBox[2] / 2;
            float y1 = pBox[1] - pBox[3] / 2;
            float x2 = pBox[0] + pBox[2] / 2;
            float y2 = pBox[1] + pBox[3] / 2;
            Box box = {x1, y1, x2, y2, prob, label};
            bboxes.push_back(box);
            // printf("\n");
        }
    }
    std::cout << bboxes.size() << std::endl;
    // std::sort(bboxes.begin(), bboxes.end(), cmp());
    // std::priority_queue<Box, std::vector<Box>, cmp> pq;
    std::vector<Box> res = nms(bboxes, nms_threshold);
    std::cout << res.size() << std::endl;
    for (auto box:res) {
        box.print();
    }

    auto matrix_vec = loadWeight("matrix.data");
    for (unsigned int i = 0; i < matrix_vec.size(); i++) {
        std::cout << matrix_vec[i] << " ";
    }
    std::cout << std::endl;
    float *matrix_inv = matrix_vec.data() + 6;
    cv::Mat img = cv::imread("bus.jpg");
    for (auto box:res) {
        int x1 = matrix_inv[0] * box.x1 + matrix_inv[1] * box.y1 + matrix_inv[2];
        int y1 = matrix_inv[3] * box.x1 + matrix_inv[4] * box.y1 + matrix_inv[5];
        int x2 = matrix_inv[0] * box.x2 + matrix_inv[1] * box.y2 + matrix_inv[2];
        int y2 = matrix_inv[3] * box.x2 + matrix_inv[4] * box.y2 + matrix_inv[5];
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
        cv::putText(img, std::to_string(box.label), cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
    }
    cv::imwrite("bus_result.jpg", img);
#if testiou
        // 测试用例
    Box box1 = {0, 0, 2, 2};
    Box box2 = {1, 1, 3, 3};  // 部分重叠
    Box box3 = {0, 0, 2, 2};  // 完全重叠
    Box box4 = {3, 3, 5, 5};  // 不重叠

    std::cout << "IoU of box1 and box2: " << iou(box1, box2) << std::endl;
    std::cout << "IoU of box1 and box3: " << iou(box1, box3) << std::endl;
    std::cout << "IoU of box1 and box4: " << iou(box1, box4) << std::endl;

#endif

    return 0;
}
// 1 x 25200 x 85
// 2142000