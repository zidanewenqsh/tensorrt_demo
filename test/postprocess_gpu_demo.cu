#include <atomic>
#include <cuda_runtime.h>
#include <memory>
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
// #define NUM_BOX_ELEMENT 7

static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

std::vector<cv::Scalar> colors = {
    cv::Scalar(255, 0, 0),      // 蓝色
    cv::Scalar(0, 255, 0),      // 绿色
    cv::Scalar(0, 0, 255),      // 红色
    cv::Scalar(0, 255, 255),    // 黄色
    cv::Scalar(255, 0, 255),    // 洋红色（品红）
    cv::Scalar(255, 255, 0),    // 青色
    cv::Scalar(0, 165, 255),    // 橙色
    cv::Scalar(128, 0, 128),    // 紫色
    cv::Scalar(255, 192, 203),  // 粉色
    cv::Scalar(128, 128, 128)   // 灰色
};
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
typedef struct box_s {
    float x1, y1, x2, y2;
    float prob; // 概率
    int label; // 类别
    int remove; // 是否保留
    void print() {
        printf("x1: %f, y1: %f, x2: %f, y2: %f, prob: %f, label: %d, remove: %d\n", x1, y1, x2, y2, prob, label, remove);
    }
}Box;
__global__ void filter_boxes_kernel(float* boxes, Box *filtered_boxes, float* d_matrix, int num_boxes, int num_probs, 
    float conf_threshold, float nms_threshold, int *box_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return ;

    // int count = atomicAdd(box_count, 1);
    // Box *bboxes = filtered_boxes + count;
    float* box = boxes + idx * num_probs;
    float prob = box[4]; // 置信度

    if (prob < conf_threshold) return;

    // 找到最大类别概率
    float max_class_prob = 0.0f;
    int label = -1;
    for (int i = 5; i < num_probs; ++i) {
        if (box[i] > max_class_prob) {
            max_class_prob = box[i];
            label = i - 5;
        }
    }

    prob *= max_class_prob;
    if (prob < conf_threshold) return;

    int count = atomicAdd(box_count, 1);
    Box *bboxes = filtered_boxes + count;
    // 计算边界框坐标
    float x1 = box[0] - box[2] / 2;
    float y1 = box[1] - box[3] / 2;
    float x2 = box[0] + box[2] / 2;
    float y2 = box[1] + box[3] / 2;
#if 1
    x1 = d_matrix[0] * x1 + d_matrix[1] * y1 + d_matrix[2];
    y1 = d_matrix[3] * x1 + d_matrix[4] * y1 + d_matrix[5];
    x2 = d_matrix[0] * x2 + d_matrix[1] * y2 + d_matrix[2];
    y2 = d_matrix[3] * x2 + d_matrix[4] * y2 + d_matrix[5];
#endif
    bboxes->x1 = x1;
    bboxes->y1 = y1;
    bboxes->x2 = x2;
    bboxes->y2 = y2;
    bboxes->prob = prob;
    bboxes->label = label;
    // printf("x1: %f, y1: %f, x2: %f, y2: %f, prob: %f, label: %d, count: %d\n", x1, y1, x2, y2, prob, label, count);
    __syncthreads();
    #if 1
    for (int i = 0; i < count - 1; i++) {
        Box *bbox = filtered_boxes + i;
        if (bboxes->remove || bbox->remove || bbox->label != label) continue;
        float xx1 = max(bbox->x1, x1);
        float yy1 = max(bbox->y1, y1);
        float xx2 = min(bbox->x2, x2);
        float yy2 = min(bbox->y2, y2);
        float w = max(0.0f, xx2 - xx1 + 1);
        float h = max(0.0f, yy2 - yy1 + 1);
        float inter = w * h;
        if (inter == 0) continue;
        float ovr = inter / ((bbox->x2 - bbox->x1 + 1) * (bbox->y2 - bbox->y1 + 1) + (x2 - x1 + 1) * (y2 - y1 + 1) - inter);
        if (ovr < nms_threshold) continue;
        if (prob > bbox->prob) {
            bbox->remove = 1;
        } else {
            bboxes->remove = 1;
        }
        // box->remove = 1;
        // if (ovr > nms_threshold) {
        //     if (prob > bbox->prob) {
        //         bbox->remove = 1;
        //     } else {
        //         bboxes->remove = 1;
        //     }
        // }
    }
    #endif
    // bboxes->remove = 0;

    // float* out_box = filtered_boxes + count * NUM_BOX_ELEMENT;
    // out_box[0] = x1;
    // out_box[1] = y1;
    // out_box[2] = x2;
    // out_box[3] = y2;
    // out_box[4] = prob;
    // out_box[5] = static_cast<float>(label);
    // out_box[6] = atomic;
}
#if 0
__global__ void nms(Box *filtered_boxes, int box_count,  float nms_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= box_count) return;
    auto box = filtered_boxes + idx;
    if (box->remove) return;
    float x1 = box->x1;
    float y1 = box->y1;
    float x2 = box->x2;
    float y2 = box->y2;
    float prob = box->prob;
    int label = box->label;
    int remove = box->remove;
    // 1--x1: 110.145782, y1: 235.767029, x2: 224.202972, y2: 535.154358, prob: 0.805094, label: 0
    printf("1--x1: %f, y1: %f, x2: %f, y2: %f, prob: %f, label: %d, remove:%d, idx: %d\n", x1, y1, x2, y2, prob, label, remove, idx);
    int flag = 0;
    if (x1 - 110 < 1 && y1 - 235 < 1 && x2 - 224 < 1 && y2 - 535 < 1 && prob - 0.805094 < 1){
        if (x1 - 110 > 0 && y1 - 235 > 0 && x2 - 224 > 0 && y2 - 535 > 0 && prob - 0.805094 > 0 ) flag = 1;
    }
    for (int i = 0; i < box_count - 1; i++) {
        if (i == idx) continue;
        int flag_ = 0;
        Box *bbox = filtered_boxes + i;
        float x1_ = bbox->x1;
        float y1_ = bbox->y1;
        float x2_ = bbox->x2;
        float y2_ = bbox->y2;
        float prob_ = bbox->prob;
        if (x1_ - 110 < 1 && y1_ - 235 < 1 && x2_ - 224 < 1 && y2_ - 535 < 1 && prob_ - 0.805094 < 1){
            if (x1_ - 110 > 0 && y1_ - 235 > 0 && x2_ - 224 > 0 && y2_ - 535 > 0 && prob_ - 0.805094 > 0 ) flag_ = 1;
        }
        if (bbox->remove || bbox->label != label) continue;
        float xx1 = max(bbox->x1, x1);
        float yy1 = max(bbox->y1, y1);
        float xx2 = min(bbox->x2, x2);
        float yy2 = min(bbox->y2, y2);
        float w = max(0.0f, xx2 - xx1);
        float h = max(0.0f, yy2 - yy1);
        float inter = w * h;
        if (inter == 0) continue;
        float ovr = inter / ((bbox->x2 - bbox->x1) * (bbox->y2 - bbox->y1) + (x2 - x1) * (y2 - y1) - inter);
        if (ovr < nms_threshold || prob >= bbox->prob) continue;
        // if (prob > bbox->prob) continue; 
        
        if (flag) {
            printf("ovr: %f, nms_threshold: %f\n", ovr, nms_threshold);  
            printf("3--x1: %f, y1: %f, x2: %f, y2: %f, prob: %f, label: %d, remove:%d, idx: %d\n", x1, y1, x2, y2, prob, label, remove, idx);
            printf("5--x1: %f, y1: %f, x2: %f, y2: %f, prob: %f, label: %d, remove:%d, idx: %d\n", x1_, y1_, x2_, y2_, prob, label, remove, idx);
        }
        if (flag_) {
            printf("4--x1: %f, y1: %f, x2: %f, y2: %f, prob: %f, label: %d, remove:%d, idx: %d\n", x1, y1, x2, y2, prob, label, remove, idx);
            printf("6--x1: %f, y1: %f, x2: %f, y2: %f, prob: %f, label: %d, remove:%d, idx: %d\n", x1_, y1_, x2_, y2_, prob, label, remove, idx);
        }
        box->remove = 1;
        return ;
        
    }
    printf("2--x1: %f, y1: %f, x2: %f, y2: %f, prob: %f, label: %d, remove:%d, idx: %d\n", x1, y1, x2, y2, prob, label, remove, idx);
    for (int i = idx + 1; i < box_count - 1; i++) {
        if (i == idx) continue;
        Box *bbox = filtered_boxes + i;
        if (bbox->remove || bbox->label != label) continue;
        float xx1 = max(bbox->x1, x1);
        float yy1 = max(bbox->y1, y1);
        float xx2 = min(bbox->x2, x2);
        float yy2 = min(bbox->y2, y2);
        float w = max(0.0f, xx2 - xx1);
        float h = max(0.0f, yy2 - yy1);
        float inter = w * h;
        if (inter == 0) continue;
        float ovr = inter / ((bbox->x2 - bbox->x1) * (bbox->y2 - bbox->y1) + (x2 - x1) * (y2 - y1) - inter);
        if (ovr < nms_threshold) continue;
        bbox->remove = 1;
    }
}
#endif
void post_process_cuda(float* data, Box *h_filtered_boxes, int *h_box_count, float*matrix_inv, int output_batch, int output_numbox, int output_numprob, 
    float confidence_threshold, float nms_threshold) {
    int output_numel = output_batch * output_numbox * output_numprob;

    // 分配和初始化 GPU 内存
    float* d_data;
    float* d_matrix;
    cudaMalloc(&d_data, output_numel * sizeof(float));
    cudaMalloc(&d_matrix, 6 * sizeof(float));
    cudaMemcpy(d_data, data, output_numel * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix, matrix_inv, 6 * sizeof(float), cudaMemcpyHostToDevice);

    // 分配 GPU 内存以存储过滤后的边界框
    Box* d_filtered_boxes;
    int* d_box_count;
    cudaMalloc(&d_filtered_boxes, output_numbox * sizeof(Box)); // 假设每个边界框有 6 个值
    cudaMalloc(&d_box_count, sizeof(int));
    cudaMemset(d_box_count, 0, sizeof(int));
    cudaMemset(d_filtered_boxes, 0, output_numbox * sizeof(Box));

    // 调用 Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_numbox + threadsPerBlock - 1) / threadsPerBlock;
    filter_boxes_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_filtered_boxes, d_matrix, 
        output_numbox, output_numprob, confidence_threshold, nms_threshold, d_box_count);
    // 检查是否有错误发生
    cudaError_t error1 = cudaGetLastError();
    if (error1 != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(error1) << std::endl;
    }
    
    // 同步设备以确保所有操作都已完成
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    }
    // box_count


    // 将过滤后的边界框复制回主机
    // int h_box_count;
    cudaMemcpy(h_box_count, d_box_count, sizeof(int), cudaMemcpyDeviceToHost);
    // nms
    threadsPerBlock = 256;
    blocksPerGrid = (*h_box_count + threadsPerBlock - 1) / threadsPerBlock;
    #if 0
    cudaMemcpy(h_filtered_boxes, d_filtered_boxes, (*h_box_count) * sizeof(Box), cudaMemcpyDeviceToHost);
    // sort h_filetered_boxes
    std::sort(h_filtered_boxes, h_filtered_boxes + (*h_box_count), [](const Box &a, const Box &b) {
        return a.prob > b.prob;
    });
    cudaMemcpy(d_filtered_boxes, h_filtered_boxes, (*h_box_count) * sizeof(Box), cudaMemcpyHostToDevice);
    #endif
    #if 0
    nms<<<blocksPerGrid, threadsPerBlock>>>(d_filtered_boxes,  *h_box_count, nms_threshold);   
    // h_filtered_boxes.resize(h_box_count * 6);
    #endif
    cudaMemcpy(h_filtered_boxes, d_filtered_boxes, (*h_box_count) * sizeof(Box), cudaMemcpyDeviceToHost);

    // 清理
    cudaFree(d_data);
    cudaFree(d_filtered_boxes);
    cudaFree(d_box_count);
}
int main() {
    std::vector<Box> bboxes;

    auto data = loadWeight("yolov5output.data");
    auto matrix_vec = loadWeight("matrix.data");
    for (unsigned int i = 0; i < matrix_vec.size(); i++) {
        std::cout << matrix_vec[i] << " ";
    }
    std::cout << std::endl;
    float *matrix_inv = matrix_vec.data() + 6;
    cv::Mat img = cv::imread("bus.jpg");
#if 1
        // 假设 data 是从模型输出的数据
    // std::vector<float> h_filtered_boxes;
    int output_batch = 1;
    int output_numbox = 25200;
    int output_numprob = 85;
    float confidence_threshold = 0.25; // 根据需要调整
    float nms_threshold = 0.5;

    auto h_filtered_boxes = std::shared_ptr<Box[]>(new Box[output_numbox]);
    int h_box_count;
    // 调用封装的函数
    // postprocess_cuda(data.data(), h_filtered_boxes.get(), &h_box_count, output_batch, output_numbox, output_numprob, confidence_threshold);
    // std::vector<Box> res;
    post_process_cuda(data.data(), h_filtered_boxes.get(), &h_box_count, matrix_inv,
            output_batch, output_numbox, output_numprob, confidence_threshold, nms_threshold);

    printf("h_box_count: %d\n", h_box_count);
    // void postprocess_cuda(float* data, float* h_filtered_boxes, int *h_box_count, int output_batch, int output_numbox, int output_numprob, float confidence_threshold);
    // 处理 h_filtered_boxes...
    for (int i = 0; i < 50; i++) {
        auto box = h_filtered_boxes[i];
        if (box.remove == 1) continue;
        printf("i: %d\n", i);
        box.print();
        #if 1
        int x1 = (int)box.x1;
        int y1 = (int)box.y1;
        int x2 = (int)box.x2;
        int y2 = (int)box.y2;
        #else
        int x1 = matrix_inv[0] * box.x1 + matrix_inv[1] * box.y1 + matrix_inv[2];
        int y1 = matrix_inv[3] * box.x1 + matrix_inv[4] * box.y1 + matrix_inv[5];
        int x2 = matrix_inv[0] * box.x2 + matrix_inv[1] * box.y2 + matrix_inv[2];
        int y2 = matrix_inv[3] * box.x2 + matrix_inv[4] * box.y2 + matrix_inv[5];
        #endif
        // cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
        // cv::putText(img, std::to_string(box.label), cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        auto name  = cocolabels[box.label];
        auto caption   = cv::format("%s %.2f", name, box.prob);
        auto color = colors[box.label % 10];
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
        // cv::putText(img_draw, std::to_string(box.label), cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        cv::putText(img, caption, cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
    }
    cv::imwrite("bus_result.jpg", img);
#else
    float confidence_threshold = 0.25;
    int output_batch = 1;
    int output_numbox = 25200;
    int output_numprob = 85;
    // int num_classes = output_numprob - 5;
    int output_numel = output_batch * output_numbox * output_numprob;
    // 假设 data 是从模型输出的数据，output_batch, output_numbox, output_numprob 是模型输出的维度
    float* d_data;
    cudaMalloc(&d_data, output_numel * sizeof(float));
    cudaMemcpy(d_data, data.data(), output_numel * sizeof(float), cudaMemcpyHostToDevice);

    // 分配 GPU 内存以存储过滤后的边界框
    float* d_filtered_boxes;
    int* d_box_count;
    cudaMalloc(&d_filtered_boxes, output_numbox * 6 * sizeof(float)); // 假设每个边界框有 6 个值
    cudaMalloc(&d_box_count, sizeof(int));
    cudaMemset(d_box_count, 0, sizeof(int));

    // 调用 Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_numbox + threadsPerBlock - 1) / threadsPerBlock;
    filter_boxes_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, output_numbox, output_numprob, confidence_threshold, d_filtered_boxes, d_box_count);

    // 将过滤后的边界框复制回主机
    int h_box_count;
    cudaMemcpy(&h_box_count, d_box_count, sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<float> h_filtered_boxes(h_box_count * 6);
    printf("h_box_count: %d\n", h_box_count);
    cudaMemcpy(h_filtered_boxes.data(), d_filtered_boxes, h_box_count * 6 * sizeof(float), cudaMemcpyDeviceToHost);

    // 清理
    cudaFree(d_data);
    cudaFree(d_filtered_boxes);
    cudaFree(d_box_count);
    return 0;
#endif
}