#include "yolov5.h"

int main() {
    std::string name = kModelName;
    Yolov5 yolo(name, 1<<28);
    std::string img_file = "bus.jpg";
    // yolo.forward_image(img_file);
    // printf("totalsize:%d\n", totalsize);
    std::string txtfile = "output.txt";
    auto lines = readLinesFromFile(txtfile);
    for (auto line:lines) {
        printf("imgpath:%s\n", line.c_str());
        yolo.forward_image(line);
    }
    return 0;
}