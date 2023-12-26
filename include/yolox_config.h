#ifndef __CONFIG_H__
#define __CONFIG_H__

const static char* kModelName = "yolox_nano";
const static char* kInputTensorName = "images";
const static char* kOutputTensorName = "output";
constexpr static int kInputH = 416;
constexpr static int kInputW = 416;
constexpr static int kImageHMax = 1024;
constexpr static int kImageWMax = 1024;
constexpr static int kChannel = 3;
constexpr static int kMaxBatch = 1;
const static float kNmsThresh = 0.5f;
const static float kConfThresh = 0.3f;

#endif