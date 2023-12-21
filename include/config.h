#ifndef __CONFIG_H__
#define __CONFIG_H__

const static char* kInputTensorName = "images";
const static char* kOutputTensorName = "output0";
constexpr static int kInputH = 640;
constexpr static int kInputW = 640;
const static float kNmsThresh = 0.7f;
const static float kConfThresh = 0.4f;

#endif