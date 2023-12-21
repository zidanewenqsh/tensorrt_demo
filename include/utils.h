#ifndef __UTILS_H__
#define __UTILS_H__
#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <cassert>
#include <NvInfer.h>

void saveWeight(const std::string &filename, const float *data, const int size);
std::vector<float> loadWeight(const std::string &filename);
#endif