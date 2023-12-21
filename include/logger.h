#ifndef __LOGGER_H__
#define __LOGGER_H__
#include <NvInfer.h>
#include <iostream>
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;



#endif