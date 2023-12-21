#include "utils.h"
void saveWeight(const std::string &filename, const float *data, const int size) {
    std::ofstream outfile(filename, std::ios::binary);
    assert(outfile.is_open() && "saveData failed");
    outfile.write(reinterpret_cast<const char *>(&size), sizeof(int));
    outfile.write(reinterpret_cast<const char *>(data), size * sizeof(float));
    outfile.close();
}

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