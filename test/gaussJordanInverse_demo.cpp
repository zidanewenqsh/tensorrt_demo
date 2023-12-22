/*
 * 高斯-约旦消元法求逆矩阵
Matrix 类封装了矩阵的数据和操作。
invert 方法实现了高斯-约旦消元法来求逆矩阵。
multiply 方法用于矩阵乘法。
isIdentity 方法检查矩阵是否为单位矩阵。
print 方法用于打印矩阵。
在 main 函数中，创建了一个 Matrix 对象 A，计算了它的逆矩阵，并验证了逆矩阵与原矩阵的乘积是否为单位矩阵。
*/
#include <iostream>
#include <vector>
#include <cmath>

const double EPSILON = 1e-10;

class Matrix {
public:
    Matrix(const std::vector<std::vector<double>>& values) : mat(values) {}

    bool invert() {
        int n = mat.size();
        std::vector<std::vector<double>> inverse(n, std::vector<double>(n, 0.0));

        // 初始化逆矩阵为单位矩阵
        for (int i = 0; i < n; ++i) {
            inverse[i][i] = 1.0;
        }

        // 进行高斯-约旦消元
        for (int i = 0; i < n; ++i) {
            // 寻找主元
            int pivot = i;
            for (int j = i + 1; j < n; ++j) {
                if (std::abs(mat[j][i]) > std::abs(mat[pivot][i])) {
                    pivot = j;
                }
            }

            if (std::abs(mat[pivot][i]) < EPSILON) {
                return false; // 矩阵不可逆
            }

            // 交换行
            std::swap(mat[i], mat[pivot]);
            std::swap(inverse[i], inverse[pivot]);

            // 归一化行
            double div = mat[i][i];
            for (int j = 0; j < n; ++j) {
                mat[i][j] /= div;
                inverse[i][j] /= div;
            }

            // 消元
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    double factor = mat[j][i];
                    for (int k = 0; k < n; ++k) {
                        mat[j][k] -= factor * mat[i][k];
                        inverse[j][k] -= factor * inverse[i][k];
                    }
                }
            }
        }

        mat = inverse;
        return true;
    }

    Matrix multiply(const Matrix& other) const {
        int n = mat.size();
        std::vector<std::vector<double>> result(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    result[i][j] += mat[i][k] * other.mat[k][j];
                }
            }
        }
        return Matrix(result);
    }

    bool isIdentity() const {
        int n = mat.size();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    if (std::abs(mat[i][j] - 1) > EPSILON) return false;
                } else {
                    if (std::abs(mat[i][j]) > EPSILON) return false;
                }
            }
        }
        return true;
    }

    void print() const {
        for (const auto& row : mat) {
            for (double elem : row) {
                std::cout << elem << " ";
            }
            std::cout << std::endl;
        }
    }

private:
    std::vector<std::vector<double>> mat;
};

int main() {
    std::vector<std::vector<double>> values = {
        {4, 7, 2},
        {3, 6, 1},
        {2, 5, 3}
    };

    Matrix A(values);
    Matrix originalA(values);
    if (A.invert()) {
        std::cout << "Inverse matrix:" << std::endl;
        A.print();

        // 验证逆矩阵
        Matrix identity = A.multiply(originalA);
        if (identity.isIdentity()) {
            std::cout << "Verification successful: A * A_inv is an identity matrix." << std::endl;
        } else {
            std::cout << "Verification failed: A * A_inv is not an identity matrix." << std::endl;
        }
    } else {
        std::cout << "Matrix is singular and cannot be inverted." << std::endl;
    }

    return 0;
}
