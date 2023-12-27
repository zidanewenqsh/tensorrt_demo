#include <iostream>
#include <vector>
using namespace std;

// 函数：打印矩阵
void printMatrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

// 函数：高斯消元法求逆矩阵
bool inverseMatrix(vector<vector<double>>& matrix) {
    int n = matrix.size();
    vector<vector<double>> inv(n, vector<double>(n, 0.0));

    // 初始化单位矩阵
    for (int i = 0; i < n; ++i) {
        inv[i][i] = 1.0;
    }

    // 高斯消元
    for (int i = 0; i < n; ++i) {
        // 寻找主元
        int maxRow = i;
        for (int k = i + 1; k < n; ++k) {
            if (abs(matrix[k][i]) > abs(matrix[maxRow][i])) {
                maxRow = k;
            }
        }
        swap(matrix[i], matrix[maxRow]);
        swap(inv[i], inv[maxRow]);

        // 检查主元是否为0
        if (abs(matrix[i][i]) < 1e-8) {
            return false; // 矩阵不可逆
        }

        // 消元
        for (int j = i + 1; j < n; ++j) {
            double factor = matrix[j][i] / matrix[i][i];
            for (int k = 0; k < n; ++k) {
                matrix[j][k] -= factor * matrix[i][k];
                inv[j][k] -= factor * inv[i][k];
            }
        }
    }

    // 回代
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i - 1; j >= 0; --j) {
            double factor = matrix[j][i] / matrix[i][i];
            for (int k = n - 1; k >= 0; --k) {
                matrix[j][k] -= factor * matrix[i][k];
                inv[j][k] -= factor * inv[i][k];
            }
        }
    }

    // 归一化对角线
    for (int i = 0; i < n; ++i) {
        double factor = matrix[i][i];
        for (int j = 0; j < n; ++j) {
            inv[i][j] /= factor;
        }
    }

    matrix = inv;
    return true;
}

int main() {
    // vector<vector<double>> matrix = {
    //     {4, 7},
    //     {2, 6}
    // };
    std::vector<std::vector<double>> matrix = {
        {4, 7, 2},
        {3, 6, 1},
        {2, 5, 3}
    };

    if (inverseMatrix(matrix)) {
        cout << "逆矩阵是：" << endl;
        printMatrix(matrix);
    } else {
        cout << "矩阵不可逆" << endl;
    }

    return 0;
}
