#include "matrix.h"
#include <cassert>
#include <iostream>
using namespace std;
Matrix::Matrix(const int rows, const int cols) {
    assert(rows > 0 && cols > 0);
    data.resize(rows);
    for (auto &v : data)
        v.resize(cols, 0);
}

void Matrix::print() {
    cout << "--------------------------------\n";
    for (auto &i : data) {
        for (auto &j : i)
            cout << j << " ";
        cout << "\n";
    }
    cout << "--------------------------------\n";
}

float& Matrix::at(uint i, uint j)
{
    assert(i < data.size());
    assert(j < data.front().size());
    return data[i][j];
}
int Matrix::cols() const { return data[0].size(); }
int Matrix::rows() const { return data.size(); }

Matrix Matrix::operator+(const Matrix &rhs) {
    assert(rows() == rhs.rows());
    assert(cols() == rhs.cols());
    Matrix answer(rows(), cols());
    for (int i = 0; i < rows(); ++i)
        for (int j = 0; j < cols(); ++j)
            answer.data[i][j] = data[i][j] + rhs.data[i][j];
    return answer;
}

Matrix Matrix::operator*(const Matrix &rhs) {
    assert(cols() == rhs.rows());
    Matrix answer(rows(), rhs.cols());
    for (int i = 0; i < answer.rows(); ++i)
        for (int j = 0; j < answer.cols(); ++j)
            for (int k = 0; k < cols(); ++k)
                answer.data[i][j] += data[i][k] * rhs.data[k][j];
    return answer;
}
