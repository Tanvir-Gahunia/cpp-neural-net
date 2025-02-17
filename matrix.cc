#include "matrix.h"
#include <cassert>
#include <iostream>
#include <random>
using namespace std;
Matrix::Matrix(const int rows, const int cols) {
    assert(rows > 0 && cols > 0);
    data.resize(rows);
    for (auto &v : data)
        v.resize(cols, 0);
}

Matrix::Matrix(const std::vector<std::vector<float>> &data) : data{data} {}

void Matrix::print() const {
    cout << "--------------------------------\n";
    for (auto &i : data) {
        for (auto &j : i)
            cout << j << " ";
        cout << "\n";
    }
    cout << "--------------------------------\n";
}
void Matrix::activation(float (*activation_func)(float)) {
    for (auto &i : data)
        for (auto &e : i)
            e = activation_func(e);
}

Matrix Matrix::sumRows() const
{
    Matrix a(1, cols());
    for(int i = 0; i < rows(); ++i)
        for (int j = 0; j < cols(); ++j)
            a.data[0][j] += data[i][j];
    return a;
}

Matrix Matrix::element_wise_product(const Matrix& rhs) const
{
    assert(rows() == rhs.rows());
    assert(cols() == rhs.cols());
    Matrix a(rows(), cols());
    for(int i = 0; i < rows(); ++i)
        for(int j = 0; j < cols(); ++j)
            a.data[i][j] = data[i][j] * rhs.data[i][j];
    return a;
}

Matrix Matrix::broadcast(uint r) const
{
    assert(rows() == 1);
    Matrix a(r, cols());
    for(int i = 0; i < r; ++i)
        a.data[i] = data[0];
    return a;
}

Matrix Matrix::row_to_matrix(uint i) const {
    assert(i < rows());
    Matrix a(1, cols());
    for (int j = 0; j < cols(); ++j)
        a.at(0, j) = data[i][j];
    return a;
}

Matrix Matrix::sub_matrix(uint rstart, uint cstart, uint rend, uint cend) const {
    assert(rstart <= rend);
    assert(cstart <= cend);
    assert(rend < rows());
    assert(cend < cols());

    Matrix a(1+rend - rstart, 1+cend - cstart);
    for (int i = rstart; i <= rend; ++i)
        for (int j = cstart; j <= cend; ++j)
            a.at(i-rstart, j-cstart) = data[i][j];
    return a;
}

float &Matrix::at(uint i, uint j) {
    assert(i < rows());
    assert(j < cols());
    return data[i][j];
}

const float &Matrix::at(uint i, uint j) const {
    assert(i < rows());
    assert(j < cols());
    return data[i][j];
}

int Matrix::cols() const { return data[0].size(); }
int Matrix::rows() const { return data.size(); }

Matrix Matrix::operator+(const Matrix &rhs) const {
    assert(rows() == rhs.rows());
    assert(cols() == rhs.cols());
    Matrix answer(rows(), cols());
    for (int i = 0; i < rows(); ++i)
        for (int j = 0; j < cols(); ++j)
            answer.data[i][j] = data[i][j] + rhs.data[i][j];
    return answer;
}

Matrix Matrix::operator-(const Matrix &rhs) const {
    assert(rows() == rhs.rows());
    assert(cols() == rhs.cols());
    Matrix answer(rows(), cols());
    for (int i = 0; i < rows(); ++i)
        for (int j = 0; j < cols(); ++j)
            answer.data[i][j] = data[i][j] - rhs.data[i][j];
    return answer;
}


Matrix Matrix::transpose() const
{
    Matrix a(cols(), rows());
    for(int i = 0; i < rows(); ++i)
        for(int j = 0; j < cols(); ++j)
            a.at(j, i) = data[i][j];
    return a;
}

void Matrix::rand(float lower, float upper) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(lower, upper);
    for (auto &i : data)
        for (auto &j : i)
            j = dist(gen);
}

Matrix& Matrix::operator*(float f)
{
    for(int i = 0; i < rows(); ++i)
        for(int j = 0; j < cols(); ++j)
            data[i][j]*=f;
    return *this;
}

Matrix Matrix::operator*(const Matrix &rhs) const {
    assert(cols() == rhs.rows());
    Matrix answer(rows(), rhs.cols());
    for (int i = 0; i < answer.rows(); ++i)
        for (int j = 0; j < answer.cols(); ++j)
            for (int k = 0; k < cols(); ++k)
                answer.data[i][j] += data[i][k] * rhs.data[k][j];
    return answer;
}
