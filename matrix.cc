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

void Matrix::print() {
    cout << "--------------------------------\n";
    for (auto &i : data) {
        for (auto &j : i)
            cout << j << " ";
        cout << "\n";
    }
    cout << "--------------------------------\n";
}
void Matrix::activation(float (*activation_func)(float))
{
    for(auto& i : data)
        for(auto& e : i)
            e = activation_func(e);
}


Matrix Matrix::row_to_matrix(uint i)
{
    assert(i < rows());
    Matrix a(1, cols());
    for(int j = 0; j < cols(); ++j)
        a.at(0, j) = data[i][j];
    return a;
}

float &Matrix::at(uint i, uint j) {
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

void Matrix::rand(float lower, float upper) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(lower, upper);
    for (auto &i : data)
        for (auto &j : i)
            j = dist(gen);
            
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
