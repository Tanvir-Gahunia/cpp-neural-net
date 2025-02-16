#ifndef MATRIX_H
#define MATRIX_H
#include "vector"
class Matrix
{
    std::vector<std::vector<float> > data;
public:
    Matrix() = default;
    Matrix(const int rows, const int cols);
    Matrix(const std::vector<std::vector<float> >&);
    void print() const;
    Matrix operator+(const Matrix&) const;
    Matrix operator-(const Matrix&) const;
    Matrix operator*(const Matrix&) const;
    Matrix row_to_matrix(uint) const;
    Matrix sub_matrix(uint, uint, uint, uint) const;
    void rand(float, float);
    float& at(uint i, uint j);
    const float& at(uint i, uint j) const;
    int rows() const;
    int cols() const;
    void activation(float (*activation_func)(float));
};

class Neuron;

class NN;
#endif
