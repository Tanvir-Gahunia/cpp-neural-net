#ifndef MATRIX_H
#define MATRIX_H
#include "vector"
class Matrix
{
    std::vector<std::vector<float> > data;
public:
    Matrix() = default;
    Matrix(const std::vector<std::vector<float> >&);
    Matrix(const int rows, const int cols);
    Matrix& operator=(const Matrix&) = default;
    Matrix transpose() const;
    Matrix row_to_matrix(uint) const;
    Matrix sub_matrix(uint, uint, uint, uint) const;
    Matrix operator+(const Matrix&) const;
    Matrix operator-(const Matrix&) const;
    Matrix operator*(const Matrix&) const;
    Matrix& operator*(float);
    Matrix element_wise_product(const Matrix&) const;
    Matrix sumRows() const;
    Matrix broadcast(uint) const;
    float& at(uint i, uint j);
    const float& at(uint i, uint j) const;
    int rows() const;
    int cols() const;
    void rand(float, float);
    void rand_he(uint);
    void print() const;
    void activation(float (*activation_func)(float));
};
#endif
