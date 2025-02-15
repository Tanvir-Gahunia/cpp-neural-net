#include "vector"
class Matrix
{
    std::vector<std::vector<float> > data;
public:
    Matrix(const int rows, const int cols);
    void print();
    Matrix operator+(const Matrix&);
    Matrix operator*(const Matrix&);
    Matrix row_to_matrix(uint);
    void rand(float, float);
    float& at(uint i, uint j);
    int rows() const;
    int cols() const;
    void activation(float (*activation_func)(float));
};

class Neuron;

class NN;

