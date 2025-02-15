#include "vector"
class Matrix
{
    std::vector<std::vector<float> > data;
public:
    Matrix(const int rows, const int cols);
    void print();
    Matrix operator+(const Matrix&);
    Matrix operator*(const Matrix&);
    float& at(uint i, uint j);
    int rows() const;
    int cols() const;
};

class Neuron;

class NN;

