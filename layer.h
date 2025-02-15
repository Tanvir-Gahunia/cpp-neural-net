#include "matrix.h"
#include "activation_func.h"

class Layer
{
    Matrix weights, biases;
public:
    Layer(uint, uint, bool);
    Matrix feed_forward(const Matrix& in, float(*activation_function)(float));
};



