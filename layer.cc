#include "layer.h"

Layer::Layer(uint in_size, uint out_size, bool rand) : weights(in_size, out_size), biases(in_size, out_size)
{
    if (!rand) return;
    weights.rand(0, 1);
    biases.rand(0, 1);
}
Matrix Layer::feed_forward(const Matrix& in, float(*activation_function)(float))
{
    Matrix a = in * weights + biases;
    a.activation(activation_function);
    return a;
}

