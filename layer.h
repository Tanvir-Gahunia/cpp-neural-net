#ifndef LAYER_H
#define LAYER_H
#include "matrix.h"
#include "activation_func.h"
class NeuralNet;

class Layer
{
    Matrix weights, biases;
public:
    Layer(uint, uint, bool);
    Matrix feed_forward(const Matrix& in, float(*activation_function)(float)) const;
    Layer train_layer(const NeuralNet&, const Matrix&, const Matrix&, const float, const float);
    void learn(const Layer&, const float);
};
#endif



