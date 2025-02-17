#ifndef LAYER_H
#define LAYER_H
#include "matrix.h"
#include "activation_func.h"
class NeuralNet;

class Layer
{
    Matrix weights, biases;
    mutable Matrix activations, pre_activations;
public:
    Layer(uint, uint, bool);
    Layer(Matrix, Matrix);
    Matrix feed_forward(const Matrix& in, float(*activation_function)(float)) const;
    Layer train_layer(const NeuralNet&, const Matrix&, const Matrix&, const float, const float);
    void backprop(const Matrix& dA, const Matrix& A_prev, Matrix& dW, Matrix& db, Matrix& dA_prev, float (*activation_derivative)(float));
    void learn(const Layer&, const float);
    const Matrix& get_activations() const;
};
#endif



