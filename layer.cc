#include "layer.h"
#include "neuralnet.h"
#include <cassert>

Layer::Layer(uint in_size, uint out_size, bool rand)
    : weights(in_size, out_size), biases(1, out_size) {
    if (!rand)
        return;
    weights.rand(0, 1);
    biases.rand(0, 1);
}

Layer::Layer(Matrix weights, Matrix biases) : weights{weights}, biases{biases}
{}


Matrix Layer::feed_forward(const Matrix &in,
                           float (*activation_function)(float)) const {

    pre_activations = (in * weights) + biases.broadcast(in.rows());
    activations = pre_activations;
    activations.activation(activation_function);
    return activations;
}


void Layer::backprop(const Matrix& dA, const Matrix& A_prev, Matrix& dW, Matrix& db, Matrix& dA_prev, float (*activation_derivative)(float))
{
    Matrix dZ = pre_activations;
    dZ.activation(activation_derivative);
    dZ = dZ.element_wise_product(dA);
    dW =  (A_prev.transpose() * dZ) * (1.0 / A_prev.rows());
    db = dZ.sumRows() * (1.0 / A_prev.rows());
    dA_prev = dZ * weights.transpose();
}

void Layer::learn(const Layer &gradient, const float learning_rate) {
    assert(weights.rows() == gradient.weights.rows());
    assert(weights.cols() == gradient.weights.cols());
    assert(biases.rows() == gradient.biases.rows());
    assert(biases.cols() == gradient.biases.cols());
    for (int i = 0; i < weights.rows(); ++i)
        for (int j = 0; j < weights.cols(); ++j)
            weights.at(i, j) -= learning_rate * gradient.weights.at(i, j);

    for (int i = 0; i < biases.rows(); ++i)
        for (int j = 0; j < biases.cols(); ++j)
            biases.at(i, j) -= learning_rate * gradient.biases.at(i, j);
}

const Matrix& Layer::get_activations() const {return activations;}
