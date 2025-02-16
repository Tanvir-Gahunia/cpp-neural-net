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
Matrix Layer::feed_forward(const Matrix &in,
                           float (*activation_function)(float)) const {
    Matrix a = (in * weights) + biases;
    a.activation(activation_function);
    return a;
}

Layer Layer::train_layer(const NeuralNet &model, const Matrix &inputs,
                         const Matrix &outputs, const float wiggle_amount,
                         const float cost) {
    Layer gradient{(uint)weights.rows(), (uint)weights.cols(), false};
    float saved;
    for (int i = 0; i < weights.rows(); ++i)
        for (int j = 0; j < weights.cols(); ++j) {
            saved = weights.at(i, j);
            weights.at(i, j) += wiggle_amount;
            gradient.weights.at(i, j) =
                (model.cost(inputs, outputs) - cost) / wiggle_amount;
            weights.at(i, j) = saved;
        }
    for (int i = 0; i < biases.rows(); ++i)
        for (int j = 0; j < biases.cols(); ++j) {
            saved = biases.at(i, j);
            biases.at(i, j) += wiggle_amount;
            gradient.biases.at(i, j) =
                (model.cost(inputs, outputs) - cost) / wiggle_amount;
            biases.at(i, j) = saved;
        }
    return gradient;
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
