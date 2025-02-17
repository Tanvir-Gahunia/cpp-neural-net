#ifndef NN_H
#define NN_H
#include "activation_func.h"
#include "matrix.h"
#include "layer.h"
#include <vector>

class NeuralNet
{
    activation_func af;
    std::vector<Layer> layers;
public:
    NeuralNet(const std::vector<uint>&, struct activation_func);
    NeuralNet(const std::vector<Layer>&, struct activation_func);
    NeuralNet(const std::vector<Matrix>&, const std::vector<Matrix>&, struct activation_func);
    Matrix feed_forward(Matrix) const;
    float cost(const Matrix&, const Matrix&) const;
    NeuralNet train_network(const Matrix&, const Matrix&, const float);
    void learn(const NeuralNet&, const float);
};
#endif

