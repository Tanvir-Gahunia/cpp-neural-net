#ifndef NN_H
#define NN_H
#include "matrix.h"
#include "layer.h"
#include <vector>

class NeuralNet
{
    std::vector<Layer> layers;
public:
    NeuralNet(const std::vector<uint>&);
    NeuralNet(const std::vector<Layer>&);
    Matrix feed_forward(Matrix) const;
    float cost(const Matrix&, const Matrix&) const;
    NeuralNet train_network(const Matrix&, const Matrix&, const float);
    void learn(const NeuralNet&, const float);
};
#endif

