#include "neuralnet.h"
#include "layer.h"
#include <cassert>
#include <iostream>

NeuralNet::NeuralNet(const std::vector<uint>& topology)
{
    for(int i = 0; i < topology.size() - 1; ++i)
        layers.emplace_back(topology[i], topology[i+1], true);
}


NeuralNet::NeuralNet(const std::vector<Layer>& in)
{
    for(auto& layer : in)
        layers.push_back(layer);
}

Matrix NeuralNet::feed_forward(Matrix in) const
{
    for(auto& layer : layers)
        in = layer.feed_forward(in, sigmoid);
    return in;
}

float NeuralNet::cost(const Matrix& inputs, const Matrix& outputs) const
{
    assert(inputs.rows() == outputs.rows()); // better be the case that we have outputs for each input
    float cost = 0;
    for (int i = 0; i < inputs.rows(); ++i) {
        Matrix a = feed_forward(inputs.row_to_matrix(i));
        assert(outputs.cols() == a.cols()); // size of output array and model better match
        for (int j = 0; j < outputs.cols(); ++j) {
            float diff = a.at(0, j) - outputs.row_to_matrix(i).at(0, j);
            cost += diff * diff;
        }
    }
    return cost / inputs.rows();
}

NeuralNet NeuralNet::train_network(const Matrix& inputs, const Matrix& outputs, const float wiggle_amount)
{
    std::vector<Layer> gradients;
    float cost_so_far = cost(inputs, outputs);
    std::cout << "Cost is " << cost_so_far << std::endl;
    for(auto& layer : layers)
        gradients.push_back(layer.train_layer(*this, inputs, outputs, wiggle_amount, cost_so_far));
    return NeuralNet{gradients};
}


void NeuralNet::learn(const NeuralNet& gradients, const float learning_rate)
{
    assert(gradients.layers.size() == layers.size());
    for(int i = 0; i < layers.size(); ++i)
        layers[i].learn(gradients.layers[i], learning_rate);
}




