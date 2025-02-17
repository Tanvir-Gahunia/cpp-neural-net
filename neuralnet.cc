#include "neuralnet.h"
#include "activation_func.h"
#include "layer.h"
#include <cassert>
#include <iostream>

NeuralNet::NeuralNet(const std::vector<uint>& topology, struct activation_func af) : af{af}
{
    for(int i = 0; i < topology.size() - 1; ++i)
        layers.emplace_back(topology[i], topology[i+1], true);
}


NeuralNet::NeuralNet(const std::vector<Layer>& in, struct activation_func af) : af(af)
{
    for(auto& layer : in)
        layers.push_back(layer);
}

NeuralNet::NeuralNet(const std::vector<Matrix>& weights, const std::vector<Matrix>& biases, struct activation_func af) : af{af}
{
    assert(weights.size() == biases.size());
    for(int i = 0; i < weights.size(); ++i)
    {
        assert(weights[i].cols() == biases[i].cols());
        layers.emplace_back(weights[i], biases[i]);
    }
}


Matrix NeuralNet::feed_forward(Matrix in) const
{
    for(auto& layer : layers)
        in = layer.feed_forward(in, af.activation_f);
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

NeuralNet NeuralNet::train_network(const Matrix& inputs, const Matrix& outputs)
{
    std::vector<Matrix> dWs, dbs;
    Matrix dA, dA_prev;
    float cost_so_far = cost(inputs, outputs);
    std::cout << "Cost is " << cost_so_far << std::endl;

    Matrix A = inputs;
    for(auto& layer : layers)
        A = layer.feed_forward(A, af.activation_f);
    dA = A - outputs;

    for (int i = layers.size() - 1; i >= 0; --i) {
        Matrix dW, db;
        layers[i].backprop(dA, (i == 0 ? inputs : layers[i - 1].get_activations()), dW, db, dA_prev, af.activation_f_derivative);
        dWs.push_back(dW);
        dbs.push_back(db);
        dA = dA_prev; 
    }
    std::reverse(dWs.begin(), dWs.end());
    std::reverse(dbs.begin(), dbs.end());
    return NeuralNet(dWs, dbs, af);
}


void NeuralNet::learn(const NeuralNet& gradients, const float learning_rate)
{
    assert(gradients.layers.size() == layers.size());
    for(int i = 0; i < layers.size(); ++i)
        layers[i].learn(gradients.layers[i], learning_rate);
}




