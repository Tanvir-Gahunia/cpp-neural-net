#include "matrix.h"
#include "neuralnet.h"
#include "activation_func.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
void import_data_entry(std::vector<std::vector<float> >& training, std::string& s)
{
    training.emplace_back();
    std::istringstream ss(s);
    float tmp;
    while(ss >> tmp)
    {
        training.back().push_back(tmp);
        if (ss.peek() != ',') return;
        ss.ignore();
    }
}

int main(int argc, char *argv[]) {
    std::ifstream in("training.csv");
    std::vector<std::vector<float> >training;
    std::string s;
    while(getline(in, s)) import_data_entry(training, s);
    Matrix training_set(training);
    activation_func af(ReLu, ReLuDerivative);
    NeuralNet nn({784, 128, 64, 10}, af);
    Matrix inputs = training_set.sub_matrix(0, 1, training.size() - 1, training.front().size() - 1);
    inputs = inputs * (1.0f / 255.0f);
    Matrix outputs(training.size(), 10);
    for(int i = 0; i < training.size(); ++i)
        outputs.at(i, training_set.at(i, 0)) = 1.0;
    float learning_rate = 1e-1;
    for(int i = 0; i < 1000; ++i)
        nn.learn(nn.train_network(inputs, outputs), learning_rate);

    return 0;
}
