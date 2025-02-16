#include "matrix.h"
#include "neuralnet.h"
#include <cassert>
#include <iostream>

int main(int argc, char *argv[]) {
    Matrix training({{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}});
    Matrix inputs = training.sub_matrix(0, 0, 3, 1);
    Matrix outputs = training.sub_matrix(0, 2, 3, 2);
    NeuralNet nn ({2, 2, 1});
    for(int i = 0; i < 100*1000; ++i)
        nn.learn(nn.train_network(inputs, outputs, 1e-1), 1e-1);
    return 0;
}
