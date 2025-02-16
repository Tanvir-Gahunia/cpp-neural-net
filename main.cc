#include "matrix.h"
#include "neuralnet.h"
#include <iostream>

int main(int argc, char *argv[]) {
    Matrix training({{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}});
    Matrix inputs = training.sub_matrix(0, 0, 3, 1);
    inputs.print();
    Matrix outputs = training.sub_matrix(0, 2, 3, 2);
    outputs.print();
    NeuralNet nn ({2, 2, 1});
    for(int i = 0; i < 100*1000; ++i)
        nn.learn(nn.train_network(inputs, outputs, 1e-1), 1e-1);

    for (int i = 0; i < inputs.rows(); ++i) {
        Matrix a = nn.feed_forward(inputs.row_to_matrix(i));
        std::cout << inputs.row_to_matrix(i).at(0, 0) << '^' << inputs.row_to_matrix(i).at(0, 1) << " is " << a.at(0, 0) << std::endl;
    }
    return 0;
}
