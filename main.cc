#include "matrix.h"
#include "neuralnet.h"
#include "activation_func.h"
#include <iostream>

int main(int argc, char *argv[]) {
    //Matrix training({{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}});
    Matrix training({{0, 0}, {1, 2}, {2, 4}, {6, 12}});
    //Matrix inputs = training.sub_matrix(0, 0, 3, 1);
    Matrix inputs = training.sub_matrix(0, 0, 3, 0);
    inputs.print();
    //Matrix outputs = training.sub_matrix(0, 2, 3, 2);
    Matrix outputs = training.sub_matrix(0, 1, 3, 1);
    outputs.print();
    //activation_func af(sigmoid, sigmoidDerivative);
    activation_func af(ReLu, ReLuDerivative);
    //NeuralNet nn ({2, 2, 1}, af);
    NeuralNet nn ({1, 1}, af);
    for(int i = 0; i < 100*1000; ++i)
        nn.learn(nn.train_network(inputs, outputs), 1e-1);

    for (int i = 0; i < inputs.rows(); ++i) {
        Matrix a = nn.feed_forward(inputs.row_to_matrix(i));
        //std::cout << inputs.row_to_matrix(i).at(0, 0) << " ^ " << inputs.row_to_matrix(i).at(0, 1) << " is " << a.at(0, 0) << std::endl;
        std::cout << inputs.row_to_matrix(i).at(0, 0) << " * 2 is " << a.at(0, 0) << std::endl;
    }
    return 0;
}
