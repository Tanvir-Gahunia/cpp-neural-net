#include "activation_func.h"
#include "matrix.h"
#include <cassert>
#include <iostream>

struct Xor {
    Matrix a0{1, 2};
    Matrix w1{2, 2};
    Matrix b1{1, 2};
    Matrix a1{1, 2};
    Matrix w2{2, 1};
    Matrix b2{1, 1};
    Matrix final{1, 1};
};

float feed_forward(Xor m, Matrix in) {
    m.a1 = in * m.w1 + m.b1;
    m.a1.activation(sigmoid);
    m.final = m.a1 * m.w2 + m.b2;
    m.final.activation(sigmoid);
    return m.final.at(0, 0);
}

float cost(Xor m, Matrix inputs, Matrix outputs)
{
    assert(inputs.rows() == outputs.rows()); // better be the case that we have
    // outputs for each input
    assert(outputs.cols() == m.final.cols()); // size of output array and model
    // final should match
    float cost = 0;
    for(int i = 0; i < inputs.rows(); ++i)
    {
        feed_forward(m, inputs.row_to_matrix(i));

        for(int j = 0; j < outputs.cols(); ++j)
        {
            float diff = m.final.at(0, j) - outputs.row_to_matrix(i).at(0, j);
            cost += diff*diff;
        }
    }
    return cost / inputs.rows();
}

int main(int argc, char *argv[]) {
    Xor model;
    model.w1.rand(0, 1);
    model.b1.rand(0, 1);
    model.a1.rand(0, 1);
    model.w2.rand(0, 1);
    model.b2.rand(0, 1);
    model.final.rand(0, 1);



    return 0;
}
