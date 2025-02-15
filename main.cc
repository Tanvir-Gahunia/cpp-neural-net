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

float feed_forward(Xor& m, const Matrix& in) {
    m.a1 = in * m.w1 + m.b1;
    m.a1.activation(sigmoid);
    m.final = m.a1 * m.w2 + m.b2;
    m.final.activation(sigmoid);
    return m.final.at(0, 0);
}

float calculate_cost(Xor& m, const Matrix& inputs, const Matrix& outputs) {
    assert(inputs.rows() == outputs.rows()); // better be the case that we have
    // outputs for each input
    assert(outputs.cols() == m.final.cols()); // size of output array and model
    // final should match
    float cost = 0;
    for (int i = 0; i < inputs.rows(); ++i) {
        feed_forward(m, inputs.row_to_matrix(i));

        for (int j = 0; j < outputs.cols(); ++j) {
            float diff = m.final.at(0, j) - outputs.row_to_matrix(i).at(0, j);
            cost += diff * diff;
        }
    }
    return cost / inputs.rows();
}

Xor train(Xor& m, const Matrix& inputs, const Matrix& outputs, const float wiggle_amount)
{
    Xor gradients;
    float saved;
    float cost = calculate_cost(m, inputs, outputs);
    std::cout << "Cost is " << cost << std::endl;
    for(int i = 0; i < m.w1.rows(); ++i)
        for(int j = 0; j < m.w1.cols(); ++j)
        {
            saved = m.w1.at(i, j);
            m.w1.at(i, j) += wiggle_amount;
            gradients.w1.at(i, j) = (calculate_cost(m, inputs, outputs) - cost)/wiggle_amount;
            m.w1.at(i, j) = saved;
        }

    for(int i = 0; i < m.b1.rows(); ++i)
        for(int j = 0; j < m.b1.cols(); ++j)
        {
            saved = m.b1.at(i, j);
            m.b1.at(i, j) += wiggle_amount;
            gradients.b1.at(i, j) = (calculate_cost(m, inputs, outputs) - cost)/wiggle_amount;
            m.b1.at(i, j) = saved;
        }

    for(int i = 0; i < m.w2.rows(); ++i)
        for(int j = 0; j < m.w2.cols(); ++j)
        {
            saved = m.w2.at(i, j);
            m.w2.at(i, j) += wiggle_amount;
            gradients.w2.at(i, j) = (calculate_cost(m, inputs, outputs) - cost)/wiggle_amount;
            m.w2.at(i, j) = saved;
        }


    for(int i = 0; i < m.b2.rows(); ++i)
        for(int j = 0; j < m.b2.cols(); ++j)
        {
            saved = m.b2.at(i, j);
            m.b2.at(i, j) += wiggle_amount;
            gradients.b2.at(i, j) = (calculate_cost(m, inputs, outputs) - cost)/wiggle_amount;
            m.b2.at(i, j) = saved;
        }
    return gradients;
}

void learn(Xor& m, const Xor gradients, const float learningrate)
{

     for(int i = 0; i < m.w1.rows(); ++i)
        for(int j = 0; j < m.w1.cols(); ++j)
            m.w1.at(i, j) -= learningrate * gradients.w1.at(i, j);


     for(int i = 0; i < m.b1.rows(); ++i)
        for(int j = 0; j < m.b1.cols(); ++j)
            m.b1.at(i, j) -= learningrate * gradients.b1.at(i, j);


     for(int i = 0; i < m.w2.rows(); ++i)
        for(int j = 0; j < m.w2.cols(); ++j)
            m.w2.at(i, j) -= learningrate * gradients.w2.at(i, j);


     for(int i = 0; i < m.b2.rows(); ++i)
        for(int j = 0; j < m.b2.cols(); ++j)
            m.b2.at(i, j) -= learningrate * gradients.b2.at(i, j);

}

int main(int argc, char *argv[]) {
    Xor model;
    model.w1.rand(0, 1);
    model.b1.rand(0, 1);
    model.a1.rand(0, 1);
    model.w2.rand(0, 1);
    model.b2.rand(0, 1);
    model.final.rand(0, 1);
    Matrix training({{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}});
    Matrix inputs = training.sub_matrix(0, 0, 3, 1);
    Matrix outputs = training.sub_matrix(0, 2, 3, 2);
    for(int i = 0; i < 100*1000; ++i)
        learn(model, train(model, inputs, outputs, 1e-1), 1e-1);

    for (int i = 0; i < inputs.rows(); ++i) {
        const Matrix input_row = inputs.row_to_matrix(i);
        float output = feed_forward(model, input_row);
        std::cout << "Input " << std::endl;
        for(int j = 0; j < input_row.cols(); ++j)
            std::cout << input_row.at(0, j) << " ";
        std::cout << std::endl;
        std::cout << "Output is " << output << std::endl;

    }

    
    return 0;
}
