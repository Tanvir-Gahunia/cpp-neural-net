#ifndef ACTIVATION_H
#define ACTIVATION_H

struct activation_func
{
    float(*activation_f)(float) = nullptr;
    float(*activation_f_derivative)(float) = nullptr;
};

float sigmoid(float x);

float sigmoidDerivative(float x);

float ReLu(float x);

float ReLuDerivative(float x);



#endif

