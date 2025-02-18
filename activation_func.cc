#include "activation_func.h"
#include <cmath>
float sigmoid(float x)
{
    return (1.0 / (1.0 + exp(-x)));
}

float sigmoidDerivative(float x)
{
    float sig = sigmoid(x);
    return sig * (1 - sig);
}

float LeakyReLu(float x)
{
    return x > 0 ? x : 0.01f * x;
}

float LeakyReLuDerivative(float x)
{
    return x > 0 ? 1.0 : 0.01f;
}


