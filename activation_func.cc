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


