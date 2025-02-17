#!/bin/bash

clang++ -o exec -std=c++20 -Wall -Werror -O3 -pedantic activation_func.cc layer.cc main.cc matrix.cc neuralnet.cc
