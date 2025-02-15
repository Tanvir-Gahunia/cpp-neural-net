#!/bin/bash

set -xe

clang++ *.cc -o exec -std=c++20 -Wall -Werror -O3
