#!/bin/bash
/bin/echo "##" $(whoami) is compiling
echo "########## Compiling"
g++ -fopenmp ACS_parallel.cpp
echo "########## Executing"
./a.out
echo "########## Done"