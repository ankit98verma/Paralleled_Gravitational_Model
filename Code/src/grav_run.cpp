/*
 * CUDA blur
 * Kevin Yuh, 2014 
 * Revised by Nailen Matschke, 2016
 * Revised by Loko Kung, 2018
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <time.h>

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

#include "blur.cuh"

using std::cerr;
using std::cout;
using std::endl;


int main(int argc, char **argv) {
    return large_gauss_test(argc, argv);
}


