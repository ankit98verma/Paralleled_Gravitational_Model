/*
 * grav_run: Ankit Verma, Garima Aggarwal, 2020
 * 
 * This file runs the CPU implementation and GPU implementation
 * of the gravitational field calculation.
 * 
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

#include "grav_cuda.cuh"
#include "grav_cpu.h"

using std::cerr;
using std::cout;
using std::endl;


int main(int argc, char **argv) {
	cout << "Running" << endl;

	cuda_call_kernel();
	get_grav_pot();

    return 1;
}


