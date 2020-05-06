/* 
 * CUDA blur
 */
#ifndef _GRAV_CUDA_C_
	#define _GRAV_CUDA_C_
#endif

#include "cuda_header.cuh"
#include "grav_cuda.cuh"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "grav_cpu.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

// CUDA_CALLABLE
// void template1(...) {
    
// }

__global__ void create_icoshpere_kernal(triangle * faces, float radius, int depth) {

    
}

float cuda_call_kernel() {
	cout << "Running from grav_cuda" << endl;
    return -1;
}
