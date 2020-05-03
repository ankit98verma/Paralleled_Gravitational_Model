/* 
 * CUDA blur
 */

#include "grav_cuda.cuh"

#include <cstdio>
#include <cstdlib>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;


#include <cuda_runtime.h>
#include "cuda_header.cuh"
#include "device_launch_parameters.h"



// CUDA_CALLABLE
// void template1(...) {
    
// }

// __global__
// void template2(...) {
    
// }

float cuda_call_kernel() {
	cout << "Running from grav_cuda" << endl;
    return -1;
}
