/* 
 * grav_cuda.cuh
 * Kevin Yuh, 2014 
 * Revised by Nailen Matschke, 2016
 * Revised by Loko Kung, 2018
 */

#ifndef _GRAV_CUDA_CUH_
#define _GRAV_CUDA_CUH_

typedef unsigned int uint;
#include "cuda_header.cuh"

// CUDA_CALLABLE void template1(...);

// float template2(...);

float cuda_call_kernel();
#endif
