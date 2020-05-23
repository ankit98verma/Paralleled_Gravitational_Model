/*
 * grav_cuda.cuh
 * Kevin Yuh, 2014
 * Revised by Nailen Matschke, 2016
 * Revised by Loko Kung, 2018
 */

#ifndef _GRAV_CUDA_CUH_
#define _GRAV_CUDA_CUH_

#include "cuda_header.cuh"
#include "grav_cpu.hpp"

#undef  GLOBAL
#ifdef _GRAV_CUDA_C_
#define GLOBAL
#else
#define GLOBAL  extern
#endif

GLOBAL triangle * dev_faces_in;
GLOBAL triangle * dev_faces_out;
GLOBAL triangle * gpu_out_faces;

void cuda_cpy_input_data();
void cuda_cpy_output_data();
void cudacall_icosphere_naive(int);
void cudacall_icosphere(int);




void free_gpu_memory();
#endif
