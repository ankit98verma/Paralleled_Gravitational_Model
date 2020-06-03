/*
 * grav_cuda.cuh
 * Kevin Yuh, 2014
 * Revised by Nailen Matschke, 2016
 * Revised by Loko Kung, 2018
 */

#ifndef _GRAV_CUDA_CUH_
#define _GRAV_CUDA_CUH_


#include "device_launch_parameters.h"
#include "grav_cpu.hpp"
#include "cuda_calls_helper.h"

#undef  GLOBAL
#ifdef _GRAV_CUDA_C_
#define GLOBAL
#else
#define GLOBAL  extern
#endif


GLOBAL triangle * gpu_out_faces;
GLOBAL vertex * gpu_out_vertices;

GLOBAL vertex * dev_vertices;
GLOBAL float * gpu_out_potential;
GLOBAL vertex * dev_vertices_ico;

void cuda_cpy_input_data();

void cudacall_icosphere_naive(int);
void cudacall_icosphere(int);

void cudacall_naive_fill_vertices(int);
void cudacall_fill_vertices(int);

void cuda_cpy_output_data();
void free_gpu_memory();


void cuda_cpy_input_data_potential(int);

void naive_cudacall_gravitational(int);
void optimal_cudacall_gravitational1(int);
void optimal_cudacall_gravitational2(int);
void optimal_cudacall_gravitational3();
void optimal_cudacall_gravitational4();

void cuda_cpy_output_data_potential();
void free_gpu_memory_potential();
#endif
