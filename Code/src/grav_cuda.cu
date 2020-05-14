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

#define BW	1024

CUDA_CALLABLE
void break_triangle(triangle face, triangle * res) {
    vertex v[3];
    for(int i=0; i<3; i++){
    }
}

__global__ void create_icoshpere_kernal(triangle * faces, float radius, int depth) {

	extern __shared__ float shmem[];

	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    // unsigned tid = threadIdx.x;

	int depth_c;
	int th_len;
	triangle res[4];
	for(depth_c=0; depth_c<depth; depth_c++){
		th_len = 20*powf(4, depth_c);
		if(idx < th_len){
			break_triangle(faces[idx], res);
		}
	}
    
}

float cuda_call_kernel(triangle * faces, float radius, int depth) {
	// each thread works on one face
	int ths = 20*pow(4, depth);
	int n_blocks = std::min(65535, (ths + BW  - 1) / BW);

	create_icoshpere_kernal<<<n_blocks, BW, BW*sizeof(float)>>>(faces, radius, depth);

	cout << "Running from grav_cuda" << endl;
    return -1;
}
