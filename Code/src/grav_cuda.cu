/*
 * CUDA blur
 */
#ifndef _GRAV_CUDA_C_
	#define _GRAV_CUDA_C_
#endif

#include "grav_cuda.cuh"
#include "device_launch_parameters.h"
#include "cuda_header.cuh"

#include "grav_cpu.hpp"
#include "cuda_calls_helper.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;




__device__ void break_triangle(triangle face_tmp, vertex * v_tmp, float radius) {
	float x_tmp, y_tmp, z_tmp, scale;
    for(int i=0; i<3; i++){
    	x_tmp = (face_tmp.v[i].x + face_tmp.v[(i+1)%3].x)/2;
		y_tmp = (face_tmp.v[i].y + face_tmp.v[(i+1)%3].y)/2;
		z_tmp = (face_tmp.v[i].z + face_tmp.v[(i+1)%3].z)/2;
		scale = radius/sqrtf(x_tmp*x_tmp + y_tmp*y_tmp + z_tmp*z_tmp);
		v_tmp[i].x = x_tmp*scale;
		v_tmp[i].y = y_tmp*scale;
		v_tmp[i].z = z_tmp*scale;
    }
}

__global__ void refine_icosphere_naive_kernal(triangle * faces, const float radius, const unsigned int depth) {

	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numthrds = blockDim.x * gridDim.x;

	unsigned int  write_offset;

	vertex v_tmp[3];

	const unsigned int th_len = 20*pow(4, depth);
	while(idx < th_len){

		triangle tri_tmp = faces[idx];
		write_offset = th_len + 3*idx;

		break_triangle(tri_tmp, v_tmp, radius);
		// got the mid points of the vertices now make new triangles
		faces[idx].v[1] = v_tmp[0];
		faces[idx].v[2] = v_tmp[2];

		// adding triangle V[0], P1, V[1]
		faces[write_offset].v[0] = v_tmp[0];
		faces[write_offset].v[1] = tri_tmp.v[1];
		faces[write_offset].v[2] = v_tmp[1];
		write_offset++;

		//adding triangle P2, V[1], V[2]
		faces[write_offset].v[0] = v_tmp[1];
		faces[write_offset].v[1] = tri_tmp.v[2];
		faces[write_offset].v[2] = v_tmp[2];
		write_offset++;

		//adding triangle V[0], V[1], V[2]
		faces[write_offset].v[0] = v_tmp[0];
		faces[write_offset].v[1] = v_tmp[1];
		faces[write_offset].v[2] = v_tmp[2];
		write_offset++;

		idx += numthrds;
	}

}

void cudacall_icosphere_naive(int thread_num) {
	// each thread works on one face
	for(int i=0; i<max_depth; i++){
		int ths = 20*pow(4, i);
		int n_blocks = std::min(65535, (ths + thread_num  - 1) / thread_num);
		refine_icosphere_naive_kernal<<<n_blocks, thread_num>>>(dev_faces_in, radius, i);
	}

}


typedef void (*func_ptr_sub_triangle_t)(triangle, vertex *, triangle *);

__device__ void sub_triangle_top(triangle face_tmp, vertex * v_tmp, triangle * res) {
    res->v[0] = face_tmp.v[0];
    res->v[1] = v_tmp[0];
    res->v[2] = v_tmp[2];
}

__device__ void sub_triangle_left(triangle face_tmp, vertex * v_tmp, triangle * res) {
    res->v[0] = v_tmp[0];
    res->v[1] = face_tmp.v[1];
    res->v[2] = v_tmp[1];
}

__device__ void sub_triangle_right(triangle face_tmp, vertex * v_tmp, triangle * res) {
    res->v[0] = v_tmp[1];
    res->v[1] = face_tmp.v[2];
    res->v[2] = v_tmp[2];
}

__device__ void sub_triangle_center(triangle face_tmp, vertex * v_tmp, triangle * res) {
    res->v[0] = v_tmp[0];
    res->v[1] = v_tmp[1];
    res->v[2] = v_tmp[2];
}

__device__ func_ptr_sub_triangle_t funcs2[4] = {sub_triangle_top, sub_triangle_left, sub_triangle_right, sub_triangle_center};


__global__ void refine_icosphere_kernal(triangle * faces, const float radius, const unsigned int th_len, triangle * faces_out) {

	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numthrds = blockDim.x * gridDim.x;

	vertex v_tmp[3];

	while(idx < 4*th_len){
		int tri_ind = idx/4;
		int sub_tri_ind = idx%4;

		break_triangle(faces[tri_ind], v_tmp, radius);

		funcs2[sub_tri_ind](faces[tri_ind], v_tmp, &faces_out[idx]);

		idx += numthrds;
	}
}

// variables local to this file
int ind2;
triangle* pointers[2];
void cudacall_icosphere(int thread_num) {

	// each thread creates a sub triangle
	int ths, n_blocks, ind1;
	for(int i=0; i<max_depth; i++){
		ths = 20*pow(4, i);
		n_blocks = std::min(65535, (ths + thread_num  - 1) / thread_num);
		ind1 = i%2;
		ind2 = (i+1)%2;
		refine_icosphere_kernal<<<n_blocks, thread_num>>>(pointers[ind1], radius, ths, pointers[ind2]);
	}
}

void cuda_cpy_input_data(){
	gpu_out_faces = (triangle *)malloc(faces_length*sizeof(triangle));
	CUDA_CALL(cudaMalloc((void **)&dev_faces_in, faces_length * sizeof(triangle)));
	CUDA_CALL(cudaMalloc((void **)&dev_faces_out, faces_length * sizeof(triangle)));
	CUDA_CALL(cudaMemcpy(dev_faces_in, faces_init, ICOSPHERE_INIT_FACE_LEN*sizeof(triangle), cudaMemcpyHostToDevice));

	pointers[0] = dev_faces_in;
	pointers[1] = dev_faces_out;
}

void spherical_harmonics(int thread_num) {

	// each thread creates a sub triangle
	int ths, n_blocks, ind1;
	for(int i=0; i<max_depth; i++){
		ths = 20*pow(4, i);
		n_blocks = std::min(65535, (ths + thread_num  - 1) / thread_num);
		ind1 = i%2;
		ind2 = (i+1)%2;
		refine_icosphere_kernal<<<n_blocks, thread_num>>>(pointers[ind1], radius, ths, pointers[ind2]);
	}
}

void cuda_cpy_output_data(){
	CUDA_CALL(cudaMemcpy(gpu_out_faces, pointers[ind2], faces_length*sizeof(triangle), cudaMemcpyDeviceToHost));
}

void free_gpu_memory(){
	CUDA_CALL(cudaFree(dev_faces_in));
	CUDA_CALL(cudaFree(dev_faces_out));
	free(gpu_out_faces);
}
