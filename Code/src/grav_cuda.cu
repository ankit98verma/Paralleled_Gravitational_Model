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
		refine_icosphere_naive_kernal<<<n_blocks, thread_num>>>(dev_faces, radius, i);
	}
	
}


typedef void (*func_ptr_sub_triangle_t)(triangle, vertex *, triangle *, int, int);

__device__ void sub_triangle_top(triangle face_tmp, vertex * v_tmp, triangle * faces_arr, int ind0, int ind) {
	// int res_ind = ind0;
    faces_arr[ind0].v[0] = face_tmp.v[0];
    faces_arr[ind0].v[1] = v_tmp[0];
    faces_arr[ind0].v[2] = v_tmp[2];
}

__device__ void sub_triangle_left(triangle face_tmp, vertex * v_tmp, triangle * faces_arr, int ind0, int ind) {
	// int res_ind = ind;
    faces_arr[ind].v[0] = v_tmp[0];
    faces_arr[ind].v[1] = face_tmp.v[1];
    faces_arr[ind].v[2] = v_tmp[1];
}

__device__ void sub_triangle_right(triangle face_tmp, vertex * v_tmp, triangle * faces_arr, int ind0, int ind) {
	// int res_ind = ind;
    faces_arr[ind].v[0] = v_tmp[1];
    faces_arr[ind].v[1] = face_tmp.v[2];
    faces_arr[ind].v[2] = v_tmp[2];
}

__device__ void sub_triangle_center(triangle face_tmp, vertex * v_tmp, triangle * faces_arr, int ind0, int ind) {
	// int res_ind = ind;
    faces_arr[ind].v[0] = v_tmp[0];
    faces_arr[ind].v[1] = v_tmp[1];
    faces_arr[ind].v[2] = v_tmp[2];
}

__device__ func_ptr_sub_triangle_t funcs2[4] = {sub_triangle_top, sub_triangle_left, sub_triangle_right, sub_triangle_center};


__global__ void refine_icosphere_kernal(triangle * faces, const float radius, const unsigned int th_len) {

	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numthrds = blockDim.x * gridDim.x;

	unsigned int write_offset0, write_offset;
	
	vertex v_tmp[3];
		
	while(idx < 4*th_len){
		int tri_ind = idx/4;
		int sub_tri_ind = idx%4;

		triangle tri_tmp = faces[tri_ind];
		break_triangle(tri_tmp, v_tmp, radius);
		
		write_offset0 = tri_ind;
		write_offset = th_len + 3*tri_ind + (sub_tri_ind-1);

		funcs2[sub_tri_ind](tri_tmp, v_tmp, faces, write_offset0, write_offset);
	
		idx += numthrds;
	}

}
void cudacall_icosphere(int thread_num) {
	// each thread creates a sub triangle
	for(int i=0; i<max_depth; i++){
		int ths = 20*pow(4, i);
		int n_blocks = std::min(65535, (ths + thread_num  - 1) / thread_num);
		refine_icosphere_kernal<<<n_blocks, thread_num>>>(dev_faces, radius, ths);
	}

}

void cuda_cpy_input_data(){
	gpu_out_faces = (triangle *)malloc(faces_length*sizeof(triangle));
	CUDA_CALL(cudaMalloc((void **)&dev_faces, faces_length * sizeof(triangle)));
	CUDA_CALL(cudaMemcpy(dev_faces, faces_init, ICOSPHERE_INIT_FACE_LEN*sizeof(triangle), cudaMemcpyHostToDevice));
}

void cuda_cpy_output_data(){
	CUDA_CALL(cudaMemcpy(gpu_out_faces, dev_faces, faces_length*sizeof(triangle), cudaMemcpyDeviceToHost));
}

void free_gpu_memory(){
	CUDA_CALL(cudaFree(dev_faces));
	free(gpu_out_faces);
}