/* 
 * CUDA blur
 */
#ifndef _GRAV_CUDA_C_
	#define _GRAV_CUDA_C_
#endif

#include "grav_cuda.cuh"
#include "device_launch_parameters.h"
#include "cuda_header.cuh"

#include "grav_cpu.h"
#include "cuda_calls_helper.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;


CUDA_CALLABLE
void break_triangle(triangle face_tmp, vertex * v_tmp, float radius) {
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

__global__ void refine_icosphere_naive_kernal(triangle * faces, float radius, unsigned int depth) {

	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numthrds = blockDim.x * gridDim.x;

	unsigned int depth_c, th_len, write_offset;

	vertex v_tmp[3];
	depth_c  = depth;
		
	th_len = 20*pow(4, depth_c);
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


__global__ void refine_icosphere_kernal(triangle * faces, float radius, unsigned int depth) {

	// TODO implement the shared memory
	
	// extern __shared__ float shmem[];

	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numthrds = blockDim.x * gridDim.x;
    // unsigned tid = threadIdx.x;

	unsigned int depth_c, th_len, write_offset;

	vertex v_tmp[3];
	vertex v_storage[12];
	depth_c  = depth;
		
	th_len = 20*pow(4, depth_c);
	while(idx < 4*th_len){
		int tri_ind = idx/4;
		int sub_tri_ind = (int)idx%4;

		triangle tri_tmp = faces[tri_ind];
		write_offset = (sub_tri_ind == 0)*tri_ind + (sub_tri_ind!=0)*(th_len + 3*tri_ind + (idx%4-1));
		break_triangle(tri_tmp, v_tmp, radius);

		v_storage[0] = tri_tmp.v[0];
		v_storage[1] = v_tmp[0];
		v_storage[2] = v_tmp[2];
		
		v_storage[3] = v_tmp[0];
		v_storage[4] = tri_tmp.v[1];
		v_storage[5] = v_tmp[1];

		v_storage[6] = v_tmp[1];
		v_storage[7] = tri_tmp.v[2];
		v_storage[8] = v_tmp[2];
		
		v_storage[9] = v_tmp[0];
		v_storage[10] = v_tmp[1];
		v_storage[11] = v_tmp[2];

	
		faces[write_offset].v[0] = v_storage[3*sub_tri_ind];
		faces[write_offset].v[1] = v_storage[3*sub_tri_ind+1];
		faces[write_offset].v[2] = v_storage[3*sub_tri_ind+2];
	
		idx += numthrds;
	}

}
void cudacall_icosphere(int thread_num) {
	// each thread works on one face
	for(int i=0; i<max_depth; i++){
		int ths = 20*pow(4, i);

		// int * dev_res, * gpu_res_out;
		// CUDA_CALL(cudaMalloc((void **)&dev_res, 4*ths * sizeof(int)));
		// gpu_res_out = (int *)malloc(4*ths*sizeof(int));
		thread_num = thread_num - thread_num%4;
		int n_blocks = std::min(65535, (ths + thread_num  - 1) / thread_num);
		refine_icosphere_kernal<<<n_blocks, thread_num>>>(dev_faces, radius, i);

		// CUDA_CALL(cudaMemcpy(gpu_res_out, dev_res, 4*ths*sizeof(int), cudaMemcpyDeviceToHost));
		// CUDA_CALL(cudaFree(dev_res));
		// for(int j=0; j<4*ths; j++){
		// 	printf("j: %d is_not_zero %d, \n", j, gpu_res_out[j]);
		// }
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
}