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

#define BW	1024

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

__global__ void create_icoshpere_kernal(triangle * faces, float radius, int depth) {

	extern __shared__ float shmem[];

	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // unsigned tid = threadIdx.x;

	unsigned int depth_c, th_len, write_offset;

	vertex v_tmp[3];
	for(depth_c=0; depth_c<depth; depth_c++){
		
		th_len = 20*pow(4, depth_c);
		
		if(idx < th_len){
			triangle tri_tmp = faces[idx];
			write_offset = th_len + 3*idx;
			
			break_triangle(tri_tmp, v_tmp, radius);
			// got the mid points of the vertices now make new triangles
			faces[idx].v[1].x = v_tmp[0].x;
			faces[idx].v[1].y = v_tmp[0].y;
			faces[idx].v[1].z = v_tmp[0].z;

			faces[idx].v[2].x = v_tmp[2].x;
			faces[idx].v[2].y = v_tmp[2].y;
			faces[idx].v[2].z = v_tmp[2].z;

			// adding triangle V[0], P1, V[1]
			faces[write_offset].v[0].x = v_tmp[0].x;
			faces[write_offset].v[0].y = v_tmp[0].y;
			faces[write_offset].v[0].z = v_tmp[0].z;

			faces[write_offset].v[1].x = tri_tmp.v[1].x;
			faces[write_offset].v[1].y = tri_tmp.v[1].y;
			faces[write_offset].v[1].z = tri_tmp.v[1].z;

			faces[write_offset].v[2].x = v_tmp[1].x;
			faces[write_offset].v[2].y = v_tmp[1].y;
			faces[write_offset].v[2].z = v_tmp[1].z;
			write_offset++;

			//adding triangle P2, V[1], V[2]
			faces[write_offset].v[0].x = v_tmp[1].x;
			faces[write_offset].v[0].y = v_tmp[1].y;
			faces[write_offset].v[0].z = v_tmp[1].z;

			faces[write_offset].v[1].x = tri_tmp.v[2].x;
			faces[write_offset].v[1].y = tri_tmp.v[2].y;
			faces[write_offset].v[1].z = tri_tmp.v[2].z;

			faces[write_offset].v[2].x = v_tmp[2].x;
			faces[write_offset].v[2].y = v_tmp[2].y;
			faces[write_offset].v[2].z = v_tmp[2].z;
			write_offset++;

			//adding triangle V[0], V[1], V[2]
			faces[write_offset].v[0].x = v_tmp[0].x;
			faces[write_offset].v[0].y = v_tmp[0].y;
			faces[write_offset].v[0].z = v_tmp[0].z;

			faces[write_offset].v[1].x = v_tmp[1].x;
			faces[write_offset].v[1].y = v_tmp[1].y;
			faces[write_offset].v[1].z = v_tmp[1].z;

			faces[write_offset].v[2].x = v_tmp[2].x;
			faces[write_offset].v[2].y = v_tmp[2].y;
			faces[write_offset].v[2].z = v_tmp[2].z;
			write_offset++;
		}
		__syncthreads();
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

void cuda_call_kernel() {

	// each thread works on one face
	int ths = 20*pow(4, max_depth);
	int n_blocks = std::min(65535, (ths + BW  - 1) / BW);

	printf("radius: %f, max_depth: %d\n", radius, max_depth);
	create_icoshpere_kernal<<<n_blocks, BW, BW*sizeof(float)>>>(dev_faces, radius, max_depth);
	
}
