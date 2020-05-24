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


float* dev_coeff;
float* dev_potential;
vertex* dev_vertices;


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

// __device__ void gpu_facprod(int n, int m, float * p){


//     for (int i = n-m+1; i<=n+m; i++){
//         *p = *p/i;
//     }
// }


__device__ void gpu_spherical_harmonics(float radius, int n_sph, vertex dev_R_vec, float* dev_coeff, float* U, int thread_index){

    // Define pseudo coefficients
    float Radius_sq = powf(radius,2);
    float rho = powf(R_eq,2)/Radius_sq;

    float x0 = R_eq*dev_R_vec.x/Radius_sq;
    float y0 = R_eq*dev_R_vec.y/Radius_sq;
    float z0 = R_eq*dev_R_vec.z/Radius_sq;

    //Initialize Intermediary Matrices
    float V[n_sph+1][n_sph+1];
    float W[n_sph+1][n_sph+1];

    // Calculate zonal terms V(n, 0). Set W(n,0)=0.0
    V[0][0] = R_eq /sqrtf(Radius_sq);
    W[0][0] = 0.0;

    V[1][0] = z0 * V[0][0];
    W[1][0] = 0.0;

    for (int n=2; n<n_sph+1; n++){
        V[n][0] = ((2*n-1)*z0*V[n-1][0] - (n-1)*rho*V[n-2][0])/n;
        W[n][0] = 0.0;
    } // Eqn 3.30


    //Calculate tesseral and sectoral terms
    for (int m = 1; m < n_sph + 1; m++){
        // Eqn 3.29
        V[m][m] = (2*m-1)*(x0*V[m-1][m-1] - y0*W[m-1][m-1]);
        W[m][m] = (2*m-1)*(x0*W[m-1][m-1] + y0*V[m-1][m-1]);

        // n=m+1 (only one term)
        if (m < n_sph){
            V[m+1][m] = (2*m+1)*z0*V[m][m];
            W[m+1][m] = (2*m+1)*z0*W[m][m];
        }

        for (int n = m+2; n<n_sph+1; n++){
            V[n][m] = ((2*n-1)*z0*V[n-1][m]-(n+m-1)*rho*V[n-2][m])/(n-m);
            W[n][m] = ((2*n-1)*z0*W[n-1][m]-(n+m-1)*rho*W[n-2][m])/(n-m);
        }
    }

    float pseudo_coeff[n_sph+1][n_sph+2];
    for (int row =0; row<n_sph+1; row++){
        for (int col= 0; col<n_sph+2; col++){
            pseudo_coeff[row][col] = dev_coeff[row*(n_sph+1) + col];
        }
    }

    // Calculate potential
    float C = 0; // Cnm coeff
    float S = 0; // Snm coeff
    float N = 0; // normalisation number
    float p = 0;
    U[thread_index] = 0; //potential
    for (int m=0; m<n_sph+1; m++){
        for (int n = m; n<n_sph+1; n++){
            C = 0;
            S = 0;
            if (m==0){
                N = sqrt(2*n+1);
                C = N*pseudo_coeff[n][0];
                U[thread_index] = C*V[n][0];
            }
            else {
                // gpu_facprod(n,m,&p);
            	for (int i = n-m+1; i<=n+m; i++){
			        p = p/i;
			    }
                N = sqrt((2)*(2*n+1)*p);
                C = N*pseudo_coeff[n][m];
                S = N*pseudo_coeff[n_sph-n][n_sph-m+1];
            }
            U[thread_index] = U[thread_index] + C*V[n][m] + S*W[n][m];
            // Calculation of the Gravitational Potential Calculation model
        }
    }
    U[thread_index] = U[thread_index]*mhu/R_eq;
}


__global__
void kerne_gravitational(int g_vertices_length, float g_radius, float n_sph, float* dev_coeff, vertex* dev_vertices, float* dev_potential){

    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    while (thread_index < g_vertices_length) {

        gpu_spherical_harmonics(g_radius, n_sph, dev_vertices[thread_index], dev_coeff, dev_potential, thread_index);
        thread_index += blockDim.x * gridDim.x;
    }
}

void cudacall_gravitational(int thread_num){

    int n_blocks = vertices_length/thread_num;
    n_blocks = std::min(65535, n_blocks);
    kerne_gravitational<<<n_blocks, thread_num>>>(vertices_length, radius, N_SPHERICAL, dev_coeff, dev_vertices, dev_potential);

}

void cuda_cpy_input_data(){
	gpu_out_faces = (triangle *)malloc(faces_length*sizeof(triangle));
	CUDA_CALL(cudaMalloc((void **)&dev_faces_in, faces_length * sizeof(triangle)));
	CUDA_CALL(cudaMalloc((void **)&dev_faces_out, faces_length * sizeof(triangle)));
	CUDA_CALL(cudaMemcpy(dev_faces_in, faces_init, ICOSPHERE_INIT_FACE_LEN*sizeof(triangle), cudaMemcpyHostToDevice));

	pointers[0] = dev_faces_in;
	pointers[1] = dev_faces_out;

	// GARIMA DATA

	// GPU Coefficient file
	CUDA_CALL(cudaMalloc((void**) &dev_coeff, sizeof(float) * 2*N_coeff));
    CUDA_CALL(cudaMemcpy(dev_coeff, coeff, sizeof(float) * 2*N_coeff, cudaMemcpyHostToDevice));

    // Vertices
    CUDA_CALL(cudaMalloc((void**) &dev_vertices, sizeof(vertex) * vertices_length));
    CUDA_CALL(cudaMemcpy(dev_vertices, vertices, sizeof(vertex) * vertices_length, cudaMemcpyHostToDevice));


    // OUTput potential - to be compared with CPU values
    CUDA_CALL(cudaMalloc((void**) &dev_potential, sizeof(float) * vertices_length));
    gpu_out_potential = (float*) malloc(sizeof(float) * vertices_length);
}

void cuda_cpy_output_data(){
	CUDA_CALL(cudaMemcpy(gpu_out_faces, pointers[ind2], faces_length*sizeof(triangle), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(gpu_out_potential, dev_potential, vertices_length*sizeof(float), cudaMemcpyDeviceToHost));
}

void free_gpu_memory(){
	CUDA_CALL(cudaFree(dev_faces_in));
	CUDA_CALL(cudaFree(dev_faces_out));
	free(gpu_out_faces);
	CUDA_CALL(cudaFree(dev_coeff));
	CUDA_CALL(cudaFree(dev_potential));
	CUDA_CALL(cudaFree(dev_vertices));
	free(gpu_out_potential);
}
