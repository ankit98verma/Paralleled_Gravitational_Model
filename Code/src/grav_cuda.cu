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
#include <fstream>
#include <iostream>
#include <math.h>
using namespace std;


// variables local to this file
float* dev_coeff;
float* dev_potential;
vertex* dev_vertices;

int ind2_faces;
triangle * pointers[2];
triangle * dev_faces_in;
triangle * dev_faces_out;

int * pointers_inds[2];
int ind2_inds;
int * dev_face_vert_ind;
int * dev_face_vert_ind_res;


float * pointers_sums[2];
int ind2_sums;
float * dev_face_sums;
float * dev_face_sums_res;

triangle * dev_verties;


void cuda_cpy_input_data(){
    gpu_out_faces = (triangle *)malloc(faces_length*sizeof(triangle));
    CUDA_CALL(cudaMalloc((void **)&dev_faces_in, faces_length * sizeof(triangle)));
    CUDA_CALL(cudaMalloc((void **)&dev_faces_out, faces_length * sizeof(triangle)));
    
    CUDA_CALL(cudaMalloc((void **)&dev_face_vert_ind, 3*faces_length * sizeof(int)));
    CUDA_CALL(cudaMalloc((void **)&dev_face_vert_ind_res, 3*faces_length * sizeof(int)));
    
    CUDA_CALL(cudaMalloc((void **)&dev_face_sums, 3*faces_length * sizeof(float)));    
    CUDA_CALL(cudaMalloc((void**) &dev_face_sums_res, 3*faces_length* sizeof(float)));

    CUDA_CALL(cudaMemcpy(dev_faces_in, faces_init, ICOSPHERE_INIT_FACE_LEN*sizeof(triangle), cudaMemcpyHostToDevice));

    ind2_faces = 0;
    pointers[0] = dev_faces_in;
    pointers[1] = dev_faces_out;

    ind2_sums = 0;
    pointers_sums[0] = dev_face_sums;
    pointers_sums[1] = dev_face_sums_res;

    ind2_inds = 0;
    pointers_inds[0] = dev_face_vert_ind;
    pointers_inds[1] = dev_face_vert_ind_res;

    // GARIMA DATA

    // GPU Coefficient file
    CUDA_CALL(cudaMalloc((void**) &dev_coeff, sizeof(float) * 2*N_coeff));
    CUDA_CALL(cudaMemcpy(dev_coeff, coeff, sizeof(float) * 2*N_coeff, cudaMemcpyHostToDevice));

    // Vertices
    CUDA_CALL(cudaMalloc((void**) &dev_vertices, sizeof(vertex) * vertices_length));
    CUDA_CALL(cudaMemcpy(dev_vertices, vertices, sizeof(vertex) * vertices_length, cudaMemcpyHostToDevice));

    // OUTput potential - to be compared with CPU values
    CUDA_CALL(cudaMalloc((void**) &dev_potential, sizeof(float) * vertices_length));
    CUDA_CALL(cudaMemset(dev_potential, 0, vertices_length* sizeof(float)));
    gpu_out_potential = (float*) malloc(sizeof(float) * vertices_length);

}

void cuda_cpy_output_data(){
    CUDA_CALL(cudaMemcpy(gpu_out_faces, pointers[ind2_faces], faces_length*sizeof(triangle), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(gpu_out_potential, dev_potential, vertices_length*sizeof(float), cudaMemcpyDeviceToHost));
}

void free_gpu_memory(){
    CUDA_CALL(cudaFree(dev_faces_in));
    CUDA_CALL(cudaFree(dev_faces_out));
    
    CUDA_CALL(cudaFree(dev_face_vert_ind));
    CUDA_CALL(cudaFree(dev_face_vert_ind_res));
    
    CUDA_CALL(cudaFree(dev_face_sums));
    CUDA_CALL(cudaFree(dev_face_sums_res));


    CUDA_CALL(cudaFree(dev_coeff));
    CUDA_CALL(cudaFree(dev_potential));
    CUDA_CALL(cudaFree(dev_vertices));
    free(gpu_out_faces);
    free(gpu_out_potential);
}


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


__global__ void refine_icosphere_kernal(triangle * faces, float * sums, const float radius, const unsigned int th_len, triangle * faces_out) {
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


__global__ 
void kernal_fill_sums_inds(vertex * vs, float * sums, int * inds, const unsigned int vertices_length){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numthrds = blockDim.x * gridDim.x;

    while(idx < vertices_length){
        sums[idx] = vs[idx].x + vs[idx].y + vs[idx].z;
        inds[idx] = idx;
        idx += numthrds;
    }
}

void cudacall_icosphere(int thread_num) {
	// each thread creates a sub triangle
	int ths, n_blocks, ind1;
	for(int i=0; i<max_depth; i++){
		ths = 20*pow(4, i);
		n_blocks = std::min(65535, (ths + thread_num  - 1) / thread_num);
		ind1 = i%2;
		ind2_faces = (i+1)%2;
		refine_icosphere_kernal<<<n_blocks, thread_num>>>(pointers[ind1], dev_face_sums, radius, ths, pointers[ind2_faces]);
	}
    int len = 3*faces_length;
    n_blocks = std::min(65535, (len + thread_num  - 1) / thread_num);
    kernal_fill_sums_inds<<<n_blocks, thread_num>>>((vertex *)pointers[ind2_faces], dev_face_sums, dev_face_vert_ind, len);
}


__device__
void dev_merge(float * s, float * r, int * ind, int * ind_res, unsigned int idx, unsigned int start, unsigned int end){
    unsigned int c=idx;
    unsigned int i=idx;unsigned int j=start;
    while(j<end && i<start){
        if(s[i] <= s[j]){
            r[c] = s[i];
            ind_res[c] = ind[i];
            i++;
        }
        else{
            r[c] = s[j];
            ind_res[c] = ind[j];
            j++;
        }
        c++;
    }
    while(i < start){
        r[c] = s[i];
        ind_res[c] = ind[i];
        c++;i++;
    }
    
    while(j < end){
        r[c] = s[j];
        ind_res[c] = ind[j];
        c++;j++;
    }
}

__global__
void kernal_merge_navie_sort(float * sums, float * res, int * ind, int * ind_res, const unsigned int length, const unsigned int r){
    
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numthrds = blockDim.x * gridDim.x;

    const unsigned int stride = r/2;
    
    while(idx < length){
        if(idx%r == 0)
            dev_merge(sums, res, ind, ind_res, idx, (unsigned int)min(length, idx + stride), (unsigned int)min(length, idx+r));
        idx += numthrds;
    }
}

__global__
void kernal_merge_sort(float * sums, float * res, int * ind, int * ind_res, const unsigned int length, const unsigned int r){
    
    __shared__ float sh_sums[1024];
    __shared__ float sh_res[1024];
    __shared__ int sh_ind[1024];
    __shared__ int sh_indres[1024];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numthrds = blockDim.x * gridDim.x;

    const int stride = r/2;

    int id = threadIdx.x;
    int t_len = min(1024, length - blockIdx.x * blockDim.x);
    
    while(idx < length){
        // copy to shared mem
        sh_sums[threadIdx.x] = sums[idx];
        sh_ind[threadIdx.x] = ind[idx];

        __syncthreads();
        
        if(id%r == 0)
            dev_merge(sh_sums, sh_res, sh_ind, sh_indres, id, min(t_len, id + stride), min(t_len, id+r));
        
        __syncthreads();
        // copy result to global mem
        res[idx] = sh_res[threadIdx.x];
        ind_res[idx] = sh_indres[threadIdx.x];
        __syncthreads();
        idx += numthrds;
    }
}

// doesn't work
__global__
void kernal_merge_chuncks(float * sums, float * res, int * ind, int * ind_res, const unsigned int length, const unsigned int r){

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numthrds = blockDim.x * gridDim.x;

    const unsigned int stride = r/2;

    unsigned int id;
    while(idx*r < length){
        id = idx*r;
        dev_merge(sums, res, ind, ind_res, id, min(length, id + stride), min(length, id+r));
        idx += numthrds;
    }
}

__global__ 
void kernal_update_faces(vertex * f_in, vertex * f_out, int * inds, const unsigned int vertices_length){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numthrds = blockDim.x * gridDim.x;

    while(idx < vertices_length){
        f_out[idx] = f_in[inds[idx]];
        idx += numthrds;
    }
}

void cudacall_sort(int thread_num) {
    
    unsigned int len = 3*faces_length;
    int n_blocks = min(65535, (len + thread_num  - 1) / thread_num);
    
    unsigned int l = ceil(log2(len)), ind1;

    for(int i=0; i<l; i++){
        ind1 = i%2;
        ind2_sums = (i+1)%2;
        ind2_inds = ind2_sums;
        unsigned int r = pow(2, i+1);
        // kernal_merge_navie_sort<<<n_blocks, thread_num>>>(pointers_sums[ind1], pointers_sums[ind2_sums], pointers_inds[ind1], pointers_inds[ind2_inds], len, r);
        kernal_merge_sort<<<n_blocks, thread_num>>>(pointers_sums[ind1], pointers_sums[ind2_sums], pointers_inds[ind1], pointers_inds[ind2_inds], len, r);
        
    }    
    
    // now sort the chunks of 1024 floats
    l = ceil(log2(n_blocks));
    for(int i=0; i<l; i++){
        ind1 = (ind1+1)%2;
        ind2_sums = (ind2_sums+1)%2;
        ind2_inds = ind2_sums;
        unsigned int r = pow(2, i+1)*1024;
        kernal_merge_navie_sort<<<n_blocks, thread_num>>>(pointers_sums[ind1], pointers_sums[ind2_sums], pointers_inds[ind1], pointers_inds[ind2_inds], len, r);
    }
    
    // CUDA_CALL(cudaMemcpy(sums, pointers_sums[ind2_sums], len*sizeof(float), cudaMemcpyDeviceToHost));
    // CUDA_CALL(cudaMemcpy(tmp, pointers_inds[ind2_inds], len*sizeof(int), cudaMemcpyDeviceToHost));
    
    // working
    n_blocks = std::min(65535, ((int)len + thread_num  - 1) / thread_num);
    int out = (ind2_faces + 1) %2;
    kernal_update_faces<<<n_blocks, thread_num>>>((vertex *)pointers[ind2_faces], (vertex *)pointers[out], pointers_inds[ind2_inds], len);
    cudaDeviceSynchronize();
    ind2_faces = out;
}


__device__ void gpu_spherical_harmonics(float radius, const int n_sph, vertex dev_R_vec, float* dev_coeff, float* U, int thread_index){

    float dev_V[21*21];
    float dev_W[21*21];

    // Define pseudo coefficients
    float Radius_sq = powf(radius,2);
    float rho = powf(R_eq,2)/Radius_sq;

    float x0 = R_eq*dev_R_vec.x/Radius_sq;
    float y0 = R_eq*dev_R_vec.y/Radius_sq;
    float z0 = R_eq*dev_R_vec.z/Radius_sq;

    // Calculate zonal terms V(n, 0). Set W(n,0)=0.0
    dev_V[0]= R_eq /sqrtf(Radius_sq);
    dev_W[0] = 0.0;

    dev_V[1*(n_sph+1) + 0] = z0 *dev_V[0];
    dev_W[1*(n_sph+1) + 0] = 0.0;

    for (int n=2; n<n_sph+1; n++){
        dev_V[n*(n_sph+1) + 0] = ((2*n-1)*z0*dev_V[(n-1)*(n_sph+1) + 0] - (n-1)*rho*dev_V[(n-2)*(n_sph+1) + 0])/n;
        dev_W[n*(n_sph+1) + 0] = 0.0;
    } // Eqn 3.30


    //Calculate tesseral and sectoral terms
    for (int m = 1; m < n_sph + 1; m++){
        // Eqn 3.29
        dev_V[m*(n_sph+1) + m] = (2*m-1)*(x0*dev_V[(m-1)*(n_sph+1) + (m-1)] - y0*dev_W[(m-1)*(n_sph+1) + (m-1)]);
        dev_W[m*(n_sph+1) + m] = (2*m-1)*(x0*dev_W[(m-1)*(n_sph+1) + (m-1)] + y0*dev_V[(m-1)*(n_sph+1) + (m-1)]);

        // n=m+1 (only one term)
        if (m < n_sph){
            dev_V[(m+1)*(n_sph+1) + (m)] = (2*m+1)*z0*dev_V[m*(n_sph+1) + m];
            dev_W[(m+1)*(n_sph+1) + (m)] = (2*m+1)*z0*dev_W[m*(n_sph+1) + m] ;
        }

        for (int n = m+2; n<n_sph+1; n++){
            dev_V[n*(n_sph+1) + m] = ((2*n-1)*z0*dev_V[(n-1)*(n_sph+1) + m]-(n+m-1)*rho*dev_V[(n-2)*(n_sph+1) + m])/(n-m);
            dev_W[n*(n_sph+1) + m] = ((2*n-1)*z0*dev_W[(n-1)*(n_sph+1) + m]-(n+m-1)*rho*dev_W[(n-2)*(n_sph+1) + m])/(n-m);
        }
    }

    // Calculate potential
    float C = 0; // Cnm coeff
    float S = 0; // Snm coeff
    float N = 0; // normalisation number
    float p = 1.0;
    U[thread_index] = 0; //potential
    for (int m=0; m<n_sph+1; m++){
        for (int n = m; n<n_sph+1; n++){
            C = 0;
            S = 0;
            if (m==0){
                N = sqrtf(2*n+1);
                C = N*dev_coeff[n*(n_sph+2)+0];
//                U[thread_index] = C*dev_V[n*(n_sph+1) + 0];
            }
            else {
                p = 1.0;
                // gpu_facprod(n,m,&p);
                for (int i = n-m+1; i<=n+m; i++){
                    p = p/i;
                }
                N = sqrtf((2)*(2*n+1)*p);
                C = N*dev_coeff[n*(n_sph+2)+m];
                S = N*dev_coeff[(n_sph-n)*(n_sph+2)+ (n_sph-m+1)];
            }
            U[thread_index] = U[thread_index] + C*dev_V[n*(n_sph+1) + m] + S*dev_W[n*(n_sph+1) + m];
            // Calculation of the Gravitational Potential Calculation model
        }
    }
    U[thread_index] = U[thread_index]*mhu/R_eq;
}


__global__
void naive_kernel_gravitational(int g_vertices_length, float g_radius, const int n_sph, float* dev_coeff, vertex* dev_vertices, float* dev_potential){


    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    while (thread_index < g_vertices_length) {

        gpu_spherical_harmonics(g_radius, n_sph, dev_vertices[thread_index], dev_coeff, dev_potential, thread_index);
        thread_index += blockDim.x * gridDim.x;
    }

}

__global__
void optimal_kernel_gravitational(int g_vertices_length, float radius, float eq_R, const int n_sph, float* dev_coeff, vertex* dev_vertices, float* U, int* M, int* N){


//    int thread_index = (blockIdx.x * blockDim.x + threadIdx.x);
//    int potential_index = (blockIdx.x * blockDim.x + threadIdx.x)%231;

    int potential_index = blockIdx.x;

    float dev_V[21*21];
    float dev_W[21*21];

    // Define pseudo coefficients
    float Radius_sq = powf(radius,2);
    float rho = powf(eq_R,2)/Radius_sq;

    float x0 = eq_R*dev_vertices[potential_index].x/Radius_sq;
    float y0 = eq_R*dev_vertices[potential_index].y/Radius_sq;
    float z0 = eq_R*dev_vertices[potential_index].z/Radius_sq;

    // Calculate zonal terms V(n, 0). Set W(n,0)=0.0
    dev_V[0]= eq_R/radius;
    dev_W[0] = 0.0;

    dev_V[1*(n_sph+1) + 0] = z0 *dev_V[0];
    dev_W[1*(n_sph+1) + 0] = 0.0;

    for (int n=2; n<n_sph+1; n++){
        dev_V[n*(n_sph+1) + 0] = ((2*n-1)*z0*dev_V[(n-1)*(n_sph+1) + 0] - (n-1)*rho*dev_V[(n-2)*(n_sph+1) + 0])/n;
        dev_W[n*(n_sph+1) + 0] = 0.0;
    } // Eqn 3.30

    //Calculate tesseral and sectoral terms
    for (int m = 1; m < n_sph + 1; m++){
        // Eqn 3.29
        dev_V[m*(n_sph+1) + m] = (2*m-1)*(x0*dev_V[(m-1)*(n_sph+1) + (m-1)] - y0*dev_W[(m-1)*(n_sph+1) + (m-1)]);
        dev_W[m*(n_sph+1) + m] = (2*m-1)*(x0*dev_W[(m-1)*(n_sph+1) + (m-1)] + y0*dev_V[(m-1)*(n_sph+1) + (m-1)]);

        // n=m+1 (only one term)
        if (m < n_sph){
            dev_V[(m+1)*(n_sph+1) + (m)] = (2*m+1)*z0*dev_V[m*(n_sph+1) + m];
            dev_W[(m+1)*(n_sph+1) + (m)] = (2*m+1)*z0*dev_W[m*(n_sph+1) + m] ;
        }

        for (int n = m+2; n<n_sph+1; n++){
            dev_V[n*(n_sph+1) + m] = ((2*n-1)*z0*dev_V[(n-1)*(n_sph+1) + m]-(n+m-1)*rho*dev_V[(n-2)*(n_sph+1) + m])/(n-m);
            dev_W[n*(n_sph+1) + m] = ((2*n-1)*z0*dev_W[(n-1)*(n_sph+1) + m]-(n+m-1)*rho*dev_W[(n-2)*(n_sph+1) + m])/(n-m);
        }
    }

        // thread index for the block and shared memory
        unsigned int tid = threadIdx.x;

        __shared__ float shmem[256]; //stores CV+SW
        shmem[tid] = 0.0; //potential

        // Calculate potential
        float C = 0; // Cnm coeff
        float S = 0; // Snm coeff
        float Norm = 0; // normalisation number
        float p = 1.0;

        if (tid<N_coeff){
            int n = N[tid];
            int m = M[tid];

            if (m==0){
                    Norm = sqrtf(2*n+1);
                    C = Norm*dev_coeff[n*(n_sph+2)+0];
                    shmem[tid] = C*dev_V[n*(n_sph+1) + 0];
                }
                else {
                    p = 1.0;
                    for (int i = n-m+1; i<=n+m; i++){
                        p = p/i;
                    }
                    Norm = sqrtf((2)*(2*n+1)*p);
                    C = Norm*dev_coeff[n*(n_sph+2)+m];
                    S = Norm*dev_coeff[(n_sph-n)*(n_sph+2)+ (n_sph-m+1)];
                    shmem[tid] = C*dev_V[n*(n_sph+1) + m] + S*dev_W[n*(n_sph+1) + m];
                }
        }
        // Calculation of the Gravitational Potential Calculation model

        // sync threads before commencing the stages of reduction
        __syncthreads();

        // Reduction #3: Sequential Addressing
        // Ref: Presentation "Optimizing Parallel Reduction in CUDA", by Mark Harris.
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // conduct the summation
            shmem[tid] = shmem[tid] + shmem[tid + s];
//                atomicAdd(&shmem[tid], shmem[tid + s]);
        }
        // Sync threads after every stage of reduction
        __syncthreads();
    }

//    U[potential_index] = shmem[0]*mhu/R_eq;
    U[potential_index] = shmem[0];
//        thread_index += blockDim.x * gridDim.x;
//    }

}

void optimal_cudacall_gravitational(int thread_num){

//    int n_blocks = ceil(vertices_length*1.0/thread_num);
//    n_blocks = std::min(65535, n_blocks);

    int len = vertices_length;
//    int n_blocks = std::min(65535, (len + thread_num  - 1) / thread_num);
    int n_blocks = std::min(65535, len);
    cout<<"\n Number of blocks \t"<<n_blocks<<'\n';

    int M[N_coeff];
    int N[N_coeff];

    int k = 0;
    for (int n=0;n<N_SPHERICAL+1;n++){
        for (int m=0;m<n+1;m++){
            N[k] = n;
            M[k] = m;
            k++;
        }
    }

    int* dev_M;
    int* dev_N;

    CUDA_CALL(cudaMalloc((void**) &dev_N, sizeof(int) * N_coeff));
    CUDA_CALL(cudaMalloc((void**) &dev_M, sizeof(int) * N_coeff));
    CUDA_CALL(cudaMemcpy(dev_N, N, sizeof(int) * N_coeff, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_M, M, sizeof(int) * N_coeff, cudaMemcpyHostToDevice));
    optimal_kernel_gravitational<<<n_blocks, thread_num>>>(vertices_length, radius, R_eq, N_SPHERICAL, dev_coeff, dev_vertices, dev_potential, dev_M, dev_N);
    CUDA_CALL(cudaFree(dev_M));
    CUDA_CALL(cudaFree(dev_N));
}





void naive_cudacall_gravitational(int thread_num){

//    int n_blocks = ceil(vertices_length*1.0/thread_num);
//    n_blocks = std::min(65535, n_blocks);

    int len = vertices_length;
    int n_blocks = std::min(65535, (len + thread_num  - 1) / thread_num);

    naive_kernel_gravitational<<<n_blocks, thread_num>>>(vertices_length, radius, N_SPHERICAL, dev_coeff, dev_vertices, dev_potential);
}

