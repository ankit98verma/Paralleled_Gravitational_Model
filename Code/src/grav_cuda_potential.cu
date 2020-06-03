/*
 * CUDA blur
 */
#ifndef _GRAV_CUDA_POTENTIAL_C_
	#define _GRAV_CUDA_POTENTIAL_C_
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


/*******************************************************************************
 * Function:        cuda_cpy_input_data_potential
 *
 * Description:     Dynamic memory allocation in CUDA and also copy information
 *                  host to device
 *
 * Arguments:       null
 *
 * Return Values:   null
*******************************************************************************/
void cuda_cpy_input_data_potential(){
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


/*******************************************************************************
 * Function:        cuda_cpy_output_data_potential
 *
 * Description:     Copy information from device to host
 *
 * Arguments:       null
 *
 * Return Values:   null
*******************************************************************************/
void cuda_cpy_output_data_potential(){
    CUDA_CALL(cudaMemcpy(gpu_out_potential, dev_potential, vertices_length*sizeof(float), cudaMemcpyDeviceToHost));
}

/*******************************************************************************
 * Function:        free_gpu_memory_potential
 *
 * Description:     Copy information from device to host
 *
 * Arguments:       null
 *
 * Return Values:   null
*******************************************************************************/
void free_gpu_memory_potential(){
    CUDA_CALL(cudaFree(dev_coeff));
    CUDA_CALL(cudaFree(dev_potential));
    CUDA_CALL(cudaFree(dev_vertices));
    free(gpu_out_potential);
}

/*******************************************************************************
 * Function:        naive_gpu_spherical_harmonics
 *
 * Description:     Naive implementation of the CPU potential calculation
 *
 * Arguments:       float radius: radius of the sphere
 *                  const int n_sph: degree of potential
 *                  vertex dev_R_vec: vertex on the sphere
 *                  float* dev_coeff: coefficient matrix
 *                  float* U: potential
 *                  int thread_index: stores the vertex index actually
 *
 * Return Values:   null
*******************************************************************************/
__device__ void naive_gpu_spherical_harmonics(float radius, const int n_sph, vertex dev_R_vec, float* dev_coeff, float* U, int thread_index){

    /*
    Naive implementation of he CPU code in GPU.
    Computes the potential for every vertex in each thread.
    Effective when large number of vertices on the sphere- can then use GPU effectively
    Each thread handles each vertex independently
    Number of thread/block = 512
    */

    // 462 float array gets stores in the global memory and hence is slower than shared memory access.
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

    // Stores the potential for each vertex
    U[thread_index] = U[thread_index]*mhu/R_eq;
}


/*******************************************************************************
 * Function:        naive_kernel_gravitational
 *
 * Description:     Kernel: Naive implementation of the CPU potential calculation
 *
 * Arguments:       int g_vertices_length: Total number of vertices on the sphere
 *                  float g_radius: radius of the sphere
 *                  const int n_sph: degree of potential
 *                  vertex* dev_vertices: vertices on the sphere
 *                  float* dev_coeff: coefficient matrix
 *                  float* dev_potential: potential
 *
 * Return Values:   null
*******************************************************************************/
__global__
void naive_kernel_gravitational(int g_vertices_length, float g_radius, const int n_sph, float* dev_coeff, vertex* dev_vertices, float* dev_potential){


    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    while (thread_index < g_vertices_length) {

        naive_gpu_spherical_harmonics(g_radius, n_sph, dev_vertices[thread_index], dev_coeff, dev_potential, thread_index);
        thread_index += blockDim.x * gridDim.x;
    }
}

/*******************************************************************************
 * Function:        optimal_kernel_gravitational
 *
 * Description:     Optimal implementation: Attempt 1
 *
 * Arguments:       int g_vertices_length: Total number of vertices on the sphere
 *                  float radius: radius of the sphere
 *                  float eq_R: Radius of the Earth
 *                  const int n_sph: degree of potential
 *                  float* dev_coeff: coefficient matrix
 *                  vertex* dev_vertices: vertices on the sphere
 *                  float* U: potential
 *                  float* M: M stores the indice for Vnm
 *                  float* N: N stores the indice for Vnm
 *
 * Return Values:   null
*******************************************************************************/
__global__
void optimal_kernel_gravitational1(int g_vertices_length, float radius, float eq_R, const int n_sph, float* dev_coeff, vertex* dev_vertices, float* U, int* M, int* N){

    /*
    Every block is aasigned to every vertex.
    Threads/block = 256;
    Every thread contains calculates one of the 231 terms used in the summation of the potential
    shmem shares stores the 231 terms of the potential calculation
    U[thread_index]  = summation of the 231 terms
    U[thread_index]  = calculated from the reduction method #3
    This method should give improved computational time when number of vertices on the sphere is small, even
    though it wouldnt be a huge time savior.
    In case of small number of vertices, GPU overload happens because of 256 threads assigned to each vertex.
    No shared memory for V[21*21], W[21*21].
    Max shared memory: 256*4bytes = 1KB
    Drawback: Not very effective for higher number of vertices
              Repeative computation of V, W in each thread which is actually common to all threads in a block.
              Unnecessary global memory access for V,W to avoid thread divergence
    */

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
       }
        // Sync threads after every stage of reduction
        __syncthreads();
    }

    U[potential_index] = shmem[0]*mhu/R_eq;

}



/*******************************************************************************
 * Function:        optimal_kernel_gravitationa2
 *
 * Description:     Optimal implementation: Attempt 2
 *
 * Arguments:       int g_vertices_length: Total number of vertices on the sphere
 *                  float radius: radius of the sphere
 *                  float eq_R: Radius of the Earth
 *                  const int n_sph: degree of potential
 *                  float* dev_coeff: coefficient matrix
 *                  vertex* dev_vertices: vertices on the sphere
 *                  float* U: potential
 *                  float* M: M stores the indice for Vnm
 *                  float* N: N stores the indice for Vnm
 *
 * Return Values:   null
*******************************************************************************/
__global__
void optimal_kernel_gravitational2(int g_vertices_length, float radius, float eq_R, const int n_sph, float* dev_coeff, vertex* dev_vertices, float* U, int* M, int* N){

    /*
    Every block is assigned to every vertex.
    Threads/block = 256;
    Every thread contains calculates one of the 231 terms used in the summation of the potential
    shmem shares stores the 231 terms of the potential calculation
    U[thread_index]  = summation of the 231 terms
    U[thread_index]  = calculated from the reduction method #3
    This method should give improved computational time when number of vertices on the sphere is small, even
    though it wouldnt be a huge time savior.
    In case of small number of vertices, GPU overload happens because of 256 threads assigned to each vertex.
    Shared memory for V[21*21], W[21*21].
    Max shared memory: (256 + 462 + 462)*4bytes ~ 4.7KB
    Shared memory for V,W. Shared across threads in a block.
    Advantages: Reduces the computational time immensely for less number of vertices.
    */

    int potential_index = blockIdx.x;

    __shared__ float dev_V[21*21];
    __shared__ float dev_W[21*21];

    // Define pseudo coefficients
    float Radius_sq = powf(radius,2);
    float rho = powf(eq_R,2)/Radius_sq;

    float x0 = eq_R*dev_vertices[potential_index].x/Radius_sq;
    float y0 = eq_R*dev_vertices[potential_index].y/Radius_sq;
    float z0 = eq_R*dev_vertices[potential_index].z/Radius_sq;

    // Calculate zonal terms V(n, 0). Set W(n,0)=0.0
    dev_V[0]= eq_R/radius;
    dev_W[0] = 0.0;

    __syncthreads();

    dev_V[1*(n_sph+1) + 0] = z0 *dev_V[0];
    dev_W[1*(n_sph+1) + 0] = 0.0;

    __syncthreads();

    for (int n=2; n<n_sph+1; n++){
        dev_V[n*(n_sph+1) + 0] = ((2*n-1)*z0*dev_V[(n-1)*(n_sph+1) + 0] - (n-1)*rho*dev_V[(n-2)*(n_sph+1) + 0])/n;
        dev_W[n*(n_sph+1) + 0] = 0.0;
        __syncthreads();

    } // Eqn 3.30

    //Calculate tesseral and sectoral terms
    for (int m = 1; m < n_sph + 1; m++){
        // Eqn 3.29
        dev_V[m*(n_sph+1) + m] = (2*m-1)*(x0*dev_V[(m-1)*(n_sph+1) + (m-1)] - y0*dev_W[(m-1)*(n_sph+1) + (m-1)]);
        __syncthreads();

        dev_W[m*(n_sph+1) + m] = (2*m-1)*(x0*dev_W[(m-1)*(n_sph+1) + (m-1)] + y0*dev_V[(m-1)*(n_sph+1) + (m-1)]);
        __syncthreads();

        // n=m+1 (only one term)
        if (m < n_sph){
            dev_V[(m+1)*(n_sph+1) + (m)] = (2*m+1)*z0*dev_V[m*(n_sph+1) + m];
            dev_W[(m+1)*(n_sph+1) + (m)] = (2*m+1)*z0*dev_W[m*(n_sph+1) + m] ;
            __syncthreads();

        }

        for (int n = m+2; n<n_sph+1; n++){
            dev_V[n*(n_sph+1) + m] = ((2*n-1)*z0*dev_V[(n-1)*(n_sph+1) + m]-(n+m-1)*rho*dev_V[(n-2)*(n_sph+1) + m])/(n-m);
            dev_W[n*(n_sph+1) + m] = ((2*n-1)*z0*dev_W[(n-1)*(n_sph+1) + m]-(n+m-1)*rho*dev_W[(n-2)*(n_sph+1) + m])/(n-m);
            __syncthreads();

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



/*******************************************************************************
 * Function:        optimal_kernel_gravitational
 *
 * Description:     Optimal implementation: Attempt 3
 *
 * Arguments:       int g_vertices_length: Total number of vertices on the sphere
 *                  float radius: radius of the sphere
 *                  float eq_R: Radius of the Earth
 *                  const int n_sph: degree of potential
 *                  float* dev_coeff: coefficient matrix
 *                  vertex* dev_vertices: vertices on the sphere
 *                  float* U: potential
 *
 * Return Values:   null
*******************************************************************************/
__global__
void optimal_kernel_gravitational3(int g_vertices_length, float radius, float eq_R, const int n_sph, float* dev_coeff, vertex* dev_vertices, float* U){

    /*
    Every block is assigned to 16 consecutive vertices.
    Threads/block = 32;
    First 16 threads conpute V for the 16 vertices.
    Next 16 threads in the block compute W for the 16 vertices
    This method should give improved computational time when fairly large number
    of vertices on the sphere though it wouldnt be a huge time savior.
    Shared memory for VW[21*22*16]
    Max shared memory: (21*22*16)*4bytes ~ 28.875KB < 48KB
    Shared memory for V,W. Shared across threads in a block.
    Advantages: Reduces the computational time immensely for fairly large number of vertices.
    Drawback: Thread warping because first 16 threads compute different from next 16 threads in the block

    WHY 32 threads only::: Limited by the Shared memory
    */

    int thread_index = (blockIdx.x * blockDim.x + threadIdx.x);
    int tid = threadIdx.x;

    int vertex_index = thread_index-16*blockIdx.x;
    if(tid>=16)
        vertex_index = vertex_index-16;


    __shared__ float dev_VW[21*22*16];

    // Define pseudo coefficients
    float Radius_sq = powf(radius,2);
    float rho = powf(eq_R,2)/Radius_sq;

    float x0 = eq_R*dev_vertices[vertex_index].x/Radius_sq;
    float y0 = eq_R*dev_vertices[vertex_index].y/Radius_sq;
    float z0 = eq_R*dev_vertices[vertex_index].z/Radius_sq;

    // Calculate zonal terms V(n, 0). Set W(n,0)=0.0
    int n_coeffi = (n_sph+1)*(n_sph+2);
    if (tid<16)
        dev_VW[tid*n_coeffi + 0*(n_sph+2)+0] = eq_R/radius;
    else
        dev_VW[(tid-16)*n_coeffi + (n_sph-0)*(n_sph+2)+ (n_sph+1-0)] = 0.0;

    __syncthreads();

    if (tid<16)
        dev_VW[tid*n_coeffi + 1*(n_sph+2)+0] = z0 *eq_R/radius;
    else
        dev_VW[(tid-16)*n_coeffi + (n_sph-1)*(n_sph+2)+ (n_sph+1-0)] = 0.0;

    __syncthreads();

    if (tid<16){
        for (int n=2; n<n_sph+1; n++){
            dev_VW[tid*n_coeffi + n*(n_sph+2)+0] = ((2*n-1)*z0*dev_VW[tid*n_coeffi + (n-1)*(n_sph+2)+0] - (n-1)*rho*dev_VW[tid*n_coeffi + (n-2)*(n_sph+2)+0])/n;
//            __syncthreads();
        } // Eqn 3.30
    }
    else{
        for (int n=2; n<n_sph+1; n++){
            dev_VW[(tid-16)*n_coeffi + (n_sph-n)*(n_sph+2)+ (n_sph+1-0)] = 0.0;
        } // Eqn 3.30

    }

    //Calculate tesseral and sectoral terms
    for (int m = 1; m < n_sph + 1; m++){
        // Eqn 3.29
        if(tid<16){
            dev_VW[tid*n_coeffi + m*(n_sph+2)+m] = (2*m-1)*(x0*dev_VW[tid*n_coeffi + (m-1)*(n_sph+2)+m-1]- y0*dev_VW[(tid)*n_coeffi + (n_sph-(m-1))*(n_sph+2)+ (n_sph+1-(m-1))]);
//            __syncthreads();
        }
        else{
            dev_VW[(tid-16)*n_coeffi + (n_sph-(m))*(n_sph+2)+ (n_sph+1-(m))] = (2*m-1)*(x0*dev_VW[(tid-16)*n_coeffi + (n_sph-(m-1))*(n_sph+2)+ (n_sph+1-(m-1))] + y0*dev_VW[(tid-16)*n_coeffi + (m-1)*(n_sph+2)+m-1]);
//            __syncthreads();
        }
    }

    if(tid<16){
        for (int m = 1; m < n_sph + 1; m++){
            // n=m+1 (only one term)
            if (m < n_sph){
                dev_VW[tid*n_coeffi + (m+1)*(n_sph+2)+m] = (2*m+1)*z0*dev_VW[tid*n_coeffi + m*(n_sph+2)+m];

//                __syncthreads();

            }

            for (int n = m+2; n<n_sph+1; n++){
                dev_VW[tid*n_coeffi + n*(n_sph+2)+m] = ((2*n-1)*z0*dev_VW[tid*n_coeffi + (n-1)*(n_sph+2)+m]-(n+m-1)*rho*dev_VW[tid*n_coeffi + (n-2)*(n_sph+2)+m])/(n-m);
//                __syncthreads();
            }
        }
    }
    else{
        for (int m = 1; m < n_sph + 1; m++){
            // n=m+1 (only one term)
            if (m < n_sph){
                dev_VW[(tid-16)*n_coeffi + (n_sph-(m+1))*(n_sph+2)+ (n_sph+1-(m))] = (2*m+1)*z0*dev_VW[(tid-16)*n_coeffi + (n_sph-(m))*(n_sph+2)+ (n_sph+1-(m))];
//                __syncthreads();

            }

            for (int n = m+2; n<n_sph+1; n++){
                dev_VW[(tid-16)*n_coeffi + (n_sph-(n))*(n_sph+2)+ (n_sph+1-(m))] = ((2*n-1)*z0*dev_VW[(tid-16)*n_coeffi + (n_sph-(n-1))*(n_sph+2)+ (n_sph+1-(m))]-(n+m-1)*rho*dev_VW[(tid-16)*n_coeffi + (n_sph-(n-2))*(n_sph+2)+ (n_sph+1-(m))])/(n-m);
//                __syncthreads();
            }
        }
    }
    __syncthreads();

    __shared__ float shmem[2*16]; //stores CV+SW
    shmem[tid] = 0.0; //potential

    // Calculate potential
    float C = 0; // Cnm coeff
    float S = 0; // Snm coeff
    float N = 0; // normalisation number
    float p = 1.0;
//    U[vertex_index] = 0.0; //potential
    for (int m=0; m<n_sph+1; m++){
        for (int n = m; n<n_sph+1; n++){
//            C = 0;
            S = 0;
            if (m==0){
                N = sqrtf(2*n+1);
                C = N*dev_coeff[n*(n_sph+2)+0];
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
            if(tid<16){
                shmem[tid] = shmem[tid] + C*dev_VW[tid*n_coeffi + n*(n_sph+2)+m];
                __syncthreads();
            }
            else{
                shmem[tid] = shmem[tid] + S*dev_VW[(tid-16)*n_coeffi + (n_sph-(n))*(n_sph+2)+ (n_sph+1-(m))];
            // Calculation of the Gravitational Potential Calculation model
            __syncthreads();
            }
        }
    }

    if(tid<16){
        U[vertex_index] = shmem[tid] + shmem[tid+16];
        __syncthreads();
        U[vertex_index] = U[vertex_index]*mhu/R_eq;
    }

}


/*******************************************************************************
 * Function:        optimal_kernel_gravitational
 *
 * Description:     Optimal implementation: Attempt 4
 *
 * Arguments:       int g_vertices_length: Total number of vertices on the sphere
 *                  float radius: radius of the sphere
 *                  float eq_R: Radius of the Earth
 *                  const int n_sph: degree of potential
 *                  float* dev_coeff: coefficient matrix
 *                  vertex* dev_vertices: vertices on the sphere
 *                  float* U: potential
 *
 * Return Values:   null
*******************************************************************************/
// __global__
// void optimal_kernel_gravitational4(int g_vertices_length, float radius, float eq_R, const int n_sph, float* dev_coeff, vertex* dev_vertices, float* U){

    /*
    Every block is assigned to 32 consecutive vertices.
    Threads/block = 64;
    First 16 threads conpute V for the 16 vertices.
    Next 16 threads in the block compute W for the 16 vertices
    This method should give improved computational time when fairly large number
    of vertices on the sphere though it wouldnt be a huge time savior.
    Shared memory for VW[21*22*32]
    Max shared memory: (21*22*32)*4bytes ~ 57.75KB < 64KB
    NOTE:::::::::: This would work if the shared memory can be expanded to 64 KB.
    Shared memory for V,W. Shared across threads in a block.
    Advantages: Reduces the computational time immensely for fairly large number of vertices.
                No Thread warping because 32 threads do the same action
                Couldnt verify due to limited shared memory :(
    */

//     int thread_index = (blockIdx.x * blockDim.x + threadIdx.x);
//     int tid = threadIdx.x;
//     int blk_hlf_dim = blockDim.x/2;


//     int vertex_index = thread_index-blk_hlf_dim*blockIdx.x;
//     if(tid>=blk_hlf_dim)
//         vertex_index = vertex_index-blk_hlf_dim;


//     __shared__ float dev_VW[21*22*32];

//     // Define pseudo coefficients
//     float Radius_sq = powf(radius,2);
//     float rho = powf(eq_R,2)/Radius_sq;

//     float x0 = eq_R*dev_vertices[vertex_index].x/Radius_sq;
//     float y0 = eq_R*dev_vertices[vertex_index].y/Radius_sq;
//     float z0 = eq_R*dev_vertices[vertex_index].z/Radius_sq;

//     // Calculate zonal terms V(n, 0). Set W(n,0)=0.0

//     if (tid<blk_hlf_dim)
//         dev_VW[tid*(n_sph+1)*(n_sph+2) + 0*(n_sph+2)+0] = eq_R/radius;
//     else
//         dev_VW[(tid-blk_hlf_dim)*(n_sph+1)*(n_sph+2) + (n_sph-0)*(n_sph+2)+ (n_sph+1-0)] = 0.0;

//     __syncthreads();

//     if (tid<blk_hlf_dim)
//         dev_VW[tid*(n_sph+1)*(n_sph+2) + 1*(n_sph+2)+0] = z0 *eq_R/radius;
//     else
//         dev_VW[(tid-blk_hlf_dim)*(n_sph+1)*(n_sph+2) + (n_sph-1)*(n_sph+2)+ (n_sph+1-0)] = 0.0;

//     __syncthreads();

//     if (tid<blk_hlf_dim){
//         for (int n=2; n<n_sph+1; n++){
//             dev_VW[tid*(n_sph+1)*(n_sph+2) + n*(n_sph+2)+0] = ((2*n-1)*z0*dev_VW[tid*(n_sph+1)*(n_sph+2) + (n-1)*(n_sph+2)+0] - (n-1)*rho*dev_VW[tid*(n_sph+1)*(n_sph+2) + (n-2)*(n_sph+2)+0])/n;
// //            __syncthreads();
//         } // Eqn 3.30
//     }
//     else{
//         for (int n=2; n<n_sph+1; n++){
//             dev_VW[(tid-blk_hlf_dim)*(n_sph+1)*(n_sph+2) + (n_sph-n)*(n_sph+2)+ (n_sph+1-0)] = 0.0;
//         } // Eqn 3.30

//     }

//     //Calculate tesseral and sectoral terms
//     for (int m = 1; m < n_sph + 1; m++){
//         // Eqn 3.29
//         if(tid<blk_hlf_dim){
//             dev_VW[tid*(n_sph+1)*(n_sph+2) + m*(n_sph+2)+m] = (2*m-1)*(x0*dev_VW[tid*(n_sph+1)*(n_sph+2) + (m-1)*(n_sph+2)+m-1]- y0*dev_VW[(tid)*(n_sph+1)*(n_sph+2) + (n_sph-(m-1))*(n_sph+2)+ (n_sph+1-(m-1))]);
// //            __syncthreads();
//         }
//         else{
//             dev_VW[(tid-blk_hlf_dim)*(n_sph+1)*(n_sph+2) + (n_sph-(m))*(n_sph+2)+ (n_sph+1-(m))] = (2*m-1)*(x0*dev_VW[(tid-blk_hlf_dim)*(n_sph+1)*(n_sph+2) + (n_sph-(m-1))*(n_sph+2)+ (n_sph+1-(m-1))] + y0*dev_VW[(tid-blk_hlf_dim)*(n_sph+1)*(n_sph+2) + (m-1)*(n_sph+2)+m-1]);
// //            __syncthreads();
//         }
//     }

//     if(tid<blk_hlf_dim){
//         for (int m = 1; m < n_sph + 1; m++){
//             // n=m+1 (only one term)
//             if (m < n_sph){
//                 dev_VW[tid*(n_sph+1)*(n_sph+2) + (m+1)*(n_sph+2)+m] = (2*m+1)*z0*dev_VW[tid*(n_sph+1)*(n_sph+2) + m*(n_sph+2)+m];

// //                __syncthreads();

//             }

//             for (int n = m+2; n<n_sph+1; n++){
//                 dev_VW[tid*(n_sph+1)*(n_sph+2) + n*(n_sph+2)+m] = ((2*n-1)*z0*dev_VW[tid*(n_sph+1)*(n_sph+2) + (n-1)*(n_sph+2)+m]-(n+m-1)*rho*dev_VW[tid*(n_sph+1)*(n_sph+2) + (n-2)*(n_sph+2)+m])/(n-m);
// //                __syncthreads();
//             }
//         }
//     }
//     else{
//         for (int m = 1; m < n_sph + 1; m++){
//             // n=m+1 (only one term)
//             if (m < n_sph){
//                 dev_VW[(tid-blk_hlf_dim)*(n_sph+1)*(n_sph+2) + (n_sph-(m+1))*(n_sph+2)+ (n_sph+1-(m))] = (2*m+1)*z0*dev_VW[(tid-blk_hlf_dim)*(n_sph+1)*(n_sph+2) + (n_sph-(m))*(n_sph+2)+ (n_sph+1-(m))];
// //                __syncthreads();

//             }

//             for (int n = m+2; n<n_sph+1; n++){
//                 dev_VW[(tid-blk_hlf_dim)*(n_sph+1)*(n_sph+2) + (n_sph-(n))*(n_sph+2)+ (n_sph+1-(m))] = ((2*n-1)*z0*dev_VW[(tid-blk_hlf_dim)*(n_sph+1)*(n_sph+2) + (n_sph-(n-1))*(n_sph+2)+ (n_sph+1-(m))]-(n+m-1)*rho*dev_VW[(tid-blk_hlf_dim)*(n_sph+1)*(n_sph+2) + (n_sph-(n-2))*(n_sph+2)+ (n_sph+1-(m))])/(n-m);
// //                __syncthreads();
//             }
//         }
//     }
//     __syncthreads();

//     __shared__ float shmem[2*32]; //stores CV+SW
//     shmem[tid] = 0.0; //potential

//     // Calculate potential
//     float C = 0; // Cnm coeff
//     float S = 0; // Snm coeff
//     float N = 0; // normalisation number
//     float p = 1.0;
// //    U[vertex_index] = 0.0; //potential
//     for (int m=0; m<n_sph+1; m++){
//         for (int n = m; n<n_sph+1; n++){
// //            C = 0;
//             S = 0;
//             if (m==0){
//                 N = sqrtf(2*n+1);
//                 C = N*dev_coeff[n*(n_sph+2)+0];
//             }
//             else {
//                 p = 1.0;
//                 // gpu_facprod(n,m,&p);
//                 for (int i = n-m+1; i<=n+m; i++){
//                     p = p/i;
//                 }
//                 N = sqrtf((2)*(2*n+1)*p);
//                 C = N*dev_coeff[n*(n_sph+2)+m];
//                 S = N*dev_coeff[(n_sph-n)*(n_sph+2)+ (n_sph-m+1)];
//             }
//             if(tid<blk_hlf_dim){
//                 shmem[tid] = shmem[tid] + C*dev_VW[tid*(n_sph+1)*(n_sph+2) + n*(n_sph+2)+m];
//                 __syncthreads();
//             }
//             else{
//                 shmem[tid] = shmem[tid] + S*dev_VW[(tid-blk_hlf_dim)*(n_sph+1)*(n_sph+2) + (n_sph-(n))*(n_sph+2)+ (n_sph+1-(m))];
//             // Calculation of the Gravitational Potential Calculation model
//             __syncthreads();
//             }
//         }
//     }

//     if(tid<blk_hlf_dim){
//         U[vertex_index] = shmem[tid] + shmem[tid+blk_hlf_dim];
//         __syncthreads();
//         U[vertex_index] = U[vertex_index]*mhu/R_eq;
//     }

// }



/*******************************************************************************
 * Function:        naive_cudacall_gravitational
 *
 * Description:     CPU call for Naive kernel implementation
 *
 * Arguments:       int thread_index: stores the vertex index actually
 *
 * Return Values:   null
*******************************************************************************/
void naive_cudacall_gravitational(int thread_num){

    int len = vertices_length;
    int n_blocks = std::min(65535, (len + thread_num  - 1) / thread_num);
    cout<<"\n Number of blocks \t"<<n_blocks<<'\n';
    naive_kernel_gravitational<<<n_blocks, thread_num>>>(vertices_length, radius, N_SPHERICAL, dev_coeff, dev_vertices, dev_potential);
}

/*******************************************************************************
 * Function:        optimal_cudacall_gravitational1
 *
 * Description:     CPU call for Optimal1 kernel implementation
 *
 * Arguments:       int thread_index: stores the vertex index actually
 *
 * Return Values:   null
*******************************************************************************/
void optimal_cudacall_gravitational1(int thread_num){
    int len = vertices_length;
    int n_blocks = std::min(65535, len);

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
    optimal_kernel_gravitational1<<<n_blocks, thread_num>>>(vertices_length, radius, R_eq, N_SPHERICAL, dev_coeff, dev_vertices, dev_potential, dev_M, dev_N);
    CUDA_CALL(cudaFree(dev_M));
    CUDA_CALL(cudaFree(dev_N));
}


/*******************************************************************************
 * Function:        optimal_cudacall_gravitational2
 *
 * Description:     CPU call for Optimal2 kernel implementation
 *
 * Arguments:       int thread_index: stores the vertex index actually
 *
 * Return Values:   null
*******************************************************************************/
void optimal_cudacall_gravitational2(int thread_num){

    int len = vertices_length;
    int n_blocks = std::min(65535, len);

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


/*******************************************************************************
 * Function:        optimal_cudacall_gravitational3
 *
 * Description:     CPU call for Optimal3 kernel implementation
 *
 * Arguments:       null
 *
 * Return Values:   null
*******************************************************************************/
void optimal_cudacall_gravitational3(){

    // Number of threads/ block = 32;
    // Number of vertices/block = 16;
    // Compute V, W in shared memory and separately
    int len = vertices_length;
    int n_blocks = ceil(len*1.0/16);
    n_blocks = std::min(65535,  n_blocks);
    cout<<"\n Number of blocks \t"<<n_blocks<<'\n';
    optimal_kernel_gravitational3<<<n_blocks, 32>>>(vertices_length, radius, R_eq, N_SPHERICAL, dev_coeff, dev_vertices, dev_potential);

}


/*******************************************************************************
 * Function:        optimal_cudacall_gravitational4
 *
 * Description:     CPU call for Optimal4 kernel implementation
 *
 * Arguments:       null
 *
 * Return Values:   null
*******************************************************************************/
void optimal_cudacall_gravitational4(){

    // Number of threads/ block = 64;
    // Number of vertices/block = 32;
    // Compute V, W in shared memory and separately

    int len = vertices_length;
    int n_blocks = ceil(len*1.0/32);
    n_blocks = std::min(65535,  n_blocks);
    cout<<"\n Number of blocks \t"<<n_blocks<<'\n';
    // cudaFuncSetCacheConfig(optimal_kernel_gravitational4, cudaFuncCachePreferShared);
    // optimal_kernel_gravitational4<<<n_blocks, 64>>>(vertices_length, radius, R_eq, N_SPHERICAL, dev_coeff, dev_vertices, dev_potential);
}

