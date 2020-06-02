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
#include <curand.h>
using namespace std;

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

__global__ void kernal_update_faces(vertex * f_in, vertex * f_out, int * inds, const unsigned int vertices_length);
__global__ void kernal_fill_sums_inds(vertex * vs, float * sums, int * inds, const unsigned int vertices_length);

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
}

void cuda_cpy_output_data(){
    CUDA_CALL(cudaMemcpy(gpu_out_faces, pointers[ind2_faces], faces_length*sizeof(triangle), cudaMemcpyDeviceToHost));
}

void free_gpu_memory(){
    CUDA_CALL(cudaFree(dev_faces_in));
    CUDA_CALL(cudaFree(dev_faces_out));

    CUDA_CALL(cudaFree(dev_face_vert_ind));
    CUDA_CALL(cudaFree(dev_face_vert_ind_res));

    CUDA_CALL(cudaFree(dev_face_sums));
    CUDA_CALL(cudaFree(dev_face_sums_res));
    free(gpu_out_faces);
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

__device__
void get_first_greatest(float * arr, int len, float a, int * res_fg)
{
    int first = 0, last = len - 1;
    while (first <= last)
    {
        int mid = (first + last) / 2;
        if (arr[mid] > a)
            last = mid - 1;
        else
            first = mid + 1;
    }
    res_fg[0] =  last + 1 == len ? len : last + 1;

}

__device__
void get_last_smallest(float * arr, int len, float a, int * res_ls)
{
    int first = 0, last = len - 1;
    while (first <= last)
    {
        int mid = (first + last) / 2;
        if (arr[mid] >= a)
            last = mid - 1;
        else
            first = mid + 1;
    }
    res_ls[0] = first - 1 < 0 ? -1 : first - 1;
}


__global__
void kernal_merge_chuncks(float * sums, float * res, int * ind, int * ind_res, const unsigned int length, const unsigned int r){
    
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numthrds = blockDim.x * gridDim.x;
    const int stride = r/2;
    
    int tmp_res[1];
    
    int k;
    int local_k;
    int arr_len;
    int arr_start;
    int arr_ind_L, arr_ind_HE, final_index;
    
    while(idx < length){
        final_index = idx;
        k = idx % r;
        local_k = k%stride;
        arr_ind_L = idx - local_k + stride;    // arr2 
        arr_ind_HE = idx - local_k - stride;   // arr1 

        if(k < stride && arr_ind_L < length){
            // an arr 1 element
            arr_len = min(stride, length - arr_ind_L);
            arr_start = idx - local_k;

            get_last_smallest(&sums[arr_ind_L], arr_len, sums[idx], tmp_res);

            final_index = local_k + tmp_res[0] + 1 + arr_start;

        }else if( k>=stride && 0 <= arr_ind_HE){
            // an arr 2 element
            arr_len = min(stride, length - arr_ind_HE);
            arr_start = idx - local_k - stride;

            get_first_greatest(&sums[arr_ind_HE], arr_len, sums[idx], tmp_res);
            
            final_index = local_k + tmp_res[0] + arr_start;
        }else{
            res[idx] = -1;
        }
        
        res[final_index] = sums[idx];
        // now place the element
        // res[final_index] = sums[idx];
        ind_res[final_index] = ind[idx];
        
        idx += numthrds;
    }

}

void cudacall_sort(int thread_num) {

    // generate the random numbers for sums

    // init the generator 
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    // set seed
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    unsigned int len = 3*faces_length;
    // float sums[len];
    
    // generate random numbers
    // curandGenerateUniform(gen, pointers_sums[0], len);


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
    cout << "l: "<< l << endl;
    for(int i=0; i<l; i++){
        ind1 = (ind1+1)%2;
        ind2_sums = (ind2_sums+1)%2;
        ind2_inds = ind2_sums;
        unsigned int r = pow(2, i+1)*1024;
        // kernal_merge_navie_sort<<<n_blocks, thread_num>>>(pointers_sums[ind1], pointers_sums[ind2_sums], pointers_inds[ind1], pointers_inds[ind2_inds], len, r);
        kernal_merge_chuncks<<<n_blocks, thread_num>>>(pointers_sums[ind1], pointers_sums[ind2_sums], pointers_inds[ind1], pointers_inds[ind2_inds], len, r);
    }

    // CUDA_CALL(cudaMemcpy(sums, pointers_sums[ind2_sums], len*sizeof(float), cudaMemcpyDeviceToHost));
    // CUDA_CALL(cudaMemcpy(tmp, pointers_inds[ind2_inds], len*sizeof(int), cudaMemcpyDeviceToHost));

    // export the sum:
    // cout << "Exporting: sums.csv"<<endl;

    // string filename1 = "sums.csv";
    // ofstream obj_stream;
    // obj_stream.open(filename1);
    // obj_stream << "sums" << endl;
    // cout <<"-----------------------" << endl;
    // for(unsigned int i=0; i< 3*faces_length; i++){
    //     obj_stream << sums[i] << endl;
    // }
    // obj_stream.close();

    // working
    n_blocks = std::min(65535, ((int)len + thread_num  - 1) / thread_num);
    int out = (ind2_faces + 1) %2;
    kernal_update_faces<<<n_blocks, thread_num>>>((vertex *)pointers[ind2_faces], (vertex *)pointers[out], pointers_inds[ind2_inds], len);
    cudaDeviceSynchronize();
    ind2_faces = out;

    // destroy random generator
    curandDestroyGenerator(gen);
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