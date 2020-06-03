/*
 * CUDA blur
 */
#ifndef _GRAV_CUDA_C_
    #define _GRAV_CUDA_C_
#endif

#include <math.h>

#include "grav_cuda.cuh"


/* Local variables */

int ind2_faces;             // denote the index of the pointer which points to the most updated faces array
triangle * pointers[2];
triangle * dev_faces;
triangle * dev_faces_cpy;

int * pointers_inds[2];
int ind2_inds;
int * dev_face_vert_ind;
int * dev_face_vert_ind_cpy;
int * dev_face_vert_ind_cpy2;


float * pointers_sums[2];
int ind2_sums;
float * dev_face_sums;
float * dev_face_sums_cpy;

__global__ void kernal_update_faces(vertex * f_in, vertex * f_out, int * inds, const unsigned int vertices_length);
__global__ void kernal_fill_sums_inds(vertex * vs, float * sums, int * inds, const unsigned int vertices_length);

void cuda_remove_duplicates(int thread_num);

void cuda_cpy_input_data(){
    gpu_out_faces = (triangle *)malloc(faces_length*sizeof(triangle));
    CUDA_CALL(cudaMalloc((void **)&dev_faces, faces_length * sizeof(triangle)));
    CUDA_CALL(cudaMalloc((void **)&dev_faces_cpy, faces_length * sizeof(triangle)));

    CUDA_CALL(cudaMalloc((void **)&dev_face_vert_ind, 3*faces_length * sizeof(int)));
    CUDA_CALL(cudaMalloc((void **)&dev_face_vert_ind_cpy, 3*faces_length * sizeof(int)));
    CUDA_CALL(cudaMalloc((void **)&dev_face_vert_ind_cpy2, 3*faces_length * sizeof(int)));

    CUDA_CALL(cudaMalloc((void **)&dev_face_sums, 3*faces_length * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**) &dev_face_sums_cpy, 3*faces_length* sizeof(float)));

    CUDA_CALL(cudaMalloc((void**) &dev_vertices_ico, vertices_length * sizeof(vertex)));

    CUDA_CALL(cudaMemcpy(dev_faces, faces_init, ICOSPHERE_INIT_FACE_LEN*sizeof(triangle), cudaMemcpyHostToDevice));

    ind2_faces = 0;
    pointers[0] = dev_faces;
    pointers[1] = dev_faces_cpy;

    ind2_sums = 0;
    pointers_sums[0] = dev_face_sums;
    pointers_sums[1] = dev_face_sums_cpy;

    ind2_inds = 0;
    pointers_inds[0] = dev_face_vert_ind;
    pointers_inds[1] = dev_face_vert_ind_cpy;

    gpu_out_vertices = (vertex *) malloc(vertices_length*sizeof(vertex));
}

void cuda_cpy_output_data(){
    CUDA_CALL(cudaMemcpy(gpu_out_faces, pointers[ind2_faces], faces_length*sizeof(triangle), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(gpu_out_vertices, dev_vertices_ico, vertices_length*sizeof(vertex), cudaMemcpyDeviceToHost));
}

void free_gpu_memory(){
    CUDA_CALL(cudaFree(dev_faces));
    CUDA_CALL(cudaFree(dev_faces_cpy));

    CUDA_CALL(cudaFree(dev_face_vert_ind));
    CUDA_CALL(cudaFree(dev_face_vert_ind_cpy));
    CUDA_CALL(cudaFree(dev_face_vert_ind_cpy2));

    CUDA_CALL(cudaFree(dev_face_sums));
    CUDA_CALL(cudaFree(dev_face_sums_cpy));

    CUDA_CALL(cudaFree(dev_vertices_ico));

    free(gpu_out_faces);
    free(gpu_out_vertices);
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
    int n_blocks, ths;
    for(int i=0; i<max_depth; i++){
        ths = 20*pow(4, i);
        n_blocks = std::min(65535, (ths + thread_num  - 1) / thread_num);
        refine_icosphere_naive_kernal<<<n_blocks, thread_num>>>(dev_faces, radius, i);
    }
    int len = 3*faces_length;
    n_blocks = std::min(65535, (len + thread_num  - 1) / thread_num);
    kernal_fill_sums_inds<<<n_blocks, thread_num>>>((vertex *)pointers[ind2_faces], dev_face_sums, dev_face_vert_ind, len);
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

__device__ func_ptr_sub_triangle_t funcs_list[4] = {sub_triangle_top, sub_triangle_left, sub_triangle_right, sub_triangle_center};


__global__ void refine_icosphere_kernal(triangle * faces, float * sums, const float radius, const unsigned int th_len, triangle * faces_out) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numthrds = blockDim.x * gridDim.x;

    vertex v_tmp[3];
    triangle v;

    while(idx < 4*th_len){
        int tri_ind = idx/4;
        int sub_tri_ind = idx%4;
        v = faces[tri_ind];
        break_triangle(v, v_tmp, radius);

        funcs_list[sub_tri_ind](v, v_tmp, &faces_out[idx]);

        idx += numthrds;
    }

}

void cudacall_icosphere(int thread_num) {
    // each thread creates a sub triangle
    int ths, n_blocks, ind1;
    for(int i=0; i<max_depth; i++){
        ths = 20*pow(4, i);
        n_blocks = std::min(65535, (ths + 4*thread_num  - 1) / thread_num);
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
        k = idx % r;
        local_k = k%stride;
        arr_ind_L = idx - local_k + stride;    // arr2 
        arr_ind_HE = idx - local_k - stride;   // arr1 

        if(k < stride && arr_ind_L < length){
            // an arr 1 element
            arr_len = min(stride, length - arr_ind_L);
            arr_start = idx - local_k + 1;

            get_last_smallest(&sums[arr_ind_L], arr_len, sums[idx], tmp_res);

            final_index = local_k + tmp_res[0] + arr_start;

        }else if( k>=stride && 0 <= arr_ind_HE){
            // an arr 2 element
            arr_len = min(stride, length - arr_ind_HE);
            arr_start = idx - local_k - stride;

            get_first_greatest(&sums[arr_ind_HE], arr_len, sums[idx], tmp_res);
            
            final_index = local_k + tmp_res[0] + arr_start;
        }
        
        // now place the element
        res[final_index] = sums[idx];
        ind_res[final_index] = ind[idx];
        
        idx += numthrds;
    }

}

__global__
void kernal_mark_duplicates(vertex * v, float * sums, int * ind, int * ind_res, int length){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numthrds = blockDim.x * gridDim.x;
    
    vertex v1, v2; int flag;
    int start, end; float tmp;
    while(idx < length){
        // find the end location
        end = idx + 1;
        while(abs(sums[end] - sums[idx]) < EPSILON && end < length){
            end++;
        }
        v1 = v[idx];
        flag = 1;
        start = idx+1;
        while(start < end){
            // check if vertices are same or not
            v2 = v[start];
            tmp = abs(v1.x - v2.x) + abs(v1.y - v2.y) + abs(v1.z - v2.z); 
            if(tmp < 3*EPSILON){
                flag = 0;
                ind_res[idx] = -1;
                break;
            }
            start++;
        }
        if(flag==1){
            ind_res[idx] = idx;
        }
        idx += numthrds;
    }
}

__global__
void kernal_count_shifts(int * inds, int * inds_res, int length){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numthrds = blockDim.x * gridDim.x;
    
    int i, j;
    while(idx < length){
        // find the end location
        
        if(inds[idx] == -1){
            inds_res[idx] = 0; 
        }
        else{
            i = idx - 1;
            j=0;
            while(i >= 0){
                i--;
                j++;
                if(inds[i] != -1){
                    break;
                }
            }
            inds_res[idx] = j;
        }
        
        idx += numthrds;
    }
}

__global__
void kernal_prefix_sum(int * inds, int * inds_res, int length, const unsigned int stride){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numthrds = blockDim.x * gridDim.x;
    
    int i;
    while(idx < length){
        // find the end location
        i = idx-stride;
        if(i >= 0){
            inds_res[idx] = inds[idx] + inds[i];
        }else{
            inds_res[idx] = inds[idx];
        }
        
        idx += numthrds;
    }
}

__global__ 
void kernal_fill_vertices(vertex * v_in, vertex * v_out, int * inds, int * shifts, int length, float radius){
    
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numthrds = blockDim.x * gridDim.x;
    
    int index;vertex tmp; float scale;
    while(idx < length){
        if(inds[idx] != -1){
            index = idx - shifts[idx];
            tmp = v_in[idx];
            scale = radius/sqrtf(tmp.x*tmp.x + tmp.y*tmp.y + tmp.z*tmp.z);
            tmp.x *= scale;
            tmp.y *= scale;
            tmp.z *= scale;
            v_out[index] = tmp;   
        }
        idx += numthrds;
    }
}

void cudacall_naive_fill_vertices(int thread_num){

    unsigned int len = 3*faces_length;
    int n_blocks = min(65535, (len + thread_num  - 1) / thread_num);

    unsigned int l = ceil(log2(len)), ind1;
    for(int i=0; i<l; i++){
        ind1 = i%2;
        ind2_sums = (i+1)%2;
        ind2_inds = ind2_sums;
        unsigned int r = pow(2, i+1);
        kernal_merge_navie_sort<<<n_blocks, thread_num>>>(pointers_sums[ind1], pointers_sums[ind2_sums], pointers_inds[ind1], pointers_inds[ind2_inds], len, r);

    }
    cuda_remove_duplicates(thread_num);
}

void cudacall_fill_vertices(int thread_num) {
    
    unsigned int len = 3*faces_length;
    int n_blocks = min(65535, (len + thread_num  - 1) / thread_num);

    unsigned int l = ceil(log2(len)), ind1;
    for(int i=0; i<l; i++){
        ind1 = i%2;
        ind2_sums = (i+1)%2;
        ind2_inds = ind2_sums;
        unsigned int r = pow(2, i+1);
        kernal_merge_sort<<<n_blocks, thread_num>>>(pointers_sums[ind1], pointers_sums[ind2_sums], pointers_inds[ind1], pointers_inds[ind2_inds], len, r);

    }

    // now sort the chunks of 1024 floats
    l = ceil(log2(n_blocks));
    for(int i=0; i<l; i++){
        ind1 = (ind1+1)%2;
        ind2_sums = (ind2_sums+1)%2;
        ind2_inds = ind2_sums;
        unsigned int r = pow(2, i+1)*1024;
        kernal_merge_chuncks<<<n_blocks, thread_num>>>(pointers_sums[ind1], pointers_sums[ind2_sums], pointers_inds[ind1], pointers_inds[ind2_inds], len, r);
    }

    cuda_remove_duplicates(thread_num);
}

void cuda_remove_duplicates(int thread_num){

    unsigned int len = 3*faces_length;
    int n_blocks = min(65535, (len + thread_num  - 1) / thread_num);

    // update the vertices position
    n_blocks = std::min(65535, ((int)len + thread_num  - 1) / thread_num);
    int out = (ind2_faces + 1) %2;
    kernal_update_faces<<<n_blocks, thread_num>>>((vertex *)pointers[ind2_faces], (vertex *)pointers[out], pointers_inds[ind2_inds], len);
    ind2_faces = out;

    // mark the duplicate vertices
    out  = (ind2_inds + 1)%2;
    kernal_mark_duplicates<<<n_blocks, thread_num>>>
                            ((vertex *)pointers[ind2_faces], pointers_sums[ind2_sums], pointers_inds[ind2_inds], pointers_inds[out], len);
    ind2_inds = out;

    int * markers = pointers_inds[ind2_inds];

    // count the shift required.
    out  = (ind2_inds + 1)%2;
    kernal_count_shifts<<<n_blocks, thread_num>>>
                            (pointers_inds[ind2_inds], pointers_inds[out], len);
    pointers_inds[ind2_inds] = dev_face_vert_ind_cpy2;
    ind2_inds = out;

    // commutate the shifts required.
    int l = ceil(log2(len));
    // l = 1;
    int ind1 = ind2_inds-1;
    for(int i=0; i<l; i++){
        ind1 = (1+ind1)%2;
        ind2_inds = (ind2_inds+1)%2;
        unsigned int r = pow(2, i);
        kernal_prefix_sum<<<n_blocks, thread_num>>>(pointers_inds[ind1], pointers_inds[ind2_inds], len, r);
    }
    
    // fill the vertices now
    kernal_fill_vertices<<<n_blocks, thread_num>>>((vertex *) pointers[ind2_faces], dev_vertices_ico, markers, pointers_inds[ind2_inds], len, radius);
}

__global__
void kernal_update_faces(vertex * f_in, vertex * f_out, int * inds, const unsigned int vertices_length){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numthrds = blockDim.x * gridDim.x;

    while(idx < vertices_length){
        f_out[idx] = f_in[inds[idx]];
        inds[idx] = idx;
        idx += numthrds;
    }
}