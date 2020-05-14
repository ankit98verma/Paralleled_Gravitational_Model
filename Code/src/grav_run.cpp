/*
 * grav_run: Ankit Verma, Garima Aggarwal, 2020
 *
 * This file runs the CPU implementation and GPU implementation
 * of the gravitational field calculation.
 *
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <time.h>

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

#include "grav_cuda.cuh"
#include "grav_cpu.h"
#include "helper_cuda.h"

using std::cerr;
using std::cout;
using std::endl;


// (From Eric's code)
cudaEvent_t start;
cudaEvent_t stop;
#define START_TIMER() {                         \
      CUDA_CALL(cudaEventCreate(&start));       \
      CUDA_CALL(cudaEventCreate(&stop));        \
      CUDA_CALL(cudaEventRecord(start));        \
    }

#define STOP_RECORD_TIMER(name) {                           \
      CUDA_CALL(cudaEventRecord(stop));                     \
      CUDA_CALL(cudaEventSynchronize(stop));                \
      CUDA_CALL(cudaEventElapsedTime(&name, start, stop));  \
      CUDA_CALL(cudaEventDestroy(start));                   \
      CUDA_CALL(cudaEventDestroy(stop));                    \
    }

int check_args(int argc, char **argv){
	if (argc != 4){
        printf("Usage: ./grav [depth] [thread_per_block] [number_of_blocks]\n");
        return 1;
    }
    return 0;
}

float time_profile_cpu(){
	float cpu_time_ms = 0;
	float cpu_time_icosphere_ms = -1;
	float cpu_time_fill_vertices_ms = -1;
	float cpu_time_sort_ms = -1;
	float cpu_time_grav_pot_ms = -1;

	START_TIMER();
		create_icoshpere();
	STOP_RECORD_TIMER(cpu_time_icosphere_ms);
	
	START_TIMER();
		fill_vertices();
	STOP_RECORD_TIMER(cpu_time_fill_vertices_ms);
	
	START_TIMER();
		quickSort_points(0, vertices_length-1);
	STOP_RECORD_TIMER(cpu_time_sort_ms);
    
    START_TIMER();
    	fill_common_theta();
    	get_grav_pot();
    STOP_RECORD_TIMER(cpu_time_grav_pot_ms);
    cpu_time_ms = cpu_time_icosphere_ms + cpu_time_fill_vertices_ms + cpu_time_sort_ms + cpu_time_grav_pot_ms;
    printf("Icosphere generation time: %f ms\n", cpu_time_icosphere_ms);
    printf("Fill vertices time: %f ms\n", cpu_time_fill_vertices_ms);
	printf("Sorting time: %f ms\n", cpu_time_sort_ms);
	printf("Gravitational potential time: %f ms\n", cpu_time_grav_pot_ms);

	return cpu_time_ms;
}


// this function should be called only after calling time_profile_cpu
float time_profile_gpu(){
	float gpu_time_ms = 0;
	float gpu_time_icosphere = -1;
	float gpu_time_indata_cpy = -1;
	float gpu_time_outdata_cpy = -1;
	
	START_TIMER();
		cuda_cpy_input_data();
	STOP_RECORD_TIMER(gpu_time_indata_cpy);
	
	START_TIMER();
		cuda_call_kernel();
	STOP_RECORD_TIMER(gpu_time_icosphere);

	START_TIMER();
		cuda_cpy_output_data();
	STOP_RECORD_TIMER(gpu_time_outdata_cpy);
	
	printf("GPU Input data copy time: %f ms\n", gpu_time_indata_cpy);
    printf("GPU Icosphere generation time: %f ms\n", gpu_time_icosphere);
	printf("GPU Output data copy time: %f ms\n", gpu_time_outdata_cpy);
	
	gpu_time_ms = gpu_time_icosphere + gpu_time_outdata_cpy + gpu_time_indata_cpy;

	return gpu_time_ms;
}

void verify_gpu_output(){

	vertex * v = (vertex *)faces;
	vertex * gpu_out_v = (vertex *)gpu_out_faces;
	bool success = true;
    for (unsigned int i=0; i<3*faces_length; i++){
        if (fabs(gpu_out_v[i].x - v[i].x) >= epsilon){
            success = false;
            cerr << "Incorrect X output at face " << int(i/3) << " Vertex "<< i%3<< ": " << v[i].x << ", "
                << gpu_out_v[i].x << endl;
        }
        if (fabs(gpu_out_v[i].y - v[i].y) >= epsilon){
            success = false;
            cerr << "Incorrect Y output at face " << int(i/3) << " Vertex "<< i%3<< ": " << v[i].y << ", "
                << gpu_out_v[i].y << endl;
        }
        if (fabs(gpu_out_v[i].z - v[i].z) >= epsilon){
            success = false;
            cerr << "Incorrect Z output at face " << int(i/3) << " Vertex "<< i%3<< ": " << v[i].z << ", "
                << gpu_out_v[i].z << endl;
        }
    }
    if(success){
        cout << "Successful output" << endl;
    }
    else{
    	cout << "******** NON Successful output ********" << endl;
    }
}

int main(int argc, char **argv) {
	if(check_args(argc, argv))
		return 1;

	int depth = atoi(argv[1]);
	int thread_num = atoi(argv[2]);
	int block_num = atoi(argv[3]);
	cout << "\nThread per block:"<< thread_num << endl;
	cout << "Number of blocks:"<< block_num << "\n" << endl;

	init_vars(depth, 1);
	allocate_cpu_mem();
	init_icosphere();

	// calculate the distance b/w two points of icosphere
	float norm0 = faces[0].v[0].x*faces[0].v[0].x + faces[0].v[0].y*faces[0].v[0].y + faces[0].v[0].z*faces[0].v[0].z;
	float norm1 = faces[0].v[1].x*faces[0].v[1].x + faces[0].v[1].y*faces[0].v[1].y + faces[0].v[1].z*faces[0].v[1].z;
	float ang = acosf((faces[0].v[0].x*faces[0].v[1].x + faces[0].v[0].y*faces[0].v[1].y + faces[0].v[0].z*faces[0].v[1].z)/(norm1*norm0));
	float dis = radius*ang;
	cout << "Distance b/w any two points of icosphere is: " << dis << " (unit is same as radius)\n" << endl;
	
	
	cout << "\n----------Running CPU Code----------\n" << endl;
	float cpu_time = time_profile_cpu();
	cout << "\n----------Running GPU Code----------\n" << endl;
	float gpu_time = time_profile_gpu();
	cout << "\n----------Verifying GPU Output----------\n" << endl;
	verify_gpu_output();

	cout << "\nTime taken by the CPU is: " << cpu_time << " milliseconds" << endl;
	cout << "Time taken by the GPU is: " << gpu_time << " milliseconds" << endl;
	cout << "Speed up factor: " << cpu_time/gpu_time << "\n" << endl;

	export_csv(faces, "utilities/vertices.csv", "utilities/cpu_edges.csv", "utilities/vertices_sph.csv");
	export_csv(gpu_out_faces, "utilities/vertices.csv", "utilities/gpu_edges.csv", "utilities/vertices_sph.csv");
	free_cpu_memory();
    return 1;
}


