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

float time_profile_cpu(int depth, float radius){
	init_vars(depth, radius);
	init_icosphere();
	float cpu_time_ms = -1;

	START_TIMER();
	
	create_icoshpere();
	fill_vertices();
	quickSort_points(0, vertices_length-1);
	fill_common_theta();

	// Garima TODO: Call Gravitational potential calculating function here
	// the "vertex * vertices" is an array of vertices a which you have to 
	// get the gravitational potential. The length of this array is given 
	// by "vertices_length". 
	// So pass "vertices" and "vertices_length" to your function.
	// To access ith vertex use: vertices[i].x, vertices[i].y and vertices[i].z
	
	STOP_RECORD_TIMER(cpu_time_ms);

	return cpu_time_ms;
}


float time_profile_gpu(int depth, float radius, int thread_num, int block_num){
	return -1;
}
int main(int argc, char **argv) {
	if(check_args(argc, argv))
		return 1;
	
	int depth = atoi(argv[1]);
	int thread_num = atoi(argv[2]);
	int block_num = atoi(argv[3]);
	cout << "\nThread per block:"<< thread_num << endl;
	cout << "Number of blocks:"<< block_num << "\n" << endl;
	
	float cpu_time = time_profile_cpu(depth, 1);
	
	export_csv("utilities/vertices.csv", "utilities/edges.csv", "utilities/vertices_sph.csv");
	cout << "\n\nTime taken by the CPU is: " << cpu_time << " milliseconds\n\n" << endl;
	

	free_cpu_memory();
    return 1;
}


