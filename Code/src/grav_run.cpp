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

#include "ta_utilities.hpp"
#include "grav_cpu.hpp"
#include "grav_cuda.cuh"
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
	if (argc != 3){
        printf("Usage: ./grav [depth] [thread_per_block] \n");
        return 1;
    }
    return 0;
}

float time_profile_cpu(bool verbose){
	float cpu_time_ms = 0;
	float cpu_time_icosphere_ms = 0;
	float cpu_time_fill_vertices_ms = 0;
	float cpu_time_grav_pot_ms = 0;

	START_TIMER();
		create_icoshpere();
	STOP_RECORD_TIMER(cpu_time_icosphere_ms);

	START_TIMER();
		fill_vertices();
	STOP_RECORD_TIMER(cpu_time_fill_vertices_ms);

	START_TIMER();
    	get_grav_pot();
    STOP_RECORD_TIMER(cpu_time_grav_pot_ms);
    cpu_time_ms = cpu_time_icosphere_ms + cpu_time_fill_vertices_ms + cpu_time_grav_pot_ms;
    
    if(verbose){
	    printf("Icosphere generation time: %f ms\n", cpu_time_icosphere_ms);
	    printf("Fill vertices time: %f ms\n", cpu_time_fill_vertices_ms);
		printf("Gravitational potential time: %f ms\n", cpu_time_grav_pot_ms);
    }

	return cpu_time_ms;
}

// this function should be called only after calling time_profile_cpu
float time_profile_gpu(int thread_num, bool verbose){
	float gpu_time_ms = 0;
	float gpu_time_icosphere = -1, gpu_time_icosphere2 =-1;
	float gpu_time_indata_cpy = -1;
	float gpu_time_outdata_cpy = -1;
	float gpu_time_gravitational = -1;


	START_TIMER();
		cuda_cpy_input_data();
	STOP_RECORD_TIMER(gpu_time_indata_cpy);

	START_TIMER();
		cudacall_icosphere_naive(thread_num);
	STOP_RECORD_TIMER(gpu_time_icosphere);
	cudaError err = cudaGetLastError();
    if (cudaSuccess != err){
        cerr << "Error " << cudaGetErrorString(err) << endl;
    }else{
    	if(verbose)
        	cerr << "No kernel error detected" << endl;
    }

	free_gpu_memory();
	START_TIMER();
		cuda_cpy_input_data();
	STOP_RECORD_TIMER(gpu_time_indata_cpy);

	START_TIMER();
		cudacall_icosphere(thread_num);
	STOP_RECORD_TIMER(gpu_time_icosphere2);
	err = cudaGetLastError();
    if (cudaSuccess != err){
        cerr << "Error " << cudaGetErrorString(err) << endl;
    }else{
    	if(verbose)
        	cerr << "No kernel error detected" << endl;
    }

    // COMPUTING GRAVITATIONAL POTENTIAL
    START_TIMER();
        cudacall_gravitational(thread_num);
    STOP_RECORD_TIMER(gpu_time_gravitational);

	START_TIMER();
		cuda_cpy_output_data();
	STOP_RECORD_TIMER(gpu_time_outdata_cpy);



	if(verbose){
		printf("GPU Input data copy time: %f ms\n", gpu_time_indata_cpy);
	    printf("GPU Naive Icosphere generation time: %f ms\n", gpu_time_icosphere);
	    printf("GPU Icosphere generation time: %f ms\n", gpu_time_icosphere2);
		printf("GPU potential calculation: %f ms\n", gpu_time_gravitational);
		printf("GPU Output data copy time: %f ms\n", gpu_time_outdata_cpy);
	}

	gpu_time_ms = gpu_time_icosphere + gpu_time_outdata_cpy + gpu_time_indata_cpy + gpu_time_gravitational;

	return gpu_time_ms;
}

void verify_gpu_potential(bool verbose){

	float* gpu_potential = gpu_out_potential;

	bool success = true;
    for (unsigned int i=0; i<vertices_length; i++){
        if (fabs(gpu_potential[i] - potential[i]) >= epsilon){
            success = false;
            cerr << "Incorrect potential calculation at " << i << " Vertex: "<< gpu_potential[i]<< ", " << potential[i] << endl;
        }
    }

    if(success){
    	if(verbose)
        	cout << "--------Successful output--------" << endl;
    }
    else{
    	cout << "******** Unsuccessful output ********" << endl;
    }
}

void verify_gpu_output(bool verbose){

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
    	if(verbose)
        	cout << "--------Successful output--------" << endl;
    }
    else{
    	cout << "******** Unsuccessful output ********" << endl;
    }
}

void run(int depth, int thread_num, int n_sph, float radius, bool verbose){

//	N_SPHERICAL = atoi(argv[2]);
	if(thread_num > 1024){
		cout << "Thread per block exceeds its maximum limit of 1024.\n Using 1024 threads per block" << endl; 
	}
	if(verbose)
		cout << "\nThread per block:"<< thread_num << endl;

	init_vars(depth, 1);
	allocate_cpu_mem(verbose);
	init_icosphere();

	if(verbose)
		cout << "\n----------Running CPU Code----------\n" << endl;
	float cpu_time = time_profile_cpu(verbose);
	
	if(verbose)
		cout << "\n----------Running GPU Code----------\n" << endl;
	float gpu_time = time_profile_gpu(thread_num, verbose);
	if(verbose)
		cout << "\n----------Verifying GPU Icosphere----------\n" << endl;
	verify_gpu_output(verbose);
	if(verbose)
		cout << "\n----------Verifying GPU Potential ----------\n" << endl;
	verify_gpu_potential(verbose);

//    for (int i=0; i<N_SPHERICAL; i++)
//        cout<<coeff[i][i]<<'\n';
//
//	for (int i=0; i<vertices_length; i++)
//    {
//        cout<<potential[i]<<'\n';
//    }

	if(verbose){
		cout << "\nTime taken by the CPU is: " << cpu_time << " milliseconds" << endl;
		cout << "Time taken by the GPU is: " << gpu_time << " milliseconds" << endl;
		cout << "Speed up factor: " << cpu_time/gpu_time << "\n" << endl;
	}

	// calculate the distance b/w two points of icosphere
	float norm0 = faces[0].v[0].x*faces[0].v[0].x + faces[0].v[0].y*faces[0].v[0].y + faces[0].v[0].z*faces[0].v[0].z;
	float norm1 = faces[0].v[1].x*faces[0].v[1].x + faces[0].v[1].y*faces[0].v[1].y + faces[0].v[1].z*faces[0].v[1].z;
	float ang = acosf((faces[0].v[0].x*faces[0].v[1].x + faces[0].v[0].y*faces[0].v[1].y + faces[0].v[0].z*faces[0].v[1].z)/(norm1*norm0));
	float dis = radius*ang;
	if(verbose)
		cout << "Distance b/w any two points of icosphere is: " << dis << " (unit is same as radius)\n" << endl;

	// export_csv(faces, "utilities/vertices.csv", "utilities/cpu_edges.csv", "utilities/vertices_sph.csv");
	// export_csv(gpu_out_faces, "utilities/vertices.csv", "utilities/gpu_edges.csv", "utilities/vertices_sph.csv");
	free_cpu_memory();
	free_gpu_memory();
}

int main(int argc, char **argv) {

	// TA_Utilities::select_coldest_GPU();
	if(check_args(argc, argv))
		return 0;
	
	run(atoi(argv[1]), atoi(argv[2]), 0, 1, false);

	
    return 1;
}


