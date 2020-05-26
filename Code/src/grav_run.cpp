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
#include <iomanip>

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

#include "ta_utilities.hpp"
#include "grav_cpu.hpp"
#include "grav_cuda.cuh"
#include "helper_cuda.h"

using namespace std;

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


/*******************************************************************************
 * Function:        chech_args
 *
 * Description:     Checks for the user inputs arguments to run the file
 *
 * Arguments:       int argc, char argv
 *
 * Return Values:   0
*******************************************************************************/

int check_args(int argc, char **argv){
	if (argc != 3){
        // printf("Usage: ./grav [depth] [thread_per_block] \n");
        printf("Usage: ./grav [depth] [verbose: 0/1] \n");
        return 1;
    }
    return 0;
}



/*******************************************************************************
 * Function:        time_profile_cpu
 *
 * Description:     RUNS the CPU code
 *
 * Arguments:       bool verbose: If true then it will prints messages on the c
 *                  console
 *
 * Return Values:   CPU computational time
*******************************************************************************/
void time_profile_cpu(bool verbose, float * res){
	
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

    if(verbose){
	    printf("Icosphere generation time: %f ms\n", cpu_time_icosphere_ms);
	    printf("Fill vertices time: %f ms\n", cpu_time_fill_vertices_ms);
		printf("Gravitational potential time: %f ms\n", cpu_time_grav_pot_ms);
    }
    res[0] = cpu_time_icosphere_ms+cpu_time_fill_vertices_ms;
    res[1] = cpu_time_grav_pot_ms;
}


/*******************************************************************************
 * Function:        time_profile_gpu
 *
 * Description:     RUNS the GPU code
 *
 * Arguments:       bool verbose: If true then it will prints messages on the c
 *                  console
 *
 * Return Values:   GPU computational time
*******************************************************************************/
void time_profile_gpu(int thread_num, bool verbose, float * res){
	
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
    err = cudaGetLastError();
    if (cudaSuccess != err){
        cerr << "Error " << cudaGetErrorString(err) << endl;
    }else{
    	if(verbose)
        	cerr << "No kernel error detected" << endl;
    }

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

	// gpu_time_ms = gpu_time_icosphere + gpu_time_outdata_cpy + gpu_time_indata_cpy + gpu_time_gravitational;

	res[0] = gpu_time_icosphere + gpu_time_outdata_cpy + gpu_time_indata_cpy;
	res[1] = gpu_time_gravitational;
}



/*******************************************************************************
 * Function:        verify_gpu_potential
 *
 * Description:     Computes the difference between the CPU potential and GPU
 *                  potential
 *
 * Arguments:       bool verbose: If true then it will prints messages on the c
 *                  console
 *
 * Return Values:   none
*******************************************************************************/
void verify_gpu_potential(bool verbose){

	float* gpu_potential = gpu_out_potential;

	bool success = true;
    for (unsigned int i=0; i<vertices_length; i++){
        if (fabs(gpu_potential[i] - potential[i]) >= epsilon_pot){
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



/*******************************************************************************
 * Function:        verify_gpu_output
 *
 * Description:     Computes the difference between the CPU and GPU vertices
 *
 * Arguments:       bool verbose: If true then it will prints messages on the c
 *                  console
 *
 * Return Values:   none
*******************************************************************************/
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

/*******************************************************************************
 * Function:        output_potential
 *
 * Description:     Stores the vertices and corresponding potential in a MATLAB
 *                  compatible .mat file
 *
 * Arguments:       bool verbose: If true message will be printed else not.
 *
 * Return Values:   none
*******************************************************************************/

void output_potential(bool verbose){
	if(verbose)
   		cout<<"Exporting: results/output_potential.mat" << endl;
    
    std::ofstream f;
    f.open("results/output_potential.mat", std::ios::out);

    for (unsigned int i=0; i<vertices_length; i++){
        f<<i<<'\t'<<vertices[i].x<<'\t'<<vertices[i].y<<'\t'<<vertices[i].z<<'\t'<<potential[i]<<'\n';
    }
    f.close();
}


/*******************************************************************************
 * Function:        run
 *
 * Description:     Stores the vertices and corresponding potential in a MATLAB
 *                  compatible .mat file
 *
 * Arguments:       int depth - needed for icosphere calculation
 *                  int thread_num - number of threads per block
 *                  float radius - radius of the sphere
 *                  bool verbose: If true then it will prints messages on the c
 *                  console
 *
 * Return Values:   none
*******************************************************************************/
void run(int depth, int thread_num, float radius, bool verbose, float * cpu_res, float * gpu_res){

//	N_SPHERICAL = atoi(argv[2]);
	if(thread_num > 1024){
		cout << "Thread per block exceeds its maximum limit of 1024.\n Using 1024 threads per block" << endl;
	}
	if(verbose)
		cout << "\nThread per block:"<< thread_num << endl;

	init_vars(depth, radius);
	allocate_cpu_mem(verbose);
	init_icosphere();

	if(verbose)
		cout << "\n----------Running CPU Code----------\n" << endl;
	time_profile_cpu(verbose, cpu_res);

	// if(verbose)
	// 	cout << "\n----------Running GPU Code----------\n" << endl;
	// float gpu_time = time_profile_gpu(thread_num, verbose, gpu_res);
	// if(verbose)
	// 	cout << "\n----------Verifying GPU Icosphere----------\n" << endl;
	// verify_gpu_output(verbose);

	/************************** TMP *****************************/
	// if(verbose)
	// 	cout << "\n----------Verifying Sorted GPU Icosphere----------\n" << endl;
	// quickSort((void *)faces, 0, 3*faces_length-1, partition_sum);
	
	// cudacall_sort(thread_num);
	// cudaError err = cudaGetLastError();
 //    if (cudaSuccess != err){
 //        cerr << "Error " << cudaGetErrorString(err) << endl;
 //    }else{
 //    	if(verbose)
 //        	cerr << "No kernel error detected" << endl;
 //    }
 //    cuda_cpy_output_data();
    
 //    verify_gpu_output(verbose);

	// if(verbose)
	// 	cout << "\n----------Verifying GPU Potential ----------\n" << endl;
	// verify_gpu_potential(verbose);

	float cpu_time = cpu_res[0] +  cpu_res[1];
	if(verbose){
		cout << "\nTime taken by the CPU is: " << cpu_time << " milliseconds" << endl;
		// cout << "Time taken by the GPU is: " << gpu_time << " milliseconds" << endl;
		// cout << "Speed up factor: " << cpu_time/gpu_time << "\n" << endl;
	}

	// calculate the distance b/w two points of icosphere
	float norm0 = faces[0].v[0].x*faces[0].v[0].x + faces[0].v[0].y*faces[0].v[0].y + faces[0].v[0].z*faces[0].v[0].z;
	float norm1 = faces[0].v[1].x*faces[0].v[1].x + faces[0].v[1].y*faces[0].v[1].y + faces[0].v[1].z*faces[0].v[1].z;
	float ang = acosf((faces[0].v[0].x*faces[0].v[1].x + faces[0].v[0].y*faces[0].v[1].y + faces[0].v[0].z*faces[0].v[1].z)/(norm1*norm0));
	float dis = radius*ang;
	if(verbose)
		cout << "Distance b/w any two points of icosphere is: " << dis << " (unit is same as radius)\n" << endl;

    // output_potential(verbose);
	// export_csv(faces, "results/vertices.csv", "results/cpu_edges.csv", verbose);
	// export_csv(gpu_out_faces, "results/vertices.csv", "results/gpu_edges.csv", verbose);
	free_cpu_memory();
	// free_gpu_memory();

}


/*******************************************************************************
 * Function:        main
 *
 * Description:     Run the main function
 *
 * Arguments:       int argc, char argv
 *
 * Return Values:   int 1 if code executes successfully else 0.
*******************************************************************************/
int main(int argc, char **argv) {

	// TA_Utilities::select_coldest_GPU();
	if(check_args(argc, argv))
		return 0;

	int len = atoi(argv[1]);
	// if (len > 10){
	// 	cout << "It is recommend to give depth < 10. For the depth 9 alone CPU takes around 60 seconds!" << endl;
	// 	cout << "Exiting the code" << endl;
	// 	return 0;
	// }

	cout << "Running from depth 0 to depth " << len << " (both inclusive)" << endl;
	if((bool)atoi(argv[2]))
		cout << "Verbose ON" << endl;
	else
		cout << "Verbose OFF" << endl;

	float cpu_times[len][2];
	
	float tmp_res[2];
	for(int i=0; i<=len; i++){
		cout << "Running for depth: " << i << endl;
		run(i, 512, 1, (bool)atoi(argv[2]), cpu_times[i], tmp_res);	
	}

	// print table
	cout << "\n**********************************************************************************" << endl;
	cout << "|"<< setw(7)<<"Depth"<<setw(2)<<"|"
		 << setw(25)<<"|"<<setw(20)<<"CPU time (ms)"<<setw(4)<<"|"<<setw(23)<<"|" << endl;
	cout << "|--------|------------------------|-----------------------|----------------------|" << endl;
	cout << "|"<<setw(30)<<"|Icosphere (ms){Ankit}" << setw(27) 
					<< "|Potential (ms){Garima} " << "|"<< setw(14) 
					<< "Total (ms)" << setw(9) << "|"<< endl;
	for(int i=0; i<=len; i++){	
		cout 	<< "|"
				<< setw(5) << right<<i << setw(4)<< "|"
				<< setw(15) << right << cpu_times[i][0]<< setw(10) <<"|"
				<< setw(15) << right << cpu_times[i][1] << setw(9)<< "|"
				<< setw(15) << right << cpu_times[i][0]+cpu_times[i][1] << setw(8)<< "|"
				<< endl;
	}
	cout << "**********************************************************************************\n" << endl;



    return 1;
}
// \t|\t \t|\tTotal (ms)\t\t|" << endl;

