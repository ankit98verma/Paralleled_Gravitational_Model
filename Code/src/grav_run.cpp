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

using std::cerr;
using std::cout;
using std::endl;


int check_args(int argc, char **argv){
	if (argc != 4){
        printf("Usage: ./grav [depth] [thread_per_block] [number_of_blocks]\n");
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
	if(check_args(argc, argv))
		return 1;
	
	int depth = atoi(argv[1]);
	int thread_num = atoi(argv[2]);
	int block_num = atoi(argv[3]);
	
	cout << "Depth: "<< depth << endl;
	cout << "Thread per block:"<< thread_num << endl;
	cout << "Number of blocks:"<< block_num << endl;
	
	init_vars(depth, 1);

	free_memory();
    return 1;
}


