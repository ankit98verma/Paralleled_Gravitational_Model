/*
 * CPU implementation of gravitational field calculator
 * header file.
 */


#ifndef _GRAV_RUN_H_
#define _GRAV_RUN_H_


#include <cstdio>
#include <cstdlib>
#include <iostream>

#define ICO_NAVIE	0
#define ICO_OPT1	1

#define GEO_POTENTIAL_NAVIE	0
#define GEO_POTENTIAL_OP1	1
#define GEO_POTENTIAL_OP2	2
#define GEO_POTENTIAL_OP3	3


#define ICOSPHERE_GPU_THREAD_NUM		1024
#define GEOPOTENTIAL_NAVIE_THREAD_NUM	512
#define GEOPOTENTIAL_OPT1_THREAD_NUM	256
#define GEOPOTENTIAL_OPT2_THREAD_NUM	256
#define GEOPOTENTIAL_OPT3_THREAD_NUM	


#endif // CUDA_HEADER_CUH_