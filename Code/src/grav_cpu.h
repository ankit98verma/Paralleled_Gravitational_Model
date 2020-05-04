/* 
 * CPU implementation of gravitational field calculator
 * header file.
 */


#ifndef _GRAV_CPU_H_
#define _GRAV_CPU_H_


void init_vars(int depth, int r);
void create_icoshpere();
int get_grav_pot();
int free_memory();


#endif // CUDA_HEADER_CUH_