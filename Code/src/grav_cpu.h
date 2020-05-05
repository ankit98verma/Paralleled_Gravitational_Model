/* 
 * CPU implementation of gravitational field calculator
 * header file.
 */


#ifndef _GRAV_CPU_H_
#define _GRAV_CPU_H_

using std::string;

void init_vars(int depth, int r);
void init_icosphere();

void create_icoshpere();
int get_grav_pot();

void export_csv(string filename);

void free_memory();


#endif // CUDA_HEADER_CUH_