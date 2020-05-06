/* 
 * CPU implementation of gravitational field calculator
 * header file.
 */


#ifndef _GRAV_CPU_H_
#define _GRAV_CPU_H_

using std::string;

struct triangle
{
	float x[3];
	float y[3];
	float z[3];
};

typedef struct triangle triangle;
void init_vars(int depth, int r);
void init_icosphere();

void create_icoshpere();
int get_grav_pot();

void export_csv(string filename1, string filename2);

void free_memory();


#endif // CUDA_HEADER_CUH_