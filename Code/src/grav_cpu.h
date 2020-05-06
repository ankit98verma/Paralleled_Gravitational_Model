/* 
 * CPU implementation of gravitational field calculator
 * header file.
 */


#ifndef _GRAV_CPU_H_
#define _GRAV_CPU_H_

using std::string;

#undef  GLOBAL
#ifdef _GRAV_CPU_C_
#define GLOBAL
#else
#define GLOBAL  extern
#endif

struct vertex
{
	float x;
	float y;
	float z;
};

struct triangle
{
	struct vertex v0;
	struct vertex v1;
	struct vertex v2;
};
typedef struct triangle triangle;

// faces of the icosphere
GLOBAL triangle * faces;
GLOBAL int faces_length;
GLOBAL int curr_faces_count;

// vertices of the icosphere
GLOBAL float * vertices_x;
GLOBAL float * vertices_y;
GLOBAL float * vertices_z;
GLOBAL int vertices_length;

// The depth of the icosphere
GLOBAL int max_depth;

// Information of sphere
GLOBAL int radius;

GLOBAL float epsilon;

void init_vars(int depth, int r);
void init_icosphere();

void create_icoshpere();
void fill_vertices();

void export_csv(string filename1, string filename2);

int get_grav_pot();

void free_memory();


#endif // CUDA_HEADER_CUH_