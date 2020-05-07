/* 
 * CPU implementation of gravitational field calculator
 * header file.
 */


#ifndef _GRAV_CPU_H_
#define _GRAV_CPU_H_

#include <cstdio>
#include <cstdlib>
#include <iostream>

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
typedef struct vertex vertex;

struct point_sph
{
	float r;
	float theta;
	float phi;
};
typedef struct point_sph point_sph;

struct triangle
{
	vertex v0;
	vertex v1;
	vertex v2;
};
typedef struct triangle triangle;

// faces of the icosphere
GLOBAL triangle * faces;
GLOBAL int faces_length;
GLOBAL int curr_faces_count;

// vertices of the icosphere
GLOBAL vertex * vertices;
GLOBAL point_sph * vertices_sph;
GLOBAL int vertices_length;

// The depth of the icosphere
GLOBAL int max_depth;

// Information of sphere
GLOBAL float radius;

GLOBAL float epsilon;

void init_vars(int depth, int r);
void init_icosphere();

void create_icoshpere();
void fill_vertices();
void quickSort_points(int low, int high);

void export_csv(string filename1, string filename2, string filename3);

void get_grav_pot(vertex * vertices, int vertices_length);

void free_cpu_memory();


#endif // CUDA_HEADER_CUH_