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

#define ICOSPHERE_INIT_FACE_LEN		20

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
    vertex v[3];
};
typedef struct triangle triangle;


// faces of the icosphere
GLOBAL triangle * faces;
GLOBAL triangle faces_init[ICOSPHERE_INIT_FACE_LEN];
GLOBAL unsigned int faces_length;

// vertices of the icosphere
GLOBAL vertex * vertices;
GLOBAL point_sph * vertices_sph;
GLOBAL unsigned int vertices_length;
GLOBAL float * potential;

GLOBAL int * common_thetas_count;
GLOBAL int * cummulative_common_theta_count;
GLOBAL unsigned int common_thetas_length;

// The depth of the icosphere
GLOBAL unsigned int max_depth;

// Information of sphere
GLOBAL float radius;

GLOBAL float epsilon;

void init_vars(unsigned int depth, float r);
void allocate_cpu_mem();
void init_icosphere();

void create_icoshpere();
void create_icoshpere2();
void fill_vertices();
void quickSort_points(int low, int high);
void fill_common_theta();

int partition_theta(void * arr_in, int low, int high);
void quickSort(void * arr, int low, int high, int partition_fun(void *, int, int));

void export_csv(triangle * f, string filename1, string filename2, string filename3);

//void get_grav_pot(vertex * vertices, int vertices_length);
void get_grav_pot();

void free_cpu_memory();


#endif // CUDA_HEADER_CUH_
