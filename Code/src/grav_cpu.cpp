/*
 * grav_run: Ankit Verma, Garima Aggarwal, 2020
 * 
 * This file contains the code for gravitational field calculation
 * by using CPU.
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

#include "grav_cpu.h"

using std::cerr;
using std::cout;
using std::endl;

#define	PI			3.1415926f;

/*Reference: http://www.songho.ca/opengl/gl_sphere.html*/
float const H_ANG = PI/180*72;
// elevation = 26.565 degree
float const ELE_ANG = atanf(1.0f / 2);

// Stores the vertices of icosphere
float * ico_x;
float * ico_y;
float * ico_z;

// Stores the edges of the icosphere
int * edges;

// The depth of the icosphere
int max_depth = 0;

// Information of sphere
int radius;

void init_vars(int depth, int r){
	// Todo: allcate the memory for variables
	max_depth = depth;
	radius = radius;

	/* Reference: https://en.wikipedia.org/wiki/Geodesic_polyhedron */
	int T = depth*depth;
	int num_edges = 30*T;
	int num_vertices = 2 + 10*T;
	
	edges = (int *)malloc(2*num_edges*sizeof(int));
	ico_x = (float *)malloc(num_vertices*sizeof(float));
	ico_y = (float *)malloc(num_vertices*sizeof(float));
	ico_z = (float *)malloc(num_vertices*sizeof(float));

	cout << "Depth: " << depth << endl;
	cout << "Faces: " << 20*T << endl;
	cout << "Total number of edges: " << num_edges << endl;
	cout << "Total number of vertices: " << num_vertices << endl;

}

void create_icoshpere(){
	/* Reference: http://www.songho.ca/opengl/gl_sphere.html*/

	//Todo: Add initial vertices and edges
	ico_x[0] = 0;
	ico_y[0] = 0;
	ico_z[0] = 0;

	float z = radius*sinf(ELE_ANG);
	float xy = radius*sinf(H_ANG);

	float hAng1 = PI/2;
	float hAng2 = PI/2 + H_ANG/2;
	for(int i=1; i<=10; i+=2){
		ico_x[i] = xy*cosf(hAng1);
		ico_x[i+1] = xy*cosf(hAng2);

		ico_y[i] = xy*sinf(hAng1);
		ico_y[i+1] = xy*sinf(hAng2);

		ico_z[i] = z;
		ico_z[i+1] = -z;
	}
	//Todo: generate icosphere of depth
}

int get_grav_pot(){
	cout << "Running from grav_cpu" << endl;
    return -1;
}

int free_memory(){
	free(ico_x);
	free(ico_y);
	free(ico_z);
	free(edges);
}


