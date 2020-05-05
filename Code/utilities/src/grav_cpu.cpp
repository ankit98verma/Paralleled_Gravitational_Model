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
using std::ofstream;
using std::string;

#define	PI			3.1415926f

/*Reference: http://www.songho.ca/opengl/gl_sphere.html*/
float const H_ANG = PI/180*72;
// elevation = 26.565 degree
float const ELE_ANG = atanf(1.0f / 2);

// Stores the vertices of icosphere
float * ico_x;
float * ico_y;
float * ico_z;
int vertices_length;

// Stores the edges of the icosphere
int * edges;
int edge_length;

// The depth of the icosphere
int max_depth = 0;

// Information of sphere
int radius;

void init_vars(int depth, int r){
	// Todo: allcate the memory for variables
	max_depth = depth;
	radius = r;

	/* Reference: https://en.wikipedia.org/wiki/Geodesic_polyhedron */
	int T = depth*depth;
	edge_length = 2*30*T;
	vertices_length = 2 + 10*T;
	
	edges = (int *)malloc(edge_length*sizeof(int));
	ico_x = (float *)malloc(vertices_length*sizeof(float));
	ico_y = (float *)malloc(vertices_length*sizeof(float));
	ico_z = (float *)malloc(vertices_length*sizeof(float));

	cout << "Depth: " << depth << endl;
	cout << "Faces: " << 20*T << endl;
	cout << "Total number of edges: " << edge_length/2 << endl;
	cout << "Total number of vertices: " << vertices_length << endl;

}

void init_icosphere(){
	//Todo: Add initial vertices and edges
	ico_x[0] = 0;
	ico_y[0] = 0;
	ico_z[0] = radius;

	float z = radius*sinf(ELE_ANG);
	float xy = radius*cosf(ELE_ANG);

	float hAng1 = PI/2;
	float hAng2 = PI/2 + H_ANG/2;

	int c = 0;
	for(int i=1; i<=10; i+=2){
		ico_x[i] = xy*cosf(hAng1);
		ico_x[i+1] = xy*cosf(hAng2);

		ico_y[i] = xy*sinf(hAng1);
		ico_y[i+1] = xy*sinf(hAng2);

		ico_z[i] = z;
		ico_z[i+1] = -z;

		edges[c] = 0;
		edges[c+1] = i;

		edges[c+2] = 11;
		edges[c+3] = i+1;

		edges[c+4] = i;
		edges[c+5] = i+1;

		edges[c+6] = i+1;
		edges[c+7] = (i+2)%10;

		edges[c+8] = i;
		edges[c+9] = (i+2)%10;

		edges[c+10] = i+1;
		if(i+3 > 10){
			edges[c+11] = (i+3)%10;	
		}else{
			edges[c+11] = (i+3);	
		}
		

		// cout << edges[c]  << endl;
		// cout << edges[c+1] << endl;


		// cout << edges[c+2] << endl;
		// cout << edges[c+3] << endl;


		// cout << edges[c+4] << endl;
		// cout << edges[c+5] << endl;


		// cout << edges[c+6] << endl;
		// cout << edges[c+7] << endl;


		// cout << edges[c+8] << endl;
		// cout << edges[c+9] << endl;

		// cout << edges[c+10] << endl;
		// cout << edges[c+11] << endl;

		hAng1 += H_ANG;
		hAng2 += H_ANG;
		c += 12;
	}
	ico_x[11] = 0;
	ico_y[11] = 0;
	ico_z[11] = -radius;
	
}

void create_icoshpere(){
	/* Reference: http://www.songho.ca/opengl/gl_sphere.html*/

	
	//Todo: generate icosphere of depth



}

void export_csv(string filename1, string filename2){
	cout << "Exporting: " << filename1 << ", " << filename2 <<endl;

	ofstream obj_stream;
	obj_stream.open(filename1);
	obj_stream << "x, y, z" << endl;
	for(int i=0; i< vertices_length; i++){
		obj_stream << ico_x[i] << ", " << ico_y[i] << ", " << ico_z[i] << endl;
	}
	obj_stream.close();

	ofstream obj_stream2;
	obj_stream2.open(filename2);
	obj_stream2 << "x1, y1, z1, x2, y2, z2" << endl;
	for(int i=0; i< edge_length; i+=2){
		int p1 = edges[i];
		int p2 = edges[i+1];
		obj_stream2 << 	ico_x[p1] << ", " << ico_y[p1] << ", " << ico_z[p1] << ", " <<
						ico_x[p2] << ", " << ico_y[p2] << ", " << ico_z[p2] << endl;
	}
	obj_stream2.close();
}

int get_grav_pot(){
	cout << "Running from grav_cpu" << endl;
    return -1;
}

void free_memory(){
	free(ico_x);
	free(ico_y);
	free(ico_z);
	free(edges);
}


