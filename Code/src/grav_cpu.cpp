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
int curr_vertices_count;

// Stores the edges of the icosphere
int * faces;
int faces_length;
int curr_faces_count;

// The depth of the icosphere
int max_depth = 0;

// Information of sphere
int radius;

void init_vars(int depth, int r){
	// Todo: allcate the memory for variables
	max_depth = depth;
	radius = r;

	/* Reference: https://en.wikipedia.org/wiki/Geodesic_polyhedron */
	int T = (depth+1)*(depth+1);
	faces_length = 3*20*T;
	vertices_length = 2 + 10*T;

	curr_vertices_count = 0;
	curr_faces_count = 0;
	
	faces = (int *)malloc(faces_length*sizeof(int));
	ico_x = (float *)malloc(vertices_length*sizeof(float));
	ico_y = (float *)malloc(vertices_length*sizeof(float));
	ico_z = (float *)malloc(vertices_length*sizeof(float));

	cout << "Depth: " << depth << endl;
	cout << "Faces: " << 20*T << endl;
	cout << "Total number of faces: " << faces_length/3 << endl;
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

		faces[c] = 0;
		faces[c+1] = i;
		faces[c+2] = (i+2)%10;

		faces[c+3] = 11;
		faces[c+4] = i+1;
		if(i+3>10)
			faces[c+5] = (i+3)%10;
		else
			faces[c+5] = i+3;
		
		faces[c+6] = i;
		faces[c+7] = i+1;
		if(i+2>10)
			faces[c+8] = (i+2)%10;
		else
			faces[c+8] = (i+2);

		faces[c+9] = i+1;
		if(i+2>10)
			faces[c+10] = (i+2)%10;
		else
			faces[c+10] = i+2;
		if(i+3>10)
			faces[c+11] = (i+3)%10;
		else
			faces[c+11] = (i+3);		


		hAng1 += H_ANG;
		hAng2 += H_ANG;
		c += 12;
	}
	cout << "c: "<< c << endl;
	ico_x[11] = 0;
	ico_y[11] = 0;
	ico_z[11] = -radius;

	curr_vertices_count = 12;
	curr_faces_count = c;
	
}

void create_icoshpere(){
	/* Reference: http://www.songho.ca/opengl/gl_sphere.html*/

	//Todo: generate icosphere of depth
	for(int i=1; i<=max_depth; i++){
		cout << "Adding to depth: " << i << endl;
		float a = curr_faces_count;
		// go through every edge and divide the edge into half
		for(int i=0; i< a; i+=2){
			// int p1 = edges[i];
			// int p2 = edges[i+1];

			// float mid_x = (ico_x[p1] + ico_x[p2])/2;
			// float mid_y = (ico_y[p1] + ico_y[p2])/2;
			// float mid_z = (ico_z[p1] + ico_z[p2])/2;
			// float scale = radius/sqrtf(mid_x*mid_x + mid_y*mid_y + mid_z*mid_z);

			// mid_x *= scale;
			// mid_y *= scale;
			// mid_z *= scale;

			// // add the new vertex 
			// ico_x[curr_vertices_count] = mid_x;
			// ico_y[curr_vertices_count] = mid_y;
			// ico_z[curr_vertices_count] = mid_z;

			// // remove the current edge and insert two new edges
			// edges[i] = p1;
			// edges[i+1] = curr_vertices_count;

			// edges[curr_faces_count] = p2;
			// edges[curr_faces_count+1] = curr_vertices_count;

			// curr_vertices_count++;
			// curr_faces_count+=2;

			// cout << "curr_vertices_count: " << curr_vertices_count << endl;
			// cout << "curr_faces_count: " << curr_vertices_count << endl;
		}
	}
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
	for(int i=0; i< curr_faces_count; i+=3){
		int p1 = faces[i];
		int p2 = faces[i+1];
		int p3 = faces[i+2];
		obj_stream2 << 	ico_x[p1] << ", " << ico_y[p1] << ", " << ico_z[p1] << ", " <<
						ico_x[p2] << ", " << ico_y[p2] << ", " << ico_z[p2] << endl;
		obj_stream2 << 	ico_x[p3] << ", " << ico_y[p3] << ", " << ico_z[p3] << ", " <<
						ico_x[p2] << ", " << ico_y[p2] << ", " << ico_z[p2] << endl;
		obj_stream2 << 	ico_x[p3] << ", " << ico_y[p3] << ", " << ico_z[p3] << ", " <<
						ico_x[p1] << ", " << ico_y[p1] << ", " << ico_z[p1] << endl;
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
	free(faces);
}


