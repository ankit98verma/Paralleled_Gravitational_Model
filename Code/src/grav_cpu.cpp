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

// faces of the icosphere
triangle * faces;
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

	faces_length = 20*pow(4, depth);
	
	curr_faces_count = 0;
	
	faces = (triangle *)malloc(faces_length*sizeof(triangle));
	
	cout << "Depth: " << depth << endl;
	cout << "Faces: " << faces_length << endl;
	cout << "Size of faces array: " << faces_length*sizeof(triangle)/4 << " words" << endl;
}

void init_icosphere(){
	//Todo: Add initial vertices and edges
	float z = radius*sinf(ELE_ANG);
	float xy = radius*cosf(ELE_ANG);

	float hAng1 = PI/2;
	float hAng2 = PI/2 + H_ANG/2;

	int c = 0;
	for(int i=0; i<5; i++){
		float x1 = xy*cosf(hAng1);
		float x2 = xy*cosf(hAng2);

		float y1 = xy*sinf(hAng1);
		float y2 = xy*sinf(hAng2);

		hAng1 += H_ANG;
		hAng2 += H_ANG;

		float x3 = xy*cosf(hAng1);
		float x4 = xy*cosf(hAng2);

		float y3 = xy*sinf(hAng1);
		float y4 = xy*sinf(hAng2);

		faces[c].x[0] = 0;
		faces[c].y[0] = 0;
		faces[c].z[0] = radius;
		
		faces[c].x[1] = x1;
		faces[c].y[1] = y1;
		faces[c].z[1] = z;
		
		faces[c].x[2] = x3;
		faces[c].y[2] = y3;
		faces[c].z[2] = z;
		c++;

		faces[c].x[0] = 0;
		faces[c].y[0] = 0;
		faces[c].z[0] = -radius;

		faces[c].x[1] = x2;
		faces[c].y[1] = y2;
		faces[c].z[1] = -z;

		faces[c].x[2] = x4;
		faces[c].y[2] = y4;
		faces[c].z[2] = -z;
		c++;

		faces[c].x[0] = x1;
		faces[c].y[0] = y1;
		faces[c].z[0] = z;

		faces[c].x[1] = x2;
		faces[c].y[1] = y2;
		faces[c].z[1] = -z;

		faces[c].x[2] = x3;
		faces[c].y[2] = y3;
		faces[c].z[2] = z;
		c++;


		faces[c].x[0] = x2;
		faces[c].y[0] = y2;
		faces[c].z[0] = -z;

		faces[c].x[1] = x3;
		faces[c].y[1] = y3;
		faces[c].z[1] = z;

		faces[c].x[2] = x4;
		faces[c].y[2] = y4;
		faces[c].z[2] = -z;
		c++;


	}
	curr_faces_count = c;
	cout << "curr_faces_count: " << curr_faces_count << endl;
}

void get_triangs(triangle tmp, triangle * v1){
	float x_tmp, y_tmp, z_tmp, scale;
	for(int i=0; i<3; i++){
		int ind1 = (i+1)%3;
		x_tmp = (tmp.x[i] + tmp.x[ind1])/2;
		y_tmp = (tmp.y[i] + tmp.y[ind1])/2;
		z_tmp = (tmp.z[i] + tmp.z[ind1])/2;

		scale = radius/sqrtf(x_tmp*x_tmp + y_tmp*y_tmp + z_tmp*z_tmp);
		v1->x[i] = x_tmp*scale;
		v1->y[i] = y_tmp*scale;
		v1->z[i] = z_tmp*scale;
	}
}

void create_icoshpere(){
	/* Reference: http://www.songho.ca/opengl/gl_sphere.html*/

	triangle triag_tmp;
	//Todo: generate icosphere of depth
	for(int j=1; j<=max_depth; j++){
		cout << "Adding to depth: " << j << " Starting with Curr face count: " << curr_faces_count<< endl;
		int a = curr_faces_count;
		// go through every edge and divide the edge into half
		for(int i=0; i<a; i++){
			triangle tri_i = faces[i];
			/* compute 3 new vertices by spliting half on each edge
	        *         P0       
	        *        / \       
	        *  V[0] *---* V[2]
	        *      / \ / \     
	        *    P1---*---P2 
	        *         V[1]  
	        */ 
			get_triangs(tri_i, &triag_tmp);
			
			//adding triangle P0, V[0], V[2]
			faces[i].x[1] = triag_tmp.x[0];
			faces[i].y[1] = triag_tmp.y[0];
			faces[i].z[1] = triag_tmp.z[0];

			faces[i].x[2] = triag_tmp.x[2];
			faces[i].y[2] = triag_tmp.y[2];
			faces[i].z[2] = triag_tmp.z[2];

			//adding triangle P1, V[0], V[1]
			faces[curr_faces_count].x[0] = triag_tmp.x[0];
			faces[curr_faces_count].y[0] = triag_tmp.y[0];
			faces[curr_faces_count].z[0] = triag_tmp.z[0];

			faces[curr_faces_count].x[1] = tri_i.x[1];
			faces[curr_faces_count].y[1] = tri_i.y[1];
			faces[curr_faces_count].z[1] = tri_i.z[1];

			faces[curr_faces_count].x[2] = triag_tmp.x[1];
			faces[curr_faces_count].y[2] = triag_tmp.y[1];
			faces[curr_faces_count].z[2] = triag_tmp.z[1];
			curr_faces_count++;

			//adding triangle P2, V[1], V[2]
			faces[curr_faces_count].x[0] = triag_tmp.x[1];
			faces[curr_faces_count].y[0] = triag_tmp.y[1];
			faces[curr_faces_count].z[0] = triag_tmp.z[1];

			faces[curr_faces_count].x[1] = tri_i.x[2];
			faces[curr_faces_count].y[1] = tri_i.y[2];
			faces[curr_faces_count].z[1] = tri_i.z[2];

			faces[curr_faces_count].x[2] = triag_tmp.x[2];
			faces[curr_faces_count].y[2] = triag_tmp.y[2];
			faces[curr_faces_count].z[2] = triag_tmp.z[2];
			curr_faces_count++;


			//adding triangle V[0], V[1], V[2]
			faces[curr_faces_count].x[0] = triag_tmp.x[0];
			faces[curr_faces_count].y[0] = triag_tmp.y[0];
			faces[curr_faces_count].z[0] = triag_tmp.z[0];

			faces[curr_faces_count].x[1] = triag_tmp.x[1];
			faces[curr_faces_count].y[1] = triag_tmp.y[1];
			faces[curr_faces_count].z[1] = triag_tmp.z[1];

			faces[curr_faces_count].x[2] = triag_tmp.x[2];
			faces[curr_faces_count].y[2] = triag_tmp.y[2];
			faces[curr_faces_count].z[2] = triag_tmp.z[2];
			curr_faces_count++;

			if(curr_faces_count > faces_length)
				cout << "EXCEEDED face count" << endl;

		}
		
	}
	cout << "Final curr face count: "<< curr_faces_count<< endl;
}

void export_csv(string filename1, string filename2){
	cout << "Exporting: " << filename1 << ", " << filename2 <<endl;

	ofstream obj_stream;
	obj_stream.open(filename1);
	obj_stream << "x, y, z" << endl;
	for(int i=0; i< curr_faces_count; i++){
		obj_stream << faces[i].x[0] << ", " << faces[i].y[0] << ", " << faces[i].z[0] << endl;
		obj_stream << faces[i].x[1] << ", " << faces[i].y[1] << ", " << faces[i].z[1] << endl;
		obj_stream << faces[i].x[2] << ", " << faces[i].y[2] << ", " << faces[i].z[2] << endl;
	}
	obj_stream.close();

	ofstream obj_stream2;
	obj_stream2.open(filename2);
	obj_stream2 << "x1, y1, z1, x2, y2, z2" << endl;
	for(int i=0; i< curr_faces_count; i++){
		triangle triangle_tmp = faces[i];
		obj_stream2 << 	triangle_tmp.x[0] << ", " << triangle_tmp.y[0] << ", " << triangle_tmp.z[0] << ", " <<
						triangle_tmp.x[1] << ", " << triangle_tmp.y[1] << ", " << triangle_tmp.z[1] << endl;
		obj_stream2 << 	triangle_tmp.x[0] << ", " << triangle_tmp.y[0] << ", " << triangle_tmp.z[0] << ", " <<
						triangle_tmp.x[2] << ", " << triangle_tmp.y[2] << ", " << triangle_tmp.z[2] << endl;
		obj_stream2 << 	triangle_tmp.x[1] << ", " << triangle_tmp.y[1] << ", " << triangle_tmp.z[1] << ", " <<
						triangle_tmp.x[2] << ", " << triangle_tmp.y[2] << ", " << triangle_tmp.z[2] << endl;
	}
	obj_stream2.close();
}

int get_grav_pot(){
	cout << "Running from grav_cpu" << endl;
    return -1;
}

void free_memory(){
	free(faces);
}


