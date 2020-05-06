/*
 * grav_run: Ankit Verma, Garima Aggarwal, 2020
 * 
 * This file contains the code for gravitational field calculation
 * by using CPU.
 * 
 */

#ifndef _GRAV_CPU_C_
	#define _GRAV_CPU_C_
#endif
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



void init_vars(int depth, int r){
	
	epsilon = 1e-6;

	// Todo: allcate the memory for variables
	max_depth = depth;
	radius = r;

	faces_length = 20*pow(4, depth);
	int edges_len = faces_length*3/2;
	vertices_length = edges_len - faces_length + 2;

	vertices_x = (float *)malloc(vertices_length*sizeof(triangle));
	vertices_y = (float *)malloc(vertices_length*sizeof(triangle));
	vertices_z = (float *)malloc(vertices_length*sizeof(triangle));

	curr_faces_count = 0;
	
	faces = (triangle *)malloc(faces_length*sizeof(triangle));
	
	cout << "Depth: " << depth << endl;
	cout << "Faces: " << faces_length << endl;
	cout << "Size of faces array: " << faces_length*sizeof(triangle)/4 << " words" << endl;
	cout << "Number of vertices: "<< vertices_length << endl;
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

		faces[c].v0.x = 0;
		faces[c].v0.y = 0;
		faces[c].v0.z = radius;
		
		faces[c].v1.x = x1;
		faces[c].v1.y = y1;
		faces[c].v1.z = z;
		
		faces[c].v2.x = x3;
		faces[c].v2.y = y3;
		faces[c].v2.z = z;
		c++;

		faces[c].v0.x = 0;
		faces[c].v0.y = 0;
		faces[c].v0.z = -radius;

		faces[c].v1.x = x2;
		faces[c].v1.y = y2;
		faces[c].v1.z = -z;

		faces[c].v2.x = x4;
		faces[c].v2.y = y4;
		faces[c].v2.z = -z;
		c++;

		faces[c].v0.x = x1;
		faces[c].v0.y = y1;
		faces[c].v0.z = z;

		faces[c].v1.x = x2;
		faces[c].v1.y = y2;
		faces[c].v1.z = -z;

		faces[c].v2.x = x3;
		faces[c].v2.y = y3;
		faces[c].v2.z = z;
		c++;

		faces[c].v0.x = x2;
		faces[c].v0.y = y2;
		faces[c].v0.z = -z;

		faces[c].v1.x = x3;
		faces[c].v1.y = y3;
		faces[c].v1.z = z;

		faces[c].v2.x = x4;
		faces[c].v2.y = y4;
		faces[c].v2.z = -z;
		c++;


	}
	curr_faces_count = c;
	cout << "curr_faces_count: " << curr_faces_count << endl;
}

void get_triangs(triangle tmp, triangle * tri){
	float x_tmp, y_tmp, z_tmp, scale;

	x_tmp = (tmp.v0.x + tmp.v1.x)/2;
	y_tmp = (tmp.v0.y + tmp.v1.y)/2;
	z_tmp = (tmp.v0.z + tmp.v1.z)/2;
	scale = radius/sqrtf(x_tmp*x_tmp + y_tmp*y_tmp + z_tmp*z_tmp);
	tri->v0.x = x_tmp*scale;
	tri->v0.y = y_tmp*scale;
	tri->v0.z = z_tmp*scale;

	x_tmp = (tmp.v1.x + tmp.v2.x)/2;
	y_tmp = (tmp.v1.y + tmp.v2.y)/2;
	z_tmp = (tmp.v1.z + tmp.v2.z)/2;
	scale = radius/sqrtf(x_tmp*x_tmp + y_tmp*y_tmp + z_tmp*z_tmp);
	tri->v1.x = x_tmp*scale;
	tri->v1.y = y_tmp*scale;
	tri->v1.z = z_tmp*scale;

	x_tmp = (tmp.v2.x + tmp.v0.x)/2;
	y_tmp = (tmp.v2.y + tmp.v0.y)/2;
	z_tmp = (tmp.v2.z + tmp.v0.z)/2;
	scale = radius/sqrtf(x_tmp*x_tmp + y_tmp*y_tmp + z_tmp*z_tmp);
	tri->v2.x = x_tmp*scale;
	tri->v2.y = y_tmp*scale;
	tri->v2.z = z_tmp*scale;
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
			faces[i].v1.x = triag_tmp.v0.x;
			faces[i].v1.y = triag_tmp.v0.y;
			faces[i].v1.z = triag_tmp.v0.z;

			faces[i].v2.x = triag_tmp.v2.x;
			faces[i].v2.y = triag_tmp.v2.y;
			faces[i].v2.z = triag_tmp.v2.z;

			//adding triangle P1, V[0], V[1]
			faces[curr_faces_count].v0.x = triag_tmp.v0.x;
			faces[curr_faces_count].v0.y = triag_tmp.v0.y;
			faces[curr_faces_count].v0.z = triag_tmp.v0.z;

			faces[curr_faces_count].v1.x = tri_i.v1.x;
			faces[curr_faces_count].v1.y = tri_i.v1.y;
			faces[curr_faces_count].v1.z = tri_i.v1.z;

			faces[curr_faces_count].v2.x = triag_tmp.v1.x;
			faces[curr_faces_count].v2.y = triag_tmp.v1.y;
			faces[curr_faces_count].v2.z = triag_tmp.v1.z;
			curr_faces_count++;

			//adding triangle P2, V[1], V[2]
			faces[curr_faces_count].v0.x = triag_tmp.v1.x;
			faces[curr_faces_count].v0.y = triag_tmp.v1.y;
			faces[curr_faces_count].v0.z = triag_tmp.v1.z;

			faces[curr_faces_count].v1.x = tri_i.v2.x;
			faces[curr_faces_count].v1.y = tri_i.v2.y;
			faces[curr_faces_count].v1.z = tri_i.v2.z;

			faces[curr_faces_count].v2.x = triag_tmp.v2.x;
			faces[curr_faces_count].v2.y = triag_tmp.v2.y;
			faces[curr_faces_count].v2.z = triag_tmp.v2.z;
			curr_faces_count++;
			
			//adding triangle V[0], V[1], V[2]
			faces[curr_faces_count].v0.x = triag_tmp.v0.x;
			faces[curr_faces_count].v0.y = triag_tmp.v0.y;
			faces[curr_faces_count].v0.z = triag_tmp.v0.z;

			faces[curr_faces_count].v1.x = triag_tmp.v1.x;
			faces[curr_faces_count].v1.y = triag_tmp.v1.y;
			faces[curr_faces_count].v1.z = triag_tmp.v1.z;

			faces[curr_faces_count].v2.x = triag_tmp.v2.x;
			faces[curr_faces_count].v2.y = triag_tmp.v2.y;
			faces[curr_faces_count].v2.z = triag_tmp.v2.z;
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
		obj_stream << faces[i].v0.x << ", " << faces[i].v0.y << ", " << faces[i].v0.z << endl;
		obj_stream << faces[i].v1.x << ", " << faces[i].v1.y << ", " << faces[i].v1.z << endl;
		obj_stream << faces[i].v2.x << ", " << faces[i].v2.y << ", " << faces[i].v2.z << endl;
	}
	obj_stream.close();

	ofstream obj_stream2;
	obj_stream2.open(filename2);
	obj_stream2 << "x1, y1, z1, x2, y2, z2" << endl;
	for(int i=0; i< curr_faces_count; i++){
		triangle triangle_tmp = faces[i];
		obj_stream2 << 	triangle_tmp.v0.x << ", " << triangle_tmp.v0.y << ", " << triangle_tmp.v0.z << ", " <<
						triangle_tmp.v1.x << ", " << triangle_tmp.v1.y << ", " << triangle_tmp.v1.z << endl;
		obj_stream2 << 	triangle_tmp.v0.x << ", " << triangle_tmp.v0.y << ", " << triangle_tmp.v0.z << ", " <<
						triangle_tmp.v2.x << ", " << triangle_tmp.v2.y << ", " << triangle_tmp.v2.z << endl;
		obj_stream2 << 	triangle_tmp.v1.x << ", " << triangle_tmp.v1.y << ", " << triangle_tmp.v1.z << ", " <<
						triangle_tmp.v2.x << ", " << triangle_tmp.v2.y << ", " << triangle_tmp.v2.z << endl;
	}
	obj_stream2.close();
}

// void fill_vertices(){
// 	int c = 0, is_add;
// 	for(int i=0; i<faces_length; i++){
// 		for(int j=0; j<c; j++){
// 			is_add = 1;
// 			for(int k=0; k<3; k++){
// 				float t = fabs(faces[i].x[k] - vertices_x[j] + faces[i].y[k] - vertices_y[j] +
// 					faces[i].z[k] - vertices_z[j]);
// 				if(t <= 3*epsilon){
// 					is_add = 0;
// 					break;
// 				}
// 				if(is_add){
// 					vertices_x[c] = faces[i].x[k];
// 					vertices_y[c] = faces[i].z[k];
// 					vertices_z[c] = faces[i].z[k];
// 					c++;
// 				}
// 			}
// 		}
// 	}
// 	cout << "Total number of vertices: " << c << endl;
// }
int get_grav_pot(){
	cout << "Running from grav_cpu" << endl;
    return -1;
}

void free_memory(){
	free(faces);
	free(vertices_x);
	free(vertices_y);
	free(vertices_z);
}


