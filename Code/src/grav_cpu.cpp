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
unsigned int curr_faces_count;

triangle * faces_copy;

void quickSort_faces(int low, int high);

void init_vars(unsigned int depth, float r){
	epsilon = 1e-6;
	max_depth = depth;
	radius = r;
}

void allocate_cpu_mem(){
	faces_length = 20*pow(4, max_depth);
	vertices_length = faces_length/2 + 2;
	vertices = (vertex *)malloc(vertices_length*sizeof(vertex));
	vertices_sph = (point_sph *)malloc(vertices_length*sizeof(point_sph));
	common_thetas_count = (int *)malloc(vertices_length*sizeof(int));
	common_thetas_length = 0;
	potential = (float*) malloc(vertices_length*sizeof(float));

	curr_faces_count = 0;
	faces = (triangle *)malloc(faces_length*sizeof(triangle));
	faces_copy = (triangle *)malloc(faces_length*sizeof(triangle));

	cout << "\nDepth: " << max_depth << endl;
	cout << "Faces: " << faces_length << endl;
	cout << "Number of vertices: "<< vertices_length << "\n" << endl;
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

		faces_init[c].v[0].x = 0;
		faces_init[c].v[0].y = 0;
		faces_init[c].v[0].z = radius;

		faces_init[c].v[1].x = x1;
		faces_init[c].v[1].y = y1;
		faces_init[c].v[1].z = z;

		faces_init[c].v[2].x = x3;
		faces_init[c].v[2].y = y3;
		faces_init[c].v[2].z = z;
		c++;

		faces_init[c].v[0].x = 0;
		faces_init[c].v[0].y = 0;
		faces_init[c].v[0].z = -radius;

		faces_init[c].v[1].x = x2;
		faces_init[c].v[1].y = y2;
		faces_init[c].v[1].z = -z;

		faces_init[c].v[2].x = x4;
		faces_init[c].v[2].y = y4;
		faces_init[c].v[2].z = -z;
		c++;

		faces_init[c].v[0].x = x1;
		faces_init[c].v[0].y = y1;
		faces_init[c].v[0].z = z;

		faces_init[c].v[1].x = x2;
		faces_init[c].v[1].y = y2;
		faces_init[c].v[1].z = -z;

		faces_init[c].v[2].x = x3;
		faces_init[c].v[2].y = y3;
		faces_init[c].v[2].z = z;
		c++;

		faces_init[c].v[0].x = x2;
		faces_init[c].v[0].y = y2;
		faces_init[c].v[0].z = -z;

		faces_init[c].v[1].x = x3;
		faces_init[c].v[1].y = y3;
		faces_init[c].v[1].z = z;

		faces_init[c].v[2].x = x4;
		faces_init[c].v[2].y = y4;
		faces_init[c].v[2].z = -z;
		c++;
	}
	curr_faces_count = c;
}

void get_midpoints(triangle tmp, triangle * tri){
	float x_tmp, y_tmp, z_tmp, scale;
	for(int i=0; i<3;i++){
		x_tmp = (tmp.v[i].x + tmp.v[(i+1)%3].x)/2;
		y_tmp = (tmp.v[i].y + tmp.v[(i+1)%3].y)/2;
		z_tmp = (tmp.v[i].z + tmp.v[(i+1)%3].z)/2;
		scale = radius/sqrtf(x_tmp*x_tmp + y_tmp*y_tmp + z_tmp*z_tmp);
		tri->v[i].x = x_tmp*scale;
		tri->v[i].y = y_tmp*scale;
		tri->v[i].z = z_tmp*scale;
	}
}

void create_icoshpere(){
	/* Reference: http://www.songho.ca/opengl/gl_sphere.html*/
	memcpy(faces, faces_init, ICOSPHERE_INIT_FACE_LEN*sizeof(triangle));
	triangle triag_tmp;
	//Todo: generate icosphere of depth
	for(unsigned int j=1; j<=max_depth; j++){
		// cout << "Adding to depth: " << j << " Starting with Curr face count: " << curr_faces_count<< endl;
		unsigned int a = curr_faces_count;
		// go through every face and divide the face into four equal parts
		for(unsigned int i=0; i<a; i++){
			triangle tri_i = faces[i];
			/* compute 3 new vertices by splitting half on each edge
	        *         P0
	        *        / \
	        *  V[0] *---* V[2]
	        *      / \ / \
	        *    P1---*---P2
	        *         V[1]
	        */
			get_midpoints(tri_i, &triag_tmp);

			//adding triangle P0, V[0], V[2]
			faces[i].v[1].x = triag_tmp.v[0].x;
			faces[i].v[1].y = triag_tmp.v[0].y;
			faces[i].v[1].z = triag_tmp.v[0].z;

			faces[i].v[2].x = triag_tmp.v[2].x;
			faces[i].v[2].y = triag_tmp.v[2].y;
			faces[i].v[2].z = triag_tmp.v[2].z;

			//adding triangle V[0], P1, V[1]
			faces[curr_faces_count].v[0].x = triag_tmp.v[0].x;
			faces[curr_faces_count].v[0].y = triag_tmp.v[0].y;
			faces[curr_faces_count].v[0].z = triag_tmp.v[0].z;

			faces[curr_faces_count].v[1].x = tri_i.v[1].x;
			faces[curr_faces_count].v[1].y = tri_i.v[1].y;
			faces[curr_faces_count].v[1].z = tri_i.v[1].z;

			faces[curr_faces_count].v[2].x = triag_tmp.v[1].x;
			faces[curr_faces_count].v[2].y = triag_tmp.v[1].y;
			faces[curr_faces_count].v[2].z = triag_tmp.v[1].z;
			curr_faces_count++;

			//adding triangle P2, V[1], V[2]
			faces[curr_faces_count].v[0].x = triag_tmp.v[1].x;
			faces[curr_faces_count].v[0].y = triag_tmp.v[1].y;
			faces[curr_faces_count].v[0].z = triag_tmp.v[1].z;

			faces[curr_faces_count].v[1].x = tri_i.v[2].x;
			faces[curr_faces_count].v[1].y = tri_i.v[2].y;
			faces[curr_faces_count].v[1].z = tri_i.v[2].z;

			faces[curr_faces_count].v[2].x = triag_tmp.v[2].x;
			faces[curr_faces_count].v[2].y = triag_tmp.v[2].y;
			faces[curr_faces_count].v[2].z = triag_tmp.v[2].z;
			curr_faces_count++;

			//adding triangle V[0], V[1], V[2]
			faces[curr_faces_count].v[0].x = triag_tmp.v[0].x;
			faces[curr_faces_count].v[0].y = triag_tmp.v[0].y;
			faces[curr_faces_count].v[0].z = triag_tmp.v[0].z;

			faces[curr_faces_count].v[1].x = triag_tmp.v[1].x;
			faces[curr_faces_count].v[1].y = triag_tmp.v[1].y;
			faces[curr_faces_count].v[1].z = triag_tmp.v[1].z;

			faces[curr_faces_count].v[2].x = triag_tmp.v[2].x;
			faces[curr_faces_count].v[2].y = triag_tmp.v[2].y;
			faces[curr_faces_count].v[2].z = triag_tmp.v[2].z;
			curr_faces_count++;
		}
	}
	memcpy(faces_copy, faces, faces_length*sizeof(triangle));
	// cout << "Final curr face count: "<< curr_faces_count<< endl;
}

void export_csv(triangle * f, string filename1, string filename2, string filename3){
	cout << "Exporting: " << filename1 << ", " << filename2 <<endl;

	ofstream obj_stream;
	obj_stream.open(filename1);
	obj_stream << "x, y, z" << endl;
	ofstream obj_stream3;
	obj_stream3.open(filename3);
	obj_stream3 << "r, theta, phi" << endl;
	for(unsigned int i=0; i< vertices_length; i++){
		obj_stream << vertices[i].x << ", " << vertices[i].y << ", " << vertices[i].z << endl;
		obj_stream3 << 	vertices_sph[i].r << ", " << vertices_sph[i].theta << ", " << vertices_sph[i].phi << endl;
	}
	obj_stream.close();
	obj_stream3.close();

	ofstream obj_stream2;
	obj_stream2.open(filename2);
	obj_stream2 << "x1, y1, z1, x2, y2, z2" << endl;
	for(unsigned int i=0; i< curr_faces_count; i++){
		triangle triangle_tmp = f[i];
		for(int j=0; j<3;j++)
			obj_stream2 << 	triangle_tmp.v[j].x << ", " << triangle_tmp.v[j].y << ", " << triangle_tmp.v[j].z << ", " <<
							triangle_tmp.v[(j+1)%3].x << ", " << triangle_tmp.v[(j+1)%3].y << ", " << triangle_tmp.v[(j+1)%3].z << endl;
	}
	obj_stream2.close();
}

void fill_vertices(){
	quickSort_faces(0, 3*faces_length-1);

	int is_add=1;
	vertex * all_vs = (vertex *)faces;
	vertices[0].x = all_vs[0].x;
	vertices[0].y = all_vs[0].y;
	vertices[0].z = all_vs[0].z;
	unsigned int c_start = 0, c_end = 1;

	vertices_sph[c_end].r = 1;
	vertices_sph[c_end].theta = atan2f(vertices[c_end].z, sqrtf(vertices[c_end].x*vertices[c_end].x + vertices[c_end].y*vertices[c_end].y));
	vertices_sph[c_end].phi = atan2f(vertices[c_end].y, vertices[c_end].x);
	for(unsigned int i=1; i<3*faces_length; i++){
		float sum_i = all_vs[i].x+all_vs[i].y+all_vs[i].z;
		float sum_i_1 = all_vs[i-1].x+all_vs[i-1].y+all_vs[i-1].z;
		if((sum_i - sum_i_1) <= epsilon){
			is_add = 1;
			for(unsigned int j=c_start; j<c_end; j++){
				float t = 	fabs(vertices[j].x - all_vs[i].x) + fabs(vertices[j].y - all_vs[i].y) +
						fabs(vertices[j].z - vertices[j].z);
				if(t <= epsilon){
					is_add = 0;
					break;
				}
			}
		}else{
			is_add = 1;
			c_start = c_end;
		}
		if(is_add){
			vertices[c_end].x = all_vs[i].x;
			vertices[c_end].y = all_vs[i].y;
			vertices[c_end].z = all_vs[i].z;
			vertices_sph[c_end].r = 1;
			vertices_sph[c_end].theta = atan2f(vertices[c_end].z, sqrtf(vertices[c_end].x*vertices[c_end].x + vertices[c_end].y*vertices[c_end].y));
			vertices_sph[c_end].phi = atan2f(vertices[c_end].y, vertices[c_end].x);
			c_end++;
		}
	}
	memcpy(faces, faces_copy, faces_length*sizeof(triangle));
}

int partition(point_sph * arr, int low, int high)  {

    point_sph pivot = arr[high]; // pivot
    int i = (low - 1); // Index of smaller element
  	point_sph tmp;
    for (int j = low; j <= high - 1; j++)
    {
        // If current element is smaller than the pivot
        if (arr[j].theta < pivot.theta)
        {
            i++; // increment index of smaller element
            tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }
    tmp = arr[high];
    arr[high] = arr[i+1];
    arr[i+1] = tmp;
    return (i + 1);
}
void quickSort_points(int low, int high)
{
	if(low < high){
		/* pi is partitioning index, arr[p] is now
	    at right place */
	    int pi = partition(vertices_sph, low, high);

	    // Separately sort elements before
	    // partition and after partition
	    quickSort_points(low, pi - 1);
	    quickSort_points(pi + 1, high);
	}

}

int partition(vertex * arr, int low, int high)  {

    vertex pivot = arr[high]; // pivot
    int i = (low - 1); // Index of smaller element
  	vertex tmp;
    for (int j = low; j <= high - 1; j++)
    {
        // If current element is smaller than the pivot
        float sum_j = arr[j].x+ arr[j].y + arr[j].z;
        float sum_p = pivot.x+ pivot.y + pivot.z;
        if (sum_j < sum_p)
        {
            i++; // increment index of smaller element
            tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }
    tmp = arr[high];
    arr[high] = arr[i+1];
    arr[i+1] = tmp;
    return (i + 1);
}
void quickSort_faces(int low, int high)
{
	if(low < high){
		/* pi is partitioning index, arr[p] is now
	    at right place */
	    int pi = partition((vertex *)faces, low, high);

	    // Separately sort elements before
	    // partition and after partition
	    quickSort_faces(low, pi - 1);
	    quickSort_faces(pi + 1, high);
	}

}

void fill_common_theta(){
	float prev = vertices_sph[0].theta;
	common_thetas_count[common_thetas_length]++;
	for(unsigned int i=1; i<vertices_length; i++){
		if(fabs(prev - vertices_sph[i].theta) > epsilon){
			common_thetas_length++;
			prev = vertices_sph[i].theta;
		}
		common_thetas_count[common_thetas_length]++;
	}
}

void find_sin_array(float theta, float* sine_array){
    // COMPUTES the sin array to be used
    // sin_array = (1, sin(theta), (sin(theta))^2, (sin(theta))^3, (sin(theta))^4/...)
    // powers needed till (sin(theta))^10--- therefore 12 components
    sine_array[0] = 1;

    for (int i=1; i<11; i++){
        sine_array[i] = sine_array[i-1]*sin(theta);
    }
}

void legendre(float theta, float* P){
    // COMPUTE P_n(sin(that))
    float sine_array[11];
    find_sin_array(theta, sine_array);

    P[0] = 0;
    P[1] = 0;
    P[2] = 0;
    P[3] = 2.5*sine_array[3] - 1.5*sine_array[1];
    P[4] = (35*sine_array[4] - 30*sine_array[2] + 3)/8;
    P[5] = (63*sine_array[5] - 70*sine_array[3] + 15*sine_array[1])/8;
    P[6] = (231*sine_array[6] - 315*sine_array[4] + 105*sine_array[2] - 5)/16;
    P[7] = (429*sine_array[7] - 693*sine_array[5] + 315*sine_array[3] - 35*sine_array[1])/16;
    P[8] = (6435*sine_array[8] - 12012*sine_array[6] + 6930*sine_array[4] - 1260*sine_array[2] + 35)/128;
    P[9] = (12155*sine_array[9] - 25740*sine_array[7] + 18018*sine_array[5] - 4620*sine_array[3] + 315*sine_array[1])/128;
    P[10] = (46189*sine_array[10] - 109395*sine_array[8] + 90090*sine_array[6] - 30030*sine_array[4] + 3465*sine_array[2] - 63)/256;
}

void potential_cal_ZONAL(float theta, float* pot_coeff){
    // INPUT: r- radius vector magnitude
    // theta - latitude in radians
    // NOTE: Zonal harmonic is a function of theta alone
    float LEGENDRE[11];
    legendre(theta, LEGENDRE);

    float ZONAL_J[11] = {0, 0, 0, 0.2541e-05, 0.1617999e-05, 0.22800004e-06, -0.5519908e-06, 0.3519996e-6, 0.2049998e-06, 0.153999e-06, 0.23699982e-06};

    for (int i=0; i<11; i++){
        pot_coeff[i] = ZONAL_J[i]*LEGENDRE[i];
    }
}

void cummulative_theta_count(){
    // Finds the cumulative of common_theta_count and store in cummulative_common_theta_count
    cummulative_common_theta_count= (int *)malloc(common_thetas_length*sizeof(int));
    cummulative_common_theta_count[0] = common_thetas_count[0];
    for (unsigned int i=1; i<common_thetas_length; i++){
       cummulative_common_theta_count[i] = common_thetas_count[i] + cummulative_common_theta_count[i-1];
    }

}

void get_grav_pot(){
    // To access ith vertex use: vertices_sph[i].r, vertices_sph[i].theta and vertices_sph[i].phi
	// The vertices_sph are sorted with respect to theta
	// common_thetas_count gives the count of thetas with epsilon = 1e-6.
	// common_thetas_length gives the length of the common_thetas_count array.
	// Hence common_thetas_length effectively gives total number of unique thetas present in vertices_sph array

    cummulative_theta_count();

	// Get potential coeff
	float pot_coeff[11];

	for (unsigned int i=0; i<common_thetas_length; i++)
    {
        int Theta_indice = cummulative_common_theta_count[i];
        float THETA = vertices_sph[Theta_indice-1].theta; // Finds the common theta value
        potential_cal_ZONAL(THETA, pot_coeff);

        if (i==0){
            for (int j=0; j<cummulative_common_theta_count[i]; j++){
                potential[j] = 0;
                for (int k=3; k<11; k++)
                    potential[j] = potential[j] + pot_coeff[k]/pow(vertices_sph[j].r,k+1);
//                cout<<"\n potential" <<j <<'\t'<<potential[j];
            }
        }
        else{
             for (int j=cummulative_common_theta_count[i-1]; j<cummulative_common_theta_count[i]; j++){
                potential[j] = 0;
                for (int k=3; k<11; k++)
                    potential[j] = potential[j] + pot_coeff[k]/pow(vertices_sph[j].r,k+1);
//                cout<<"\n potential" <<j <<'\t'<<potential[j];
            }
        }
    }
}

void free_cpu_memory(){
	free(faces);
	free(vertices);
	free(vertices_sph);
	free(potential);
	free(cummulative_common_theta_count);
	free(common_thetas_count);
}


