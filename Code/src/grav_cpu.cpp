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
int curr_faces_count;

void init_vars(int depth, int r){

	epsilon = 1e-6;

	// Todo: allcate the memory for variables
	max_depth = depth;
	radius = r;

	faces_length = 20*pow(4, depth);
	vertices_length = faces_length/2 + 2;
	vertices = (vertex *)malloc(vertices_length*sizeof(vertex));
	
	vertices_sph = (point_sph *)malloc(vertices_length*sizeof(point_sph));
	
	common_thetas_count = (int *)malloc(vertices_length*sizeof(int));
	common_thetas_length = 0;
	
	curr_faces_count = 0;
	faces = (int *)malloc(3*faces_length*sizeof(int));

	cout << "\nDepth: " << depth << endl;
	cout << "Faces: " << faces_length << endl;
	// cout << "Size of faces array: " << faces_length*sizeof(triangle)/4 << " words" << endl;
	cout << "Number of vertices: "<< vertices_length << "\n" << endl;
	// cout << "Size of vertices array: " << vertices_length*sizeof(vertices)/4 << " words" << endl;
	faces_length = 3*faces_length;

	potential = (float*) malloc(vertices_length*sizeof(float));
}

void init_icosphere(){
	//Todo: Add initial vertices and edges
	float z = radius*sinf(ELE_ANG);
	float xy = radius*cosf(ELE_ANG);

	float hAng1 = PI/2;
	float hAng2 = PI/2 + H_ANG/2;

	vertices[0].x = 0;
	vertices[0].y = 0;
	vertices[0].z = radius;
	int c = 0;
	for(int i=1; i<=10; i+=2){
		float x1 = xy*cosf(hAng1);
		float x2 = xy*cosf(hAng2);

		float y1 = xy*sinf(hAng1);
		float y2 = xy*sinf(hAng2);

		hAng1 += H_ANG;
		hAng2 += H_ANG;

		vertices[i].x = x1;
		vertices[i].y = y1;
		vertices[i].z = z;

		vertices[i+1].x = x2;
		vertices[i+1].y = y2;
		vertices[i+1].z = -z;

		faces[c] = 0;
		faces[c+1] = i;
		faces[c+2] = (i+2)%10;
		c+=3;
		
		faces[c] = 11;
		faces[c+1] = i+1;
		if(i+3 > 10)
			faces[c+2] = (i+3)%10;
		else
			faces[c+2] = (i+3);
		c+=3;

		faces[c] = i;
		faces[c+1] = i+1;
		faces[c+2] = (i+2)%10;
		c+=3;

		faces[c] = i+1;
		faces[c+1] = (i+2)%10;
		if(i+3 > 10)
			faces[c+2] = (i+3)%10;
		else
			faces[c+2] = (i+3);
		c+=3;
	}
	curr_faces_count = c;
	vertices[11].x = 0;
	vertices[11].y = 0;
	vertices[11].z = -radius;
}

void get_midpoints(vertex * tmp, vertex * tri){
	float x_tmp, y_tmp, z_tmp, scale;

	x_tmp = (tmp[0].x + tmp[1].x)/2;
	y_tmp = (tmp[0].y + tmp[1].y)/2;
	z_tmp = (tmp[0].z + tmp[1].z)/2;
	scale = radius/sqrtf(x_tmp*x_tmp + y_tmp*y_tmp + z_tmp*z_tmp);
	tri[0].x = x_tmp*scale;
	tri[0].y = y_tmp*scale;
	tri[0].z = z_tmp*scale;

	x_tmp = (tmp[1].x + tmp[2].x)/2;
	y_tmp = (tmp[1].y + tmp[2].y)/2;
	z_tmp = (tmp[1].z + tmp[2].z)/2;
	scale = radius/sqrtf(x_tmp*x_tmp + y_tmp*y_tmp + z_tmp*z_tmp);
	tri[1].x = x_tmp*scale;
	tri[1].y = y_tmp*scale;
	tri[1].z = z_tmp*scale;

	x_tmp = (tmp[2].x + tmp[0].x)/2;
	y_tmp = (tmp[2].y + tmp[0].y)/2;
	z_tmp = (tmp[2].z + tmp[0].z)/2;
	scale = radius/sqrtf(x_tmp*x_tmp + y_tmp*y_tmp + z_tmp*z_tmp);
	tri[2].x = x_tmp*scale;
	tri[2].y = y_tmp*scale;
	tri[2].z = z_tmp*scale;
}

void create_icoshpere(){
	/* Reference: http://www.songho.ca/opengl/gl_sphere.html*/

	vertices triag_tmp[3];
	//Todo: generate icosphere of depth
	for(int j=1; j<=max_depth; j++){
		// cout << "Adding to depth: " << j << " Starting with Curr face count: " << curr_faces_count<< endl;
		int a = curr_faces_count;
		// go through every edge and divide the edge into half
		for(int i=0; i<a; i+=3){
			vertices tri_i[3];
			tri_i[0] = vertices[faces[i]];
			tri_i[1] = vertices[faces[1]];
			tri_i[2] = vertices[faces[2]];
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
			// faces[i].v1.x = triag_tmp.v0.x;
			// faces[i].v1.y = triag_tmp.v0.y;
			// faces[i].v1.z = triag_tmp.v0.z;

			// faces[i].v2.x = triag_tmp.v2.x;
			// faces[i].v2.y = triag_tmp.v2.y;
			// faces[i].v2.z = triag_tmp.v2.z;

			// //adding triangle P1, V[0], V[1]
			// faces[curr_faces_count].v0.x = triag_tmp.v0.x;
			// faces[curr_faces_count].v0.y = triag_tmp.v0.y;
			// faces[curr_faces_count].v0.z = triag_tmp.v0.z;

			// faces[curr_faces_count].v1.x = tri_i.v1.x;
			// faces[curr_faces_count].v1.y = tri_i.v1.y;
			// faces[curr_faces_count].v1.z = tri_i.v1.z;

			// faces[curr_faces_count].v2.x = triag_tmp.v1.x;
			// faces[curr_faces_count].v2.y = triag_tmp.v1.y;
			// faces[curr_faces_count].v2.z = triag_tmp.v1.z;
			// curr_faces_count++;

			// //adding triangle P2, V[1], V[2]
			// faces[curr_faces_count].v0.x = triag_tmp.v1.x;
			// faces[curr_faces_count].v0.y = triag_tmp.v1.y;
			// faces[curr_faces_count].v0.z = triag_tmp.v1.z;

			// faces[curr_faces_count].v1.x = tri_i.v2.x;
			// faces[curr_faces_count].v1.y = tri_i.v2.y;
			// faces[curr_faces_count].v1.z = tri_i.v2.z;

			// faces[curr_faces_count].v2.x = triag_tmp.v2.x;
			// faces[curr_faces_count].v2.y = triag_tmp.v2.y;
			// faces[curr_faces_count].v2.z = triag_tmp.v2.z;
			// curr_faces_count++;

			// //adding triangle V[0], V[1], V[2]
			// faces[curr_faces_count].v0.x = triag_tmp.v0.x;
			// faces[curr_faces_count].v0.y = triag_tmp.v0.y;
			// faces[curr_faces_count].v0.z = triag_tmp.v0.z;

			// faces[curr_faces_count].v1.x = triag_tmp.v1.x;
			// faces[curr_faces_count].v1.y = triag_tmp.v1.y;
			// faces[curr_faces_count].v1.z = triag_tmp.v1.z;

			// faces[curr_faces_count].v2.x = triag_tmp.v2.x;
			// faces[curr_faces_count].v2.y = triag_tmp.v2.y;
			// faces[curr_faces_count].v2.z = triag_tmp.v2.z;
			// curr_faces_count++;
		}
	}
	cout << "Final curr face count: "<< curr_faces_count<< endl;
}

void export_csv(string filename1, string filename2, string filename3){
	cout << "Exporting: " << filename1 << ", " << filename2 <<endl;

	ofstream obj_stream;
	obj_stream.open(filename1);
	obj_stream << "x, y, z" << endl;
	ofstream obj_stream3;
	obj_stream3.open(filename3);
	obj_stream3 << "r, theta, phi" << endl;
	for(int i=0; i< vertices_length; i++){
		obj_stream << vertices[i].x << ", " << vertices[i].y << ", " << vertices[i].z << endl;
		// obj_stream3 << 	vertices_sph[i].r << ", " << vertices_sph[i].theta << ", " << vertices_sph[i].phi << endl;
	}
	obj_stream.close();
	obj_stream3.close();

	ofstream obj_stream2;
	obj_stream2.open(filename2);
	obj_stream2 << "x1, y1, z1, x2, y2, z2" << endl;
	for(int i=0; i< curr_faces_count; i+=3){
		vertex v1 = vertices[faces[i]];
		vertex v2 = vertices[faces[i+1]];
		vertex v3 = vertices[faces[i+2]];
		obj_stream2 << 	v1.x << ", " << v1.y << ", " << v1.z << ", " <<
						v2.x << ", " << v2.y << ", " << v2.z << endl;
		obj_stream2 << 	v1.x << ", " << v1.y << ", " << v1.z << ", " <<
						v3.x << ", " << v3.y << ", " << v3.z << endl;
		obj_stream2 << 	v3.x << ", " << v3.y << ", " << v3.z << ", " <<
						v2.x << ", " << v2.y << ", " << v2.z << endl;
	}
	obj_stream2.close();
}

void fill_vertices(){
	int c = 0, is_add;
	vertex * all_vs = (vertex *)faces;
	for(int i=0; i<3*faces_length; i++){
		is_add = 1;
		for(int j=0; j<c; j++){
			float t = 	fabs(all_vs[i].x - vertices[j].x) + fabs(all_vs[i].y - vertices[j].y) +
						fabs(all_vs[i].z - vertices[j].z);
			if(t <= 3*epsilon){
				is_add = 0;
				break;
			}
		}
		if(is_add){
			vertices[c].x = all_vs[i].x;
			vertices[c].y = all_vs[i].y;
			vertices[c].z = all_vs[i].z;
			vertices_sph[c].r = 1;
			vertices_sph[c].theta = atan2f(vertices[c].z, sqrtf(vertices[c].x*vertices[c].x + vertices[c].y*vertices[c].y));
			vertices_sph[c].phi = atan2f(vertices[c].y, vertices[c].x);
			c++;
		}
	}
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

void fill_common_theta(){
	float prev = vertices_sph[0].theta;
	common_thetas_count[common_thetas_length]++;
	for(int i=1; i<vertices_length; i++){
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
    for (int i=1; i<common_thetas_length; i++){
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

	for (int i=0; i<common_thetas_length; i++)
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


