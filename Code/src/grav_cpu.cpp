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
#include <sstream>


#include "grav_cpu.hpp"

using namespace std;


/*Reference for this file: http://www.songho.ca/opengl/gl_sphere.html*/

float const H_ANG = PI/180*72;
float const ELE_ANG = atanf(1.0f / 2);	// elevation = 26.565 degree
unsigned int curr_faces_count;

triangle * faces_copy;


/* Decleration of local functions */
// int partition_sum(void * arr, int low, int high);
void get_coefficients();

/*******************************************************************************
 * Function:        init_vars
 *
 * Description:     This function initializes global variables. This should be
 *					the first function to be called from this file.
 *
 * Arguments:       unsigned int depth: The maximum depth of the icosphere
 *					float r: The radius of sphere
 *
 * Return Values:   None.
 *
*******************************************************************************/
void init_vars(unsigned int depth, float r){
	max_depth = depth;
	radius = r;
	// Get coefficients for the Potential function calculations
	get_coefficients();
}

/*******************************************************************************
 * Function:        allocate_cpu_mem
 *
 * Description:     This function dynamically allocate memory for the variables
 *					in the CPU memory. This function should be called only after
 *					"init_vars" function. This should be the second function to
 *					be called from this file.
 *
 * Arguments:       bool verbose: If true then it will prints messages on the c
 *									console.
 *
 * Return Values:   None.
 *
*******************************************************************************/
void allocate_cpu_mem(bool verbose){
    faces_length = 20*pow(4, max_depth);
	vertices_length = faces_length/2 + 2;
	vertices = (vertex *)malloc(vertices_length*sizeof(vertex));
	potential = (float*) malloc(vertices_length*sizeof(float));
	curr_faces_count = 0;
	faces = (triangle *)malloc(faces_length*sizeof(triangle));
	faces_copy = (triangle *)malloc(faces_length*sizeof(triangle));

	if(verbose){
		cout << "\nDepth: " << max_depth << endl;
		cout << "Faces: " << faces_length << endl;
		cout << "Number of vertices: "<< vertices_length << "\n" << endl;
	}
}

/*******************************************************************************
 * Function:        init_icosphere
 *
 * Description:     This function dynamically allocate memory for the variables
 *					in the CPU memory. This function should be called only after
 *					"init_vars" function. This should be the second function to
 *					be called from this file.
 *
 * Arguments:       bool verbose: If true then it will prints messages on the c
 *									console.
 *
 * Return Values:   None.
 *
*******************************************************************************/
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

void create_icoshpere_navie(){
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

void create_icoshpere(){
	/* Reference: http://www.songho.ca/opengl/gl_sphere.html*/
	memcpy(faces, faces_init, ICOSPHERE_INIT_FACE_LEN*sizeof(triangle));
	memcpy(faces_copy, faces_init, ICOSPHERE_INIT_FACE_LEN*sizeof(triangle));

	triangle triag_tmp;
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
			faces_copy[4*i].v[0] = tri_i.v[0];
			faces_copy[4*i].v[1] = triag_tmp.v[0];
			faces_copy[4*i].v[2] = triag_tmp.v[2];

			//adding triangle V[0], P1, V[1]
			faces_copy[4*i+1].v[0] = triag_tmp.v[0];
			faces_copy[4*i+1].v[1] = tri_i.v[1];
			faces_copy[4*i+1].v[2] = triag_tmp.v[1];

			//adding triangle P2, V[1], V[2]
			faces_copy[4*i+2].v[0] = triag_tmp.v[1];
			faces_copy[4*i+2].v[1] = tri_i.v[2];
			faces_copy[4*i+2].v[2] = triag_tmp.v[2];

			//adding triangle V[0], V[1], V[2]
			faces_copy[4*i+3].v[0] = triag_tmp.v[0];
			faces_copy[4*i+3].v[1] = triag_tmp.v[1];
			faces_copy[4*i+3].v[2] = triag_tmp.v[2];

			curr_faces_count+=3;
		}
		memcpy(faces, faces_copy, curr_faces_count*sizeof(triangle));
	}
}

void export_csv(triangle * f, string filename1, string filename2, bool verbose){
	if(verbose)
		cout << "Exporting: " << filename1 << ", " << filename2 <<endl;

	ofstream obj_stream;
	obj_stream.open(filename1);
	obj_stream << "x, y, z" << endl;
	for(unsigned int i=0; i< vertices_length; i++){
		obj_stream << vertices[i].x << ", " << vertices[i].y << ", " << vertices[i].z << endl;
	}
	obj_stream.close();

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
	quickSort((void *)faces, 0, 3*faces_length-1, partition_sum);

	int is_add=1;
	vertex * all_vs = (vertex *)faces;
	vertices[0].x = all_vs[0].x;	
	vertices[0].y = all_vs[0].y;
	vertices[0].z = all_vs[0].z;
	unsigned int c_start = 0, c_end = 1;
	for(unsigned int i=1; i<3*faces_length; i++){
		float sum_i = all_vs[i].x+all_vs[i].y+all_vs[i].z;
		float sum_i_1 = all_vs[i-1].x+all_vs[i-1].y+all_vs[i-1].z;
		if((sum_i - sum_i_1) <= EPSILON){
			is_add = 1;
			for(unsigned int j=c_start; j<c_end; j++){
				float t = 	fabs(vertices[j].x - all_vs[i].x) + fabs(vertices[j].y - all_vs[i].y) +
						fabs(vertices[j].z - vertices[j].z);
				if(t <= EPSILON){
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
			c_end++;
		}
	}
	memcpy(faces, faces_copy, faces_length*sizeof(triangle));
}

int partition_sum(void * arr_in, int low, int high){
	vertex * arr = (vertex *)arr_in;
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

void quickSort(void * arr, int low, int high, int partition_fun(void *, int, int)){
	if(low < high){
		/* pi is partitioning index, arr[p] is now
	    at right place */
	    int pi = partition_fun(arr, low, high);

	    // Separately sort elements before
	    // partition and after partition
	    quickSort(arr, low, pi - 1, partition_fun);
	    quickSort(arr, pi + 1, high, partition_fun);
	}

}

/*******************************************************************************
 * Function:        facprod
 *
 * Description:     computes (n+m)!/(n-m)!
 *
 * Arguments:       int n, int m- from geoppotential coefficients
 *
 * Return Values:   float p = (n+m)!/(n-m)!
 *
 * References:      O. Montenbruck, and E. Gill, _Satellite Orbits: Models,
                    Methods and Applications_, 2012, p.56-68.
 *
*******************************************************************************/
float facprod(int n, int m){
    float p = 1.0; // initialisation of p

    for (int i = n-m+1; i<=n+m; i++)
        p = p/i;

    return p;
}



/*******************************************************************************
 * Function:        get_coefficients
 *
 * Description:     Gets the geopotential coefficients from the GRAVITY_MODEL.txt
 *
 * Arguments:       none
 *
 * Return Values:   none
 *
 * References:      O. Montenbruck, and E. Gill, _Satellite Orbits: Models,
                    Methods and Applications_, 2012, p.56-68.
 *
*******************************************************************************/
void get_coefficients(){

    // Read the file from GRAVITY_MODEL
    ifstream file("resources/GRAVITY_MODEL.txt");

    // Pseudo data_definition
    int N_N = (N_SPHERICAL+1)*(N_SPHERICAL+2)/2;
    float data[N_N][4];

    // Read data from the file
	string line;
	int row=0;
    while (!file.eof())
    {

        getline(file, line);
        stringstream iss(line);
        for (int col = 0; col < 4; ++col)
        {
            iss >> data[row][col];
        }
        row++;
    }

    // Systematically store the data in
    // coeff[N_SPHERICAL+1][N_SPHERICAL+2] matrix
    int k=0;
    // Store the Cnm coefficients
    for (int i=0; i<N_SPHERICAL+1; i++)
        for (int j=0; j<=i; j++)
        {
            coeff[i][j] = data[k][2];
            k++;
        }

    k=0;
    // Store the Snm coefficients
    for (int i = N_SPHERICAL; i>=0; i--)
        for (int j=N_SPHERICAL+1; j>i; j--)
        {
            coeff[i][j] = data[k][3];
            k++;
        }
    file.close();
}



/*******************************************************************************
 * Function:        spherical_harmonics
 *
 * Description:     This function calculates the geopotential value at specific
                    locations on the spherical model
 *
 * Arguments:       vertex R_vec ==  vertex on the sphere. has information on x,y,z
 *
 * Return Values:   float U == potential value
 *
 * References:      O. Montenbruck, and E. Gill, _Satellite Orbits: Models,
                    Methods and Applications_, 2012, p.56-68.
 *
*******************************************************************************/
float spherical_harmonics(vertex R_vec){

    // Define pseudo coefficients
    float Radius_sq = pow(radius,2);
    float rho = pow(R_eq,2)/Radius_sq;

    float x0 = R_eq*R_vec.x/Radius_sq;
    float y0 = R_eq*R_vec.y/Radius_sq;
    float z0 = R_eq*R_vec.z/Radius_sq;

    //Initialize Intermediary Matrices
    float V[N_SPHERICAL+1][N_SPHERICAL+1];
    float W[N_SPHERICAL+1][N_SPHERICAL+1];

    for (int m=0; m<N_SPHERICAL+1; m++){
        for (int n = m; n<N_SPHERICAL+1; n++){
            V[n][m] = 0.0;
            W[n][m] = 0.0;
        }
    }

    // Calculate zonal terms V(n, 0). Set W(n,0)=0.0
    V[0][0] = R_eq /sqrt(Radius_sq);
    W[0][0] = 0.0;

    V[1][0] = z0 * V[0][0];
    W[1][0] = 0.0;

    for (int n=2; n<N_SPHERICAL+1; n++){
        V[n][0] = ((2*n-1)*z0*V[n-1][0] - (n-1)*rho*V[n-2][0])/n;
        W[n][0] = 0.0;
    } // Eqn 3.30


    //Calculate tesseral and sectoral terms
    for (int m = 1; m < N_SPHERICAL + 1; m++){
        // Eqn 3.29
        V[m][m] = (2*m-1)*(x0*V[m-1][m-1] - y0*W[m-1][m-1]);
        W[m][m] = (2*m-1)*(x0*W[m-1][m-1] + y0*V[m-1][m-1]);

        // n=m+1 (only one term)
        if (m < N_SPHERICAL){
            V[m+1][m] = (2*m+1)*z0*V[m][m];
            W[m+1][m] = (2*m+1)*z0*W[m][m];
        }

        for (int n = m+2; n<N_SPHERICAL+1; n++){
            V[n][m] = ((2*n-1)*z0*V[n-1][m]-(n+m-1)*rho*V[n-2][m])/(n-m);
            W[n][m] = ((2*n-1)*z0*W[n-1][m]-(n+m-1)*rho*W[n-2][m])/(n-m);
        }
    }

//    for (int m=0; m<N_SPHERICAL+1; m++){
//        for (int n = m; n<N_SPHERICAL+1; n++){
//        }
//    }

    // Calculate potential
    float C = 0; // Cnm coeff
    float S = 0; // Snm coeff
    float N = 0; // normalisation number
    float U = 0; //potential
    float p = 0;
    for (int m=0; m<N_SPHERICAL+1; m++){
        for (int n = m; n<N_SPHERICAL+1; n++){
            C = 0;
            S = 0;
            if (m==0){
                N = sqrt(2*n+1);
                C = N*coeff[n][0];
//                U = C*V[n][0];
            }
            else {
                p = facprod(n,m);
                N = sqrt(2*(2*n+1)*p);
                C = N*coeff[n][m];
                S = N*coeff[N_SPHERICAL-n][N_SPHERICAL-m+1];
            }
            U = U + C*V[n][m] + S*W[n][m];
            // Calculation of the Gravitational Potential Calculation model
        }
    }
    U = U*mhu/R_eq;
    return U;
}



/*******************************************************************************
 * Function:        get_grav_pot
 *
 * Description:     This function calculates the geopotential value at all
                    locations on the spherical model
 *
 * Arguments:       none
 *
 * Return Values:   none
 *
*******************************************************************************/

void get_grav_pot(){
    for (unsigned int i=0; i<vertices_length; i++){
        potential[i] = spherical_harmonics(vertices[i]);
    }
}



void free_cpu_memory(){

    // Free malloc arrays
	free(faces);
	free(faces_copy);
	free(vertices);
	free(potential);
}

