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


#include "grav_cpu.hpp"

using std::cerr;
using std::cout;
using std::endl;
using std::ofstream;
using std::string;

#define	PI			3.1415926f
#define R_eq    6378.1363
#define mhu 398600 // in km^3/s^2
#define N_SPHERICAL 20
#define N_N (N_SPHERICAL+1)*(N_SPHERICAL+2)/2


/*Reference: http://www.songho.ca/opengl/gl_sphere.html*/
float const H_ANG = PI/180*72;
// elevation = 26.565 degree
float const ELE_ANG = atanf(1.0f / 2);
unsigned int curr_faces_count;
class path;

triangle * faces_copy;

int partition_sum(void * arr, int low, int high);

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
			faces[i].v[1] = triag_tmp.v[0];
			faces[i].v[2] = triag_tmp.v[2];

			//adding triangle V[0], P1, V[1]
			faces[curr_faces_count].v[0] = triag_tmp.v[0];
			faces[curr_faces_count].v[1] = tri_i.v[1];
			faces[curr_faces_count].v[2] = triag_tmp.v[1];
			curr_faces_count++;

			//adding triangle P2, V[1], V[2]
			faces[curr_faces_count].v[0] = triag_tmp.v[1];
			faces[curr_faces_count].v[1] = tri_i.v[2];
			faces[curr_faces_count].v[2] = triag_tmp.v[2];
			curr_faces_count++;

			//adding triangle V[0], V[1], V[2]
			faces[curr_faces_count].v[0] = triag_tmp.v[0];
			faces[curr_faces_count].v[1] = triag_tmp.v[1];
			faces[curr_faces_count].v[2] = triag_tmp.v[2];
			curr_faces_count++;
		}
	}
	memcpy(faces_copy, faces, faces_length*sizeof(triangle));
	// cout << "Final curr face count: "<< curr_faces_count<< endl;
}

void create_icoshpere2(){
	/* Reference: http://www.songho.ca/opengl/gl_sphere.html*/
	memcpy(faces, faces_init, ICOSPHERE_INIT_FACE_LEN*sizeof(triangle));

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
	// memcpy(faces_copy, faces, faces_length*sizeof(triangle));
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
	quickSort((void *)faces, 0, 3*faces_length-1, partition_sum);
	// quickSort_points(0, 3*faces_length-1);

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

int partition_theta(void * arr_in, int low, int high){
	point_sph * arr = (point_sph *)arr_in;

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

void quickSort(void * arr, int low, int high, int partition_fun(void *, int, int))
{
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

float facprod(int n, int m){
    float p = 1.0;

    for (int i = n-m+1; i<=n+m; i++)
        p = p/i;

    return p;
}


void get_coefficients(float (&coeff)[N_SPHERICAL+1][N_SPHERICAL+2]){

    //[ Fetching data from utilities\jared_EGM20_model.csv file
    std::ifstream file("..\utilities\jared_EGM20_model.csv");
    float data[2][N_N];
    for(int row = 0; row < 2; ++row)
    {
        std::string line;
        std::getline(file, line);
        if ( !file.good() )
            break;

        std::stringstream iss(line);

        for (int col = 0; col < N_N; ++col)
        {
            std::string val;
            std::getline(iss, val, ',');
            if ( !iss.good() )
                break;

            std::stringstream convertor(val);
            convertor >> data[row][col];
        }
    }

    int k=0;
    for (int i=0; i<N_SPHERICAL+1; i++)
        for (int j=0; j<=i; j++)
        {
            coeff[i][j] = data[0][k];
            k++;
        }

    k=0;
    for (int i = N_SPHERICAL; i>=0; i--)
        for (int j=N_SPHERICAL+1; j>i; j--)
        {
            coeff[i][j] = data[1][k];
            k++;
        }

//    for (int row=0; row<2; row++){
//        for (int col=0; col<((N_SPHERICAL+1)*(N_SPHERICAL+2))/2; col++)
//            cout<<data[row][col]<<'\t';
//        cout<<"\n \n \n";
//
//    }
//
//    for (int i=0; i<N_SPHERICAL +1; i++){
//        cout<<"\n \n";
//        for (int j=0; j<N_SPHERICAL+2; j++)
//            cout<<coeff[i][j]<<'\t';
//    }

    file.close();
}

//[
//References:
//1. O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and Applications_, 2012, p.56-68.
float spherical_harmonics(float (&coeff)[N_SPHERICAL+1][N_SPHERICAL+2], float* R_vec ){
    //R_vec ==  cartesian coordinate vector
//    float Radius = vertices_sph[0].r; // The altitude is fixed throughout for all the points on the sphere
    float Radius = sqrt(pow(R_vec[0],2) + pow(R_vec[1],2) + pow(R_vec[2],2)); // The altitude is fixed throughout for all the points on the sphere
    float Radius_sq = pow(Radius,2);
    float rho = pow(R_eq,2)/Radius_sq;

    float x0 = R_eq*R_vec[0]/Radius_sq;
    float y0 = R_eq*R_vec[1]/Radius_sq;
    float z0 = R_eq*R_vec[2]/Radius_sq;

    //Initialize Intermediary Matrices
    float V[N_SPHERICAL+1][N_SPHERICAL+1];
    float W[N_SPHERICAL+1][N_SPHERICAL+1];

    // Calculate zonal terms V(n, 0). Set W(n,0)=0.0
    V[0][0] = R_eq /sqrt(Radius_sq);
    W[0][0] = 0.0;

    V[1][0] = z0 * V[0][0];
    W[1][0] = 0.0;

    for (int n=2; n<N_SPHERICAL+1; n++){
        V[n][0] = ((2*n-1)*z0*V[n-1][0] - (n-1)*rho*V[n-2][0+1])/n;
        W[n][0] = 0.0;
    } // Eqn 3.30


    //Calculate tesseral and sectoral terms
    for (int m = 1; m < N_SPHERICAL + 1; m++){
        // Eqn 3.29
        V[m][m] = (2*m-1)*(x0*V[m-1][m-1] - y0*W[m-1][m-1]);
        W[m][m] = (2*m-1)*(x0*W[m-1][m-1] + y0*V[m-1][m-1]);

        // n=m+1 (only one term)
        if (m <= N_SPHERICAL){
            V[m+1][m] = (2*m+1)*z0*V[m][m];
            W[m+1][m] = (2*m+1)*z0*W[m][m];
        }

        for (int n = m+2; n<N_SPHERICAL+1; n++){
            V[n][m] = ((2*n-1)*z0*V[n-1][m]-(n+m-1)*rho*V[n-2][m])/(n-m);
            W[n][m] = ((2*n-1)*z0*W[n-1][m]-(n+m-1)*rho*W[n-2][m])/(n-m);
        }
    }

    // Calculate potential
    float C = 0; // Cnm coeff
    float S = 0; // Snm coeff
    float N = 0; // normalisation number
    float U = 0; //potential
    for (int m=0; m<N_SPHERICAL+1; m++){
        for (int n = m; n<N_SPHERICAL+1; n++){
            C = 0;
            S = 0;
            if (m==0){
                N = sqrt(2*n+1);
                C = N*coeff[n][0];
                U = C*V[n][0];
            }
            else {
                N = sqrt((2)*(2*n+1)*facprod(n,m));
                C = N*coeff[n][m];
                S = N*coeff[N_SPHERICAL-n][N_SPHERICAL-m+1];
            }
            U = U + C*V[n][m] + S*W[n][m];
        }
    }
    U = U*mhu/R_eq;
//    cout<<"U in the function  "<<U<<'\n';

    return U;
}
//]

void get_grav_pot(){

    float coeff[N_SPHERICAL+1][N_SPHERICAL+2];

    for (int i=0; i<N_SPHERICAL +1; i++){
        for (int j=0; j<N_SPHERICAL+2; j++)
            coeff[i][j]=0;
    }
    get_coefficients(coeff);

//    for (int i=0; i<N_SPHERICAL +1; i++){
//        cout<<"\n \n";
//        for (int j=0; j<N_SPHERICAL+2; j++)
//            cout<<coeff[i][j]<<'\t';
//    }

    float R_vec[3];
    float U[vertices_length]; // potential at each point

    for (int i=0; i<vertices_length; i++){
        R_vec[0] = vertices[i].x;
        R_vec[1] = vertices[i].y;
        R_vec[2] = vertices[i].z;

        U[i] = spherical_harmonics(coeff, R_vec);
    }
}

void free_cpu_memory(){
	free(faces);
	free(faces_copy);
	free(vertices);
	free(vertices_sph);
	free(potential);
	free(cumulative_common_theta_count);
	free(common_thetas_count);
}


