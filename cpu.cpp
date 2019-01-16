/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 Satish Kumar
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
//#include <iostream>
//#include <ctime>
//#include <cstdlib>
//#include <cmath>
#include "cpu.hpp"
using namespace std;

void init_zero(float a[], int len) {
	for (int i = 0; i < len; i++) {
		a[i] = 0;
	}
}

void init_zero(int a[], int len) {
	for (int i = 0; i < len; i++) {
		a[i] = 0;
	}
}

void makeSPDmatrix(float A[], int row, int colume) {
	srand (time(NULL));
//srand (32424);
	for (int i = 0; i < row; i++)
	{
		A[i * colume + i]  = row * 2 * (rand() % 10 + 1) ;

		for (int j = i + 1; j < colume; j++) {
			if (i != j) {

				A[i * colume + j] = -1 * ((rand() % 10 + 1));
				A[j * colume + i] = A[i * colume + j];
			}
		}
	}
}

void makeSPDmatrix_sparse(float A[], int row, int colume) {
	srand (time(NULL));
//    srand (32424);
	for (int i = 0; i < row; i++)
	{
		A[i * colume + i]  = row * 2 * (rand() % 10 + 1) ;

		for (int j = i + 1; j < colume; j++) {
			if (i != j) {
				if ((i + j) % 5 == 0) {
					A[i * colume + j] = -1 * ((rand() % 10 + 1));
					A[j * colume + i] = A[i * colume + j];
				}
				else {
					A[i * colume + j] = 0;
					A[j * colume + i] = 0;
				}
			}
		}
	}
}

bool comparefloat (float A, float B)
{
	float EPSILON = 1e-3;
	if (copysign(1, A)*copysign(1, B) < 0) {
		return false;
	}
	float diff = fabs (A - B);
	return (diff < EPSILON) ;
}


void print_matrix(float m[], int row, int colume)
{
	int i, j;
	printf("\nMatrix Given\n");
	for (i = 0; i < row; i++)
	{
		for (j = 0; j < colume; j++) {
			printf("%3f     ", m[j + i * colume]);
		}
		printf("\n");
	}
}
void print_vector(int n, float v[])
{
	int i;
	printf("\nVector Given\n");
	for (i = 0; i < n; i++) {
		printf(" %3f ", v[i]);
	}
	printf("\n");
}
void print_vector(int n, int v[])
{
	int i;
	printf("\nVector Given\n");
	for (i = 0; i < n; i++)
		printf("  %3d  ", v[i]);
	printf("\n");
}
int matrix_x_vector(int row, int colume, float y[], float x[], float result[])
{
	int i, j;
	for (i = 0; i < row; i++)
	{
		result[i] = 0;
		for (j = 0; j < colume; j++)
		{
			result[i] +=  (x[j + i * colume] * y[j]);

		}

	}
	return 0;
}

int matrix_x_matrix(int row, int colume, float y[], float x[], float result[], int colume2nd)
{
	for (int i = 0; i < row; i++)
	{

		for (int j = 0; j < colume2nd; j++)
		{
			result[j + i * colume2nd] = 0;
			for (int k = 0; k < colume; k++) {
				result[j + i * colume2nd] +=  (x[k + i * colume] * y[j + k * colume2nd]);
			}
		}
	}
	return 0;
}


void makeMatrix(float x[], int row, int colume) {
	srand (time(NULL));
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < colume; ++j) {
			x[i * colume + j]  = rand() % 10 + 1 ;
		}
	}
}


void makeVector(float x[], int len) {
	srand (time(NULL));
//    srand (32424);
	for (int i = 0; i < len; ++i) {
		x[i]  = sin((3.14  * i )/ len);
		// x[i]  = -500;
	}
}

int compareTwoMatrix(float x[], float y[], int row, int colume) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < colume; j++) {
			if (!comparefloat(x[i * colume + j], y[i * colume + j])) {
				cout << "it is not same do something  " << x[i * colume + j] << " not same " << y[i * colume + j] << endl;

				return 0;
			}
		}
	}
	cout << "it is same " << endl;
	return 0;
}

int compareTwoVector(float x[], float y[], int len) {

	for (int i = 0; i < len; i++) {
		if (!comparefloat(x[i], y[i])) {
			cout << "it is not same do something  " << x[i] << " not same " << y[i] << endl;

			return 0;
		}
	}
	cout << "it is same " << endl;
	return 0;
}

void makeSparseMatrix(float x[], int row, int colume) {

	for (int i = 0; i < row * colume; ++i) {
		if (i % 3 == 0) {
			x[i]  = (rand() % 100 - 10) / 1.234567;

		}
		else {
			x[i] = 0;
		}
	}
}
int sparse_matrix_x_vector(int row, int size, float b[], int row_number[], int col_number[], float a[], float ab[])
{

	for (int i = 0; i < row; ++i) {
		ab[i] = 0;
	}
	for (int i = 0; i < size; i++) {
		ab[row_number[i]] += a[i] * b[col_number[i]];
	}
	return 0;
}



void makeSparseMatrix_To_vectors(float x[], int row, int colume, float non_zero[], int rowe[], int col[], int size) {
	int k = 0;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < colume; j++) {
			if (x[i * colume + j] != 0) {
				non_zero[k] = x[i * colume + j];
				rowe[k] = j;
				col[k] = i;
				k++;
			}
		}
	}
}

int size_non_zero_marix(float x[], int row, int colume) {
	int size = 0;
	for (int i = 0; i < row * colume; i++) {
		if (x[i] != 0.0) {
			size++;
		}
	}
	return size;
}


float sumMatrix(float x[], int row, int colume) {
	float sum = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < colume; j++) {
			sum += x[i * colume + j];
		}
	}
	return sum;
}

float sumVector(float x[], int size) {
	float sum = 0;
	for (int i = 0; i < size; i++) {
		sum += x[i];
	}
	return sum;
}


bool checkInArray(int colNumber, int m, int x[], int ss[]) {
	for (int p = 0; p < m; p++) {
		if (x[p] == colNumber) {
			ss[0] = p;
			return true;
		}
	}
	return false;
}

void sparse_matrix_cpu(float a_non_zero[], int a_col_number[], int a_row[], float b_non_zero[], int b_col_number[], int b_col[], int size, int size_b, int colume2ndmatrix, float ab[], int point[]) {
	for (int i = 0; i < 2 * (size + size_b); i++) {
		ab[i] = 0;
		point[i] = 0;
	}
	int m = 0;

	for (int j = 0; j < size; j++) {
		for (int k = 0; k < size_b; k++) {
			if (a_col_number[j] == b_col_number[k]) {
				int s[1];
				if (checkInArray(a_row[j]*colume2ndmatrix + b_col[k], m, point, s)) {
					ab[s[0]] += a_non_zero[j] * b_non_zero[k];
				}
				else {

					ab[m] = a_non_zero[j] * b_non_zero[k];

					point[m] = a_row[j] * colume2ndmatrix + b_col[k];

					m++;
				}
			}
		}
	}

}

void vectorToMatrix(int row, int colume2ndmatrix, int resultMatrix_size, float ab[], float ab2[], int point[]) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < colume2ndmatrix; j++) {
			ab2[i * colume2ndmatrix + j] = 0;
		}
	}
	for (int i = 0; i < resultMatrix_size; i++) {

		ab2[point[i]] += ab[i];
	}
}

int matrix_add_or_sub_matrix(int row, int colume, float y[], float x[], float result[], int colume2nd, int operation)
{
	int i, j;
	printf("\nResulted Matrix of [M]*[M]\n");
	for (i = 0; i < row; i++)
	{
		for (j = 0; j < colume; j++)
		{
			result[j + i * colume] =  x[j + i * colume] + (operation * y[j + i * colume]);
		}
	}
	return 0;
}


float vector_vector_multiplication(int entries, float v1[], float v2[])
{
	float result = 0;

	for (int i = 0; i < entries; i++) {
		result += v1[i] * v2[i];
	}
	return result;
}
bool relative_error_test (float A, float B) {

	float EPSILON = 1e-2;
	float numerical_zero = 1.0e-10;

	if (abs(B) <= numerical_zero and abs(A) <= numerical_zero) {
		return true;
	}

	if (abs(B) <= numerical_zero or abs(A) <= numerical_zero) {
		return false;
	}

	float error = abs(1.0 - A / B);

	if (error < EPSILON)
		return true;
	else
		return false;
}

void matrixToSparse(float x[], float result[], int point[], int row, int colume, int size)
{
	for (int i = 0; i < size; i++) {
		result[i] = 0;
		point[i] = 0;
	}
	int m = 0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < colume; j++)
		{
			if (x[j + i * colume] != 0)
			{
				result[m] =  x[j + i * colume];
				point[m] = j + i * colume;
				m++;
			}

		}
	}
}
bool checkSparseInArray(int m, float x[] , float y , int point, int point2[] ) {
	for (int p = 0; p < m; p++) {
		if (comparefloat(x[p], y) && point == point2[p]) {
			return true;
		}
	}
	return false;
}
int compareTwoSparseVector(float x[], float y[], int point2 [], int point [], int len) {
	for (int i = 0; i < len; i++) {
		if (!checkSparseInArray(len, x , y[i] , point[i], point2)) {
			cout << "it is not same do something" << endl;
			return 0;
		}

	}
	cout << "it is same " << endl;
	return 0;
}

void sparse_Add_Sub_Matrix(int size, int size_b, int row, float a_non_zero[], int a_col_number[], int a_row[], float b_non_zero[], int b_row_number[], int b_col[], int colume2ndmatrix, float ab[], int point[], int operation) {
	for (int i = 0; i < size + size_b; i++) {
		ab[i] = 0;
		point[i] = 0;
	}
	for (int i = 0; i < size; i++) {
		ab[i] = a_non_zero[i];
		point[i] = a_row[i] * colume2ndmatrix + a_col_number[i];
	}
	for (int i = 0; i < size_b; i++) {
		ab[i + size] = operation * b_non_zero[i];
		point[i + size] = b_row_number[i] * colume2ndmatrix + b_col[i];
	}
	for (int i = 0; i < size + size_b - 1; i++) {
		for (int j = 0; j < size + size_b; j++) {
			if (i != j && j > i) {
				if (point[i] == point[j]) {
					ab[i] += ab[j];
					ab[j] = 0;
					point[j] = 0;
				}
			}
		}
	}
}

void Gauss_elmination_cpu(float a[], float b[], float d[], int n) {

	//********* Forward elimination process**************//
	for (int i = 0; i < n - 1; i++) {
		for (int k = i + 1; k < n; k++) {
			float c = a[k * n + i] / a[i * n + i];
			for (int j = i; j < n; j++) {
				a[k * n + j] = a[k * n + j] - (c * a[i * n + j] );
			}
			b[k] = b[k] - (c * b[i] );
		}
	}

	//***************** Backward Substitution method****************//

	for (int i = n - 1; i >= 0; i--) {
		d[i] = b[i];
		for (int j = i + 1; j < n; j++) {
			if (j != i) {
				d[i] = d[i] - a[i * n + j] * d[j];

			}
		}
		d[i] = d[i] / a[i * n + i];
	}

}

void gauss_seidel_method_cpu(float a[], float x[], float b[], float x_new[], int n, int num_iter) {

	for (int k = 0; k < num_iter; k++) {
		for (int i = 0; i < n; i++ ) {
			float sum = 0;
			for (int j = 0; j < n; j++ ) {
				if ( j != i ) {
					sum += (a[i * (n) + j] * x[j]);
				}
			}
			x[i] = (b[i] - sum ) / a[i + i * (n)];
		}
	}
}

void jacobi_method_cpu(float a[], float x[], float b[], float x_new[], int n, int num_iter) {

	for (int k = 0; k < num_iter; k++) {
		for (int i = 0; i < n; i++ ) {
			float sum = 0;
			for (int j = 0; j < n; j++ ) {
				if ( j != i ) {
					sum += (a[i * (n) + j] * x[j]);
				}
			}
			x_new[i] = (b[i] - sum ) / a[i + i * (n)];
		}
		for (int i = 0; i < n; i++) {
			x[i] = x_new[i];
		}
	}
}

void reduction_interpolation_vector(float v[], int len, float result[]) {
	for (int i = 0; i <= len; i++) {
		if (i == 0) {
			result[i] = 0.5 * v[0];
			result[i + 1] = 0.5 * 2 * v[0];
		} else {
			if ((2 * i + 1) < (2 * len)) {
				result[2 * i + 1] = 0.5 * 2 * v[i];
				result[2 * i] = 0.5 * (v[i] + v[i - 1]);
			} else {

				result[2 * i] = 0.5 * v[i];
			}
		}

	}
}
void reduction_vector(float v[], int len, float result[]) {
	//    float result[len];
	for (int i = 0; i < len; i++) {
		if (((2 * i) + 2) < 2 * len) {
			result[i] = 0.25 * (v[(2 * i)] + 2 * v[(2 * i) + 1] + v[(2 * i) + 2]);
		}
		else if (((2 * i) + 1) < 2 * len) {
			result[i] = 0.25 * (v[(2 * i)] + 2 * v[(2 * i) + 1]);
		} else {
			result[i] = 0.25 * (v[(2 * i)]);
		}
	}
}
void interpolation_reduction_matrix(float m[], int len, float result2[]) {

	int len2 = len / 2;

	float *result  = new float[len2 * len];

	init_zero(result, len2 * len);

	for (int i = 0; i < len2; i++) {
		for (int j = 0; j < len; j++) {
			if (i + 1 == len2 && len % 2 != 1) {
				result[j * len2 + i] = 0.5 * (m[i * 2 + (j * len)] + 2 * m[i * 2 + (j * len) + 1]);
			}
			else {
				result[j * len2 + i] = 0.5 * (m[i * 2 + (j * len)] + 2 * m[i * 2 + (j * len) + 1] + m[i * 2 + (j * len) + 2]);
			}
		}
	}
	for (int i = 0; i < len2; i++) {
		for (int j = 0; j < len2; j++) {
			if (i + 1 == len2 && len % 2 != 1) {
				result2[i * len2 + j] = 0.25 * (result[j + 2 * (i * len2)] + 2 * result[j + len2 + 2 * (i * len2)]);
			}
			else {
				result2[i * len2 + j] = 0.25 * (result[j + 2 * (i * len2)] + 2 * result[j + len2 + 2 * (i * len2)] + result[j + 2 * len2 + 2 * (i * len2)]);
			}
		}
	}
	delete [] result;
}

void add_sub_vector(float x[], float y[], float result[], int size,  int oper) {
	for (int i = 0; i < size; i++) {
		result[i] = x[i] + oper * y[i];
	}
}


void jacobi_method_cpu_sparse_matrix(float a[], int col[], int row[], float b[], float x[], float x_new[], int n, int num_iter, int size) {


	for (int k = 0; k < num_iter; k++) {
		init_zero(x_new, n);

		for (int i = 0; i < size; i++) {
			if (row[i] != col[i]) {
				x_new[row[i]] += (a[i] * x[col[i]]);
			}
		}
		for (int i = 0; i < n; i++) {
			x[i] = 0;
		}
		for (int i = 0; i < size; i++) {
			if (row[i] == col[i] && a[i] != 0) {
				x[row[i]] = (b[col[i]] - x_new[row[i]]) / a[i];
			}

		}
	}
}

int interpolation_reduction_matrix_sparse_matrix(float m[], int col[], int row[], int size, int len, float result2[], int result2_row[], int result2_col[]) {


	int len2 = len / 2;

	float *result  = new float[len2 * len];

	init_zero(result, len2 * len);

	for (int i = 0 ; i < len2; i++) {
		for (int j = 0; j < size; j++) {
			if (i + 1 == len2 && len % 2 != 1) {
				if ((row[j] + col[j]*len) == (2 * i + (col[j] * len))) {
					result[i + col[j]* len2] +=  0.5 * m[j];
				}
				else if ((row[j] + col[j]*len) == (2 * i + (col[j] * len)) + 1) {
					result[i + col[j]* len2] +=  0.5 * 2 * m[j];
				}
			}
			else {
				if ((row[j] + col[j]*len)  == (2 * i + (col[j] * len))) {
					result[i + col[j]* len2] +=  0.5 * m[j];
				}
				else if ((row[j] + col[j]*len)  == (2 * i + (col[j] * len)) + 1) {
					result[i + col[j]* len2] +=  0.5 * 2 * m[j];
				}
				else if ((row[j] + col[j]*len)  == (2 * i + (col[j] * len)) + 2) {
					result[i + col[j]* len2] +=  0.5 * m[j];
				}
			}
		}
	}

	int point = 0;
	for (int i = 0; i < len2; i++) {
		for (int j = 0; j < len2; j++) {
			if (i + 1 == len2 && len % 2 != 1) {
				result2[point] = 0.25 * (result[j + 2 * (i * len2)] + 2 * result[j + len2 + 2 * (i * len2)]);
				result2_col[point] = j;
				result2_row[point] = i;
				if (result2[point] != 0) {
					point++;
				}
			}
			else {
				result2[point] = 0.25 * (result[j + 2 * (i * len2)] + 2 * result[j + len2 + 2 * (i * len2)] + result[j + 2 * len2 + 2 * (i * len2)]);
				result2_col[point] = j;
				result2_row[point] = i;
				if (result2[point] != 0) {
					point++;
				}
			}
		}
	}
	delete [] result;
	return point;
}

float norm(float v[], int size) {
	float sum = 0;
	for (int k = 0; k < size; k++) {
		sum += v[k] * v[k];
	}
	return sqrt(sum);
}

int mx_i(int m , int n, int i, int j ){
	return j*m+i;
}

void laplace1D ( int m, int n, float a[] ) {
	for (int j = 0; j < n; j++ ) {
		for (int i = 0; i < m; i++ ) {
			if ( j == i - 1 ) {
				a[i + j * m] = -1.0;
			}
			else if ( j == i ) {
				a[i + j * m] = 2.0;
			}
			else if ( j == i + 1 ) {
				a[i + j * m] = -1.0;
			}
			else {
				a[i + j * m] = 0.0;
			}
		}
	}

}

void laplace2D ( int m, int n, float a[] ) {
	for (int j = 0; j < n; j++ ) {
		for (int i = 0; i < m; i++ ) {
			a[i + j * m] = 0;
		}
	}
		for (int i = 0; i < m; i++ ) {
				a[i + i * m] = 4;
				if (i-1 >= 0) {
						a[(i-1)+i*m] = -1;
				}

				if (i+1 < m) {
						a[(i+1)+i*m] = -1;
				}
		}
		int j = 0;
		for (int i =0; i < m/2; i++){
			j = i+(m/2);
			a[i*m+j] = -1;
			a[j*m+i] = -1;
		}
}

void reduction_vector_sparse(float v[], int len2, float result[]) {
	//    float result[len];
	int len = len2 / 2;
	for (int i = 0; i < len; i++) {
		if (((2 * i) + 2) < len2) {
			result[i] = 0.25 * (v[(2 * i)] + 2 * v[(2 * i) + 1] + v[(2 * i) + 2]);
		}
		else if (((2 * i) + 1) < len2) {
			result[i] = 0.25 * (v[(2 * i)] + 2 * v[(2 * i) + 1]);
		} else {
			result[i] = 0.25 * (v[(2 * i)]);
		}
	}
}
