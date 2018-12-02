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
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <stdio.h>
// #include "cpu.cpp"
// using namespace std;

//init_zero: initilize all value to 0
void init_zero(float a[], int len);

//init_zero: initilize all value to 0
void init_zero(int a[], int len);

// makeSPDmatrix : make positive symmetric definite matrix
void makeSPDmatrix(float A[], int row, int colume);

// makeSPDmatrix_sparse : make positive symmetric definite sparse matrix
void makeSPDmatrix_sparse(float A[], int row, int colume);

// CompareDoubles2 : this function took two float number and check it. It is equal or not with EPSILON = 1e-3
bool comparefloat (float A, float B);

// print_matrix : It print the matrix
void print_matrix(float m[], int row, int colume);

// print_vector : It print the vector of float type
void print_vector(int n, float v[]);

// print_vector : It print the vector of int type
void print_vector(int n, int v[]);

// matrix_x_vector : this function multiply matrix and vector
int matrix_x_vector(int row, int colume, float y[], float x[], float result[]);

// matrix_x_matrix : this function multiply matrix and matrix
int matrix_x_matrix(int row, int colume, float y[], float x[], float result[], int colume2nd);

// makeMatrix : this function make matrix we put empty matrix and it put random number
void makeMatrix(float x[], int row, int colume);

// makeVector : this function make vector we put empty vector and it put random number
void makeVector(float x[], int len);

// compareTwoMatrix : this function compare two matrices are equal or not
int compareTwoMatrix(float x[], float y[], int row, int colume);

// compareTwoVector : this function compare two vector are equal or not
int compareTwoVector(float x[], float y[], int len);

// makeSparseMatrix : this function make matrix we put empty matrix and it put random number on random position and mostly are zero
void makeSparseMatrix(float x[], int row, int colume);

// sparse_matrix_x_vector : this function multiply sparse matrix with vector
int sparse_matrix_x_vector(int row, int size, float b[], int row_number[], int col_number[], float a[], float ab[]);

// makeSparseMatrix_To_vectors : this function get matrix and make a CSR format
void makeSparseMatrix_To_vectors(float x[], int row, int colume, float non_zero[], int col_number[], int col[], int size);

// size_non_zero_marix : this function return how many non zero number in sparse matrix
int size_non_zero_marix(float x[], int row, int colume);

// sumMatrix : it calculate sum of matrix element
float sumMatrix(float x[], int row, int colume);

// sumVector : it calculate sum of vector element
float sumVector(float x[], int size);

// checkInArray : it check element in array or not
bool checkInArray(int colNumber, int m, int x[], int ss[]);

// sparse_matrix_cpu : It is multipling sparse matrix with sparse matrix where both matrix in CSR format
void sparse_matrix_cpu(float a_non_zero[], int a_col_number[], int a_row[], float b_non_zero[], int b_col_number[], int b_col[], int size, int size_b, int colume2ndmatrix, float ab[], int point[]);

// vectorToMatrix : It input CSR format and return sparse matrix
void vectorToMatrix(int row, int colume2ndmatrix, int resultMatrix_size, float ab[], float ab2[], int point[]);

// matrix_add_or_sub_matrix : It add or subtract the two matrics
int matrix_add_or_sub_matrix(int row, int colume, float y[], float x[], float result[], int colume2nd, int operation);

// vector_vector_multiplication : It is dot product of two vectors
float vector_vector_multiplication(int entries, float v1[], float v2[]);

// relative_error_test: It compare two two float number with EPSILON = 1e-2
bool relative_error_test(float A, float B);

// matrixToSparse : It convert sparse matrix to two array 1. non zero value 2. the position of value in matrix
void matrixToSparse(float x[], float result[], int point[], int row, int colume, int size);

// compareTwoSparseVector : It compare two CSR format matrix
int compareTwoSparseVector(float x[], float y[], int point [], int point2 [], int len);

// checkSparseInArray : It check if the value in array for CSR format
bool checkSparseInArray(int m, float x[] , float y , int point, int point2[] );

//sparse_Add_Sub_Matrix : It add or sub two sparse matrix in CSR format
void sparse_Add_Sub_Matrix(int size_a, int size_b, int row, float a_non_zero[], int a_col_number[], int a_row[], float b_non_zero[], int b_row_number[], int b_col[], int colume2ndmatrix, float ab[], int point[], int operation);

// Gauss-elmination : It is gauss elmination method who solve linear equation
void Gauss_elmination_cpu(float a[], float b[], float x[], int n);

// jacobi_method : It is Jacobi method who solve linear equation
void jacobi_method_cpu(float a[], float x[], float b[], float x_new[], int n, int num_iter);

//relexation_interpolation_reduction_matrix: RAI R is reduction, I is interpolation, A is matrix
void interpolation_reduction_matrix(float m[], int len, float result2[]);

//reduction_reduction_vector: it is reduction of vector
void reduction_vector(float v[], int len, float result[]);

//reduction_interpolation_vector: it is interpolation of the vector
void reduction_interpolation_vector(float v[], int len, float result[]);

// add_sub_vector : it add or subtract two vectors
void add_sub_vector(float x[], float y[], float result[], int size,  int oper);

//jacobi_method_cpu_sparse_matrix: jacobi method for sparse matrix
void jacobi_method_cpu_sparse_matrix(float a[], int col[], int row[], float b[], float x[], float x_new[], int n, int num_iter, int size);

//interpolation_reduction_matrix_sparse_matrix: interpolation and reduction of matrix in cpu M[10][10] => M[5][5]
int interpolation_reduction_matrix_sparse_matrix(float m[], int col[], int row[], int size, int len, float result2[], int result2_row[], int result2_col[]);

//norm : return the norm of verctor ||x||
float norm(float v[], int size);

//reduction_reduction_vector: it is reduction of vector
void reduction_vector_sparse(float v[], int len, float result[]);










