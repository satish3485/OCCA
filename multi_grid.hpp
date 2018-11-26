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
using namespace std;
#include <occa.hpp>
#include "cpu.hpp"
#include <fstream>
// gauss_elmination : This function gauss elmination method
void gauss_elmination(int row, occa::memory o_a, occa::memory o_b, occa::memory o_ab, occa::device device);

// jacobi_method : This function jacobi method
void jacobi_method(int row, occa::memory o_a, occa::memory o_b, occa::memory o_ab, occa::device device, int iteration);


//===============================dense matrix ===============================


//relaxation_interpolation_vector : vector interpolation v[3] => v[7]
void relaxation_interpolation_vector(int row, occa::memory o_a, occa::memory o_b, occa::memory o_ab, occa::device device);

//relaxation_reduction_vector : reduction of vector v[7] => v[3]
void relaxation_reduction_vector(int row, occa::memory o_a, occa::memory o_b, occa::memory o_ab, occa::device device);

//reduction_interpolation_dense_matrix : interpolation and reduction of matrix M[10][10] => M[5][5]
void reduction_interpolation_dense_matrix(int row, occa::memory o_a, occa::memory o_b, occa::memory o_ab, occa::device device);

//multigrid_method : it is multigrid method for dense matrix in CPU
void multigrid_method(float a[], float x[], float b[], int recursion, int row, int alpha);

//dense_Matrix_Vector_Multiplication_call_gpu: matrix vector multiplication call to GPU
void dense_Matrix_Vector_Multiplication_call_gpu(int row, occa::memory o_a, occa::memory o_b, occa::memory o_c, occa::device device);

//add_sub_call_gpu: It add or subtract the vector in GPU call
void add_sub_call_gpu(int row, occa::memory o_a, occa::memory o_b, occa::memory o_c, occa::device device, int oper);

//jacobi_method_call_gpu: It is jacobi method call for multigrid method
void jacobi_method_call_gpu(int row, occa::memory o_a, occa::memory o_b, occa::memory o_c, occa::memory o_d, occa::device device,  int iteration);

//gauss_elmination_call_gpu : It is guass elimination method for multigrid method
void gauss_elmination_call_gpu(int row, occa::memory o_a, occa::memory o_b, occa::memory o_c, occa::device device);

//relaxation_reduction_vector : it is reduction of vector v[7] => v[3]
void relaxation_reduction_vector(int row, occa::memory o_b, occa::memory o_b2h, occa::device device);

//reduction_interpolation_reduction_matrix_call_gpu : interpolation and reduction of matrix in GPU M[10][10] => M[5][5]
void reduction_interpolation_reduction_matrix_call_gpu(int row, occa::memory o_a, occa::memory o_a2h, occa::device device);

//relaxation_interpolation_vector_call_gpu: interpoation of vector in GPU v[3] => v[7]
void relaxation_interpolation_vector_call_gpu(int row, occa::memory o_x2h, occa::memory o_x, occa::device device);

//multigrid_method_gpu : it is multigrid method for dense matrix in GPU
void multigrid_method_gpu(int row, occa::memory o_a, occa::memory o_b, occa::memory o_x, occa::device device, int recursion, int alpha);

//multigrid_method_once : it call once and it call to both method gpu and cpu multigrid method
void multigrid_method_once(int row, occa::memory o_a, occa::memory o_b, occa::memory o_c, occa::device device, int recursion);

//=============================== sparse  matrix ===============================

//sparse_Matrix_Vector_Multiplication_call_gpu : matrix and vector multiplication in gpu
void sparse_Matrix_Vector_Multiplication_call_gpu(int row, int size, occa::memory o_a, occa::memory o_a_row, occa::memory o_a_col, occa::memory o_b, occa::memory o_ab, occa::device device);

//jacobi_method_call_gpu_sparse_matrix : call to jacobi method in GPUs
void jacobi_method_call_gpu_sparse_matrix(int row, occa::memory o_a, occa::memory o_a_col, occa::memory o_a_row, occa::memory o_b, occa::memory o_x, occa::memory o_x_new2h, occa::device device,  int iteration, int size, occa::memory o_roww);

//reduction_interpolation_reduction_sparse_matrix_call_gpu: reduction of matrix M[10][10] => M[5][5]
int reduction_interpolation_reduction_sparse_matrix_call_gpu(int row, occa::memory o_a, occa::memory o_a_col, occa::memory o_a_row,  occa::memory o_result, occa::memory o_result_row, occa::memory o_result_col, occa::device device, int size, occa::memory o_size);

//sparse_vector_to_matrix_gpu_call : convert CSR format to matrix
void sparse_vector_to_matrix_gpu_call(int row, occa::memory o_a, occa::memory o_a_row, occa::memory o_a_col, occa::device device, int size_a, occa::memory o_aa);

//multigrid_method_gpu_sparse_matrix : multigrid method for gpu
void multigrid_method_gpu_sparse_matrix(int row, occa::memory o_a, occa::memory o_a_row, occa::memory o_a_col, occa::memory o_b, occa::memory o_x, occa::device device, int recursion, int alpha, int size_a);

//multigrid_method_sparse_matrix: multigrid method for cpu
void multigrid_method_sparse_matrix(float a_non_zero[], int a_col_number[], int a_row[], float x[], float b[], int recursion, int row, int alpha, int size_a);

//multigrid_method_once_sparse_matrix : it call once and it call to both method gpu and cpu multigrid method
void multigrid_method_once_sparse_matrix(int row, occa::memory o_a, occa::memory o_b, occa::memory o_x, occa::device device, int recursion);

//row_vector : make csr format vector for row
void row_vector(occa::memory o_a_row, int row, int size, occa::memory o_row_number);








































