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
//#include "cstdlib"
#include "sparse_matrix.hpp"
using namespace std;

void sparse_Matrix_Matrix_Multiplication(int row, int colume, int row2ndmatrix, int colume2ndmatrix, occa::memory o_a, occa::memory o_b, occa::memory o_ab, occa::device device) {

	occa::memory o_a_non_zero, o_a_col_number, o_a_row, o_b_non_zero, o_b_row_number, o_b_col;
	float *a  = new float[row * colume];
	float *b  = new float[row2ndmatrix * colume2ndmatrix];
	// float *ab3 = new float[row * colume2ndmatrix];

	makeSparseMatrix(a, row, colume);
	makeSparseMatrix(b, row2ndmatrix, colume2ndmatrix);

	//    get length of a none zero elements
	int size_a = size_non_zero_marix(a, row, colume);

	float * a_non_zero = new float[size_a];
	int * a_col_number = new int[size_a];
	//    int * a_row_number = new int[row+1];
	int * a_row = new int[size_a];


	makeSparseMatrix_To_vectors(a, row, colume, a_non_zero, a_col_number, a_row, size_a);

	//    get length of a none zero elements
	int size_b = size_non_zero_marix(b, row2ndmatrix, colume2ndmatrix);

	float * b_non_zero = new float[size_b];
	int * b_row_number = new int[size_b];
	int * b_col = new int[size_b];

	float *ab = new float[row * colume2ndmatrix];

	int resultMatrix_size = 2 * (size_a + size_b);

	float *ab3 = new float[resultMatrix_size];
	float *ab2 = new float[resultMatrix_size];
	int *point = new int[resultMatrix_size];
	int *point2 = new int[resultMatrix_size];
	makeSparseMatrix_To_vectors(b, row2ndmatrix, colume2ndmatrix, b_non_zero, b_col, b_row_number, size_b);

	occa::kernel sparse_matrix_X_Matrix_kernal;

	// Allocate memory on the device
	o_a_non_zero  = device.malloc(size_a * sizeof(float));
	o_a_col_number  = device.malloc(size_a * sizeof(int));
	o_a_row  = device.malloc(size_a * sizeof(int));
	o_b_non_zero = device.malloc(size_b * sizeof(float));
	o_b_row_number  = device.malloc(size_b * sizeof(int));
	o_b_col  = device.malloc(size_b * sizeof(int));
	o_ab  = device.malloc((row * colume2ndmatrix) * sizeof(float));

	// Compile the kernel at run-time
	sparse_matrix_X_Matrix_kernal = device.buildKernel("matrix_X_matrix.okl", "sparse_matrix_x_Matrix");

	// Copy memory to the device
	o_a_non_zero.copyFrom(a_non_zero);
	o_a_col_number.copyFrom(a_col_number);
	o_a_row.copyFrom(a_row);
	o_b_non_zero.copyFrom(b_non_zero);
	o_b_row_number.copyFrom(b_row_number);
	o_b_col.copyFrom(b_col);



	clock_t start;
	double duration, duration2;
	start = clock();


	// Launch device kernel
	sparse_matrix_X_Matrix_kernal(size_a, size_b, row, o_a_non_zero, o_a_col_number, o_a_row, o_b_non_zero, o_b_row_number, o_b_col, colume2ndmatrix, o_ab);

	duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
	cout << "timer for GPU printf: " << duration << '\n';

	// Copy result to the host
	o_ab.copyTo(ab);
	start = clock();

	sparse_matrix_cpu(a_non_zero, a_col_number, a_row, b_non_zero, b_row_number, b_col, size_a, size_b, colume2ndmatrix, ab2, point);

	duration2 = ( clock() - start ) / (double) CLOCKS_PER_SEC;
	cout << "timer for CPU printf: " << duration2 << '\n';

	cout << "speedup  : " << duration2 / duration << '\n';

	matrixToSparse(ab, ab3, point2, row, colume2ndmatrix, resultMatrix_size);

	compareTwoSparseVector(ab3, ab2, point2, point, resultMatrix_size);

	float sum = sumVector(ab3, resultMatrix_size);
	cout << "sum for gpu printf: " << sum << '\n';

	float sum2 = sumVector(ab2, resultMatrix_size);
	cout << "sum for cpu printf: " << sum2 << '\n';



//     Free host memory
	delete [] a;
	delete [] a_non_zero;
	delete [] a_row;
	delete [] a_col_number;
	delete [] b;
	delete [] b_non_zero;
	delete [] b_row_number;
	delete [] b_col;
	delete [] ab;
	delete [] ab2;
	delete [] point;
	delete [] point2;

}


void sparse_Matrix_Vector_Multiplication(int row, int colume, occa::memory o_a, occa::memory o_b, occa::memory o_ab, occa::device device) {

	occa::memory o_c, o_d, o_row_number;
	float *a  = new float[row * colume];
	float *b  = new float[colume];
	float *ab = new float[row];
	float *ab2 = new float[row];

	makeSparseMatrix(a, row, colume);
	makeVector(b, colume);

	//    get length of a none zero elements
	int size = size_non_zero_marix(a, row, colume);


	float * a_non_zero = new float[size];
	int * a_col_number = new int[size];
	int * a_row = new int[size];
	int * a_row_number = new int[row + 1];
	int k = 0;
	a_row_number[k] = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < colume; j++) {
			if (a[i * colume + j] != 0) {
				k++;
			}
		}
		a_row_number[i + 1] = k;
	}


	makeSparseMatrix_To_vectors(a, row, colume, a_non_zero, a_col_number, a_row, size);

	occa::kernel sparse_matrix_X_Vector_kernal;

	// Allocate memory on the device
	o_a  = device.malloc(size * sizeof(float));
	o_b  = device.malloc(colume * sizeof(float));
	o_c  = device.malloc(size * sizeof(int));
	o_d  = device.malloc(size * sizeof(int));
	o_ab = device.malloc((row) * sizeof(float));
	o_row_number = device.malloc((row + 1) * sizeof(int));


	// Compile the kernel at run-time
	sparse_matrix_X_Vector_kernal = device.buildKernel("matrix_X_matrix.okl",
	                                "sparse_matrix_x_Vectors"
	                                                  );

	// Copy memory to the device
	o_a.copyFrom(a_non_zero);
	o_b.copyFrom(b);
	o_c.copyFrom(a_row);
	o_d.copyFrom(a_col_number);
	o_row_number.copyFrom(a_row_number);

	clock_t start;
	double duration, duration2;
	start = clock();

	// Launch device kernel
	sparse_matrix_X_Vector_kernal(size, row, o_a, o_b, o_c, o_d, o_ab, o_row_number);

	duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
	cout << "timer for GPU printf: " << duration << '\n';

	// Copy result to the host
	o_ab.copyTo(ab);

	start = clock();

	sparse_matrix_x_vector(row, size, b, a_row, a_col_number, a_non_zero, ab2);

	duration2 = ( clock() - start ) / (double) CLOCKS_PER_SEC;
	cout << "timer for CPU printf: " << duration2 << '\n';

	cout << "speedup  : " << duration2 / duration << '\n';

	compareTwoVector(ab, ab2, row);

	// Free host memory
	delete [] a;
	delete [] a_non_zero;
	delete [] a_col_number;
	delete [] a_row;
	delete [] a_row_number;
	delete [] b;
	delete [] ab;
	delete [] ab2;

}

void sparse_Add_Sub_Matrix(int row, int colume, int row2ndmatrix, int colume2ndmatrix, occa::memory o_a, occa::memory o_b, occa::memory o_ab, occa::device device, int operation) {

	occa::memory o_a_non_zero, o_a_col_number, o_a_row, o_b_non_zero, o_b_row_number, o_b_col, o_point;
	float *a  = new float[row * colume];
	float *b  = new float[row2ndmatrix * colume2ndmatrix];
	float *ab3 = new float[row * colume2ndmatrix];
	makeSparseMatrix(a, row, colume);
	makeSparseMatrix(b, row2ndmatrix, colume2ndmatrix);

	//    get length of a none zero elements
	int size_a = size_non_zero_marix(a, row, colume);

	float * a_non_zero = new float[size_a];
	int * a_col_number = new int[size_a];
	int * a_row = new int[size_a];


	makeSparseMatrix_To_vectors(a, row, colume, a_non_zero, a_col_number, a_row, size_a);

	//    get length of a none zero elements
	int size_b = size_non_zero_marix(b, row2ndmatrix, colume2ndmatrix);

	float * b_non_zero = new float[size_b];
	int * b_row_number = new int[size_b];
	int * b_col = new int[size_b];


	//    int resultMatrix_size = 2*(size_a+size_b);
	int resultMatrix_size = size_a + size_b;

	float *ab = new float[resultMatrix_size];
	int *point = new int[resultMatrix_size];
	float *ab2 = new float[resultMatrix_size];
	int *point2 = new int[resultMatrix_size];
	makeSparseMatrix_To_vectors(b, row2ndmatrix, colume2ndmatrix, b_non_zero, b_col, b_row_number, size_b);

	occa::kernel sparse_Add_Sub_Matrix_kernal;

	// Allocate memory on the device
	o_a_non_zero  = device.malloc(size_a * sizeof(float));
	o_a_col_number  = device.malloc(size_a * sizeof(int));
	o_a_row  = device.malloc(size_a * sizeof(int));
	o_b_non_zero = device.malloc(size_b * sizeof(float));
	o_b_row_number  = device.malloc(size_b * sizeof(int));
	o_b_col  = device.malloc(size_b * sizeof(int));
	o_ab  = device.malloc((resultMatrix_size) * sizeof(float));
	o_point  = device.malloc((resultMatrix_size) * sizeof(int));

	// Compile the kernel at run-time
	sparse_Add_Sub_Matrix_kernal = device.buildKernel("matrix_X_matrix.okl", "sparse_Add_Sub_Matrix");

	// Copy memory to the device
	o_a_non_zero.copyFrom(a_non_zero);
	o_a_col_number.copyFrom(a_col_number);
	o_a_row.copyFrom(a_row);
	o_b_non_zero.copyFrom(b_non_zero);
	o_b_row_number.copyFrom(b_row_number);
	o_b_col.copyFrom(b_col);



	clock_t start;
	double duration, duration2;
	start = clock();


	// Launch device kernel
	sparse_Add_Sub_Matrix_kernal(size_a, size_b, row, o_a_non_zero, o_a_col_number, o_a_row, o_b_non_zero, o_b_row_number, o_b_col, colume2ndmatrix, o_ab, o_point, operation);


	duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
	cout << "timer for GPU printf: " << duration << '\n';

	// Copy result to the host
	o_ab.copyTo(ab);
	o_point.copyTo(point);
	start = clock();

	sparse_Add_Sub_Matrix(size_a, size_b, row, a_non_zero, a_col_number, a_row, b_non_zero, b_row_number, b_col, colume2ndmatrix, ab2, point2, operation);
	// matrix_add_or_sub_matrix(row, colume, b, a, ab3, colume2ndmatrix, operation);

	duration2 = ( clock() - start ) / (double) CLOCKS_PER_SEC;
	cout << "timer for CPU printf: " << duration2 << '\n';

	cout << "speedup  : " << duration2 / duration << '\n';

	compareTwoSparseVector(ab, ab2, point2, point, resultMatrix_size);

	float sum = sumVector(ab, resultMatrix_size);
	cout << "sum for gpu printf: " << sum << '\n';

	float sum2 = sumVector(ab2, resultMatrix_size);
	cout << "sum for cpu printf: " << sum2 << '\n';


	//     Free host memory
	delete [] a;
	delete [] a_non_zero;
	delete [] a_row;
	delete [] a_col_number;
	delete [] b;
	delete [] b_non_zero;
	delete [] b_row_number;
	delete [] b_col;
	delete [] ab;
	delete [] ab2;
	delete [] ab3;
	delete [] point;

}
