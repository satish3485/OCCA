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
#include "dense_matrix.hpp"


using namespace std;

void dense_Matrix_Vector_Multiplication(int row, int colume, occa::memory o_a, occa::memory o_b, occa::memory o_ab, occa::device device) {



    float *a  = new float[row * colume];
    float *b  = new float[colume];
    float *ab = new float[row];
    float *ab2 = new float[row];

    makeMatrix(a, row, colume);
    makeVector(b, colume);

    occa::kernel matrix_X_Vector_kernal;

    // Allocate memory on the device
    o_a  = device.malloc((row * colume) * sizeof(float));
    o_b  = device.malloc(colume * sizeof(float));
    o_ab = device.malloc(row * sizeof(float));

    // Compile the kernel at run-time
    matrix_X_Vector_kernal = device.buildKernel("matrix_X_matrix.okl",
                             "matrix_x_Vectors");

    // Copy memory to the device
    o_a.copyFrom(a);
    o_b.copyFrom(b);

    clock_t start;
    double duration, duration2;
    start = clock();

    // Launch device kernel
    matrix_X_Vector_kernal(row, colume, o_a, o_b, o_ab);
    duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "timer for GPU printf: " << duration << '\n';

    // Copy result to the host
    o_ab.copyTo(ab);

    start = clock();
    matrix_x_vector(row, colume, b, a, ab2);
    duration2 = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "timer for CPU printf: " << duration2 << '\n';

    cout << "speedup  : " << duration2 / duration << '\n';

    compareTwoVector(ab, ab2, row);

    // Free host memory
    delete [] a;
    delete [] b;
    delete [] ab;
    delete [] ab2;

}

void dense_Matrix_Multiplication(int row, int colume, int row2ndmatrix, int colume2ndmatrix, occa::memory o_a, occa::memory o_b, occa::memory o_ab, occa::device device, int operation , int mul_And_Add_Sub) {

    float *a  = new float[row * colume];
    float *b  = new float[row2ndmatrix * colume2ndmatrix];
    float *ab = new float[row * colume2ndmatrix];
    float *ab2 = new float[row * colume2ndmatrix];


    makeMatrix(a, row, colume);
    makeMatrix(b, row2ndmatrix, colume2ndmatrix);

    occa::kernel matrix_X_matrix_kernal;


    // Allocate memory on the device
    o_a  = device.malloc(row * colume * sizeof(float));
    o_b  = device.malloc((row2ndmatrix * colume2ndmatrix) * sizeof(float));
    o_ab = device.malloc((row * colume2ndmatrix) * sizeof(float));

    if (mul_And_Add_Sub == 1) {

        // Compile the kernel at run-time
        matrix_X_matrix_kernal = device.buildKernel("matrix_X_matrix.okl",
                                 "matrix_X_matrix");
    }
    else {
        matrix_X_matrix_kernal = device.buildKernel("matrix_X_matrix.okl",
                                 "add_Sub_Matrix_Multiplication");
    }

    // Copy memory to the device
    o_a.copyFrom(a);
    o_b.copyFrom(b);

    clock_t start;
    double duration, duration2;

    start = clock();
    // Launch device kernel
    matrix_X_matrix_kernal(row, colume, colume2ndmatrix, o_a, o_b, o_ab, operation);


    duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;

    cout << "timer for GPU printf: " << duration << '\n';


    // Copy result to the host
    o_ab.copyTo(ab);


    start = clock();

    if (mul_And_Add_Sub == 1) {
        matrix_x_matrix(row, colume, b, a, ab2, colume2ndmatrix);
    }
    else {
        matrix_add_or_sub_matrix(row, colume, b, a, ab2, colume2ndmatrix, operation);
    }

    duration2 = ( clock() - start ) / (double) CLOCKS_PER_SEC;

    cout << "timer for CPU printf: " << duration2 << '\n';
    cout << "speedup  : " << duration2 / duration << '\n';

    compareTwoMatrix(ab, ab2, row, colume2ndmatrix);

//    print_matrix(ab, row, colume);
//    print_matrix(ab2, row, colume);

    // Free host memory
    delete [] a;
    delete [] b;
    delete [] ab;
    delete [] ab2;

}

void vector_reduction(occa::device device, int entries) {

    occa::kernel dot_product;

    // Choosing something not divisible by 256

    int block   = 256;
    int blocks  = (entries + block - 1) / block;

    float *vec      = new float[entries];
    float *vec2      = new float[entries];
    float result_gpu = 0;
    float result = 0;
    float *blockSum = new float[blocks];
    occa::memory o_vec, o_blockSum, o_vec2;

    // Initialize device memory
    for (int i = 0; i < entries; ++i) {
        vec[i] = i * 1.11;
        vec2[i] = i * 1.11;
    }

    for (int i = 0; i < blocks; ++i) {
        blockSum[i] = 0;
    }

    // Allocate memory on the device
    o_vec      = occa::malloc(entries * sizeof(float));
    o_blockSum = occa::malloc(blocks  * sizeof(float));
    o_vec2      = occa::malloc(entries * sizeof(float));


    dot_product = occa::buildKernel("matrix_X_matrix.okl",
                                    "dot_product");

    // Host -> Device
    o_vec.copyFrom(vec);
    o_vec2.copyFrom(vec2);

    clock_t start;
    double duration, duration2;

    start = clock();

    dot_product(entries, o_vec, o_vec2, o_blockSum, block);

    duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;

    cout << "timer for GPU printf: " << duration << '\n';
    // Host <- Device
    o_blockSum.copyTo(blockSum);

    start = clock();

    result = vector_vector_multiplication(entries, vec, vec2);

    duration2 = ( clock() - start ) / (double) CLOCKS_PER_SEC;

    cout << "timer for CPU printf: " << duration2 << '\n';

    cout << "speedup  : " << duration2 / duration << '\n';


    // Finalize the reduction in the host
    for (int i = 0; i < blocks; ++i) {
        result_gpu += blockSum[i];
    }

    if (relative_error_test(result_gpu, result)) {
        cout << "it is same" << endl;
    }
    else {
        cout << "it is not same" << endl;
    }

    // Free host memory
    delete [] vec;
    delete [] vec2;
    delete [] blockSum;

}
