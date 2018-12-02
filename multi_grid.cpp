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
#include "multi_grid.hpp"
using namespace std;


void gauss_elmination(int row, occa::memory o_a, occa::memory o_b, occa::memory o_ab, occa::device device) {


    float *a  = new float[row * row];
    float *b  = new float[row];
    float *b2 = new float[row];
    float *x_cpu = new float[row];
    float *x_gpu = new float[row];

    makeSPDmatrix(a, row, row);
    makeVector(b, row);

    float *a2 = new float[row * row];

    for (int i = 0; i < (row * row); i++) {
        a2[i] = a[i];
    }
    for (int i = 0; i < row; i++) {
        b2[i] = b[i];
    }

    occa::kernel matrix_X_Vector_kernal;
    occa::memory o_c;
    // Allocate memory on the device
    o_a  = device.malloc((row * row) * sizeof(float));
    o_b  = device.malloc(row * sizeof(float));
    o_c  = device.malloc(row * sizeof(float));

    // Compile the kernel at run-time
    matrix_X_Vector_kernal = device.buildKernel("matrix_X_matrix.okl",
                             "gauss_elmination_gpu_forward");

    // Copy memory to the device
    o_a.copyFrom(a);
    o_b.copyFrom(b);
//
    clock_t start;
    double duration, duration2;
    start = clock();

    // Launch device kernel
    for (int i = 0; i < row - 1; i++) {
        matrix_X_Vector_kernal(o_a, o_b, o_c, row, i);
    }
    matrix_X_Vector_kernal = device.buildKernel("matrix_X_matrix.okl",
                             "gauss_elmination_gpu_backward");

    for (int i = row - 1; i >= 0; i--) {
        matrix_X_Vector_kernal(o_a, o_b, o_c, row, i);
    }
    duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "timer for GPU printf: " << duration << '\n';

    // Copy result to the host
    o_c.copyTo(x_gpu);

    start = clock();

    Gauss_elmination_cpu(a2, b2, x_cpu, row);
    duration2 = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "timer for CPU printf: " << duration2 << '\n';

    cout << "speedup  : " << duration2 / duration << '\n';

    compareTwoVector(x_gpu, x_cpu, row);

    // Free host memory
    delete [] a;
    delete [] b;
    delete [] a2;
    delete [] b2;
    delete [] x_cpu;
    delete [] x_gpu;

}


void jacobi_method(int row, occa::memory o_a, occa::memory o_b, occa::memory o_ab, occa::device device, int iteration) {
    float *a  = new float[row * (row)];
    float *x_new2 = new float[row];
    float *x  = new float[row];
    float *x_new  = new float[row];
    float *b = new float[row];

    makeSPDmatrix(a, row, row);
    makeVector(b, row);

    init_zero(x, row );
    init_zero(x_new, row );
    init_zero(x_new2, row );

    occa::kernel matrix_X_Vector_kernal;
    occa::memory o_c, o_d;
    // Allocate memory on the device
    o_a  = device.malloc((row * (row)) * sizeof(float));
    o_b  = device.malloc(row * sizeof(float));
    o_c  = device.malloc(row * sizeof(float));
    o_d  = device.malloc(row * sizeof(float));

    // Compile the kernel at run-time
    matrix_X_Vector_kernal = device.buildKernel("matrix_X_matrix.okl",
                             "jacobi_method_gpu");

    // Copy memory to the device
    o_a.copyFrom(a);
    o_d.copyFrom(x);
    o_b.copyFrom(b);
    //
    clock_t start;
    double duration, duration2;
    start = clock();

//    Launch device kernel;
    for (int i = 0; i < iteration; i++) {

        matrix_X_Vector_kernal(o_a, o_b, o_c, o_d, row);
    }

    duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "timer for GPU printf: " << duration << '\n';

    // Copy result to the host
    o_c.copyTo(x_new);

    init_zero(x, row);

    start = clock();

    jacobi_method_cpu(a, x, b, x_new2, row, iteration);
    duration2 = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "timer for CPU printf: " << duration2 << '\n';

    cout << "speedup  : " << duration2 / duration << '\n';

    compareTwoVector(x_new, x_new2, row);

    // Free host memory
    delete [] a;
    delete [] x_new;
    delete [] b;
    delete [] x_new2;
    delete [] x;
}

void relaxation_interpolation_vector(int row, occa::memory o_a, occa::memory o_b, occa::memory o_ab, occa::device device) {
    float *v  = new float[row * row];
    float *result = new float[(row * 2) + 1];
    float *result_cpu = new float[(row * 2) + 1];

    makeVector(v, row);

    init_zero(result, (row * 2) + 1);
    init_zero(result_cpu, (row * 2) + 1);

    occa::kernel relexation;
    occa::memory o_c;
    // Allocate memory on the device
    o_a  = device.malloc((row) * sizeof(float));
    o_b  = device.malloc(((row * 2) + 1) * sizeof(float));


    // Compile the kernel at run-time
    relexation = device.buildKernel("matrix_X_matrix.okl",
                                    "reduction_interpolation_vector_gpu");

    // Copy memory to the device
    o_a.copyFrom(v);

    clock_t start;
    double duration, duration2;
    start = clock();

    //    Launch device kernel;
    relexation(o_a, row, o_b);

    duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "timer for GPU printf: " << duration << '\n';

    // Copy result to the host
    o_b.copyTo(result);
    start = clock();

    reduction_interpolation_vector(v, row, result_cpu);
    duration2 = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "timer for CPU printf: " << duration2 << '\n';

    cout << "speedup  : " << duration2 / duration << '\n';
    compareTwoVector(result_cpu, result, (row * 2) + 1);

    // Free host memory
    delete [] v;
    delete [] result_cpu;
    delete [] result;
}
void relaxation_reduction_vector(int row, occa::memory o_a, occa::memory o_b, occa::memory o_ab, occa::device device) {
    float *v  = new float[(row * 2) + 1];
    float *result = new float[row];
    float *result_cpu = new float[row];

    makeVector(v, (row * 2) + 1);

    for (int i = 0; i < row; i++) {
        result[i] = 0;
        result_cpu[i] = 0;
    }

    occa::kernel relexation;
    occa::memory o_c;
    // Allocate memory on the device
    o_a  = device.malloc(((row * 2) + 1) * sizeof(float));
    o_b  = device.malloc(row * sizeof(float));

    // Compile the kernel at run-time
    relexation = device.buildKernel("matrix_X_matrix.okl",
                                    "reduction_reduction_vector_gpu");

    // Copy memory to the device
    o_a.copyFrom(v);
    clock_t start;
    double duration, duration2;
    start = clock();

    //    Launch device kernel;
    relexation(o_a, row, o_b);


    duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "timer for GPU printf: " << duration << '\n';

    // Copy result to the host
    o_b.copyTo(result);


    start = clock();

    reduction_vector(v, row, result_cpu);
    duration2 = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "timer for CPU printf: " << duration2 << '\n';

    cout << "speedup  : " << duration2 / duration << '\n';
    compareTwoVector(result_cpu, result, row);

    // Free host memory
    delete [] v;
    delete [] result_cpu;
    delete [] result;
}
void reduction_interpolation_dense_matrix(int row, occa::memory o_a, occa::memory o_b, occa::memory o_ab, occa::device device) {
    float *m  = new float[row * row];
    float *result = new float[(row / 2) * (row / 2)];
    float *result_cpu = new float[(row / 2) * (row / 2)];

    makeMatrix(m, row, row);

    init_zero(result, (row / 2) * (row / 2));
    init_zero(result_cpu, (row / 2) * (row / 2));

    occa::kernel relexation;
    occa::memory o_c;
    // Allocate memory on the device
    o_a  = device.malloc((row * row) * sizeof(float));
    o_b  = device.malloc((row / 2) * (row / 2) * sizeof(float));
    o_c  = device.malloc((row * (row / 2)) * sizeof(float));

    // Compile the kernel at run-time
    relexation = device.buildKernel("matrix_X_matrix.okl",
                                    "reduction_interpolation_reduction_matrix_gpu");

    // Copy memory to the device
    o_a.copyFrom(m);
    //    o_b.copyFrom(x);
    //
    clock_t start;
    double duration, duration2;
    start = clock();

    //    Launch device kernel;

    relexation(o_a, row, o_c, o_b);


    duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "timer for GPU printf: " << duration << '\n';

    // Copy result to the host
    o_b.copyTo(result);


    start = clock();

    interpolation_reduction_matrix(m, row, result_cpu);
    duration2 = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "timer for CPU printf: " << duration2 << '\n';

    cout << "speedup  : " << duration2 / duration << '\n';
    compareTwoMatrix(result_cpu, result, row / 2, row / 2);

    // Free host memory
    delete [] m;
    delete [] result_cpu;
    delete [] result;
}



void multigrid_method(float a[], float x[], float b[], int recursion, int row, int alpha) {


    if (recursion == 0 || row / 2 <= 3) {

        Gauss_elmination_cpu(a, b, x, row);
        return ;
    }

    float *x_new2 = new float[row];

    init_zero(x_new2, row);

    jacobi_method_cpu(a, x, b, x_new2, row, alpha);


    float *b2h = new float[row / 2];
    float *res1 = new float[row];
    float *x_new2h = new float[row];
    float *a2h = new float[(row / 2) * (row / 2)];

    init_zero(b2h, row / 2);
    init_zero(res1, row);
    init_zero(x_new2h, row);
    init_zero(a2h, (row / 2) * (row / 2));

    matrix_x_vector(row, row, x, a, x_new2h);

    add_sub_vector(b, x_new2h, res1, row,  -1);

    reduction_vector(res1, (row / 2), b2h);

    init_zero(x_new2h, row);

    interpolation_reduction_matrix(a, row, a2h);

    multigrid_method(a2h, x_new2h, b2h, recursion - 1, row / 2, alpha);

    float * res_int = new float[(row * 2) + 1];

    init_zero(res_int, (row * 2) + 1);

    reduction_interpolation_vector(x_new2h, row, res_int);

    add_sub_vector(x, res_int, x, row,  1);

    init_zero(x_new2, row);

    jacobi_method_cpu(a, x, b, x_new2, row, alpha);


    delete [] x_new2h;
    delete [] b2h;
    delete [] res1;
    delete [] a2h;
    delete [] res_int;
    delete [] x_new2;
}

void dense_Matrix_Vector_Multiplication_call_gpu(int row, occa::memory o_a, occa::memory o_b, occa::memory o_c, occa::device device) {

    occa::kernel matrix_X_Vector_kernal;
    // Compile the kernel at run-time
    matrix_X_Vector_kernal = device.buildKernel("matrix_X_matrix.okl",
                             "matrix_x_Vectors");

    // Launch device kernel
    matrix_X_Vector_kernal(row, row, o_a, o_b, o_c);


}

void add_sub_call_gpu(int row, occa::memory o_a, occa::memory o_b, occa::memory o_c, occa::device device, int oper) {

    occa::kernel matrix_X_Vector_kernal;
    // Compile the kernel at run-time
    matrix_X_Vector_kernal = device.buildKernel("matrix_X_matrix.okl",
                             "add_sub_vector_gpu");

    // Launch device kernel
    matrix_X_Vector_kernal(o_a, o_b, o_c, row, oper);

}
void jacobi_method_call_gpu(int row, occa::memory o_a, occa::memory o_b, occa::memory o_x, occa::memory o_d, occa::device device,  int iteration) {


    occa::kernel matrix_X_Vector_kernal;

    // Compile the kernel at run-time
    matrix_X_Vector_kernal = device.buildKernel("matrix_X_matrix.okl",
                             "jacobi_method_gpu");

    //    Launch device kernel;
    for (int i = 0; i < iteration; i++) {
        matrix_X_Vector_kernal(o_a, o_b, o_x, o_d, row);
    }

}

void gauss_elmination_call_gpu(int row, occa::memory o_a, occa::memory o_b, occa::memory o_c, occa::device device) {



    occa::kernel matrix_X_Vector_kernal;
    // Compile the kernel at run-time
    matrix_X_Vector_kernal = device.buildKernel("matrix_X_matrix.okl",
                             "gauss_elmination_gpu_forward");

    // Launch device kernel
    for (int i = 0; i < row - 1; i++) {
        matrix_X_Vector_kernal(o_a, o_b, o_c, row, i);
    }
    matrix_X_Vector_kernal = device.buildKernel("matrix_X_matrix.okl",
                             "gauss_elmination_gpu_backward");

    for (int i = row - 1; i >= 0; i--) {
        matrix_X_Vector_kernal(o_a, o_b, o_c, row, i);
    }
}

void relaxation_reduction_vector(int row, occa::memory o_b, occa::memory o_b2h, occa::device device) {

    occa::kernel relexation;

    // Compile the kernel at run-time
    relexation = device.buildKernel("matrix_X_matrix.okl",
                                    "reduction_reduction_vector_gpu");

    //    Launch device kernel;
    relexation(o_b, row, o_b2h);
}

void reduction_interpolation_reduction_matrix_call_gpu(int row, occa::memory o_a, occa::memory o_a2h, occa::device device) {

    occa::kernel relexation;
    occa::memory o_temp;
    float *result_emp = new float[row * (row / 2)];
    o_temp  = device.malloc((row * (row / 2)) * sizeof(float));
    // Compile the kernel at run-time
    relexation = device.buildKernel("matrix_X_matrix.okl",
                                    "reduction_interpolation_reduction_matrix_gpu");
    //    Launch device kernel;
    relexation(o_a, row, o_temp, o_a2h);
}

void relaxation_interpolation_vector_call_gpu(int row, occa::memory o_x2h, occa::memory o_x, occa::device device) {

    occa::kernel relexation;

    // Compile the kernel at run-time
    relexation = device.buildKernel("matrix_X_matrix.okl",
                                    "reduction_interpolation_vector_gpu");

    //    Launch device kernel;

    relexation(o_x2h, row, o_x);
}
//======================== multigrid dense matrix ===========================================

void multigrid_method_gpu(int row, occa::memory o_a, occa::memory o_b, occa::memory o_x, occa::device device, int recursion, int alpha) {
    if (recursion == 0 || row / 2 <= 3) {
        gauss_elmination_call_gpu(row, o_a, o_b, o_x, device);
        return;
    }
    occa::memory o_d, o_b2h, o_x2h, o_a2h, o_res, o_res2, o_res_result2h;
    // Allocate memory on the device

    o_d  = device.malloc(row * sizeof(float));
    o_b2h  = device.malloc((row / 2) * sizeof(float));
    o_x2h  = device.malloc((row / 2) * sizeof(float));
    o_res  = device.malloc(row * sizeof(float));
    o_res2  = device.malloc(row * sizeof(float));
    o_a2h  = device.malloc((row / 2) * (row / 2) * sizeof(float));


    jacobi_method_call_gpu(row, o_a, o_b, o_x, o_d, device, alpha);

    dense_Matrix_Vector_Multiplication_call_gpu(row, o_a, o_x, o_res2, device);

    add_sub_call_gpu(row, o_b, o_res2, o_res, device, -1);

    relaxation_reduction_vector(row / 2, o_res, o_b2h, device);

    reduction_interpolation_reduction_matrix_call_gpu(row, o_a, o_a2h, device);

    multigrid_method_gpu(row / 2, o_a2h, o_b2h, o_x2h, device, recursion - 1, alpha);

    o_res_result2h  = device.malloc(((row * 2) + 1) * sizeof(float));

    relaxation_interpolation_vector_call_gpu(row, o_x2h, o_res_result2h, device);

    add_sub_call_gpu(row, o_x, o_res_result2h, o_x, device, 1);

    jacobi_method_call_gpu(row, o_a, o_b, o_x, o_d, device, alpha);

}


void multigrid_method_once(int row, occa::memory o_a, occa::memory o_b, occa::memory o_c, occa::device device, int recursion) {

    float *a  = new float[row * row];
    float *b = new float[row];
    float *a2  = new float[row * row];
    float *b2 = new float[row];
    float *x = new float[row];
    float *x2 = new float[row];
    float *x_new2h = new float[row];

    makeSPDmatrix(a, row, row);
    makeVector(b, row);

    //==========================make csv file ==================================

    ofstream aMatrix, bvector;
    aMatrix.open( "amatrix.csv");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < row; j++) {
            if (j + 1 == row) {
                aMatrix << a[i * row + j];
            }
            else {
                aMatrix << a[i * row + j] << ",";
            }
        }
        aMatrix << "\n ";
    }
    aMatrix.close();
    bvector.open( "bvector.csv");
    for (int i = 0; i < row; i++) {
        bvector << b[i];

        bvector << "\n ";
    }
    bvector.close();
    //==========================end ============================================

    for (int i = 0; i < row * row; i++) {
        a2[i] = a[i];
    }
    for (int i = 0; i < row; i++) {
        b2[i] = b[i];
    }

    init_zero(x, row);
    init_zero(x2, row);
//    print_matrix(a,row,row);
//    print_vector(row, b);


    // Allocate memory on the device
    o_a  = device.malloc((row * row) * sizeof(float));
    o_b  = device.malloc(row * sizeof(float));
    o_c  = device.malloc(row * sizeof(float));

    // Copy memory to the device
    o_a.copyFrom(a);
    o_b.copyFrom(b);
    o_c.copyFrom(x);

    clock_t start;

//    Launch device kernel;
    double duration, duration2;
    start = clock();
    for (int i = 0; i < 30; i++) {
        multigrid_method_gpu(row, o_a, o_b, o_c, device, recursion, 10);
        // calculate absolute value residual r = ||Ax-b||
        // if < epsilon : break



        init_zero(x_new2h, row);

        occa::memory o_res, o_res2;

        o_res  = device.malloc(row * sizeof(float));
        o_res2  = device.malloc(row * sizeof(float));

        dense_Matrix_Vector_Multiplication_call_gpu(row, o_a, o_c, o_res, device);

        add_sub_call_gpu(row, o_b, o_res, o_res2, device, -1);

        o_res2.copyTo(x_new2h);

        float r_abs = norm(x_new2h, row);

        cout << "residual steps gpu  = " << r_abs << endl;

        if (r_abs < 10e-5) {
            cout << "residual  = " << r_abs << endl;
            break;
        }

    }


    duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "timer for GPU printf: " << duration << '\n';
//    print_vector(row, x);

    o_c.copyTo(x);

    start = clock();

    for (int i = 0; i < 30; i++) {
        multigrid_method(a2, x2, b2, recursion, row, 10);

        float *x_new2h2 = new float[row];

        init_zero(x_new2h, row);
        init_zero(x_new2h2, row);

        matrix_x_vector(row, row, x2, a2, x_new2h);

        add_sub_vector(x_new2h, b2, x_new2h2, row,  -1);

        float r_abs = norm(x_new2h2, row);

        cout << "residual steps i cpu  = " << r_abs << endl;

        if (r_abs < 10e-5) {
            cout << "residual  = " << r_abs << endl;
            break;
        }


        delete [] x_new2h2;

    }
    duration2 = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "timer for CPU printf: " << duration2 << '\n';

    cout << "speedup  : " << duration2 / duration << '\n';

    //==========================make csv file ==================================

    ofstream xvector, x_cpu;
    xvector.open( "x_gpu.csv");
    x_cpu.open( "x_cpu.csv");
    for (int i = 0; i < row; i++) {
        xvector << x[i];
        x_cpu << x2[i];
        xvector << "\n ";
        x_cpu << "\n";
    }
    xvector.close();
    x_cpu.close();
    //==========================end ============================================

    int cnt = 0  ;
    float max_relative = 0.0 ;

    for ( int i = 0 ; i < row ; i++ ) {

        float sol_GPU = x[i]; // the solution from CPU and GPU
        float sol_CPU = x2[i];

        if (! comparefloat(sol_GPU, sol_CPU)) { // use the function to verify if two float are equal
            cnt ++;
        }

        float relative_error = std::abs(sol_GPU - sol_CPU) / sol_GPU;

        if ( relative_error > max_relative  )
            max_relative = relative_error;

    }

    cout << "number of error: " << cnt << endl ;
    cout << "max relative error: " << max_relative << endl ;

    compareTwoVector(x, x2, row);

    delete [] a;
    delete [] b;
    delete [] x;
    delete [] a2;
    delete [] b2;
    delete [] x2;
    delete [] x_new2h;
}

//======================== multigrid sparse matrix ===========================================

void row_vector(occa::memory o_a_row, int row, int size, occa::memory o_row_number) {
    int * row_number = new int [row + 1];
    int * rowe = new int [size];

    o_a_row.copyTo(rowe);

    int k = 0;
    row_number[0] = 0;

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < size; j++) {
            if (rowe[j] != -1 && i == rowe[j] && i != j) {
                k ++;
                rowe[j] = -1;
            }
        }
        row_number[i + 1] = k + 1;
    }

    o_row_number.copyFrom(row_number);

    delete [] rowe;
    delete [] row_number;
}

void sparse_Matrix_Vector_Multiplication_call_gpu(int row, int size, occa::memory o_a, occa::memory o_a_row, occa::memory o_a_col, occa::memory o_b, occa::memory o_ab, occa::device device) {


    occa::kernel sparse_matrix_X_Vector_kernal;

    // Compile the kernel at run-time
    sparse_matrix_X_Vector_kernal = device.buildKernel("matrix_X_matrix.okl",
                                    "sparse_matrix_x_Vectors_multigrid"
                                                      );


    // Launch device kernel
    sparse_matrix_X_Vector_kernal(size, row, o_a, o_b, o_a_row, o_a_col, o_ab);

}

void jacobi_method_call_gpu_sparse_matrix(int row, occa::memory o_a, occa::memory o_a_col, occa::memory o_a_row, occa::memory o_b, occa::memory o_x, occa::device device,  int iteration, int size) {


    occa::kernel matrix_X_Vector_kernal;

    matrix_X_Vector_kernal = device.buildKernel("matrix_X_matrix.okl",
                             "init_zero_gpu");

    matrix_X_Vector_kernal(o_x, row);

    occa::memory o_x_new25;

    o_x_new25  = device.malloc((row) * sizeof(float));

    // Compile the kernel at run-time
    matrix_X_Vector_kernal = device.buildKernel("matrix_X_matrix.okl",
                             "jacobi_method_gpu_sparse_matrix");

//        Launch device kernel;
    for (int k = 0; k < iteration; k++) {

        matrix_X_Vector_kernal(o_a, o_a_col, o_a_row, o_b, o_x, o_x_new25, row, size);
    }

}


int reduction_interpolation_reduction_sparse_matrix_call_gpu(int row, occa::memory o_a, occa::memory o_a_col, occa::memory o_a_row,  occa::memory o_result2, occa::memory o_result2_row, occa::memory o_result2_col, occa::device device, int size) {

    occa::kernel reduction;
    occa::memory o_temp,  o_size, o_temp2;
    int len2 = row / 2;
    o_temp  = device.malloc((row * len2) * sizeof(float));
    o_temp2  = device.malloc((len2 * len2) * sizeof(float));
    o_size  = device.malloc(2 * sizeof(int));
    // Compile the kernel at run-time
    reduction = device.buildKernel("matrix_X_matrix.okl",
                                   "interpolation_matrix_sparse_matrix_gpu");
    //    Launch device kernel;
    reduction(o_a, o_a_col, o_a_row, size, row, len2, o_temp);

    reduction = device.buildKernel("matrix_X_matrix.okl",
                                   "reduction_matrix_sparse_matrix_gpu");

    int point = 0;

    reduction(size, row, o_result2, o_result2_row, o_result2_col, o_temp2, o_temp, o_size, len2);

    int *ab = new int[2];
    o_size.copyTo(ab);
    point = ab[0];
    delete [] ab;


    return point;

}

void sparse_vector_to_matrix_gpu_call(int row, occa::memory o_a, occa::memory o_a_row, occa::memory o_a_col, occa::device device, int size_a, occa::memory o_aa) {

    occa::kernel matrix_X_Vector_kernal;

    // Compile the kernel at run-time
    matrix_X_Vector_kernal = device.buildKernel("matrix_X_matrix.okl",
                             "sparse_vector_to_matrix_gpu");

    //Launch device kernel;
    matrix_X_Vector_kernal(o_a, o_a_row, o_a_col, row, size_a, o_aa);
}



void multigrid_method_gpu_sparse_matrix(int row, occa::memory o_a, occa::memory o_a_row, occa::memory o_a_col, occa::memory o_b, occa::memory o_x, occa::device device, int recursion, int alpha, int size_a) {
    if (recursion == 0 || row / 2 <= 3 || size_a < row ) {
        occa::memory o_aa;

        o_aa = device.malloc((row * row) * sizeof(float));

        sparse_vector_to_matrix_gpu_call(row, o_a, o_a_row, o_a_col, device, size_a, o_aa);

        gauss_elmination_call_gpu(row, o_aa, o_b, o_x, device);

        return;
    }

    occa::memory o_b2h, o_x2h, o_a2h, o_a2h_row, o_a2h_col, o_res, o_res2, o_res_result2h, o_row_number ;
    // Allocate memory on the device

    o_b2h  = device.malloc((row / 2) * sizeof(float));
    o_x2h  = device.malloc((row / 2) * sizeof(float));
    o_res  = device.malloc(row * sizeof(float));
    o_res2  = device.malloc(row * sizeof(float));

    int size_non = (size_a + row);
    o_a2h  = device.malloc(size_non * sizeof(float));
    o_a2h_row  = device.malloc(size_non * sizeof(int));
    o_a2h_col  = device.malloc(size_non * sizeof(int));


    jacobi_method_call_gpu_sparse_matrix(row, o_a, o_a_col, o_a_row, o_b, o_x, device,  alpha, size_a);

    sparse_Matrix_Vector_Multiplication_call_gpu(row, size_a, o_a, o_a_row, o_a_col, o_x, o_res, device);

    add_sub_call_gpu(row, o_b, o_res, o_res2, device, -1);

    relaxation_reduction_vector(row / 2, o_res2, o_b2h, device);

    size_non =  reduction_interpolation_reduction_sparse_matrix_call_gpu(row, o_a, o_a_col, o_a_row,  o_a2h, o_a2h_row, o_a2h_col, device, size_a);


    multigrid_method_gpu_sparse_matrix(row / 2, o_a2h, o_a2h_row, o_a2h_col, o_b2h, o_x2h, device, recursion - 1, alpha, size_non);

    o_res_result2h  = device.malloc(((row * 2) + 1) * sizeof(float));

    relaxation_interpolation_vector_call_gpu(row, o_x2h, o_res_result2h, device);

    add_sub_call_gpu(row, o_x, o_res_result2h, o_x, device, 1);

    jacobi_method_call_gpu_sparse_matrix(row, o_a, o_a_col, o_a_row, o_b, o_x, device,  alpha, size_a);
}



void multigrid_method_sparse_matrix(float a_non_zero[], int a_col_number[], int a_row[], float x[], float b[], int recursion, int row, int alpha, int size_a) {

    if (recursion == 0 || row / 2 < 3 ) {
        int *point = new int[size_a];
        float *aa  = new float [row * row];
        for (int i = 0; i < size_a; i++) {
            point[i] = a_col_number[i] * row + a_row[i];
        }

        vectorToMatrix(row, row, size_a, a_non_zero, aa, point);

        Gauss_elmination_cpu(aa, b, x, row);

        delete [] point;
        delete [] aa;

        return ;
    }


    float *x_new2 = new float[row];
    init_zero(x_new2, row);

    jacobi_method_cpu_sparse_matrix(a_non_zero, a_col_number, a_row, b, x, x_new2, row, alpha, size_a);


    float *b2h = new float[row / 2];
    float *res1 = new float[row];
    float *x_new2h = new float[row];

    init_zero(b2h, row / 2);
    init_zero(res1, row);
    init_zero(x_new2h, row);

    sparse_matrix_x_vector(row, size_a, x, a_row, a_col_number, a_non_zero, x_new2h);
    add_sub_vector(b, x_new2h, res1, row,  -1);
    reduction_vector_sparse(res1, row, b2h);

    init_zero(x_new2h, row);

    int size_non = (size_a + row) * 3;

    float *a2h = new float[size_non];
    int *a2h_row = new int[size_non];
    int *a2h_col = new int[size_non];

    init_zero(a2h, size_non);
    init_zero(a2h_row, size_non);
    init_zero(a2h_col, size_non);


    size_non = interpolation_reduction_matrix_sparse_matrix(a_non_zero, a_col_number, a_row, size_a, row, a2h, a2h_row, a2h_col);

    multigrid_method_sparse_matrix(a2h, a2h_col, a2h_row, x_new2h, b2h, recursion - 1, row / 2, alpha, size_non);

    float * res_int = new float[(row * 2) + 1];

    init_zero(res_int, (row * 2) + 1);

    reduction_interpolation_vector(x_new2h, row, res_int);

    add_sub_vector(x, res_int, x, row,  1);

    init_zero(x_new2, row);

    jacobi_method_cpu_sparse_matrix(a_non_zero, a_col_number, a_row, b, x, x_new2, row, alpha, size_a);


    delete [] x_new2h;
    delete [] b2h;
    delete [] res1;
    delete [] res_int;
    delete [] a2h;
    delete [] a2h_col;
    delete [] a2h_row;
}







void multigrid_method_once_sparse_matrix(int row, occa::memory o_a, occa::memory o_b, occa::memory o_x, occa::device device, int recursion) {

    int alpha = 10;

    float *a  = new float[row * row];
    float *b = new float[row];
    float *x = new float[row];
    float *x_new2 = new float[row];

    makeSPDmatrix_sparse(a, row, row);
    makeVector(b, row);

    //==========================make csv file ==================================

    ofstream aMatrix, bvector;
    aMatrix.open( "amatrix.csv");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < row; j++) {
            if (j + 1 == row) {
                aMatrix << a[i * row + j];
            }
            else {
                aMatrix << a[i * row + j] << ",";
            }
        }
        aMatrix << "\n ";
    }
    aMatrix.close();
    bvector.open( "bvector.csv");
    for (int i = 0; i < row; i++) {
        bvector << b[i];

        bvector << "\n ";
    }
    bvector.close();
    //==========================end ============================================

    init_zero(x, row);
    init_zero(x_new2, row);

    occa::memory o_a_row, o_a_col;

    //    get length of a none zero elements
    int size_a = size_non_zero_marix(a, row, row);

    float * a_non_zero = new float[size_a];
    int * a_col_number = new int[size_a];
    int * a_row = new int[size_a];

    init_zero(a_non_zero, size_a);
    init_zero(a_row, size_a);
    init_zero(a_col_number, size_a);
    makeSparseMatrix_To_vectors(a, row, row, a_non_zero, a_row, a_col_number, size_a);

    float *x2 = new float[row];
    init_zero(x2, row);

    // Allocate memory on the device
    o_a  = device.malloc(size_a * sizeof(float));
    o_a_row  = device.malloc(size_a * sizeof(int));
    o_a_col  = device.malloc(size_a * sizeof(int));
    o_b  = device.malloc(row * sizeof(float));
    o_x  = device.malloc(row * sizeof(float));

    o_a.copyFrom(a_non_zero);
    o_a_row.copyFrom(a_row);
    o_a_col.copyFrom(a_col_number);
    o_b.copyFrom(b);
    o_x.copyFrom(x);


    clock_t start, start2;

    double duration, duration2;
    start = clock();
    for (int i = 0; i < 10; i++) {
        multigrid_method_gpu_sparse_matrix(row, o_a, o_a_row, o_a_col, o_b, o_x, device, recursion, alpha, size_a);

        occa::memory o_res, o_res2;

        o_res  = device.malloc(row * sizeof(float));
        o_res2  = device.malloc(row * sizeof(float));

        sparse_Matrix_Vector_Multiplication_call_gpu(row, size_a, o_a, o_a_row, o_a_col, o_x, o_res, device);

        add_sub_call_gpu(row, o_b, o_res, o_res2, device, -1);

        o_res2.copyTo(x);

        float r_abs = norm(x, row);

        cout << "residual steps gpu  = " << r_abs << endl;

        if (r_abs < 10e-5) {
            cout << "residual  = " << r_abs << endl;
            break;
        }

    }



    duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "timer for GPU printf: " << duration << '\n';

    start2 = clock();

    for (int ii = 0; ii < 10; ii++) {

        multigrid_method_sparse_matrix(a_non_zero, a_col_number, a_row, x2, b, recursion, row, alpha, size_a);

        float *x_new2h2 = new float[row];

        init_zero(x_new2, row);
        init_zero(x_new2h2, row);

        sparse_matrix_x_vector(row, size_a, x2, a_row, a_col_number, a_non_zero, x_new2);

        add_sub_vector(x_new2, b, x_new2h2, row,  -1);

        float r_abs = norm(x_new2h2, row);

        cout << "residual steps i cpu  = " << r_abs << endl;

        if (r_abs < 10e-5) {
            cout << "residual  = " << r_abs << endl;
            break;
        }


        delete [] x_new2h2;

    }
    duration2 = ( clock() - start2 ) / (double) CLOCKS_PER_SEC;
    cout << "timer for CPU printf: " << duration2 << '\n';

    cout << "speedup  : " << duration2 / duration << '\n';

    o_x.copyTo(x);

    int cnt = 0  ;
    float max_relative = 0.0 ;

    for ( int i = 0 ; i < row ; i++ ) {

        float sol_GPU = x[i]; // the solution from CPU and GPU
        float sol_CPU = x2[i];

        if (! comparefloat(sol_GPU, sol_CPU)) { // use the function to verify if two float are equal
            cnt ++;
        }

        float relative_error = std::abs(sol_GPU - sol_CPU) / sol_GPU;

        if ( relative_error > max_relative  )
            max_relative = relative_error;

    }

    //==========================make csv file ==================================

    ofstream xvector, x_cpu;
    xvector.open( "x_gpu.csv");
    x_cpu.open( "x_cpu.csv");
    for (int i = 0; i < row; i++) {
        xvector << x[i];
        x_cpu << x2[i];
        xvector << "\n ";
        x_cpu << "\n";
    }
    xvector.close();
    x_cpu.close();
    //==========================end ============================================

    cout << "number of error: " << cnt << endl ;
    cout << "max relative error: " << max_relative << endl ;

    compareTwoVector(x, x2, row);

    delete [] a;
    delete [] b;
    delete [] x;
    delete [] a_non_zero;
    delete [] a_row;
    delete [] a_col_number;
    delete [] x2;
    delete [] x_new2;

}
