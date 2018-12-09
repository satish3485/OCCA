/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 SATISH KUMAR
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

#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
#define ids 0
#elif defined(__linux__)
#define ids 0
#elif defined(__APPLE__) && defined(__MACH__)
#define ids 1
#elif defined(unix) || defined(__unix__) || defined(__unix)
#define ids 0
#else
#define ids 0
#endif


#include <iostream>

#include <ctime>
#include <occa.hpp>
#include <cmath>
#include "cpu.hpp"
#include "cstdlib"
#include "dense_matrix.hpp"
#include "sparse_matrix.hpp"
#include "multi_grid.hpp"
#include <fstream>
#include <stdio.h>

occa::json parseArgs(int argc, const char **argv);
occa::json parseArgs(int argc, const char **argv) {
    // Note:
    //   occa::cli is not supported yet, please don't rely on it
    //   outside of the occa examples
    occa::cli::parser parser;
    parser
    .withDescription(
        "Example adding two vectors"
    )
    .addOption(
        occa::cli::option('d', "device",
                          "Device properties (default: \"mode: 'OpenCL'\")")
        .withArg()
        .withDefaultValue("mode: 'OpenCL'")
    )
    .addOption(
        occa::cli::option('v', "verbose",
                          "Compile kernels in verbose mode")
    );

    occa::json args = parser.parseArgs(argc, argv);
    occa::settings()["kernel/verbose"] = args["options/verbose"];

    return args;
}


int main(int argc, const char **argv) {
    occa::printModeInfo();
    /*
     Try running with OCCA_VERBOSE=1 or set
     verbose at run-time with:
     occa::settings()["kernel/verbose"] = true;
     */
    occa::json args = parseArgs(argc, argv);

//    occa::json args = parseArgs(argc, argv);

    occa::device device;
    occa::memory o_a, o_b, o_ab;

    int gpu = 0;

    cout << "" << endl;
    cout << "1. OpenCL press 1 & enter" << endl;
    cout << "2. CUDA press 2 & enter" << endl;
    cout << "3. OpenMP press 3 & enter" << endl;
    cout << "4. Threads press 4 & enter" << endl;
    cout << "" << endl;
    cout << "Please choose your choice :";
    cin >> gpu;

    //---[ Device setup with string flags ]-------------------
//      device.setup("mode: 'Serial'");
    if (gpu == 1) {
        if (ids == 1) {
            device.setup("mode       : 'OpenCL', "
                         "platform_id : 0, "
                         "device_id   : 1" );
        } else {
            device.setup("mode       : 'OpenCL', "
                         "platform_id : 0, "
                         "device_id   : 0" );
        }
    }
    else if (gpu == 2) {
        if (ids == 1) {
            device.setup("mode     : 'CUDA', "
                         "device_id : 1");
        } else {
            device.setup("mode     : 'CUDA', "
                         "device_id : 0");
        }

    }
    else if (gpu == 3) {
        device.setup("mode     : 'OpenMP', "
                     "schedule : 'compact', "
                     "chunk    : 10");
    }
    else if (gpu == 4) {
        device.setup("mode        : 'Threads', "
                     "threadCount : 4, "
                     "schedule    : 'compact', "
                     "pinnedCores : [0, 0, 1, 1]");
    }
    else {
        return 0;
    }
//    ========================================================


    int operation = 1;
    int mul_And_Add_Sub = 0;
    cout << "" << endl;
    cout << "1. Dense Matrix Vector Multiplication press 1 & enter" << endl;

    cout << "2. Dense Matrix Matrix Operation press 2 & enter" << endl;
    cout << "3. Sparse Matrix Vector Multiplication press 3 & enter" << endl;
    cout << "4. Sparse Matrix Matrix Operation press 4 & enter" << endl;
    cout << "5. Vector Vector Multiplication press 5 & enter" << endl;
    cout << "6. Gauss Elimination method press 6 & enter" << endl;
    cout << "7. Jacobi method method press 7 & enter" << endl;
    cout << "8. Reduction and Interpolation matrix press 8 & enter" << endl;
    cout << "9. Interpolation vector press 9 & enter" << endl;
    cout << "10. Reduction vector press 10 & enter" << endl;
    cout << "11. Multi-grid method press 11 & enter" << endl;
    cout << "" << endl;
    int oper = 0;
    cout << "Please choose your choice : ";
    cin >> oper;
    if (oper == 1) {
        int row = 0;
        int colume = 0;
        int vector_size = 0;
        cout << "Please enter number of row : ";
        cin >> row;
        cout << "Please enter number of colume : ";
        cin >> colume;

        vector_size = colume;
        if (colume != vector_size) {
            cout << "Colume of matrix and vector size must same" << endl;
            return 0;
        } else {
            dense_Matrix_Vector_Multiplication(row, colume, o_a, o_b, o_ab, device);
        }
    }

    else if (oper == 2) {
        int row = 0;
        int colume = 0;
        int row2ndmatrix = 0;
        int colume2ndmatrix = 0;


        int operation = 1;
        int mul_And_Add_Sub = 0;
        cout << "1. Multiplication press 1 & enter " << endl;
        cout << "2. Addition press 2 & enter " << endl;
        cout << "3. Subtraction press 3 & enter " << endl;

        cout << "Please choose your choice : ";
        cin >> operation;

        if (operation == 1) {
            cout << "Please enter number of row : ";
            cin >> row;
            cout << "Please enter number of colume : ";
            cin >> colume;

            row2ndmatrix = colume;
            cout << "Please enter number of 2nd matrix colume : ";
            cin >> colume2ndmatrix;
            dense_Matrix_Multiplication(row, colume, row2ndmatrix, colume2ndmatrix, o_a, o_b, o_ab, device, operation , 1);
        } else if (operation == 2) {
            cout << "Please enter number of row : ";
            cin >> row;
            cout << "Please enter number of colume : ";
            cin >> colume;
            row2ndmatrix = row;
            colume2ndmatrix = colume;

            dense_Matrix_Multiplication(row, colume, row2ndmatrix, colume2ndmatrix, o_a, o_b, o_ab, device, 1 , 0);
        } else if (operation == 3) {
            cout << "Please enter number of row : ";
            cin >> row;
            cout << "Please enter number of colume : ";
            cin >> colume;
            row2ndmatrix = row;
            colume2ndmatrix = colume;

            dense_Matrix_Multiplication(row, colume, row2ndmatrix, colume2ndmatrix, o_a, o_b, o_ab, device, -1 , 0);
        }
    }
    else if (oper == 3) {
        int row = 0;
        int colume = 0;
        int vector_size = 0;
        cout << "Please enter number of row : ";
        cin >> row;
        cout << "Please enter number of colume : ";
        cin >> colume;

        vector_size = colume;
        if (colume != vector_size) {
            cout << "Colume of matrix and vector size must same" << endl;
            return 0;
        } else {
            sparse_Matrix_Vector_Multiplication(row, colume, o_a, o_b, o_ab, device);
        }
    }
    else if (oper == 4) {
        int row = 0;
        int colume = 0;
        int row2ndmatrix = 0;
        int colume2ndmatrix = 0;

        int operation = 0;
        int mul_And_Add_Sub = 0;
        cout << "1. Multiplication press 1 & enter " << endl;
        cout << "2. Addition press 2 & enter " << endl;
        cout << "3. Subtraction press 3 & enter " << endl;

        cout << "Please choose your choice : ";
        cin >> operation;

        if (operation == 1) {
            cout << "Please enter number of row : ";
            cin >> row;
            cout << "Please enter number of colume : ";
            cin >> colume;

            row2ndmatrix = colume;
            cout << "Please enter number of 2nd matrix colume : ";
            cin >> colume2ndmatrix;
            sparse_Matrix_Matrix_Multiplication(row, colume, row2ndmatrix, colume2ndmatrix, o_a, o_b, o_ab, device);
        } else if (operation == 2) {
            cout << "Please enter number of row : ";
            cin >> row;
            cout << "Please enter number of colume : ";
            cin >> colume;
            row2ndmatrix = row;
            colume2ndmatrix = colume;

            sparse_Add_Sub_Matrix(row, colume, row2ndmatrix, colume2ndmatrix, o_a, o_b, o_ab, device, 1);
        }
        else if (operation == 3) {
            cout << "Please enter number of row : ";
            cin >> row;
            cout << "Please enter number of colume : ";
            cin >> colume;
            row2ndmatrix = row;
            colume2ndmatrix = colume;

            sparse_Add_Sub_Matrix(row, colume, row2ndmatrix, colume2ndmatrix, o_a, o_b, o_ab, device, -1);
        }
    }
    else if (oper == 5) {
        int vector_size = 0;
        cout << "Please enter vector length : ";
        cin >> vector_size;
        vector_reduction(device, vector_size);
    }

    else if (oper == 6) {
        int vector_size = 0;
        cout << "Please enter matrix length : ";
        cin >> vector_size;
        gauss_elmination(vector_size, o_a, o_b, o_ab, device);

    }
    else if (oper == 7) {
        int vector_size = 0;
        cout << "Please enter matrix length : ";
        cin >> vector_size;
        int recurs = 0;
        cout << "Number of iterations: ";
        cin >> recurs;
        jacobi_method(vector_size, o_a, o_b, o_ab, device, recurs);
    }
    else if (oper == 8) {
        int vector_size = 0;
        cout << "Please enter matrix length : ";
        cin >> vector_size;
        reduction_interpolation_dense_matrix(vector_size, o_a, o_b, o_ab, device);
    }
    else if (oper == 9) {
        int vector_size = 0;
        cout << "Please enter matrix length : ";
        cin >> vector_size;
        relaxation_interpolation_vector(vector_size, o_a, o_b, o_ab, device);
    }
    else if (oper == 10) {
        int vector_size = 0;
        cout << "Please enter matrix length : ";
        cin >> vector_size;
        relaxation_reduction_vector(vector_size, o_a, o_b, o_ab, device);
    }
    else if (oper == 11) {
        int choice = 0;
        cout << "1. Multi-grid method dense matrix press 1 & enter" << endl;
        cout << "2. Multi-grid method sparse matrix press 2 & enter" << endl;
        cin >> choice;
        if (choice == 1 || choice == 2) {
            int vector_size = 0;
            cout << "Please enter matrix length : ";
            cin >> vector_size;
            int recurs = 0;
            cout << "Number of recursion: ";
            cin >> recurs;
            if (choice == 1) {
                multigrid_method_once(vector_size, o_a, o_b, o_ab, device, recurs);
            }
            if (choice == 2) {
                multigrid_method_once_sparse_matrix(vector_size, o_a, o_b, o_ab, device,recurs);
            }
        }
    }
    return 0;
}
