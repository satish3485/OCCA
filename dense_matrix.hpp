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
// dense_Matrix_Vector_Multiplication : This function dot production of matrix and vector
void dense_Matrix_Vector_Multiplication(int row, int colume, occa::memory o_a, occa::memory o_b, occa::memory o_ab, occa::device device);

// dense_Matrix_Multiplication : This function dot product of matrix and matix
void dense_Matrix_Multiplication(int row, int colume, int row2ndmatrix, int colume2ndmatrix, occa::memory o_a, occa::memory o_b, occa::memory o_ab, occa::device device, int operation , int mul_And_Add_Sub);

// vector_reduction : This function is dot product of vector and vector
void vector_reduction(occa::device device, int entries);
